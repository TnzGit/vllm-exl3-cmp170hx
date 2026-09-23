#!/usr/bin/env python3
"""Derive exact-length C2/C4 inputs from archived turns and freeze their IDs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
HELPER = HERE / "r0_c2_c4_mtp_matched_load.py"
SPEC = importlib.util.spec_from_file_location("r0_matched_load", HELPER)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

TOKENIZER_FILES = {
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "tokenizer.model", "spiece.model", "vocab.json",
    "vocab.txt", "merges.txt", "chat_template.jinja",
}
LEAD_WORDS = (
    "Amber", "Birch", "Cedar", "Dahlia", "Elm", "Flint", "Grove", "Harbor",
    "Iris", "Juniper", "Kestrel", "Linden", "Maple", "North", "Olive", "Prairie",
    "Quartz", "River", "Summit", "Timber", "Umber", "Valley", "Willow", "Yarrow",
)
FILLER_CANDIDATES = (" neutral", " context", " ordinary", " sample")
TURN_FILES = runner.TURN_FILES


class ManifestError(ValueError):
    """Input assets or requested deterministic prompts cannot satisfy contract."""


def _regular_tree(root_arg: Path) -> tuple[Path, list[Path]]:
    if root_arg.is_symlink():
        raise ManifestError(f"Path must not be a symlink: {root_arg}")
    root = root_arg.resolve(strict=True)
    if not root.is_dir():
        raise ManifestError(f"Path must be a directory: {root_arg}")
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ManifestError(f"Symlink in asset tree: {path}")
        if path.is_file():
            files.append(path)
    return root, files


def _tree_hash(root: Path, files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _file_hash(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ManifestError(f"Expected regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_tokenizer(path: Path):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ManifestError("transformers is required to load the explicit tokenizer path") from exc
    return AutoTokenizer.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=False
    )


def _encode(tokenizer: Any, text: str) -> list[int]:
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ManifestError("Tokenizer cannot encode deterministic prompt text") from exc
    if not isinstance(ids, list) or any(type(token) is not int or token < 0 for token in ids):
        raise ManifestError("Tokenizer returned invalid token IDs")
    return ids


def _decode(tokenizer: Any, ids: list[int]) -> str:
    try:
        return tokenizer.decode(
            ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ManifestError("Tokenizer cannot decode candidate prompt IDs") from exc


def derive_archived_request(
    tokenizer: Any,
    source: dict[str, Any],
    source_file: Path,
    source_label: str,
    target_tokens: int,
    lead_token: int,
    filler_id: int,
) -> tuple[list[int], str, dict[str, Any]]:
    source_ids = source.get("prompt_token_ids")
    query_span = source.get("query_span")
    facts = source.get("target_facts")
    if (not isinstance(source_ids, list) or any(type(t) is not int or t < 0 for t in source_ids)
            or not isinstance(query_span, list) or len(query_span) != 2
            or any(type(v) is not int for v in query_span)):
        raise ManifestError(f"Malformed archived token IDs or query span: {source_label}")
    query_start, query_end = query_span
    if (source.get("prompt_tokens") != len(source_ids) or query_start < 1
            or query_end <= query_start or query_end != len(source_ids)):
        raise ManifestError(f"Archived prompt/query boundaries are inconsistent: {source_label}")
    if not isinstance(facts, list) or len(facts) != 1 or not isinstance(facts[0], dict):
        raise ManifestError(f"Expected exactly one archived recovery fact: {source_label}")
    answer = facts[0].get("code")
    marker = facts[0].get("marker")
    if not isinstance(answer, str) or not answer or not isinstance(marker, str) or not marker:
        raise ManifestError(f"Archived recovery fact is malformed: {source_label}")
    query_ids = source_ids[query_start:query_end]
    decoded_query = _decode(tokenizer, query_ids)
    if decoded_query != source.get("query_text") or marker not in decoded_query:
        raise ManifestError(f"Decoded archived query does not match its recorded text: {source_label}")
    filler_count = target_tokens - len(source_ids)
    if filler_count < 0:
        raise ManifestError(f"Archived input exceeds target length: {source_label}")
    derived = source_ids[:query_start] + [filler_id] * filler_count + query_ids
    derived[0] = lead_token
    decoded = _decode(tokenizer, derived)
    if len(derived) != target_tokens or answer not in decoded or marker not in decoded:
        raise ManifestError(f"Derived input lost its archived recovery fact/query: {source_label}")
    return derived, answer, {
        "source_file": source_label,
        "source_file_sha256": _file_hash(source_file),
        "source_token_ids_sha256": runner.token_ids_sha256(source_ids),
        "source_prompt_tokens": len(source_ids),
        "source_query_span": query_span,
        "derived_query_span": [query_start + filler_count, query_start + filler_count + len(query_ids)],
        "replaced_token_0": {"from": source_ids[0], "to": lead_token},
        "lead_token_id": lead_token,
        "lead_token_text": _decode(tokenizer, [lead_token]),
        "filler_token_id": filler_id,
        "filler_count": filler_count,
        "insertion": "neutral filler IDs immediately before archived query span",
        "preserved_query_text": decoded_query,
        "recovery_marker": marker,
        "expected_single_recovery_code": answer,
        "input_classification": "derived; not historical byte-identical input",
    }


def make_manifest(args: argparse.Namespace, tokenizer_loader=_load_tokenizer) -> dict[str, Any]:
    model_root, model_files = _regular_tree(args.model_path)
    tokenizer_root, _ = _regular_tree(args.tokenizer_path)
    try:
        tokenizer_root.relative_to(model_root)
    except ValueError as exc:
        raise ManifestError(
            "Tokenizer path must be inside model path so its hash matches the runner's model-tree proof"
        ) from exc
    model_config = model_root / "config.json"
    model_pack_hash = _tree_hash(model_root, model_files)
    tokenizer_files = [path for path in model_files if path.name in TOKENIZER_FILES]
    if not tokenizer_files:
        raise ManifestError("Model tree has no recognized tokenizer assets")
    tokenizer_hash = _tree_hash(model_root, tokenizer_files)
    for path in tokenizer_files:
        try:
            path.relative_to(tokenizer_root)
        except ValueError as exc:
            raise ManifestError(
                "All recognized tokenizer assets must be under the explicit tokenizer path"
            ) from exc
    config_hash = _file_hash(model_config)

    source_root, source_files = _regular_tree(args.source_prompts)
    source_by_relative = {path.relative_to(source_root).as_posix(): path for path in source_files}
    tokenizer = tokenizer_loader(args.tokenizer_path)
    vocab_size = len(tokenizer)
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    try:
        vocab_ids = set(tokenizer.get_vocab().values())
    except (AttributeError, TypeError) as exc:
        raise ManifestError("Tokenizer does not expose a verifiable token vocabulary") from exc
    if any(type(token) is not int or token < 0 or token >= vocab_size for token in vocab_ids):
        raise ManifestError("Tokenizer vocabulary contains an out-of-range token ID")
    ordinary_ids = sorted(vocab_ids - special_ids)
    lead_count = sum(item["concurrency"] for item in runner.POINTS.values())
    filler_id = None
    for candidate in FILLER_CANDIDATES:
        candidate_ids = _encode(tokenizer, candidate)
        if len(candidate_ids) == 1 and candidate_ids[0] in ordinary_ids:
            if _encode(tokenizer, _decode(tokenizer, candidate_ids)) == candidate_ids:
                filler_id = candidate_ids[0]
                break
    if filler_id is None:
        raise ManifestError("Tokenizer has no round-trippable single-token neutral filler")

    leads: list[tuple[str, int]] = []
    for word in LEAD_WORDS:
        lead_ids = _encode(tokenizer, word)
        if (len(lead_ids) == 1 and lead_ids[0] in ordinary_ids and lead_ids[0] != filler_id
                and _decode(tokenizer, lead_ids) == word):
            if all(lead_ids[0] != token for _, token in leads):
                leads.append((word, lead_ids[0]))
        if len(leads) == lead_count:
            break
    if len(leads) != lead_count:
        raise ManifestError("Tokenizer cannot produce enough globally distinct ordinary lead tokens")

    points: dict[str, Any] = {}
    lead_index = 0
    for point, geometry in runner.POINTS.items():
        requests = []
        context = 16000 if point.endswith("16k") else 80000
        for slot in range(geometry["concurrency"]):
            _, lead_token = leads[lead_index]
            lead_index += 1
            if lead_token >= vocab_size or filler_id >= vocab_size:
                raise ManifestError("Constructed token ID is outside the tokenizer vocabulary")
            source_label = f"ctx{context}/{TURN_FILES[slot]}"
            source_path = source_by_relative.get(source_label)
            if source_path is None:
                raise ManifestError(f"Missing archived source turn: {source_label}")
            try:
                source = json.loads(source_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ManifestError(f"Cannot read archived source turn: {source_label}") from exc
            if not isinstance(source, dict) or source.get("context_limit") != context:
                raise ManifestError(f"Archived turn has unexpected context limit: {source_label}")
            ids, answer, derivation = derive_archived_request(
                tokenizer, source, source_path, source_label,
                geometry["prompt_tokens"], lead_token, filler_id,
            )
            requests.append({
                "slot": slot,
                "token_ids": ids,
                "token_ids_sha256": runner.token_ids_sha256(ids),
                "expected_answer": answer,
                "derivation": derivation,
            })
        points[point] = {**geometry, "requests": requests}

    manifest = {
        "schema": 1,
        "input_classification": "derived_from_archived_turns_not_historical_byte_identical",
        "blocked_points": runner.BLOCKED_POINTS,
        "model": args.model,
        "provenance": {
            "model_pack_sha256": model_pack_hash,
            "model_revision_sha256": args.model_revision_sha256,
            "model_config_sha256": config_hash,
            "tokenizer_sha256": tokenizer_hash,
            "installed_exl3_source_sha256": args.installed_exl3_source_sha256,
            "r0_source_commit": args.r0_source_commit,
        },
        "runtime_expectations": {
            "vllm_version": args.vllm_version,
            "exllamav3_revision": args.exllamav3_revision,
            "driver_version": args.driver_version,
            "cuda_version": args.cuda_version,
            "torch_version": args.torch_version,
            "effective_max_num_batched_tokens": args.effective_max_num_batched_tokens,
        },
        "points": points,
    }
    # Apply the same strict schema validation as the execution helper.
    import tempfile
    import os

    fd, name = tempfile.mkstemp(prefix="r0-c2-c4-manifest-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, separators=(",", ":"), sort_keys=True)
        runner.load_prompt_manifest(Path(name))
    finally:
        Path(name).unlink(missing_ok=True)
    return manifest


def _sha_arg(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise argparse.ArgumentTypeError("must be 64 lowercase hexadecimal characters")
    return value


def _commit_arg(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise argparse.ArgumentTypeError("must be a full 40-character lowercase commit")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--source-prompts", type=Path, required=True,
                        help="root containing ctx16000/ and ctx80000/ archived turn JSON files")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision-sha256", type=_sha_arg, required=True)
    parser.add_argument("--installed-exl3-source-sha256", type=_sha_arg, required=True)
    parser.add_argument("--r0-source-commit", type=_commit_arg, required=True)
    parser.add_argument("--vllm-version", required=True)
    parser.add_argument("--exllamav3-revision", type=_commit_arg, required=True)
    parser.add_argument("--driver-version", required=True)
    parser.add_argument("--cuda-version", required=True)
    parser.add_argument("--torch-version", required=True)
    parser.add_argument("--effective-max-num-batched-tokens", type=int, required=True)
    parser.add_argument("--out", type=Path, help="write manifest to this new file; default is stdout")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.effective_max_num_batched_tokens <= 0:
            raise ManifestError("effective token budget must be positive")
        manifest = make_manifest(args)
        rendered = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        if args.out:
            if args.out.exists() or args.out.is_symlink():
                raise ManifestError(f"Output must not already exist: {args.out}")
            with args.out.open("x", encoding="utf-8") as stream:
                stream.write(rendered)
        else:
            sys.stdout.write(rendered)
    except (ManifestError, runner.CellError, OSError, ValueError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
