#!/usr/bin/env python3
"""Create the exact-token C2/C4 manifest from explicit local model assets."""

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
PROMPT_PREFIX = (
    "{lead} task: Read this request carefully. Ignore the neutral context. "
    "Reply with exactly the answer code shown at the very end, with no extra text. "
    "Neutral context:"
)
PROMPT_SUFFIX = " Answer code: {answer}"
FILLER_CANDIDATES = (" neutral", " context", " ordinary", " sample")


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


def _make_prompt_ids(
    tokenizer: Any,
    point: str,
    slot: int,
    lead_word: str,
    filler_id: int,
    target_tokens: int,
) -> tuple[list[int], str]:
    answer = f"r0-c2-c4-{point}-{slot}"
    prefix = _encode(tokenizer, PROMPT_PREFIX.format(lead=lead_word))
    suffix = _encode(tokenizer, PROMPT_SUFFIX.format(answer=answer))
    filler_count = target_tokens - len(prefix) - len(suffix)
    if not prefix or filler_count < 0:
        raise ManifestError(f"Instruction and answer do not fit the {point} token budget")
    ids = prefix + [filler_id] * filler_count + suffix
    decoded = _decode(tokenizer, ids)
    if not decoded.startswith(lead_word) or not decoded.endswith(answer):
        raise ManifestError(f"Tokenizer did not preserve the {point} prompt lead/answer suffix")
    if _encode(tokenizer, decoded) != ids:
        raise ManifestError(f"Tokenizer failed exact token-ID round trip for {point} slot {slot}")
    return ids, answer


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
        lead_ids = _encode(tokenizer, PROMPT_PREFIX.format(lead=word))
        if lead_ids and lead_ids[0] in ordinary_ids and lead_ids[0] != filler_id:
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
        for slot in range(geometry["concurrency"]):
            lead_word, lead_token = leads[lead_index]
            lead_index += 1
            if lead_token >= vocab_size or filler_id >= vocab_size:
                raise ManifestError("Constructed token ID is outside the tokenizer vocabulary")
            ids, answer = _make_prompt_ids(
                tokenizer,
                point,
                slot,
                lead_word,
                filler_id,
                geometry["prompt_tokens"],
            )
            requests.append({
                "slot": slot,
                "token_ids": ids,
                "token_ids_sha256": runner.token_ids_sha256(ids),
                "expected_answer": answer,
            })
        points[point] = {**geometry, "requests": requests}

    manifest = {
        "schema": 1,
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
