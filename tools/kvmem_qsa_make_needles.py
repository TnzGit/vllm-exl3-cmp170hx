#!/usr/bin/env python3
"""Generate deterministic token-ID needle cases for K0 QSA shadow retrieval.

The generated prompts are token IDs, not text, so the recorded needle/query
spans are exact and do not depend on a decode/re-encode round trip.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


FILLER = (
    " Repository note: a worker processed a routine build artifact, checked "
    "three ordinary files, recorded a timestamp, and continued to the next "
    "task. This paragraph contains no secret, credential, code, or answer. "
)

SINGLE = {
    "early": (
        "KVMEM_NEEDLE_EARLY_918273",
        "cobalt-lantern-47",
        0.10,
    ),
    "middle": (
        "KVMEM_NEEDLE_MIDDLE_564738",
        "silver-orbit-29",
        0.50,
    ),
    "late": (
        "KVMEM_NEEDLE_LATE_192837",
        "amber-river-83",
        0.84,
    ),
}

MULTI = (
    ("KVMEM_NEEDLE_MULTI_A_314159", "violet-harbor-31", 0.12),
    ("KVMEM_NEEDLE_MULTI_B_271828", "granite-comet-72", 0.52),
    ("KVMEM_NEEDLE_MULTI_C_161803", "jade-forest-16", 0.82),
)


def _encode(tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _repeat_to_length(unit: list[int], length: int) -> list[int]:
    if length <= 0:
        return []
    if not unit:
        raise ValueError("filler tokenization is empty")
    q, r = divmod(length, len(unit))
    return unit * q + unit[:r]


def build_case(
    tokenizer,
    *,
    target_tokens: int,
    case_name: str,
    needles: Iterable[tuple[str, str, float]],
) -> dict:
    needle_specs = list(needles)
    if not needle_specs:
        raise ValueError("at least one needle is required")

    query_text = (
        "\nFinal user query: "
        + (
            f"What is the exact code stored under {needle_specs[0][0]}? "
            "Answer with only that code."
            if len(needle_specs) == 1
            else "Return the exact codes for "
            + ", ".join(x[0] for x in needle_specs)
            + " in that order, separated by commas."
        )
    )
    query_ids = _encode(tokenizer, query_text)
    filler_ids = _encode(tokenizer, FILLER)
    history_len = target_tokens - len(query_ids)
    if history_len <= 0:
        raise ValueError("target context is too small for the query")

    rendered = []
    for marker, code, frac in needle_specs:
        text = (
            f"\nImportant retained fact {marker}: "
            f"the exact recovery code is {code}. "
            "Keep this fact available for a later query.\n"
        )
        ids = _encode(tokenizer, text)
        desired = int(history_len * frac)
        rendered.append((marker, code, frac, desired, ids))
    rendered.sort(key=lambda x: x[3])

    prompt: list[int] = []
    spans = []
    for marker, code, frac, desired, ids in rendered:
        if desired < len(prompt):
            desired = len(prompt)
        prompt.extend(_repeat_to_length(filler_ids, desired - len(prompt)))
        start = len(prompt)
        prompt.extend(ids)
        end = len(prompt)
        spans.append(
            {
                "marker": marker,
                "code": code,
                "fraction": frac,
                "start": start,
                "end": end,
                "length": end - start,
            }
        )

    if len(prompt) > history_len:
        raise ValueError(
            f"needle layout exceeds history budget: {len(prompt)} > {history_len}"
        )
    prompt.extend(_repeat_to_length(filler_ids, history_len - len(prompt)))
    query_start = len(prompt)
    prompt.extend(query_ids)
    query_end = len(prompt)
    if len(prompt) != target_tokens:
        raise AssertionError("prompt construction did not hit exact target length")

    return {
        "schema": 1,
        "name": case_name,
        "target_tokens": target_tokens,
        "prompt_tokens": len(prompt),
        "prompt_token_ids": prompt,
        "history_span": [0, query_start],
        "query_span": [query_start, query_end],
        "needles": spans,
        "query_text": query_text,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--contexts",
        type=int,
        nargs="+",
        default=(160000, 240000),
    )
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema": 1,
        "model_dir": str(Path(args.model_dir).resolve()),
        "tokenizer_class": type(tok).__name__,
        "cases": [],
    }
    for ctx in args.contexts:
        case_specs = [
            (f"ctx{ctx}_single_early", (SINGLE["early"],)),
            (f"ctx{ctx}_single_middle", (SINGLE["middle"],)),
            (f"ctx{ctx}_single_late", (SINGLE["late"],)),
            (f"ctx{ctx}_multi", MULTI),
        ]
        for name, needles in case_specs:
            case = build_case(
                tok,
                target_tokens=ctx,
                case_name=name,
                needles=needles,
            )
            path = args.out_dir / f"{name}.json"
            path.write_text(json.dumps(case, separators=(",", ":")))
            manifest["cases"].append(
                {
                    "name": name,
                    "context": ctx,
                    "path": str(path.resolve()),
                    "query_span": case["query_span"],
                    "needles": case["needles"],
                }
            )

    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({
        "manifest": str(manifest_path.resolve()),
        "cases": len(manifest["cases"]),
        "contexts": args.contexts,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
