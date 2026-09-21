#!/usr/bin/env python3
"""Generate fixed-history multi-query turn suites for K1A churn analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


FILLER = (
    " Repository archive entry: a routine build completed, several ordinary "
    "files were checked, a timestamp was recorded, and processing continued. "
    "This passage carries no answer to the later retrieval questions. "
)

FACTS = (
    ("KVMEM_FACT_A_104729", "cobalt-lantern-47", 0.08),
    ("KVMEM_FACT_B_208319", "silver-orbit-29", 0.22),
    ("KVMEM_FACT_C_315947", "amber-river-83", 0.38),
    ("KVMEM_FACT_D_426853", "violet-harbor-31", 0.55),
    ("KVMEM_FACT_E_537961", "granite-comet-72", 0.72),
    ("KVMEM_FACT_F_648271", "jade-forest-16", 0.88),
)

TURN_SPECS = (
    (
        "ask_a",
        ("KVMEM_FACT_A_104729",),
        "What is the exact recovery code stored under KVMEM_FACT_A_104729? "
        "Answer with only the code.",
    ),
    (
        "ask_b",
        ("KVMEM_FACT_B_208319",),
        "What is the exact recovery code stored under KVMEM_FACT_B_208319? "
        "Answer with only the code.",
    ),
    (
        "ask_a_paraphrase",
        ("KVMEM_FACT_A_104729",),
        "Look back through the archive and return only the code associated "
        "with marker KVMEM_FACT_A_104729.",
    ),
    (
        "ask_c",
        ("KVMEM_FACT_C_315947",),
        "What is the exact recovery code stored under KVMEM_FACT_C_315947? "
        "Answer with only the code.",
    ),
    (
        "ask_d_e",
        ("KVMEM_FACT_D_426853", "KVMEM_FACT_E_537961"),
        "Return the exact recovery codes for KVMEM_FACT_D_426853 and "
        "KVMEM_FACT_E_537961 in that order, separated by a comma.",
    ),
    (
        "ask_a_repeat",
        ("KVMEM_FACT_A_104729",),
        "What is the exact recovery code stored under KVMEM_FACT_A_104729? "
        "Answer with only the code.",
    ),
)


def _enc(tok, text: str) -> list[int]:
    return list(tok.encode(text, add_special_tokens=False))


def _repeat(unit: list[int], n: int) -> list[int]:
    if n <= 0:
        return []
    if not unit:
        raise ValueError("empty filler tokenization")
    q, r = divmod(n, len(unit))
    return unit * q + unit[:r]


def build_history(tok, history_tokens: int) -> tuple[list[int], list[dict]]:
    filler = _enc(tok, FILLER)
    rendered = []
    for marker, code, frac in FACTS:
        text = (
            f"\nImportant retained fact {marker}: "
            f"the exact recovery code is {code}. "
            "Keep this fact available for later archive queries.\n"
        )
        ids = _enc(tok, text)
        rendered.append((marker, code, frac, int(history_tokens * frac), ids))
    rendered.sort(key=lambda x: x[3])

    history: list[int] = []
    facts: list[dict] = []
    for marker, code, frac, desired, ids in rendered:
        desired = max(desired, len(history))
        history.extend(_repeat(filler, desired - len(history)))
        start = len(history)
        history.extend(ids)
        end = len(history)
        facts.append(
            {
                "marker": marker,
                "code": code,
                "fraction": frac,
                "start": start,
                "end": end,
                "length": end - start,
            }
        )
    if len(history) > history_tokens:
        raise ValueError("facts exceed history token budget")
    history.extend(_repeat(filler, history_tokens - len(history)))
    assert len(history) == history_tokens
    return history, facts


def build_suite(tok, *, context: int, history_reserve: int = 512) -> dict:
    history_tokens = context - history_reserve
    if history_tokens <= 0:
        raise ValueError("context too small")
    history, facts = build_history(tok, history_tokens)
    fact_map = {f["marker"]: f for f in facts}

    turns = []
    for index, (name, markers, query_text) in enumerate(TURN_SPECS):
        query_ids = _enc(tok, "\nFinal user query: " + query_text)
        prompt = history + query_ids
        if len(prompt) > context:
            raise ValueError(
                f"turn {name} exceeds context: {len(prompt)} > {context}"
            )
        turns.append(
            {
                "index": index,
                "name": name,
                "context_limit": context,
                "history_tokens": history_tokens,
                "prompt_tokens": len(prompt),
                "prompt_token_ids": prompt,
                "history_span": [0, history_tokens],
                "query_span": [history_tokens, len(prompt)],
                "target_markers": list(markers),
                "target_facts": [fact_map[m] for m in markers],
                "query_text": query_text,
            }
        )

    return {
        "schema": 1,
        "context_limit": context,
        "history_tokens": history_tokens,
        "facts": facts,
        "turns": turns,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--contexts", type=int, nargs="+", default=(160000, 240000))
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
        "contexts": [],
    }

    for ctx in args.contexts:
        suite = build_suite(tok, context=ctx)
        ctx_dir = args.out_dir / f"ctx{ctx}"
        ctx_dir.mkdir(parents=True, exist_ok=True)
        suite_path = ctx_dir / "suite.json"
        suite_path.write_text(json.dumps(suite, separators=(",", ":")))
        turn_entries = []
        for turn in suite["turns"]:
            p = ctx_dir / f"turn_{turn['index']:02d}_{turn['name']}.json"
            p.write_text(json.dumps(turn, separators=(",", ":")))
            turn_entries.append(
                {
                    "index": turn["index"],
                    "name": turn["name"],
                    "path": str(p.resolve()),
                    "query_span": turn["query_span"],
                    "target_markers": turn["target_markers"],
                }
            )
        manifest["contexts"].append(
            {
                "context": ctx,
                "history_tokens": suite["history_tokens"],
                "suite_path": str(suite_path.resolve()),
                "turns": turn_entries,
            }
        )

    out = args.out_dir / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"manifest": str(out.resolve()), "contexts": args.contexts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
