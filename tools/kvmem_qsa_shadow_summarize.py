#!/usr/bin/env python3
"""Summarize QSA shadow selections as coarse historical working sets.

The model output is unchanged. This tool only analyzes the logical token
indices that QSA already selected for final-query rows.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


BLOCK_SIZES = (128, 256, 512)
BUDGETS = (32768, 65536)
SINK_TOKENS = 512
RECENT_TOKENS = 4096


def _load_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open() as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON") from exc
    return out


def _needle_blocks(needle: dict, block_size: int) -> set[int]:
    start = int(needle["start"])
    end = int(needle["end"])
    if end <= start:
        return set()
    return set(range(start // block_size, (end - 1) // block_size + 1))


def _mandatory_blocks(query_start: int, block_size: int) -> set[int]:
    out = set(range(0, (min(SINK_TOKENS, query_start) + block_size - 1) // block_size))
    recent_start = max(0, query_start - RECENT_TOKENS)
    if recent_start < query_start:
        out.update(
            range(
                recent_start // block_size,
                (query_start - 1) // block_size + 1,
            )
        )
    return out


def summarize_case(case: dict, records: list[dict]) -> dict:
    q0, q1 = map(int, case["query_span"])
    votes: collections.Counter[int] = collections.Counter()
    direct_tokens: collections.Counter[int] = collections.Counter()
    layers_seen: set[str] = set()
    query_rows = 0
    records_used = 0

    for rec in records:
        if int(rec.get("schema", 0)) != 1:
            continue
        if rec.get("skip_topk"):
            # MTP follower rows would duplicate target selections. K0 runs
            # no-draft, but fail closed against accidental duplicate voting.
            continue
        layer = str(rec.get("layer_name", ""))
        used_this_record = False
        for row in rec.get("rows") or []:
            pos = int(row["pos"])
            if not (q0 <= pos < q1):
                continue
            query_rows += 1
            used_this_record = True
            layers_seen.add(layer)
            for tok in row.get("selected") or []:
                tok = int(tok)
                if 0 <= tok < q0:
                    votes[tok] += 1
                    direct_tokens[tok] += 1
        if used_this_record:
            records_used += 1

    needle_direct = []
    for n in case["needles"]:
        s, e = int(n["start"]), int(n["end"])
        hits = sum(v for tok, v in direct_tokens.items() if s <= tok < e)
        needle_direct.append(
            {
                "marker": n["marker"],
                "code": n["code"],
                "start": s,
                "end": e,
                "direct_qsa_token_hit": hits > 0,
                "direct_vote_count": int(hits),
            }
        )

    policies: dict[str, dict] = {}
    for block_size in BLOCK_SIZES:
        block_votes: collections.Counter[int] = collections.Counter()
        for tok, count in votes.items():
            block_votes[tok // block_size] += count
        ranked = [
            block
            for block, _ in sorted(
                block_votes.items(),
                key=lambda kv: (-kv[1], kv[0]),
            )
        ]
        for budget in BUDGETS:
            cap = budget // block_size
            mandatory = _mandatory_blocks(q0, block_size)
            chosen = set(sorted(mandatory)[:cap])
            for block in ranked:
                if len(chosen) >= cap:
                    break
                chosen.add(block)

            needles = []
            for n in case["needles"]:
                required = _needle_blocks(n, block_size)
                hit = required & chosen
                needles.append(
                    {
                        "marker": n["marker"],
                        "required_blocks": sorted(required),
                        "selected_required_blocks": sorted(hit),
                        "block_recall": (
                            len(hit) / len(required) if required else 1.0
                        ),
                        "all_required_blocks_selected": required <= chosen,
                    }
                )
            key = f"b{block_size}_budget{budget}"
            policies[key] = {
                "block_size": block_size,
                "budget_tokens": budget,
                "budget_blocks": cap,
                "mandatory_blocks": len(mandatory),
                "voted_history_blocks": len(block_votes),
                "selected_blocks": len(chosen),
                "needles": needles,
            }

    return {
        "schema": 1,
        "case": case["name"],
        "target_tokens": case["target_tokens"],
        "query_span": case["query_span"],
        "records_total": len(records),
        "records_used": records_used,
        "query_rows_used": query_rows,
        "layers_used": len(layers_seen),
        "layers": sorted(layers_seen),
        "unique_history_tokens_voted": len(votes),
        "total_history_token_votes": int(sum(votes.values())),
        "direct_needle_hits": needle_direct,
        "policies": policies,
    }


def summarize_suite(manifest: dict, shadow_dir: Path) -> dict:
    cases = []
    for spec in manifest["cases"]:
        case = json.loads(Path(spec["path"]).read_text())
        shadow = shadow_dir / f"{spec['name']}.jsonl"
        cases.append(summarize_case(case, _load_jsonl(shadow)))

    total_needles = sum(len(c["direct_needle_hits"]) for c in cases)
    direct_hits = sum(
        int(n["direct_qsa_token_hit"])
        for c in cases
        for n in c["direct_needle_hits"]
    )
    policy_summary = {}
    for key in cases[0]["policies"] if cases else []:
        rows = [
            n
            for c in cases
            for n in c["policies"][key]["needles"]
        ]
        full = sum(int(n["all_required_blocks_selected"]) for n in rows)
        mean_recall = (
            sum(float(n["block_recall"]) for n in rows) / len(rows) if rows else 0.0
        )
        policy_summary[key] = {
            "needles": len(rows),
            "fully_recalled_needles": full,
            "fully_recalled_rate": full / len(rows) if rows else 0.0,
            "mean_block_recall": mean_recall,
        }

    primary64 = policy_summary.get("b256_budget65536", {})
    primary32 = policy_summary.get("b256_budget32768", {})
    go_signal = (
        primary64.get("fully_recalled_rate", 0.0) >= 0.99
        and primary32.get("fully_recalled_rate", 0.0) >= 0.95
    )
    return {
        "schema": 1,
        "cases": cases,
        "suite": {
            "case_count": len(cases),
            "needle_count": total_needles,
            "direct_qsa_token_hits": direct_hits,
            "direct_qsa_token_hit_rate": (
                direct_hits / total_needles if total_needles else 0.0
            ),
            "policies": policy_summary,
            "primary_policy": {
                "block_size": 256,
                "budgets": [32768, 65536],
                "go_threshold_64k": 0.99,
                "desirable_threshold_32k": 0.95,
                "go_signal": go_signal,
            },
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--case", type=Path)
    mode.add_argument("--manifest", type=Path)
    ap.add_argument("--shadow", type=Path)
    ap.add_argument("--shadow-dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.case is not None:
        if args.shadow is None:
            ap.error("--case requires --shadow")
        result = summarize_case(
            json.loads(args.case.read_text()),
            _load_jsonl(args.shadow),
        )
    else:
        if args.shadow_dir is None:
            ap.error("--manifest requires --shadow-dir")
        result = summarize_suite(
            json.loads(args.manifest.read_text()),
            args.shadow_dir,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
