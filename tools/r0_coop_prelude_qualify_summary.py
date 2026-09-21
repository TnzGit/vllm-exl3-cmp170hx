#!/usr/bin/env python3
"""Summarize EARLY-only qualification against same-run BASE cells."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


CONTEXTS = (4096, 160000, 240000)


def load(path: Path) -> dict:
    return json.load(open(path))


def gain_pct(base_ms: float, cand_ms: float) -> float:
    return (base_ms - cand_ms) / base_ms * 100.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    d = Path(args.dir)

    rows = {}
    gains = []
    parity_all = True
    valid_all = True

    for ctx in CONTEXTS:
        b = load(d / f"ref_{ctx}.json")
        c = load(d / f"parity_k3_{ctx}.json")
        bm = float(b["ms_per_output_token_median"])
        cm = float(c["ms_per_output_token_median"])
        g = gain_pct(bm, cm)
        gains.append(g)

        # parity block is populated when --parity-reference is used. If an
        # older harness omits it, final flattened parity_recheck remains the
        # second authority and the row records UNKNOWN here.
        p = c.get("parity") or {}
        pstatus = p.get("status", "UNKNOWN")
        parity_all &= pstatus == "PASS"
        valid = b.get("status") == "VALID" and c.get("status") == "VALID"
        valid_all &= valid

        rows[str(ctx)] = {
            "base_ms_per_token": bm,
            "candidate_ms_per_token": cm,
            "latency_gain_pct": round(g, 4),
            "base_tok_s": b.get("output_tok_s_median"),
            "candidate_tok_s": c.get("output_tok_s_median"),
            "base_accepted_per_pass": b.get("accepted_per_pass"),
            "candidate_accepted_per_pass": c.get("accepted_per_pass"),
            "base_emitted_per_pass": b.get("emitted_per_pass"),
            "candidate_emitted_per_pass": c.get("emitted_per_pass"),
            "base_status": b.get("status"),
            "candidate_status": c.get("status"),
            "parity": pstatus,
        }

    sentinel_gains = []
    for name in ("in", "out"):
        b = load(d / f"base_4096_sentinel_{name}.json")
        c = load(d / f"early_4096_sentinel_{name}.json")
        sentinel_gains.append(
            gain_pct(
                float(b["ms_per_output_token_median"]),
                float(c["ms_per_output_token_median"]),
            )
        )

    mean_sentinel_gain = statistics.mean(sentinel_gains)
    median_context_gain = statistics.median(gains)
    worst_context_gain = min(gains)

    qualifies = (
        parity_all
        and valid_all
        and mean_sentinel_gain >= 0.5
        and median_context_gain >= 0.5
        and worst_context_gain >= -0.5
    )

    out = {
        "contexts": rows,
        "context_gains_pct": [round(x, 4) for x in gains],
        "median_context_gain_pct": round(median_context_gain, 4),
        "sentinel_gains_pct": [round(x, 4) for x in sentinel_gains],
        "mean_4k_sentinel_gain_pct": round(mean_sentinel_gain, 4),
        "worst_context_gain_pct": round(worst_context_gain, 4),
        "parity_all": parity_all,
        "valid_all": valid_all,
        "qualification_thresholds": {
            "mean_4k_sentinel_gain_pct_min": 0.5,
            "median_context_gain_pct_min": 0.5,
            "worst_context_gain_pct_min": -0.5,
        },
        "qualified": qualifies,
    }
    print(json.dumps(out, indent=2))
    return 0 if qualifies else 3


if __name__ == "__main__":
    raise SystemExit(main())
