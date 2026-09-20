#!/usr/bin/env python3
"""Aggregate a py-spy speedscope profile into self-time hotspot tables.

Reports Python self-time per function so CPU dispatch cost can be separated
from blocking CUDA calls, without dumping hundreds of thousands of samples.

usage: python3 r0_analyze_pyspy.py profile.json [--top 30]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--filter", default=None,
                    help="only show frames whose name contains this substring")
    args = ap.parse_args()

    data = json.load(open(args.profile))
    frames = data.get("shared", {}).get("frames", [])
    profiles = data.get("profiles", [])
    if not profiles:
        print("ERROR: no profiles in file", file=sys.stderr)
        return 1

    prof = profiles[0]
    samples = prof.get("samples", [])
    weights = prof.get("weights", [])
    if not weights:
        weights = [1] * len(samples)
    total = float(sum(weights)) or 1.0

    self_time: Counter[int] = Counter()
    total_time: Counter[int] = Counter()
    for sample, w in zip(samples, weights):
        if not sample:
            continue
        self_time[sample[-1]] += w
        for f in sample:
            total_time[f] += w

    def name(i: int) -> str:
        fr = frames[i]
        return f"{fr.get('name', '?')} @ {fr.get('file', '?')}:{fr.get('line', '?')}"

    print(f"samples={len(samples)} total_weight={int(total)}")
    print("=== TOP SELF TIME (leaf frames) ===")
    for idx, w in self_time.most_common(args.top):
        pct = 100.0 * w / total
        if args.filter and args.filter not in name(idx):
            continue
        print(f"{pct:6.2f}%  {int(w):8d}  {name(idx)}")

    print("=== TOP INCLUSIVE TIME (anywhere on stack) ===")
    for idx, w in total_time.most_common(args.top):
        pct = 100.0 * w / total
        if args.filter and args.filter not in name(idx):
            continue
        print(f"{pct:6.2f}%  {int(w):8d}  {name(idx)}")

    # Group self-time by source file so plugin vs torch vs vllm vs stdlib is
    # visible at a glance.
    by_file: Counter[str] = Counter()
    for idx, w in self_time.items():
        by_file[frames[idx].get("file", "?") or "?"] += w
    print("=== SELF TIME BY FILE (top 20) ===")
    for f, w in by_file.most_common(20):
        print(f"{100.0 * w / total:6.2f}%  {int(w):8d}  {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
