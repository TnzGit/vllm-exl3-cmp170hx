"""Reproducible CPU microbenchmark of exact stable LRU victim selection."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np


def full_sort(times: np.ndarray, need: int) -> np.ndarray:
    insertion_order = np.arange(len(times), dtype=np.int64)
    return np.lexsort((insertion_order, times))[:need]


def partial_select(times: np.ndarray, need: int) -> np.ndarray:
    if need < len(times):
        cutoff = np.partition(times, need - 1)[need - 1]
        older = np.flatnonzero(times < cutoff)
        tied = np.flatnonzero(times == cutoff)[: need - len(older)]
        selected = np.concatenate((older, tied))
    else:
        selected = np.arange(len(times), dtype=np.int64)
    order = np.lexsort((selected, times[selected]))
    return selected[order]


def run_once(fn, times: np.ndarray, need: int, iterations: int) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        fn(times, need)
    return (time.perf_counter() - start) * 1000 / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=int, default=4032)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()
    if args.candidates < 2 or args.iterations < 1 or args.trials < 1:
        parser.error("candidates >= 2 and positive iterations/trials required")

    rng = np.random.default_rng(423)
    cases = []
    for need in (1, 8, 16, 64, min(512, args.candidates)):
        if need > args.candidates:
            continue
        for tied in (False, True):
            times = (
                rng.integers(0, 97, size=args.candidates, dtype=np.int64)
                if tied
                else rng.permutation(args.candidates).astype(np.int64)
            )
            if not np.array_equal(full_sort(times, need), partial_select(times, need)):
                raise RuntimeError("partial selector differs from stable full sort")
            full_ms = []
            partial_ms = []
            for trial in range(args.trials):
                funcs = (
                    (full_sort, partial_select)
                    if trial % 2 == 0
                    else (partial_select, full_sort)
                )
                timings = {
                    fn.__name__: run_once(fn, times, need, args.iterations)
                    for fn in funcs
                }
                full_ms.append(timings["full_sort"])
                partial_ms.append(timings["partial_select"])
            cases.append(
                {
                    "need": need,
                    "ties": tied,
                    "full_sort_ms": statistics.median(full_ms),
                    "partial_select_ms": statistics.median(partial_ms),
                }
            )
    print(json.dumps({"schema": 1, "candidates": args.candidates, "cases": cases}))


if __name__ == "__main__":
    main()
