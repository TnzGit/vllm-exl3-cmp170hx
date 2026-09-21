#!/usr/bin/env python3
"""Diagnose SM80 persistent_topk set stability with unique vs tied cutoffs."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch


def _jaccard(a: set[int], b: set[int]) -> float:
    u = a | b
    return len(a & b) / len(u) if u else 1.0


def _run_persistent(logits: torch.Tensor, *, k: int, repeats: int) -> dict:
    import vllm._C  # noqa: F401 - registers torch.ops._C

    if not hasattr(torch.ops._C, "persistent_topk"):
        raise RuntimeError("torch.ops._C.persistent_topk is unavailable")

    rows, width = logits.shape
    lengths = torch.full(
        (rows,), width, dtype=torch.int32, device=logits.device
    )
    workspace = torch.empty(
        (1024 * 1024,), dtype=torch.uint8, device=logits.device
    )
    output = torch.empty((rows, k), dtype=torch.int32, device=logits.device)

    row_sets: list[list[set[int]]] = [[] for _ in range(rows)]
    for _ in range(repeats):
        torch.ops._C.persistent_topk(
            logits,
            lengths,
            output,
            workspace,
            k,
            width,
        )
        torch.cuda.synchronize()
        out_cpu = output.cpu()
        for r in range(rows):
            row_sets[r].append(set(map(int, out_cpu[r].tolist())))

    summaries = []
    for r, sets in enumerate(row_sets):
        frozen = [tuple(sorted(x)) for x in sets]
        distinct = len(set(frozen))
        pairs = [
            _jaccard(a, b)
            for a, b in itertools.combinations(sets, 2)
        ]
        summaries.append(
            {
                "row": r,
                "distinct_sets": distinct,
                "min_pairwise_jaccard": min(pairs) if pairs else 1.0,
                "max_pairwise_jaccard": max(pairs) if pairs else 1.0,
                "set_size": len(sets[0]) if sets else 0,
            }
        )
    return {
        "rows": rows,
        "width": width,
        "k": k,
        "repeats": repeats,
        "row_summaries": summaries,
        "any_nondeterministic_set": any(
            x["distinct_sets"] > 1 for x in summaries
        ),
        "min_jaccard_all_rows": min(
            (x["min_pairwise_jaccard"] for x in summaries),
            default=1.0,
        ),
    }


def _make_unique(rows: int, width: int, device: torch.device) -> torch.Tensor:
    base = torch.arange(width, dtype=torch.float32, device=device)
    return base.repeat(rows, 1)


def _make_tied(
    rows: int,
    width: int,
    *,
    k: int,
    device: torch.device,
) -> torch.Tensor:
    if k < 2 or width < k * 4:
        raise ValueError("width/k too small for tied-cutoff case")
    logits = torch.zeros((rows, width), dtype=torch.float32, device=device)
    high = k // 2
    tie = k * 4
    logits[:, :high] = 2.0
    logits[:, high : high + tie] = 1.0
    return logits


def _torch_reference(
    logits: torch.Tensor, *, k: int, repeats: int
) -> dict:
    rows = logits.shape[0]
    row_sets: list[list[set[int]]] = [[] for _ in range(rows)]
    for _ in range(repeats):
        out = torch.topk(logits, k, dim=-1, sorted=False).indices.cpu()
        for r in range(rows):
            row_sets[r].append(set(map(int, out[r].tolist())))
    summaries = []
    for r, sets in enumerate(row_sets):
        frozen = [tuple(sorted(x)) for x in sets]
        summaries.append(
            {
                "row": r,
                "distinct_sets": len(set(frozen)),
                "set_size": len(sets[0]),
            }
        )
    return {
        "rows": rows,
        "k": k,
        "repeats": repeats,
        "row_summaries": summaries,
        "any_nondeterministic_set": any(
            x["distinct_sets"] > 1 for x in summaries
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--width", type=int, default=60000)
    ap.add_argument("--k", type=int, default=512)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    capability = torch.cuda.get_device_capability(device)

    unique = _make_unique(args.rows, args.width, device)
    tied = _make_tied(
        args.rows,
        args.width,
        k=args.k,
        device=device,
    )

    result = {
        "schema": 1,
        "device": {
            "name": props.name,
            "capability": list(capability),
            "sm_count": props.multi_processor_count,
        },
        "case_unique": _run_persistent(
            unique, k=args.k, repeats=args.repeats
        ),
        "case_tied": _run_persistent(
            tied, k=args.k, repeats=args.repeats
        ),
        "torch_tied_reference": _torch_reference(
            tied, k=args.k, repeats=args.repeats
        ),
    }
    result["interpretation"] = {
        "persistent_unique_stable": not result["case_unique"][
            "any_nondeterministic_set"
        ],
        "persistent_tied_unstable": result["case_tied"][
            "any_nondeterministic_set"
        ],
        "tie_path_supported": (
            not result["case_unique"]["any_nondeterministic_set"]
            and result["case_tied"]["any_nondeterministic_set"]
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
