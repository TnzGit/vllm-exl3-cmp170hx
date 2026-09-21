#!/usr/bin/env python3
"""Attribute a vLLM/EXL3 startup log into coarse wall-time phases.

This is intentionally read-only and relies on timing messages already emitted
by vLLM 0.29 plus EXL3 prescan diagnostics. It does not modify vLLM or the
plugin and does not pretend that "Loading weights" is pure disk I/O.

Important nesting:
- vLLM "Loading weights took" is inside total "Model loading took".
- total model loading also includes model construction and
  process_weights_after_loading.
- "init engine (profile, create kv cache, warmup model)" happens after model
  loading.
- CUDA graph capture is inside the init-engine phase.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


RE_FLOAT = r"([0-9]+(?:\.[0-9]+)?)"

PATTERNS = {
    "weights_s": re.compile(r"Loading weights took " + RE_FLOAT + r" seconds"),
    "model_s": re.compile(
        r"Model loading took [0-9.]+ GiB memory and " + RE_FLOAT + r" seconds"
    ),
    "engine_init_s": re.compile(
        r"init engine \(profile, create kv cache, warmup model\) took "
        + RE_FLOAT
        + r" s"
    ),
    "graph_capture_s": re.compile(
        r"Graph capturing finished in " + RE_FLOAT + r" secs"
    ),
    "checkpoint_gib": re.compile(
        r"Checkpoint size: " + RE_FLOAT + r" GiB"
    ),
    "available_ram_gib": re.compile(
        r"Available RAM: " + RE_FLOAT + r" GiB"
    ),
    "available_kv_cache_gib": re.compile(
        r"Available KV cache memory: " + RE_FLOAT + r" GiB"
    ),
    "attention_block_tokens": re.compile(
        r"Setting attention block size to ([0-9]+) tokens"
    ),
    "prefetch_s": re.compile(
        r"Prefetching checkpoint files into page cache finished in "
        + RE_FLOAT
        + r"s"
    ),
}

PRESCAN_RE = re.compile(
    r"EXL3 trellis PRESCAN ready .*? elapsed_ms=" + RE_FLOAT
)


def _last_float(pattern: re.Pattern[str], text: str) -> float | None:
    hits = pattern.findall(text)
    if not hits:
        return None
    return float(hits[-1])


def parse_log(text: str, wall: dict[str, Any] | None = None) -> dict[str, Any]:
    vals = {name: _last_float(pat, text) for name, pat in PATTERNS.items()}

    prescan_ms = [float(x) for x in PRESCAN_RE.findall(text)]
    weights_s = vals["weights_s"]
    model_s = vals["model_s"]
    engine_init_s = vals["engine_init_s"]
    graph_capture_s = vals["graph_capture_s"]

    model_non_weight_s = (
        max(0.0, model_s - weights_s)
        if model_s is not None and weights_s is not None
        else None
    )
    engine_non_graph_s = (
        max(0.0, engine_init_s - graph_capture_s)
        if engine_init_s is not None and graph_capture_s is not None
        else None
    )

    total_to_health_s = None
    if wall:
        start = wall.get("start_monotonic_s")
        health = wall.get("health_monotonic_s")
        if isinstance(start, (int, float)) and isinstance(health, (int, float)):
            total_to_health_s = float(health) - float(start)

    accounted_s = None
    outside_accounted_s = None
    if model_s is not None and engine_init_s is not None:
        accounted_s = model_s + engine_init_s
        if total_to_health_s is not None:
            outside_accounted_s = max(0.0, total_to_health_s - accounted_s)

    buckets: dict[str, float] = {}
    if weights_s is not None:
        buckets["weights_path"] = weights_s
    if model_non_weight_s is not None:
        buckets["model_construct_postload"] = model_non_weight_s
    if engine_non_graph_s is not None:
        buckets["engine_profile_cache_warmup_ex_graph"] = engine_non_graph_s
    elif engine_init_s is not None:
        buckets["engine_profile_cache_warmup_total"] = engine_init_s
    if graph_capture_s is not None:
        buckets["cuda_graph_capture"] = graph_capture_s
    if outside_accounted_s is not None:
        buckets["frontend_spawn_preflight_other"] = outside_accounted_s

    ranking = sorted(
        ({"phase": k, "seconds": v} for k, v in buckets.items()),
        key=lambda row: row["seconds"],
        reverse=True,
    )

    checkpoint_gib = vals["checkpoint_gib"]
    implied_weight_gib_s = (
        checkpoint_gib / weights_s
        if checkpoint_gib is not None and weights_s and weights_s > 0
        else None
    )

    return {
        "schema": 1,
        "timings": {
            "weights_s": weights_s,
            "model_total_s": model_s,
            "model_construct_postload_s": model_non_weight_s,
            "engine_init_s": engine_init_s,
            "graph_capture_s": graph_capture_s,
            "engine_non_graph_s": engine_non_graph_s,
            "total_to_health_s": total_to_health_s,
            "accounted_model_plus_engine_s": accounted_s,
            "frontend_spawn_preflight_other_s": outside_accounted_s,
        },
        "checkpoint": {
            "checkpoint_gib": checkpoint_gib,
            "available_ram_gib": vals["available_ram_gib"],
            "prefetch_s": vals["prefetch_s"],
            "implied_checkpoint_gib_per_weights_second": implied_weight_gib_s,
            "warning": (
                "weights_s is not pure storage throughput: it includes tensor "
                "materialization, EXL3 weight_loader work and CPU->GPU copies."
            ),
        },
        "runtime_geometry": {
            "available_kv_cache_gib": vals["available_kv_cache_gib"],
            "effective_attention_block_tokens": (
                int(vals["attention_block_tokens"])
                if vals["attention_block_tokens"] is not None
                else None
            ),
        },
        "exl3_prescan": {
            "records": len(prescan_ms),
            "sum_ms": sum(prescan_ms),
            "max_ms": max(prescan_ms) if prescan_ms else None,
            "mean_ms": (
                sum(prescan_ms) / len(prescan_ms) if prescan_ms else None
            ),
        },
        "phase_ranking": ranking,
        "interpretation": {
            "largest_phase": ranking[0]["phase"] if ranking else None,
            "largest_phase_seconds": ranking[0]["seconds"] if ranking else None,
            "weights_path_contains": [
                "safetensors page faults / storage reads",
                "tensor materialization",
                "EXL3 per-tensor loaders",
                "blocking H2D direct-fill/copies",
            ],
            "model_construct_postload_contains": [
                "module construction before first weight yield",
                "process_weights_after_loading",
                "LinearEXL3 handle construction",
                "fused MoE pointer/state setup",
            ],
            "engine_non_graph_contains": [
                "memory profile forward",
                "hybrid KV-cache sizing/allocation",
                "kernel warmup",
                "non-graph compile/warmup work",
            ],
            "long_context_clue": (
                "Compare warm 4K vs warm 161K engine_init / engine_non_graph. "
                "A stable weights_s with a larger long-context engine phase "
                "implicates hybrid KV sizing/allocation/warmup rather than "
                "checkpoint loading."
            ),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--wall", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    wall = json.loads(args.wall.read_text()) if args.wall else None
    result = parse_log(args.log.read_text(errors="replace"), wall)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
