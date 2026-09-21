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
    "torch_compile_total_s": re.compile(
        r"torch\.compile took " + RE_FLOAT + r" s in total"
    ),
    "compile_warmup_together_s": re.compile(
        r"torch\.compile and initial profiling/warmup run together took "
        + RE_FLOAT
        + r" s in total"
    ),
    "initial_profiling_warmup_s": re.compile(
        r"Initial profiling/warmup run took " + RE_FLOAT + r" s"
    ),
    "dynamo_bytecode_s": re.compile(
        r"Dynamo bytecode transform time: " + RE_FLOAT + r" s"
    ),
}

WEIGHTS_RE = re.compile(r"Loading weights took " + RE_FLOAT + r" seconds")
PRESCAN_RE = re.compile(
    r"EXL3 trellis PRESCAN ready .*? elapsed_ms=" + RE_FLOAT
)
COMPILE_GRAPH_RE = re.compile(
    r"Compiling graph\([^\n]*?\).*?took " + RE_FLOAT + r" s"
)
COMPILE_CACHE_DIR_RE = re.compile(
    r"Using cache directory: (\S+) for vLLM's torch\.compile"
)
KV_CAPACITY_RE = re.compile(
    r"GPU KV cache size: ([0-9,]+) tokens, "
    r"Maximum concurrency for ([0-9,]+) tokens per request: ([0-9.]+)x"
)


def _last_float(pattern: re.Pattern[str], text: str) -> float | None:
    hits = pattern.findall(text)
    if not hits:
        return None
    return float(hits[-1])


def parse_log(text: str, wall: dict[str, Any] | None = None) -> dict[str, Any]:
    vals = {name: _last_float(pat, text) for name, pat in PATTERNS.items()}

    weight_loads_s = [float(x) for x in WEIGHTS_RE.findall(text)]
    main_weights_s = weight_loads_s[0] if weight_loads_s else None
    auxiliary_weight_loads_s = weight_loads_s[1:]
    auxiliary_weights_s = (
        sum(auxiliary_weight_loads_s) if auxiliary_weight_loads_s else 0.0
    )
    draft_weights_s = (
        auxiliary_weight_loads_s[0]
        if len(auxiliary_weight_loads_s) == 1
        else None
    )
    total_weights_s = sum(weight_loads_s) if weight_loads_s else None

    prescan_ms = [float(x) for x in PRESCAN_RE.findall(text)]
    kv_capacity_hits = KV_CAPACITY_RE.findall(text)
    if kv_capacity_hits:
        kv_tokens_raw, kv_request_raw, kv_concurrency_raw = kv_capacity_hits[-1]
        kv_cache_size_tokens = int(kv_tokens_raw.replace(",", ""))
        kv_capacity_request_tokens = int(kv_request_raw.replace(",", ""))
        kv_max_concurrency = float(kv_concurrency_raw)
    else:
        kv_cache_size_tokens = None
        kv_capacity_request_tokens = None
        kv_max_concurrency = None
    weights_s = main_weights_s
    model_s = vals["model_s"]
    engine_init_s = vals["engine_init_s"]
    graph_capture_s = vals["graph_capture_s"]

    model_non_weight_s = (
        max(0.0, model_s - total_weights_s)
        if model_s is not None and total_weights_s is not None
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
    if main_weights_s is not None:
        buckets["main_model_weights_path"] = main_weights_s
    if auxiliary_weights_s > 0:
        buckets["auxiliary_draft_weights_path"] = auxiliary_weights_s
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
        checkpoint_gib / main_weights_s
        if checkpoint_gib is not None
        and main_weights_s
        and main_weights_s > 0
        else None
    )

    compile_graph_s = [float(x) for x in COMPILE_GRAPH_RE.findall(text)]
    compile_cache_dirs = COMPILE_CACHE_DIR_RE.findall(text)
    aot_direct_load = "Directly load AOT compilation from path" in text
    standalone_artifact_reconstruction = (
        "reconstructed serializable fn from standalone compile artifacts"
        in text
    )

    return {
        "schema": 2,
        "timings": {
            "weights_s": main_weights_s,
            "main_weights_s": main_weights_s,
            "draft_weights_s": draft_weights_s,
            "auxiliary_weights_s": auxiliary_weights_s,
            "total_weights_s": total_weights_s,
            "weight_loads_s": weight_loads_s,
            "model_total_s": model_s,
            "model_construct_postload_s": model_non_weight_s,
            "engine_init_s": engine_init_s,
            "graph_capture_s": graph_capture_s,
            "engine_non_graph_s": engine_non_graph_s,
            "total_to_health_s": total_to_health_s,
            "accounted_model_plus_engine_s": accounted_s,
            "frontend_spawn_preflight_other_s": outside_accounted_s,
            "torch_compile_total_s": vals["torch_compile_total_s"],
            "compile_warmup_together_s": vals["compile_warmup_together_s"],
            "initial_profiling_warmup_s": vals["initial_profiling_warmup_s"],
            "dynamo_bytecode_s": vals["dynamo_bytecode_s"],
            "compile_graph_s": compile_graph_s,
            "compile_graph_sum_s": sum(compile_graph_s),
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
        "compile_cache": {
            "cache_dirs": compile_cache_dirs,
            "aot_direct_load": aot_direct_load,
            "standalone_artifact_reconstruction": (
                standalone_artifact_reconstruction
            ),
            "cache_hit_evidence": bool(
                aot_direct_load or standalone_artifact_reconstruction
            ),
        },
        "runtime_geometry": {
            "available_kv_cache_gib": vals["available_kv_cache_gib"],
            "gpu_kv_cache_size_tokens": kv_cache_size_tokens,
            "capacity_request_tokens": kv_capacity_request_tokens,
            "kv_max_concurrency": kv_max_concurrency,
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
            "weight_load_contract": (
                "weights_s is the first/main model Loading weights record. "
                "Any later Loading weights records are reported separately as "
                "auxiliary/draft loads. model_construct_postload subtracts "
                "the sum of all weight-load records from model_total_s."
            ),
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
