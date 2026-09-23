#!/usr/bin/env python3
"""Validate and combine exact-context Q2E benchmark cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def summarize(run_dir: Path, contexts: list[int]) -> dict[str, Any]:
    cells = []
    errors: list[str] = []
    for context in contexts:
        cell_dir = run_dir / f"ctx{context}"
        cell = json.loads((cell_dir / "cell.json").read_text())
        worker = _jsonl(cell_dir / "worker.jsonl")
        scheduler = _jsonl(cell_dir / "scheduler.jsonl")
        runtime = [row for row in worker if row.get("event") == "q2d_streaming_runtime"]
        layers = sorted({int(row["layer_id"]) for row in runtime})
        combined_peak = max(
            (int(row.get("combined_real_pages", 0)) for row in runtime), default=0
        )
        read_peak = max(
            (int(row.get("peak_read_pages", 0)) for row in runtime), default=0
        )
        working_peak = max(
            (int(row.get("max_working_pages", 0)) for row in runtime), default=0
        )
        write_peak = max(
            (int(row.get("peak_real_write_pages", 0)) for row in scheduler),
            default=0,
        )
        roundtrip_exact = bool(runtime) and all(
            row.get("cpu_roundtrip_exact") is True for row in runtime
        )
        cell_errors = []
        if int(cell.get("context_limit", -1)) != context:
            cell_errors.append("context label mismatch")
        if cell.get("count_exact") is not True:
            cell_errors.append("API token counts are not exact")
        if cell.get("target_codes_in_order") is not True:
            cell_errors.append("target codes missing or out of order")
        if len(layers) != 12:
            cell_errors.append(f"expected 12 QSA layers, got {len(layers)}")
        if combined_peak > 4160 or working_peak > 4160:
            cell_errors.append("physical page cap exceeded")
        if read_peak > 4032 or write_peak > 128:
            cell_errors.append("READ/WRITE partition exceeded")
        if not roundtrip_exact:
            cell_errors.append("CPU roundtrip was not exact")
        errors.extend(f"ctx{context}: {error}" for error in cell_errors)
        cells.append({
            "context": context,
            "prompt_tokens": int(cell["prompt_tokens"]),
            "completion_tokens": int(cell["completion_tokens"]),
            "prefill_s": float(cell["prefill_s"]),
            "prefill_tok_s": float(cell["prefill_tok_s"]),
            "decode_tokens": int(cell["decode_tokens"]),
            "decode_s": float(cell["decode_s"]),
            "decode_tok_s": float(cell["decode_tok_s"]),
            "decode_ms_per_token": float(cell["decode_ms_per_token"]),
            "wall_s": float(cell["wall_s"]),
            "target_codes_in_order": bool(cell["target_codes_in_order"]),
            "runtime_records": len(runtime),
            "qsa_layers": layers,
            "combined_real_page_peak": combined_peak,
            "read_page_peak": read_peak,
            "write_page_peak": write_peak,
            "working_page_peak": working_peak,
            "cpu_roundtrip_exact": roundtrip_exact,
            "status": "VALID" if not cell_errors else "INVALID",
        })
    return {
        "schema": 1,
        "benchmark": "q2e_exact_context_prefill_decode",
        "contexts": contexts,
        "performance_mode": {
            "eager": True,
            "mtp": False,
            "prefix_cache": False,
            "q2e_trace": False,
            "direct_io": True,
            "consumer_sync": True,
            "max_num_batched_tokens": 1024,
            "max_model_len": 246000,
            "gpu_memory_utilization": 0.92,
        },
        "cells": cells,
        "errors": errors,
        "status": "VALID" if not errors else "INVALID",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--contexts", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.run_dir, args.contexts)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "VALID" else 3


if __name__ == "__main__":
    raise SystemExit(main())
