#!/usr/bin/env python3
"""Promote a Q2C plan to the Q2D full-source CPU reload oracle."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from vllm_exl3.kvmem_q2d_plan import validate_reload_plan


def promote(q2c: dict) -> dict:
    if int(q2c.get("schema", 0)) != 1:
        raise ValueError("Q2C plan schema mismatch")
    if q2c.get("mode") != "qsa_scheduler_owned_transition":
        raise ValueError("expected Q2C scheduler-owned transition plan")
    max_model_len = int(q2c.get("max_model_len", 161000))
    page_tokens = int(q2c["page_tokens"])
    out = {
        "schema": 1,
        "mode": "qsa_cpu_reload_shadow",
        "diagnostic_full_source_shadow": True,
        "page_tokens": page_tokens,
        "physical_page_count": int(q2c["physical_page_count"]),
        "scheduler_chunk_tokens": int(q2c["scheduler_chunk_tokens"]),
        "query_row_batch": 64,
        "staging_pages": 128,
        "max_model_len": max_model_len,
        "cpu_page_count": math.ceil(max_model_len / page_tokens),
        "expected_qsa_layers": int(q2c.get("expected_qsa_layers", 12)),
        "query_span": list(q2c["query_span"]),
        "target_facts": list(q2c.get("target_facts", [])),
        "allclose_atol": 0.02,
        "allclose_rtol": 0.01,
        "note": (
            "Correctness oracle only: stock full GPU KV remains the source and "
            "same-forward reference. A separate <=4160-page pool reloads complete "
            "history from generic vLLM CPU backing using dynamic READ tables."
        ),
    }
    return validate_reload_plan(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--q2c-plan", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    result = promote(json.loads(args.q2c_plan.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
