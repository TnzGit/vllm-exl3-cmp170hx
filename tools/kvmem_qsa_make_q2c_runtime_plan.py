#!/usr/bin/env python3
"""Promote a validated Q2B frozen resident plan to Q2C scheduler ownership."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vllm_exl3.kvmem_qsa_scheduler_runtime import validate_runtime_plan


def promote(q2b: dict) -> dict:
    if int(q2b.get("schema", 0)) != 1:
        raise ValueError("Q2B plan schema mismatch")
    if q2b.get("mode") != "qsa_cpu_backed_shadow":
        raise ValueError("expected Q2B CPU-backed shadow plan")
    out = dict(q2b)
    out["mode"] = "qsa_scheduler_owned_transition"
    out["scheduler_chunk_tokens"] = int(q2b["active_reserve_tokens"])
    out["note"] = (
        "Q2C-transition uses the same frozen 64K sticky history and 1K active "
        "reserve. The scheduler owns 16-token QSA pages; after apply_min_pos "
        "it releases nonresident historical pages into null holes. The worker "
        "publishes retained history through generic CPU backing and verifies "
        "the bounded scheduler-owned cache without an independent 130MiB shadow."
    )
    validate_runtime_plan(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--q2b-plan", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    result = promote(json.loads(args.q2b_plan.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
