#!/usr/bin/env python3
"""Extract hard Q2C dual-pool boot evidence from one vLLM serve log."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PLATFORM = (
    "Q2C dual-pool: preserving 16-token QSA pages; "
    "skipping legacy QSA/Mamba shared-page alignment."
)
CORE_RE = re.compile(
    r"Q2C dual-pool KV config: private_qsa_blocks=(\d+) "
    r"private_qsa_bytes=(\d+) regular_blocks=(\d+) regular_bytes=(\d+)"
)
WORKER_RE = re.compile(
    r"Q2C dual-pool worker allocation: stock_bytes=(\d+) "
    r"private_qsa_bytes=(\d+) private_qsa_blocks=(\d+)"
)
CAP_RE = re.compile(
    r"Maximum concurrency for [0-9,]+ tokens per request:\s*([0-9.]+)x"
)
AVAILABLE_RE = re.compile(r"Available KV cache memory:\s*([0-9.]+) GiB")


def summarize(text: str) -> dict:
    core = CORE_RE.findall(text)
    worker = WORKER_RE.findall(text)
    cap = CAP_RE.findall(text)
    available = AVAILABLE_RE.findall(text)

    expected_private_blocks = 4161
    expected_usable_blocks = 4160
    expected_layers = 12
    expected_layer_page = 32768
    expected_private_bytes = (
        expected_private_blocks * expected_layers * expected_layer_page
    )

    core_row = tuple(map(int, core[-1])) if core else None
    worker_row = tuple(map(int, worker[-1])) if worker else None
    concurrency = float(cap[-1]) if cap else None
    available_gib = float(available[-1]) if available else None
    legacy_1568 = text.count(
        "Setting attention block size to 1568 tokens "
        "to ensure that attention page size is >= mamba page size."
    )

    checks = {
        "platform_skip_seen": PLATFORM in text,
        "core_dual_pool_seen_once": len(core) >= 1,
        "worker_dual_pool_seen_once": len(worker) >= 1,
        "legacy_1568_alignment_absent": legacy_1568 == 0,
        "private_blocks_exact": bool(
            core_row and core_row[0] == expected_private_blocks
        ),
        "private_bytes_exact": bool(
            core_row and core_row[1] == expected_private_bytes
        ),
        "worker_private_matches_core": bool(
            core_row
            and worker_row
            and worker_row[1] == core_row[1]
            and worker_row[2] == core_row[0]
        ),
        "worker_stock_matches_core": bool(
            core_row and worker_row and worker_row[0] == core_row[3]
        ),
        "regular_pool_nonempty": bool(
            core_row and core_row[2] > 0 and core_row[3] > 0
        ),
        "reported_concurrency_at_least_one": bool(
            concurrency is not None and concurrency >= 1.0
        ),
    }
    gate = all(checks.values())
    return {
        "schema": 1,
        "classification": (
            "Q2C_DUAL_POOL_BOOT_GO" if gate else "Q2C_DUAL_POOL_BOOT_NO_GO"
        ),
        "dual_pool_boot_gate": gate,
        "checks": checks,
        "expected": {
            "private_qsa_blocks_total": expected_private_blocks,
            "private_qsa_usable_blocks": expected_usable_blocks,
            "qsa_layers": expected_layers,
            "page_bytes_per_layer": expected_layer_page,
            "private_qsa_bytes": expected_private_bytes,
        },
        "observed": {
            "core_private_qsa_blocks": core_row[0] if core_row else None,
            "core_private_qsa_bytes": core_row[1] if core_row else None,
            "regular_blocks": core_row[2] if core_row else None,
            "regular_bytes": core_row[3] if core_row else None,
            "worker_stock_bytes": worker_row[0] if worker_row else None,
            "worker_private_qsa_bytes": worker_row[1] if worker_row else None,
            "worker_private_qsa_blocks": worker_row[2] if worker_row else None,
            "max_concurrency": concurrency,
            "available_kv_cache_gib": available_gib,
            "legacy_1568_alignment_count": legacy_1568,
            "core_log_matches": len(core),
            "worker_log_matches": len(worker),
        },
        "interpretation": (
            "GO proves boot-time memory/config/allocation domains are split: "
            "the QSA group owns an independent 4161-block arena while the "
            "regular/Mamba groups retain a separate stock arena. It does not "
            "by itself prove request-time ownership or semantics."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    result = summarize(args.log.read_text(errors="replace"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result["dual_pool_boot_gate"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
