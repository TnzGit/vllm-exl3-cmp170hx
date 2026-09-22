#!/usr/bin/env python3
"""Emit the Q2C shared-placeholder vs dedicated-storage sizing contract."""

from __future__ import annotations

import argparse
import json
from math import ceil
from pathlib import Path
from types import SimpleNamespace

import torch

from vllm.v1.core.kv_cache_utils import _max_memory_usage_bytes_from_groups
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_exl3.kvmem_qsa_scheduler_runtime import make_qsa_runtime_spec


def build(plan: dict) -> dict:
    spec = make_qsa_runtime_spec(plan)
    qsa_specs = {f"qsa.{i}": spec for i in range(int(plan["expected_qsa_layers"]))}
    group = UniformTypeKVCacheSpecs.from_specs(qsa_specs)
    if group is None:
        raise RuntimeError("Q2C QSA specs did not form a uniform group")

    cfg = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=161000),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )
    shared_pages = group.max_memory_usage_pages(cfg)

    # Reproduce the first-live-run failure mechanism with the observed
    # hybrid-aligned 1568-token attention page. This is a diagnostic model of
    # global shared-stride charging, not a claim that QSA itself changed to 1568.
    wide = FullAttentionSpec(
        block_size=1568,
        num_kv_heads=2,
        head_size=256,
        head_size_v=256,
        dtype=torch.bfloat16,
    )
    wide_pages = ceil(161000 / 1568)
    groups = [
        KVCacheGroupSpec(list(qsa_specs), group),
        KVCacheGroupSpec(["observed-wide.0"], wide),
    ]
    new_shared_bytes = _max_memory_usage_bytes_from_groups(cfg, groups)
    old_shared_bytes = (
        int(plan["physical_page_count"]) + wide_pages
    ) * max(group.page_size_bytes, wide.page_size_bytes)

    dedicated_bytes_layer = spec.dedicated_memory_bytes_per_layer
    expected_layers = int(plan["expected_qsa_layers"])
    out = {
        "schema": 1,
        "qsa_manager_block_tokens": spec.block_size,
        "qsa_page_size_bytes": spec.page_size_bytes,
        "logical_pages_161k": spec.max_num_blocks_per_req(cfg, 161000),
        "shared_placeholder_pages_per_qsa_group": shared_pages,
        "shared_placeholder_group_page_bytes": group.page_size_bytes,
        "virtual_real_page_cap": spec.physical_page_cap,
        "virtual_null_block_id": spec.virtual_null_block_id,
        "dedicated_pages_per_layer": spec.dedicated_page_count,
        "dedicated_bytes_per_layer": dedicated_bytes_layer,
        "dedicated_mib_per_layer": dedicated_bytes_layer / 2**20,
        "dedicated_gib_all_qsa_layers": dedicated_bytes_layer * expected_layers / 2**30,
        "diagnostic_wide_block_tokens": 1568,
        "diagnostic_wide_pages_161k": wide_pages,
        "diagnostic_wide_page_bytes": wide.page_size_bytes,
        "diagnostic_old_shared_charge_gib": old_shared_bytes / 2**30,
        "diagnostic_new_shared_charge_gib": new_shared_bytes / 2**30,
        "contract_ok": bool(
            spec.block_size == 16
            and shared_pages == 1
            and spec.physical_page_cap == 4160
            and spec.dedicated_page_count == 4161
            and old_shared_bytes > 12 * 2**30
            and new_shared_bytes < 1 * 2**30
        ),
        "note": (
            "The 1568-token value is retained as a diagnostic model of the "
            "observed global hybrid stride. Q2C QSA itself remains 16-token. "
            "Real QSA storage is the model-owned dedicated buffer and is "
            "accounted by GPU memory profiling, not by shared BlockPool pages."
        ),
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    result = build(json.loads(args.plan.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result["contract_ok"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
