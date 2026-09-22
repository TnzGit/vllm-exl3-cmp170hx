#!/usr/bin/env python3
"""Read-only Q2C preflight for scheduler-visible bounded QSA KV ownership.

No engine, CUDA initialization, or installed-source mutation is performed.
The probe validates the vLLM 0.29 extension/allocator contracts that Q2C needs
before a real scheduler/cache-manager patch is attempted.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from vllm.config.cache import CacheConfig
from vllm.v1.core import kv_cache_utils
from vllm.v1.core import single_type_kv_cache_manager as single_mgr
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

from vllm_exl3.kvmem_qsa_scheduler_contract import (
    QSAResidentContractSpec,
    QSAResidentRegistryProbeManager,
    make_qsa_resident_contract_spec,
    q2c_geometry,
    register_qsa_resident_contract,
)


def _qsa_source() -> tuple[str, Path, str]:
    candidates = (
        "vllm.models.qwen4_exp.nvidia.qsa",
        "vllm.model_executor.models.qwen4_exp.nvidia.qsa",
    )
    for name in candidates:
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            continue
        path = Path(spec.origin).resolve()
        return name, path, path.read_text(encoding="utf-8")
    raise RuntimeError("cannot resolve installed Qwen4Exp QSA source")


def _probe_packed_grouping() -> dict:
    cache = CacheConfig()
    cache.kv_cache_layout = "BLHNC"
    resolved = cache.get_resolved_kv_cache_layout()
    fake = SimpleNamespace(
        cache_config=cache,
        speculative_config=None,
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen4_exp")),
    )
    resident = make_qsa_resident_contract_spec()
    stock = FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=256,
        head_size_v=256,
        dtype=torch.bfloat16,
    )
    specs = {"stock.attn": stock}
    for i in range(12):
        specs[f"qsa.{i}"] = resident
    groups = kv_cache_utils._get_packed_kv_cache_groups(fake, specs)
    if not groups:
        return {"ok": False, "reason": "packed grouping returned no groups"}
    qsa_groups = [g for g in groups if any(x.startswith("qsa.") for x in g.layer_names)]
    qsa_specs = []
    for group in qsa_groups:
        gs = group.kv_cache_spec
        if hasattr(gs, "kv_cache_specs"):
            qsa_specs.extend(
                spec for name, spec in gs.kv_cache_specs.items() if name.startswith("qsa.")
            )
        elif any(x.startswith("qsa.") for x in group.layer_names):
            qsa_specs.append(gs)
    return {
        "ok": bool(qsa_specs)
        and all(isinstance(s, QSAResidentContractSpec) for s in qsa_specs)
        and all(s.block_size == 16 and s.page_size_bytes == 32768 for s in qsa_specs),
        "resolved_layout": resolved.name,
        "group_count": len(groups),
        "qsa_group_count": len(qsa_groups),
        "qsa_layer_specs_seen": len(qsa_specs),
        "qsa_block_sizes": sorted({int(s.block_size) for s in qsa_specs}),
        "qsa_page_sizes": sorted({int(s.page_size_bytes) for s in qsa_specs}),
    }


def build_result(observed_full_block_tokens: int) -> dict:
    import vllm

    qsa_module, qsa_path, qsa_src = _qsa_source()
    utils_src = inspect.getsource(kv_cache_utils)
    mgr_src = inspect.getsource(single_mgr.SingleTypeKVCacheManager)

    register_qsa_resident_contract()
    contract_spec = make_qsa_resident_contract_spec()
    manager_cls = KVCacheSpecRegistry.get_manager_class(contract_spec)
    base_spec_cls = KVCacheSpecRegistry.get_uniform_type_base_spec(contract_spec)
    stock_probe = FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=256,
        head_size_v=256,
        dtype=torch.bfloat16,
    )
    stock_manager_cls = KVCacheSpecRegistry.get_manager_class(stock_probe)

    packed_pos = utils_src.find("if packed_groups := _get_packed_kv_cache_groups")
    unify_pos = utils_src.find("filtered_spec = unify_kv_cache_spec_page_size")
    packed_before_unify = packed_pos >= 0 and unify_pos >= 0 and packed_pos < unify_pos

    block_outermost = [
        x.name for x in KVCacheLayout if x.is_block_outermost and x.is_block_compact
    ]
    geometry = q2c_geometry(161_000)
    geometry_240k = q2c_geometry(240_000)
    ratio = observed_full_block_tokens // int(geometry["resident_page_tokens"])

    checks = {
        "vllm_029": str(getattr(vllm, "__version__", "")).startswith("0.29"),
        "qsa_currently_returns_full_attention_spec": (
            "def get_kv_cache_spec" in qsa_src and "return FullAttentionSpec(" in qsa_src
        ),
        "registry_custom_spec_registered": manager_cls is QSAResidentRegistryProbeManager,
        "registry_uniform_base_is_custom": base_spec_cls is QSAResidentContractSpec,
        "builtin_full_attention_registry_preserved": (
            stock_manager_cls is single_mgr.FullAttentionManager
        ),
        "logical_table_exceeds_physical_cap": (
            int(geometry["logical_table_pages"]) > int(geometry["physical_page_cap"])
        ),
        "q2b_physical_geometry_exact": (
            int(geometry["page_size_bytes"]) == 32768
            and int(geometry["history_pages"]) == 4096
            and int(geometry["active_pages"]) == 64
            and int(geometry["physical_page_cap"]) == 4160
            and abs(float(geometry["bounded_mib_per_layer"]) - 130.0) < 1e-9
        ),
        "observed_full_to_resident_page_ratio_exact": (
            observed_full_block_tokens % int(geometry["resident_page_tokens"]) == 0
            and ratio == 98
        ),
        "packed_group_path_precedes_page_unify": packed_before_unify,
        "block_outermost_layout_available": bool(block_outermost),
        "base_manager_supports_null_holes": all(
            needle in mgr_src
            for needle in ("self._null_block", "_remove_blocks_in_range", "blocks[i] = self._null_block")
        ),
    }

    grouping = _probe_packed_grouping()
    checks["custom_16t_spec_survives_packed_grouping"] = bool(grouping.get("ok"))
    go = all(checks.values())

    return {
        "schema": 1,
        "classification": (
            "Q2C_SCHEDULER_SHRINK_PREFLIGHT_GO"
            if go
            else "Q2C_SCHEDULER_SHRINK_PREFLIGHT_NO_GO"
        ),
        "go": go,
        "vllm": {
            "version": getattr(vllm, "__version__", None),
            "package_file": str(Path(vllm.__file__).resolve()),
        },
        "installed_qsa": {
            "module": qsa_module,
            "path": str(qsa_path),
            "current_owner_spec": "FullAttentionSpec",
        },
        "geometry_161k": geometry,
        "geometry_240k": geometry_240k,
        "prior_q2b_observation": {
            "full_source_page_tokens": observed_full_block_tokens,
            "resident_page_tokens": int(geometry["resident_page_tokens"]),
            "cross_granularity_ratio": ratio,
        },
        "registry": {
            "spec": f"{QSAResidentContractSpec.__module__}.{QSAResidentContractSpec.__qualname__}",
            "manager": f"{manager_cls.__module__}.{manager_cls.__qualname__}" if manager_cls else None,
            "uniform_base": (
                f"{base_spec_cls.__module__}.{base_spec_cls.__qualname__}"
                if base_spec_cls else None
            ),
        },
        "layout": {
            "block_outermost_compact_layouts": block_outermost,
            "packed_probe": grouping,
        },
        "checks": checks,
        "implementation_boundary": {
            "proven_by_this_preflight": [
                "vLLM 0.29 can register an out-of-tree QSA cache spec/manager",
                "logical block-table width can remain full-history while spec physical accounting is bounded",
                "16-token/32KiB QSA pages survive block-outermost packed grouping without page-size inflation",
                "base manager representation supports null holes in logical block tables",
            ],
            "not_yet_proven": [
                "real scheduler allocations stay within the 4160-page cap",
                "Q2B CPU backing is authoritative after full QSA source removal",
                "READ block-table remap and WRITE slot mapping coexist in one engine",
                "memory/concurrency improves on hardware",
                "MTP follower correctness",
            ],
            "next_runtime_shape": (
                "QSA get_kv_cache_spec -> custom 16-token resident spec; "
                "custom manager/coordinator keeps full logical row with null host-backed holes; "
                "Q2B CPU backing supplies history; active writes retain dedicated resident slots."
            ),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--observed-full-block-tokens", type=int, default=1568)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    result = build_result(args.observed_full_block_tokens)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result["go"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
