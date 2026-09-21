#!/usr/bin/env python3
"""Read-only capability preflight for the first real K1 bounded-KV prototype.

This script does not start an engine, initialize CUDA, or modify installed
vLLM. It answers which upstream ownership/transfer interfaces are present in
the local vLLM wheel so the K1 implementation can reuse them instead of
inventing a parallel block manager.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import json
from pathlib import Path
from typing import Any


CAPABILITIES = {
    "hisparse_coordinator": (
        "vllm.v1.hisparse.coordinator",
        ("HiSparseCoordinator",),
    ),
    "hisparse_connector": (
        "vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector",
        ("HiSparseConnector",),
    ),
    "hisparse_managers": (
        "vllm.v1.core.single_type_kv_cache_manager",
        (
            "HiSparseResidentManager",
            "HiSparseHotManager",
            "HiSparseSourceManager",
        ),
    ),
    "hisparse_specs": (
        "vllm.v1.kv_cache_interface",
        (
            "HiSparseResidentSpec",
            "HiSparseHotSpec",
        ),
    ),
    "offloading_connector": (
        "vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector",
        ("OffloadingConnector",),
    ),
    "cpu_offload_spec": (
        "vllm.v1.kv_offload.cpu.spec",
        ("CPUOffloadingSpec",),
    ),
    "cpu_offload_manager": (
        "vllm.v1.kv_offload.cpu.manager",
        ("CPUOffloadingManager",),
    ),
}


def _probe_module(module_name: str, attrs: tuple[str, ...]) -> dict[str, Any]:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return {
            "module": module_name,
            "found": False,
            "imported": False,
            "origin": None,
            "attrs": {name: False for name in attrs},
            "error": "module spec not found",
        }

    result: dict[str, Any] = {
        "module": module_name,
        "found": True,
        "origin": spec.origin,
        "imported": False,
        "attrs": {name: False for name in attrs},
        "error": None,
    }
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # read-only diagnostic: report, do not mask
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    result["imported"] = True
    attr_info = {}
    for name in attrs:
        obj = getattr(module, name, None)
        attr_info[name] = {
            "present": obj is not None,
            "module": getattr(obj, "__module__", None) if obj is not None else None,
            "qualname": getattr(obj, "__qualname__", None) if obj is not None else None,
        }
    result["attrs"] = attr_info
    return result


def _all_attrs_present(entry: dict[str, Any]) -> bool:
    if not entry.get("imported"):
        return False
    attrs = entry.get("attrs") or {}
    return bool(attrs) and all(
        bool(v.get("present")) if isinstance(v, dict) else bool(v)
        for v in attrs.values()
    )


def _qwen_package_root() -> Path | None:
    for name in (
        "vllm.models.qwen4_exp",
        "vllm.model_executor.models.qwen4_exp",
    ):
        spec = importlib.util.find_spec(name)
        if spec is None:
            continue
        if spec.submodule_search_locations:
            return Path(next(iter(spec.submodule_search_locations))).resolve()
        if spec.origin:
            return Path(spec.origin).resolve().parent
    return None


def _source_hits(root: Path | None) -> dict[str, list[dict[str, Any]]]:
    needles = (
        "HiSparse",
        "OffloadingConnector",
        "CPUOffloading",
        "QSAIndexer",
        "qsa_select",
        "get_kv_cache_spec",
        "KVCacheSpec",
    )
    out = {needle: [] for needle in needles}
    if root is None or not root.exists():
        return out

    for path in sorted(root.rglob("*.py")):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        for lineno, line in enumerate(lines, 1):
            for needle in needles:
                if needle in line:
                    out[needle].append(
                        {
                            "path": str(path),
                            "line": lineno,
                            "text": line.strip()[:240],
                        }
                    )
    return out


def _inspect_signatures(entries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    targets = (
        (
            "hisparse_connector",
            "HiSparseConnector",
            (
                "__init__",
                "bind_kv_cache_manager",
                "register_kv_caches",
                "start_load_kv",
                "build_connector_meta",
            ),
        ),
        (
            "hisparse_coordinator",
            "HiSparseCoordinator",
            (
                "__init__",
                "build_offload_command",
                "update_residency",
            ),
        ),
        (
            "offloading_connector",
            "OffloadingConnector",
            (
                "__init__",
                "register_kv_caches",
                "start_load_kv",
            ),
        ),
    )
    result: dict[str, Any] = {}
    for cap_name, class_name, methods in targets:
        entry = entries.get(cap_name) or {}
        if not entry.get("imported"):
            continue
        try:
            module = importlib.import_module(entry["module"])
            cls = getattr(module, class_name)
        except Exception:
            continue
        key = f"{entry['module']}.{class_name}"
        row: dict[str, Any] = {}
        for method in methods:
            obj = getattr(cls, method, None)
            if obj is None:
                row[method] = None
                continue
            try:
                row[method] = str(inspect.signature(obj))
            except Exception:
                row[method] = "<signature unavailable>"
        result[key] = row
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import vllm

    entries = {
        name: _probe_module(module, attrs)
        for name, (module, attrs) in CAPABILITIES.items()
    }
    qwen_root = _qwen_package_root()
    hits = _source_hits(qwen_root)

    hisparse_keys = (
        "hisparse_coordinator",
        "hisparse_connector",
        "hisparse_managers",
        "hisparse_specs",
    )
    hisparse_core = all(_all_attrs_present(entries[k]) for k in hisparse_keys)
    generic_offload = (
        _all_attrs_present(entries["offloading_connector"])
        and _all_attrs_present(entries["cpu_offload_spec"])
        and _all_attrs_present(entries["cpu_offload_manager"])
    )
    qwen_mentions_hisparse = bool(hits["HiSparse"])

    if hisparse_core:
        route = "adapt_hisparse_contract"
    elif generic_offload:
        route = "extend_generic_offloading_contract"
    else:
        route = "custom_vllm_patch_required"

    result = {
        "schema": 1,
        "vllm": {
            "version": getattr(vllm, "__version__", None),
            "package_file": str(Path(vllm.__file__).resolve()),
        },
        "capabilities": entries,
        "signatures": _inspect_signatures(entries),
        "qwen4_exp": {
            "package_root": str(qwen_root) if qwen_root else None,
            "source_hits": hits,
            "mentions_hisparse": qwen_mentions_hisparse,
        },
        "decision_support": {
            "hisparse_core_available": hisparse_core,
            "generic_cpu_offloading_available": generic_offload,
            "qwen4_exp_mentions_hisparse": qwen_mentions_hisparse,
            "recommended_integration_route": route,
            "warning": (
                "Availability is not compatibility. A positive HiSparse probe "
                "means its ownership/transfer contracts can be reused or adapted; "
                "it does not prove Qwen4Exp cache layout compatibility."
            ),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
