#!/usr/bin/env python3
"""Patch installed vLLM Qwen4Exp QSA for K1-Q2C scheduler ownership.

Default-off. With VLLM_QWEN_KVMEM_Q2C_PLAN set:
- QSA reports the out-of-tree 16-token QSAResidentRuntimeSpec;
- the scheduler initially owns the complete 16-token QSA history;
- at the frozen transition boundary the custom manager releases non-resident
  historical blocks into null holes while keeping the logical row complete;
- retained history is published through vLLM generic CPU offload using only a
  reusable staging window, not a second 130 MiB/layer resident shadow;
- after scheduler shrink, retained pages are zeroed and restored from CPU
  backing before sparse attention, proving the host copy is authoritative;
- active writes use the scheduler-provided slot_mapping; sparse reads use the
  same scheduler block_table after masking non-resident historical selections.

Research-only. Eager mode is required and CUDA graph capture fails closed.
"""

from __future__ import annotations

import argparse
import py_compile
import sys
from pathlib import Path


MARKER = "# KVMEM_QSA_Q2C_RUNTIME_V1"
TARGET = Path("models/qwen4_exp/nvidia/qsa.py")

IMPORT_ANCHOR = "from __future__ import annotations\n\n"
IMPORT_BLOCK = """from __future__ import annotations

import json
import os
from pathlib import Path

"""

HELPER_ANCHOR = "from .indexer_qsa import QSAIndexer\n\n"
HELPER = r'''from .indexer_qsa import QSAIndexer

# KVMEM_QSA_Q2C_RUNTIME_V1
_Q2C_PLAN_CACHE = None
_Q2C_TENSOR_CACHE = {}


def _q2c_plan():
    global _Q2C_PLAN_CACHE
    path = os.environ.get("VLLM_QWEN_KVMEM_Q2C_PLAN")
    if not path:
        return None
    cached = _Q2C_PLAN_CACHE
    if cached is not None and cached[0] == path:
        return cached[1]
    from vllm_exl3.kvmem_qsa_scheduler_runtime import load_runtime_plan
    plan = load_runtime_plan(path)
    _Q2C_PLAN_CACHE = (path, plan)
    _Q2C_TENSOR_CACHE.clear()
    return plan


def _q2c_spec(plan):
    from vllm_exl3.kvmem_qsa_scheduler_runtime import make_qsa_runtime_spec
    return make_qsa_runtime_spec(plan)


def _q2c_tensor(plan, name, values, device, dtype=torch.int64):
    key = (name, str(device))
    out = _Q2C_TENSOR_CACHE.get(key)
    if out is None:
        out = torch.tensor(values, dtype=dtype, device=device)
        _Q2C_TENSOR_CACHE[key] = out
    return out


def _q2c_stats(payload):
    path = os.environ.get("VLLM_QWEN_KVMEM_Q2C_WORKER_STATS_PATH")
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, separators=(",", ":")) + "\n")


def _q2c_apply_visibility(layer, plan, selected, positions, block_table):
    """Mask selections whose logical page is already a scheduler null hole.

    This is intentionally driven by actual block-table ownership, not merely
    by the frozen resident list. Pages in the current unprocessed chunk still
    have real block IDs and remain visible for causal prefill.
    """
    if block_table.ndim != 2 or block_table.shape[0] != 1:
        raise RuntimeError("Q2C runtime currently requires one request")

    page_tokens = int(plan["page_tokens"])
    pos = positions.to(device=selected.device, dtype=torch.int64)
    valid = selected >= 0
    if not bool(valid.any().item()):
        apply_rows = pos >= int(plan["apply_min_pos"])
        return apply_rows, 0, 0, 0

    # Only pages strictly before the current forward's first page are guaranteed
    # processed and therefore eligible for progressive reclamation.
    first_query_page = int(pos.min().item()) // page_tokens if pos.numel() else 0
    hist_end = min(first_query_page, int(plan["active_page0"]))
    resident_set = set(int(x) for x in plan["resident_pages"])
    first_hole = None
    for page in range(hist_end):
        if page not in resident_set:
            first_hole = page
            break

    null_id = (
        block_table[0, first_hole].to(torch.int64)
        if first_hole is not None
        else None
    )

    logical = selected.clamp_min(0).to(torch.int64)
    pages = torch.div(logical, page_tokens, rounding_mode="floor")
    if bool(valid.any().item()):
        max_page = int(pages[valid].max().item())
        if max_page >= int(block_table.shape[1]):
            raise RuntimeError(
                f"Q2C selected page {max_page} exceeds block-table width "
                f"{block_table.shape[1]}"
            )
    page_ids = block_table[0].index_select(0, pages.reshape(-1)).reshape_as(pages)
    historical = logical < int(plan["active_from_pos"])
    processed_history = pages < hist_end
    drop = (
        valid & historical & processed_history & (page_ids == null_id)
        if null_id is not None
        else torch.zeros_like(valid)
    )

    historical_total = int((valid & historical).sum().item())
    dropped = int(drop.sum().item())
    historical_kept = historical_total - dropped
    selected.masked_fill_(drop, -1)

    apply_rows = pos >= int(plan["apply_min_pos"])
    if bool((pos < int(plan["apply_min_pos"])).any().item()):
        layer._q2c_prefill_dropped = int(
            getattr(layer, "_q2c_prefill_dropped", 0)
        ) + dropped
        layer._q2c_prefill_historical = int(
            getattr(layer, "_q2c_prefill_historical", 0)
        ) + historical_total
    return apply_rows, historical_total, historical_kept, dropped


def _q2c_table_evidence(layer, plan, block_table, positions):
    if block_table.ndim != 2 or block_table.shape[0] != 1:
        raise RuntimeError("Q2C runtime currently requires one request")
    page_tokens = int(plan["page_tokens"])
    max_pos = int(positions.max().item()) if positions.numel() else -1
    logical_pages = max(0, (max_pos + 1 + page_tokens - 1) // page_tokens)
    logical_pages = min(logical_pages, int(block_table.shape[1]))
    active_page0 = int(plan["active_page0"])
    resident = [int(x) for x in plan["resident_pages"] if int(x) < logical_pages]
    resident_idx = _q2c_tensor(
        plan, f"resident_pages_prefix_{len(resident)}", resident, block_table.device
    )
    resident_ids = (
        block_table[0].index_select(0, resident_idx).to(torch.int64)
        if resident else torch.empty(0, dtype=torch.int64, device=block_table.device)
    )

    hist_end = min(active_page0, logical_pages)
    resident_set = set(resident)
    holes = [i for i in range(hist_end) if i not in resident_set]
    hole_idx = _q2c_tensor(plan, f"holes_{hist_end}", holes, block_table.device)
    hole_ids = (
        block_table[0].index_select(0, hole_idx).to(torch.int64)
        if holes else torch.empty(0, dtype=torch.int64, device=block_table.device)
    )
    hole_unique = torch.unique(hole_ids) if hole_ids.numel() else hole_ids
    resident_unique = torch.unique(resident_ids) if resident_ids.numel() else resident_ids

    active_end = logical_pages
    active_pages = max(0, active_end - active_page0)
    active_ids = (
        block_table[0, active_page0:active_end].to(torch.int64)
        if active_pages else torch.empty(0, dtype=torch.int64, device=block_table.device)
    )
    active_unique = torch.unique(active_ids) if active_ids.numel() else active_ids

    shrunk = bool(
        holes
        and hole_unique.numel() == 1
        and resident_unique.numel() == len(resident)
        and (
            resident_unique.numel() == 0
            or not bool(torch.isin(hole_unique, resident_unique).any().item())
        )
    )
    real_ids = torch.cat((resident_unique, active_unique))
    real_unique = torch.unique(real_ids) if real_ids.numel() else real_ids
    return {
        "shrunk": shrunk,
        "logical_pages": logical_pages,
        "resident_history_pages": len(resident),
        "active_real_pages": int(active_unique.numel()),
        "scheduler_real_pages": int(real_unique.numel()),
        "hole_pages": len(holes),
        "hole_unique_ids": int(hole_unique.numel()),
        "null_block_id": int(hole_unique[0].item()) if shrunk else None,
        "resident_physical_ids": resident_ids,
    }


def _q2c_backing(layer, plan, kv_cache):
    backing = getattr(layer, "_q2c_cpu_backing", None)
    if backing is not None:
        return backing, layer._q2c_staging
    staging_pages = int(plan.get("publication_staging_pages", 128))
    if staging_pages <= 0:
        raise RuntimeError("Q2C staging page count must be positive")
    expected = (
        staging_pages,
        int(kv_cache.shape[1]),
        int(kv_cache.shape[2]),
        int(kv_cache.shape[3]),
    )
    staging = torch.empty(expected, dtype=kv_cache.dtype, device=kv_cache.device)
    page_size_bytes = int(staging[0].numel() * staging.element_size())
    if page_size_bytes != 32768:
        raise RuntimeError(
            f"Q2C scheduler page geometry mismatch: {page_size_bytes} != 32768"
        )
    from vllm_exl3.kvmem_vllm_offload import single_tensor_cpu_backing
    backing = single_tensor_cpu_backing(
        tensor=staging,
        page_size_bytes=page_size_bytes,
        num_cpu_blocks=int(plan["resident_page_count"]),
        lineage=f"q2c:{layer.layer_name}",
    )
    layer._q2c_cpu_backing = backing
    layer._q2c_staging = staging
    layer._q2c_page_size_bytes = page_size_bytes
    return backing, staging


def _q2c_publish_history(layer, plan, kv_cache, block_table):
    if getattr(layer, "_q2c_cpu_published", False):
        return
    backing, staging = _q2c_backing(layer, plan, kv_cache)
    resident = [int(x) for x in plan["resident_pages"]]
    chunk = int(staging.shape[0])
    d2h_bytes = 0
    jobs = 0
    for start in range(0, len(resident), chunk):
        pages = resident[start : start + chunk]
        idx = torch.tensor(pages, dtype=torch.int64, device=block_table.device)
        phys = block_table[0].index_select(0, idx).to(torch.int64)
        n = len(pages)
        staging[:n].copy_(kv_cache.index_select(0, phys))
        torch.cuda.synchronize()
        obs = backing.publish(pages, list(range(n)))
        d2h_bytes += int(obs.transfer_bytes)
        jobs += int(obs.job_id != 0)
    if not backing.all_present(resident):
        raise RuntimeError("Q2C CPU backing misses published resident history")
    expected = len(resident) * int(layer._q2c_page_size_bytes)
    if d2h_bytes != expected:
        raise RuntimeError(f"Q2C D2H byte mismatch {d2h_bytes} != {expected}")
    layer._q2c_cpu_published = True
    layer._q2c_d2h_bytes = d2h_bytes
    layer._q2c_d2h_jobs = jobs


def _q2c_restore_history_from_cpu(layer, plan, kv_cache, block_table):
    if getattr(layer, "_q2c_cpu_restored_after_shrink", False):
        return
    if not getattr(layer, "_q2c_cpu_published", False):
        _q2c_publish_history(layer, plan, kv_cache, block_table)
    backing, staging = _q2c_backing(layer, plan, kv_cache)
    resident = [int(x) for x in plan["resident_pages"]]
    chunk = int(staging.shape[0])
    h2d_bytes = 0
    jobs = 0
    exact = True
    first_bad = None
    for start in range(0, len(resident), chunk):
        pages = resident[start : start + chunk]
        idx = torch.tensor(pages, dtype=torch.int64, device=block_table.device)
        phys = block_table[0].index_select(0, idx).to(torch.int64)
        n = len(pages)
        # Keep only one staging-window-sized reference at a time. This proves
        # the CPU round trip restored the bytes that were resident immediately
        # before we deliberately destroy the GPU copy, without recreating a
        # persistent 130 MiB/layer shadow.
        expected = kv_cache.index_select(0, phys).clone()
        zeros = torch.zeros_like(staging[:n])
        kv_cache.index_copy_(0, phys, zeros)
        torch.cuda.synchronize()
        obs = backing.stage_in(pages, list(range(n)))
        h2d_bytes += int(obs.transfer_bytes)
        jobs += int(obs.job_id != 0)
        same = torch.eq(staging[:n], expected).reshape(n, -1).all(dim=1)
        kv_cache.index_copy_(0, phys, staging[:n])
        if not bool(same.all().item()):
            exact = False
            bad = int(torch.nonzero(~same, as_tuple=False)[0, 0].item())
            first_bad = pages[bad]
            break
    expected = len(resident) * int(layer._q2c_page_size_bytes)
    if h2d_bytes != expected:
        raise RuntimeError(f"Q2C H2D byte mismatch {h2d_bytes} != {expected}")
    if not exact:
        raise RuntimeError(f"Q2C CPU authoritative restore mismatch page={first_bad}")
    layer._q2c_cpu_restored_after_shrink = True
    layer._q2c_h2d_bytes = h2d_bytes
    layer._q2c_h2d_jobs = jobs
    layer._q2c_restore_exact = exact


def _q2c_run(
    layer,
    impl,
    plan,
    query,
    key,
    value,
    output,
    selected,
    positions,
    main_metadata,
    side_metadata,
):
    if query.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Q2C runtime requires eager execution")

    # Writes always use scheduler ownership. Before transition this is the full
    # logical QSA row; after transition it addresses only retained/active real
    # blocks while null holes remain in the logical row.
    impl.do_kv_cache_update(
        layer, key, value, layer.kv_cache, main_metadata.slot_mapping
    )

    apply_rows = positions.to(torch.int64) >= int(plan["apply_min_pos"])
    if bool(apply_rows.any().item()):
        _q2c_publish_history(layer, plan, layer.kv_cache, main_metadata.block_table)

    evidence = _q2c_table_evidence(
        layer, plan, main_metadata.block_table, positions
    )
    if evidence["shrunk"]:
        _q2c_restore_history_from_cpu(
            layer, plan, layer.kv_cache, main_metadata.block_table
        )
        if evidence["scheduler_real_pages"] > int(plan["physical_page_count"]):
            raise RuntimeError(
                "Q2C worker sees scheduler real pages above physical cap: "
                f'{evidence["scheduler_real_pages"]} > {plan["physical_page_count"]}'
            )

    # Apply the bounded visibility policy on every forward. Before the frozen
    # query boundary this masks only pages the scheduler has already reclaimed;
    # current work pages remain real and visible.
    apply_rows, hist_total, hist_kept, dropped = _q2c_apply_visibility(
        layer, plan, selected, positions, main_metadata.block_table
    )

    impl.forward_qsa(
        layer,
        query,
        key,
        value,
        layer.kv_cache,
        main_metadata,
        output,
        token_to_req=side_metadata.token_to_req,
    )

    if bool(apply_rows.any().item()):
        _q2c_stats(
            {
                "layer": layer.layer_name,
                "phase": "shrunk" if evidence["shrunk"] else "full_source",
                "logical_pages": int(evidence["logical_pages"]),
                "scheduler_real_pages": int(evidence["scheduler_real_pages"]),
                "physical_page_cap": int(plan["physical_page_count"]),
                "resident_history_pages": int(evidence["resident_history_pages"]),
                "active_real_pages": int(evidence["active_real_pages"]),
                "hole_pages": int(evidence["hole_pages"]),
                "hole_unique_ids": int(evidence["hole_unique_ids"]),
                "null_block_id": evidence["null_block_id"],
                "historical_selected": int(hist_total),
                "historical_resident_kept": int(hist_kept),
                "historical_selected_dropped": int(dropped),
                "prefill_historical_selected": int(
                    getattr(layer, "_q2c_prefill_historical", 0)
                ),
                "prefill_historical_selected_dropped": int(
                    getattr(layer, "_q2c_prefill_dropped", 0)
                ),
                "cpu_published": bool(getattr(layer, "_q2c_cpu_published", False)),
                "cpu_restored_after_shrink": bool(
                    getattr(layer, "_q2c_cpu_restored_after_shrink", False)
                ),
                "cpu_restore_exact": bool(getattr(layer, "_q2c_restore_exact", False)),
                "d2h_bytes": int(getattr(layer, "_q2c_d2h_bytes", 0)),
                "d2h_jobs": int(getattr(layer, "_q2c_d2h_jobs", 0)),
                "h2d_bytes": int(getattr(layer, "_q2c_h2d_bytes", 0)),
                "h2d_jobs": int(getattr(layer, "_q2c_h2d_jobs", 0)),
                "staging_pages": int(
                    getattr(layer, "_q2c_staging", torch.empty(0)).shape[0]
                ),
                "staging_bytes": int(
                    getattr(layer, "_q2c_staging", torch.empty(0)).numel()
                    * getattr(layer, "_q2c_staging", torch.empty(0)).element_size()
                ),
                "private_cache_blocks": int(layer.kv_cache.shape[0]),
                "private_cache_storage_bytes": int(
                    layer.kv_cache.untyped_storage().nbytes()
                ),
                "private_cache_block_stride_bytes": int(
                    layer.kv_cache.stride(0) * layer.kv_cache.element_size()
                ),
            }
        )

'''

SPEC_ANCHOR = """    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )
"""

SPEC_BLOCK = """    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        _q2c_plan_obj = _q2c_plan()
        if _q2c_plan_obj is not None:
            return _q2c_spec(_q2c_plan_obj)
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )
"""

RUN_ANCHOR = """        impl = cast(Qwen4ExpQSAFlashAttentionImpl, self.impl)
        impl.do_kv_cache_update(
            self,
            key,
            value,
            self.kv_cache,
            main_metadata.slot_mapping,
        )
        impl.forward_qsa(
            self,
            query,
            key,
            value,
            self.kv_cache,
            main_metadata,
            output,
            token_to_req=side_metadata.token_to_req,
        )
"""

RUN_BLOCK = """        impl = cast(Qwen4ExpQSAFlashAttentionImpl, self.impl)
        _q2c_plan_obj = _q2c_plan()
        if _q2c_plan_obj is None:
            impl.do_kv_cache_update(
                self,
                key,
                value,
                self.kv_cache,
                main_metadata.slot_mapping,
            )
            impl.forward_qsa(
                self,
                query,
                key,
                value,
                self.kv_cache,
                main_metadata,
                output,
                token_to_req=side_metadata.token_to_req,
            )
        else:
            _q2c_run(
                self,
                impl,
                _q2c_plan_obj,
                query,
                key,
                value,
                output,
                selected,
                side_metadata.logical_positions[:num_tokens],
                main_metadata,
                side_metadata,
            )
"""


def patch(path: Path, *, check_only: bool = False) -> str:
    src = path.read_text(encoding="utf-8")
    if MARKER in src:
        return "already patched"
    for name, anchor in (
        ("import", IMPORT_ANCHOR),
        ("helper", HELPER_ANCHOR),
        ("spec", SPEC_ANCHOR),
        ("run", RUN_ANCHOR),
    ):
        if src.count(anchor) != 1:
            raise RuntimeError(f"{name} anchor count != 1")

    out = src.replace(IMPORT_ANCHOR, IMPORT_BLOCK, 1)
    out = out.replace(HELPER_ANCHOR, HELPER, 1)
    out = out.replace(SPEC_ANCHOR, SPEC_BLOCK, 1)
    out = out.replace(RUN_ANCHOR, RUN_BLOCK, 1)

    required = (
        MARKER,
        "VLLM_QWEN_KVMEM_Q2C_PLAN",
        "make_qsa_runtime_spec",
        "Q2C runtime requires eager execution",
        "_q2c_table_evidence",
        "_q2c_publish_history",
        "_q2c_restore_history_from_cpu",
        "single_tensor_cpu_backing",
        "index_copy_",
        "scheduler_real_pages",
        "physical_page_cap",
        "historical_selected_dropped",
        "cpu_restored_after_shrink",
        "private_cache_storage_bytes",
        "private_cache_block_stride_bytes",
    )
    missing = [x for x in required if x not in out]
    if missing:
        raise RuntimeError(f"Q2C postcondition missing: {missing}")

    if check_only:
        compile(out, str(path), "exec")
        return "anchors/postconditions validated; no files changed"

    original = path.with_suffix(path.suffix + ".kvmem_qsa_q2c_runtime.orig")
    if not original.exists():
        original.write_text(src, encoding="utf-8")
    path.write_text(out, encoding="utf-8")
    py_compile.compile(str(path), doraise=True)
    return "patched"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("vllm_root", type=Path)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    target = args.vllm_root.resolve() / TARGET
    if not target.is_file():
        print(f"ERROR: missing QSA source: {target}", file=sys.stderr)
        return 2
    try:
        status = patch(target, check_only=args.check_only)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"PASS: K1-Q2C QSA runtime {status}: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
