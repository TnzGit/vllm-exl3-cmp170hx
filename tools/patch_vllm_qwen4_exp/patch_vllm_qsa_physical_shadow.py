#!/usr/bin/env python3
"""Patch vLLM Qwen4Exp QSA for K1-Q2A physical-shadow resident cache.

Default-off. When VLLM_QWEN_KVMEM_PHYSICAL_PLAN is set, affected query/decode
rows use an independent bounded physical KV cache. The scheduler-owned full KV
cache remains intact and is used as both bootstrap source and same-forward
reference.

For each affected QSA call:
1. update the normal full KV cache;
2. mask nonresident historical selected indices exactly as K1-Q1;
3. lazily bootstrap resident historical pages D2D from the full cache;
4. write active suffix K/V into remapped resident slots;
5. compute full-cache reference attention and resident-cache attention using
   the exact same selected indices;
6. require bit-exact affected-row attention outputs;
7. publish resident output to the model.

This is an eager diagnostic. CUDA graph capture fails closed.
"""

from __future__ import annotations

import argparse
import py_compile
import sys
from pathlib import Path


MARKER = "# KVMEM_QSA_PHYSICAL_SHADOW_V1"
TARGET = Path("models/qwen4_exp/nvidia/qsa.py")

IMPORT_ANCHOR = "from __future__ import annotations\n\n"
IMPORT_BLOCK = """from __future__ import annotations

import json
import os
from pathlib import Path

"""

HELPER_ANCHOR = "from .indexer_qsa import QSAIndexer\n\n"

HELPER = r'''from .indexer_qsa import QSAIndexer

# KVMEM_QSA_PHYSICAL_SHADOW_V1
_KVMEM_PHYSICAL_PLAN_CACHE = None
_KVMEM_PHYSICAL_TENSOR_CACHE = {}


def _kvmem_physical_plan():
    global _KVMEM_PHYSICAL_PLAN_CACHE
    path = os.environ.get("VLLM_QWEN_KVMEM_PHYSICAL_PLAN")
    if not path:
        return None
    cached = _KVMEM_PHYSICAL_PLAN_CACHE
    if cached is not None and cached[0] == path:
        return cached[1]

    plan = json.loads(Path(path).read_text())
    if int(plan.get("schema", 0)) != 1:
        raise RuntimeError("KVMem physical plan schema mismatch")
    if plan.get("mode") != "qsa_physical_shadow":
        raise RuntimeError("KVMem physical plan mode mismatch")

    for field in (
        "region_tokens",
        "page_tokens",
        "resident_page_count",
        "physical_page_count",
        "active_from_pos",
        "active_page0",
        "active_reserve_pages",
    ):
        if int(plan[field]) <= 0:
            raise RuntimeError(f"KVMem physical plan invalid {field}")

    resident_pages = [int(x) for x in plan["resident_pages"]]
    if len(resident_pages) != int(plan["resident_page_count"]):
        raise RuntimeError("KVMem resident page count mismatch")
    if len(set(resident_pages)) != len(resident_pages):
        raise RuntimeError("KVMem resident pages contain duplicates")
    if any(x < 0 for x in resident_pages):
        raise RuntimeError("KVMem resident pages contain negative ids")
    if int(plan["active_from_pos"]) % int(plan["page_tokens"]):
        raise RuntimeError("KVMem active boundary must be KV-page aligned")

    normalized = dict(plan)
    normalized["resident_pages"] = resident_pages
    _KVMEM_PHYSICAL_PLAN_CACHE = (path, normalized)
    _KVMEM_PHYSICAL_TENSOR_CACHE.clear()
    return normalized


def _kvmem_plan_tensor(plan, name, values, device, dtype=torch.int64):
    key = (name, str(device))
    cached = _KVMEM_PHYSICAL_TENSOR_CACHE.get(key)
    if cached is None:
        cached = torch.tensor(values, dtype=dtype, device=device)
        _KVMEM_PHYSICAL_TENSOR_CACHE[key] = cached
    return cached


def _kvmem_stats(payload):
    path = os.environ.get("VLLM_QWEN_KVMEM_PHYSICAL_STATS_PATH")
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, separators=(",", ":")) + "\n")


def _kvmem_apply_visibility(plan, selected, positions):
    apply_min = int(plan["apply_min_pos"])
    active_from = int(plan["active_from_pos"])
    region_tokens = int(plan["region_tokens"])

    pos = positions.to(device=selected.device, dtype=torch.int64)
    apply_rows = pos >= apply_min
    if not bool(apply_rows.any().item()):
        return apply_rows, 0, 0, 0

    valid = selected >= 0
    historical = selected < active_from
    logical = selected.clamp_min(0).to(torch.int64)
    regions = torch.div(logical, region_tokens, rounding_mode="floor")
    resident_regions = _kvmem_plan_tensor(
        plan,
        "resident_regions",
        plan["resident_regions"],
        selected.device,
        torch.int64,
    )
    keep_hist = torch.isin(regions, resident_regions)
    row_mask = apply_rows.unsqueeze(1)
    drop = row_mask & valid & historical & ~keep_hist

    hist_total = int((row_mask & valid & historical).sum().item())
    hist_kept = int((row_mask & valid & historical & keep_hist).sum().item())
    dropped = int(drop.sum().item())
    selected.masked_fill_(drop, -1)
    return apply_rows, hist_total, hist_kept, dropped


def _kvmem_resident_table(layer, plan, full_table, full_block_tokens):
    if full_table.ndim != 2 or full_table.shape[0] != 1:
        raise RuntimeError("K1-Q2A currently requires exactly one request")
    resident_page_tokens = int(plan["page_tokens"])
    logical_token_capacity = int(full_table.shape[1]) * int(full_block_tokens)
    width = (
        logical_token_capacity + resident_page_tokens - 1
    ) // resident_page_tokens

    cached = getattr(layer, "_kvmem_resident_block_table", None)
    if (
        cached is not None
        and cached.device == full_table.device
        and cached.shape == (1, width)
    ):
        return cached

    table = torch.full(
        (1, width),
        -1,
        dtype=full_table.dtype,
        device=full_table.device,
    )
    resident_pages = plan["resident_pages"]
    resident_count = int(plan["resident_page_count"])
    for logical_page, physical_page in zip(
        resident_pages, range(resident_count), strict=True
    ):
        if logical_page >= width:
            raise RuntimeError("resident logical page exceeds resident table width")
        table[0, logical_page] = physical_page

    active_page0 = int(plan["active_page0"])
    reserve_pages = int(plan["active_reserve_pages"])
    if active_page0 + reserve_pages > width:
        raise RuntimeError("active reserve exceeds resident table width")
    for off in range(reserve_pages):
        table[0, active_page0 + off] = resident_count + off

    layer._kvmem_resident_block_table = table
    layer._kvmem_resident_block_table_width = width
    return table


def _kvmem_bootstrap_resident(layer, plan, full_cache, full_table):
    if full_cache.ndim != 4:
        raise RuntimeError("K1-Q2A expected a 4D QSA main KV cache")
    if full_table.ndim != 2 or full_table.shape[0] != 1:
        raise RuntimeError("K1-Q2A bootstrap currently requires one request")

    # QSA full-cache layout is [physical_block, kv_heads, block_tokens, 2*head].
    full_block_tokens = int(full_cache.shape[2])
    resident_page_tokens = int(plan["page_tokens"])
    if full_block_tokens <= 0 or full_block_tokens % resident_page_tokens:
        raise RuntimeError(
            "full QSA block size must be divisible by resident page size: "
            f"full={full_block_tokens} resident={resident_page_tokens}"
        )

    cache = getattr(layer, "_kvmem_resident_cache", None)
    expected_shape = (
        int(plan["physical_page_count"]),
        int(full_cache.shape[1]),
        resident_page_tokens,
        int(full_cache.shape[3]),
    )
    if cache is None:
        cache = torch.empty(
            expected_shape,
            dtype=full_cache.dtype,
            device=full_cache.device,
        )
        layer._kvmem_resident_cache = cache
    elif tuple(cache.shape) != expected_shape:
        raise RuntimeError(
            f"resident cache shape mismatch {tuple(cache.shape)} != {expected_shape}"
        )

    if getattr(layer, "_kvmem_resident_initialized", False):
        return cache, full_block_tokens

    resident_pages = plan["resident_pages"]
    resident_count = int(plan["resident_page_count"])
    logical_page_ids = _kvmem_plan_tensor(
        plan,
        "resident_pages",
        resident_pages,
        full_table.device,
        torch.int64,
    )

    # Every 16-token resident page is gathered from the scheduler's much larger
    # hybrid attention blocks (1568 tokens on the qualified CMP170HX runtime).
    # The algorithm is generic for any full block that is a multiple of the
    # resident page size.
    token0 = logical_page_ids * resident_page_tokens
    logical_full_block = torch.div(
        token0, full_block_tokens, rounding_mode="floor"
    )
    offset0 = torch.remainder(token0, full_block_tokens)
    if bool((offset0 + resident_page_tokens > full_block_tokens).any().item()):
        raise RuntimeError("resident page crosses a full-cache block boundary")
    if bool((logical_full_block >= full_table.shape[1]).any().item()):
        raise RuntimeError("resident history exceeds full-cache block table")

    src_physical_block = full_table[0].index_select(
        0, logical_full_block
    ).to(torch.int64)
    if bool((src_physical_block < 0).any().item()):
        raise RuntimeError("full cache lacks a historical resident page")

    token_offsets = torch.arange(
        resident_page_tokens,
        dtype=torch.int64,
        device=full_cache.device,
    )
    chunk = 128
    for start in range(0, resident_count, chunk):
        end = min(start + chunk, resident_count)
        src_blocks = src_physical_block[start:end]
        src_offsets = offset0[start:end]
        # Advanced gather result: [pages, page_tokens, kv_heads, 2*head].
        gathered = full_cache[
            src_blocks[:, None],
            :,
            src_offsets[:, None] + token_offsets[None, :],
            :,
        ]
        # Resident cache layout: [pages, kv_heads, page_tokens, 2*head].
        cache[start:end].copy_(gathered.permute(0, 2, 1, 3))

    layer._kvmem_resident_initialized = True
    layer._kvmem_resident_bootstrap_pages = resident_count
    layer._kvmem_full_block_tokens = full_block_tokens
    return cache, full_block_tokens


def _kvmem_compare_selected_inputs(
    plan,
    selected_rows,
    full_cache,
    full_table,
    resident_cache,
    resident_table,
):
    """Byte-compare K/V payloads addressed by the same logical selected tokens."""

    if full_table.shape[0] != 1 or resident_table.shape[0] != 1:
        raise RuntimeError("K1-Q2A input compare currently requires one request")

    valid = selected_rows[selected_rows >= 0].to(torch.int64)
    if valid.numel() == 0:
        return True, 0, None

    # De-duplicate to keep the diagnostic bounded even when top-k repeats the
    # same history token across many query rows/heads.
    logical_tokens = torch.unique(valid)
    full_page_tokens = int(full_cache.shape[2])
    resident_page_tokens = int(plan["page_tokens"])

    compared = 0
    first_bad = None
    chunk = 4096
    for start in range(0, int(logical_tokens.numel()), chunk):
        toks = logical_tokens[start : start + chunk]

        full_logical_block = torch.div(
            toks, full_page_tokens, rounding_mode="floor"
        )
        full_offset = torch.remainder(toks, full_page_tokens)
        full_phys = full_table[0].index_select(
            0, full_logical_block
        ).to(torch.int64)

        resident_logical_page = torch.div(
            toks, resident_page_tokens, rounding_mode="floor"
        )
        resident_offset = torch.remainder(toks, resident_page_tokens)
        resident_phys = resident_table[0].index_select(
            0, resident_logical_page
        ).to(torch.int64)

        if bool((full_phys < 0).any().item()):
            raise RuntimeError("full reference table misses a selected token")
        if bool((resident_phys < 0).any().item()):
            raise RuntimeError("resident table misses a selected token")

        full_values = full_cache[
            full_phys,
            :,
            full_offset,
            :,
        ]
        resident_values = resident_cache[
            resident_phys,
            :,
            resident_offset,
            :,
        ]
        same = torch.eq(full_values, resident_values).all(dim=(1, 2))
        compared += int(toks.numel())
        if not bool(same.all().item()):
            bad = int(torch.nonzero(~same, as_tuple=False)[0, 0].item())
            first_bad = int(toks[bad].item())
            return False, compared, first_bad

    return True, compared, first_bad


def _kvmem_active_slot_mapping(plan, positions, apply_rows, device):
    pos = positions.to(device=device, dtype=torch.int64)
    active_pos = pos[apply_rows]
    active_from = int(plan["active_from_pos"])
    page_tokens = int(plan["page_tokens"])
    resident_count = int(plan["resident_page_count"])
    reserve_pages = int(plan["active_reserve_pages"])

    rel = active_pos - active_from
    if bool((rel < 0).any().item()):
        raise RuntimeError("active slot mapping received a historical row")
    page_off = torch.div(rel, page_tokens, rounding_mode="floor")
    if bool((page_off >= reserve_pages).any().item()):
        raise RuntimeError("K1-Q2A active suffix exceeded resident reserve")
    token_off = torch.remainder(active_pos, page_tokens)
    return (resident_count + page_off) * page_tokens + token_off


def _kvmem_resident_attention(
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
    apply_rows, hist_total, hist_kept, dropped = _kvmem_apply_visibility(
        plan, selected, positions
    )

    # Full-cache update is authoritative during Q2A and happens before bootstrap,
    # so a mixed chunk that first crosses the query boundary has all historical
    # rows available as bootstrap source.
    impl.do_kv_cache_update(
        layer,
        key,
        value,
        layer.kv_cache,
        main_metadata.slot_mapping,
    )

    if not bool(apply_rows.any().item()):
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
        return

    if query.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("K1-Q2A physical shadow requires eager execution")

    resident_cache, full_block_tokens = _kvmem_bootstrap_resident(
        layer, plan, layer.kv_cache, main_metadata.block_table
    )
    resident_table = _kvmem_resident_table(
        layer,
        plan,
        main_metadata.block_table,
        full_block_tokens,
    )

    active_slots = _kvmem_active_slot_mapping(
        plan, positions, apply_rows, query.device
    )
    impl.do_kv_cache_update(
        layer,
        key[apply_rows],
        value[apply_rows],
        resident_cache,
        active_slots,
    )

    # Same-forward reference: exact same selected indices, full cache.
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

    ref_rows = output[apply_rows].clone()

    resident_query = query[apply_rows]
    resident_selected = selected[apply_rows]
    resident_req = side_metadata.token_to_req[: selected.shape[0]][apply_rows]

    key_cache, value_cache = resident_cache.transpose(1, 2).split(
        impl.head_size, dim=-1
    )
    key_cache = canonicalize_singleton_dim_strides(key_cache)
    value_cache = canonicalize_singleton_dim_strides(value_cache)

    from .ops.qsa import qsa_sparse_paged_attention

    resident_out = torch.empty_like(resident_query)
    qsa_sparse_paged_attention(
        resident_query,
        key_cache,
        value_cache,
        resident_selected,
        resident_table,
        resident_req,
        resident_out,
    )

    exact = bool(torch.equal(ref_rows, resident_out))
    max_abs = float(
        (ref_rows.float() - resident_out.float()).abs().max().item()
    )
    input_exact, input_tokens_compared, first_bad_input_token = (
        _kvmem_compare_selected_inputs(
            plan,
            resident_selected,
            layer.kv_cache,
            main_metadata.block_table,
            resident_cache,
            resident_table,
        )
    )

    output[apply_rows] = resident_out
    applied_pos = positions.to(query.device, dtype=torch.int64)[apply_rows]
    _kvmem_stats(
        {
            "schema": 1,
            "layer_name": str(layer.layer_name),
            "rows_applied": int(apply_rows.sum().item()),
            "first_pos": int(applied_pos[0].item()),
            "last_pos": int(applied_pos[-1].item()),
            "historical_selected": hist_total,
            "historical_resident_kept": hist_kept,
            "historical_selected_dropped": dropped,
            "bootstrap_pages": int(
                getattr(layer, "_kvmem_resident_bootstrap_pages", 0)
            ),
            "resident_physical_pages": int(plan["physical_page_count"]),
            "resident_page_tokens": int(plan["page_tokens"]),
            "full_block_tokens": int(full_block_tokens),
            "resident_table_width": int(
                getattr(layer, "_kvmem_resident_block_table_width", 0)
            ),
            "resident_cache_bytes": int(
                resident_cache.numel() * resident_cache.element_size()
            ),
            "input_mapping_exact": bool(input_exact),
            "input_tokens_compared": int(input_tokens_compared),
            "first_bad_input_token": first_bad_input_token,
            "attention_exact": exact,
            "attention_max_abs": max_abs,
        }
    )

    if not input_exact:
        raise RuntimeError(
            "K1-Q2A resident input mapping mismatch "
            f"first_bad_token={first_bad_input_token}"
        )
    if not exact and os.environ.get(
        "VLLM_QWEN_KVMEM_CONTINUE_INPUT_EXACT_NONEXACT", "0"
    ) != "1":
        raise RuntimeError(
            "K1-Q2A resident attention numerical mismatch "
            f"max_abs={max_abs}; input_mapping_exact=true"
        )

'''

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
        _kvmem_plan = _kvmem_physical_plan()
        if _kvmem_plan is None:
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
            _kvmem_resident_attention(
                self,
                impl,
                _kvmem_plan,
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

    if src.count(IMPORT_ANCHOR) != 1:
        raise RuntimeError("import anchor count != 1")
    if src.count(HELPER_ANCHOR) != 1:
        raise RuntimeError("helper anchor count != 1")
    if src.count(RUN_ANCHOR) != 1:
        raise RuntimeError("QSA run anchor count != 1")

    out = src.replace(IMPORT_ANCHOR, IMPORT_BLOCK, 1)
    out = out.replace(HELPER_ANCHOR, HELPER, 1)
    out = out.replace(RUN_ANCHOR, RUN_BLOCK, 1)

    required = (
        MARKER,
        "VLLM_QWEN_KVMEM_PHYSICAL_PLAN",
        "VLLM_QWEN_KVMEM_PHYSICAL_STATS_PATH",
        "_kvmem_bootstrap_resident",
        "_kvmem_active_slot_mapping",
        "qsa_sparse_paged_attention",
        "torch.equal(ref_rows, resident_out)",
        "full_block_tokens",
        "resident_page_tokens",
        "resident_table_width",
        "resident_cache_bytes",
        "input_mapping_exact",
        "input_tokens_compared",
        "VLLM_QWEN_KVMEM_CONTINUE_INPUT_EXACT_NONEXACT",
        "resident attention numerical mismatch",
    )
    missing = [x for x in required if x not in out]
    if missing:
        raise RuntimeError(f"physical-shadow postcondition missing: {missing}")

    if check_only:
        compile(out, str(path), "exec")
        return "anchors/postconditions validated; no files changed"

    original = path.with_suffix(path.suffix + ".kvmem_qsa_physical.orig")
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
    print(f"PASS: K1-Q2A QSA physical shadow {status}: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
