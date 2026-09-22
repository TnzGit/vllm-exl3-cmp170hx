"""Full-source correctness oracle for bounded CPU-backed QSA page reload."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

import torch

from vllm_exl3.kvmem_q2d_plan import load_reload_plan, validate_reload_plan


def _write_event(payload: dict[str, Any]) -> None:
    path = os.environ.get("VLLM_QWEN_KVMEM_Q2D_STATS_PATH")
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, separators=(",", ":")) + "\n")


def _new_state(
    layer: Any, plan: dict[str, Any], transfer_cache: torch.Tensor
) -> dict[str, Any]:
    old = getattr(layer, "_q2d_reload_state", None)
    if old is not None:
        old["backing"].close()
    cap = int(plan["physical_page_count"])
    staging = int(plan["staging_pages"])
    if tuple(transfer_cache.shape[:1]) != (cap + staging,):
        raise RuntimeError("Q2D transfer cache page count mismatch")
    page_bytes = int(transfer_cache[0].numel() * transfer_cache.element_size())
    if page_bytes != 32768:
        raise RuntimeError(f"Q2D page byte geometry mismatch: {page_bytes}")
    from vllm_exl3.kvmem_vllm_offload import single_tensor_cpu_backing

    backing = single_tensor_cpu_backing(
        tensor=transfer_cache,
        page_size_bytes=page_bytes,
        num_cpu_blocks=int(plan["cpu_page_count"]),
        lineage=f"q2d:{layer.layer_name}",
    )
    state = {
        "backing": backing,
        "logical_to_slot": {},
        "slot_to_logical": [None] * cap,
        "last_use": {},
        "clock": 0,
        "published": set(),
        "d2h_bytes": 0,
        "d2h_jobs": 0,
        "h2d_bytes": 0,
        "h2d_jobs": 0,
        "stage_compare_pages": 0,
        "stage_compare_exact": True,
        "peak_slots": 0,
        "next_publish_page": 0,
        "saw_forward": False,
    }
    layer._q2d_reload_state = state
    return state


def _state(layer: Any, plan: dict[str, Any], transfer_cache: torch.Tensor) -> Any:
    state = getattr(layer, "_q2d_reload_state", None)
    if state is None:
        state = _new_state(layer, plan, transfer_cache)
    return state


def _source_pages(
    logical_pages: Sequence[int],
    full_cache: torch.Tensor,
    full_table: torch.Tensor,
    page_tokens: int,
) -> torch.Tensor:
    if not logical_pages:
        return torch.empty(
            (0, full_cache.shape[1], page_tokens, full_cache.shape[3]),
            dtype=full_cache.dtype,
            device=full_cache.device,
        )
    full_block_tokens = int(full_cache.shape[2])
    if full_block_tokens % page_tokens:
        raise RuntimeError("Q2D full block does not divide into reload pages")
    logical = torch.tensor(
        logical_pages, dtype=torch.int64, device=full_table.device
    )
    token0 = logical * page_tokens
    full_logical_blocks = torch.div(
        token0, full_block_tokens, rounding_mode="floor"
    )
    offsets = torch.remainder(token0, full_block_tokens)
    if bool((full_logical_blocks >= full_table.shape[1]).any().item()):
        raise RuntimeError("Q2D source page exceeds full block table")
    physical = full_table[0].index_select(0, full_logical_blocks).to(torch.int64)
    if bool((physical < 0).any().item()):
        raise RuntimeError("Q2D source full block is missing")
    token_offsets = torch.arange(
        page_tokens, dtype=torch.int64, device=full_cache.device
    )
    gathered = full_cache[
        physical[:, None],
        :,
        offsets[:, None] + token_offsets[None, :],
        :,
    ]
    return gathered.permute(0, 2, 1, 3).contiguous()


def _allocate_slot(state: dict[str, Any], protected: set[int]) -> int:
    slots: list[int | None] = state["slot_to_logical"]
    for slot, logical in enumerate(slots):
        if logical is None:
            return slot
    candidates = [
        int(logical) for logical in state["logical_to_slot"] if logical not in protected
    ]
    if not candidates:
        raise RuntimeError("Q2D reload pool has no evictable slot")
    victim = min(candidates, key=lambda logical: state["last_use"].get(logical, -1))
    slot = int(state["logical_to_slot"].pop(victim))
    state["last_use"].pop(victim, None)
    slots[slot] = None
    return slot


def _assign_slot(state: dict[str, Any], logical: int, protected: set[int]) -> int:
    existing = state["logical_to_slot"].get(logical)
    if existing is not None:
        return int(existing)
    slot = _allocate_slot(state, protected)
    old = state["slot_to_logical"][slot]
    if old is not None:
        state["logical_to_slot"].pop(int(old), None)
        state["last_use"].pop(int(old), None)
    state["logical_to_slot"][logical] = slot
    state["slot_to_logical"][slot] = logical
    state["peak_slots"] = max(
        int(state.get("peak_slots", 0)), len(state["logical_to_slot"])
    )
    return slot


def _assign_many(
    state: dict[str, Any], logical_pages: Sequence[int], protected: set[int]
) -> list[int]:
    pages = [int(page) for page in logical_pages]
    if len(set(pages)) != len(pages):
        raise RuntimeError("Q2D bulk slot assignment contains duplicate pages")
    missing = [page for page in pages if page not in state["logical_to_slot"]]
    free = [
        slot for slot, logical in enumerate(state["slot_to_logical"])
        if logical is None
    ]
    need_victims = max(0, len(missing) - len(free))
    victims = sorted(
        (
            int(logical) for logical in state["logical_to_slot"]
            if int(logical) not in protected
        ),
        key=lambda logical: state["last_use"].get(logical, -1),
    )[:need_victims]
    if len(free) + len(victims) < len(missing):
        raise RuntimeError("Q2D reload pool has insufficient evictable slots")
    available = list(free)
    for victim in victims:
        slot = int(state["logical_to_slot"].pop(victim))
        state["last_use"].pop(victim, None)
        state["slot_to_logical"][slot] = None
        available.append(slot)
    for page, slot in zip(missing, available[:len(missing)], strict=True):
        state["logical_to_slot"][page] = slot
        state["slot_to_logical"][slot] = page
    state["peak_slots"] = max(
        int(state.get("peak_slots", 0)), len(state["logical_to_slot"])
    )
    return [int(state["logical_to_slot"][page]) for page in pages]


def _touch(state: dict[str, Any], pages: Sequence[int]) -> None:
    for page in pages:
        state["clock"] += 1
        state["last_use"][int(page)] = int(state["clock"])


def _publish_completed(
    state: dict[str, Any],
    plan: dict[str, Any],
    positions: torch.Tensor,
    full_cache: torch.Tensor,
    full_table: torch.Tensor,
    transfer_cache: torch.Tensor,
) -> int:
    page_tokens = int(plan["page_tokens"])
    cap = int(plan["physical_page_count"])
    staging_count = int(plan["staging_pages"])
    reached = int(positions.max().item()) + 1 if positions.numel() else 0
    # Publication follows the sequential single-request prefix, rather than
    # only the pages touched by this invocation. This matters when decode
    # crosses a page boundary: the just-completed previous page is no longer
    # present in ``positions`` but must become CPU-authoritative.
    completed_page_count = reached // page_tokens
    next_page = int(state["next_publish_page"])
    if completed_page_count < next_page:
        raise RuntimeError("Q2D publication high-water mark moved backwards")
    pages = list(range(next_page, completed_page_count))
    for start in range(0, len(pages), staging_count):
        chunk = pages[start : start + staging_count]
        source = _source_pages(chunk, full_cache, full_table, page_tokens)
        n = len(chunk)
        transfer_cache[cap : cap + n].copy_(source)
        torch.cuda.synchronize()
        obs = state["backing"].publish(chunk, list(range(cap, cap + n)))
        state["d2h_bytes"] += int(obs.transfer_bytes)
        state["d2h_jobs"] += int(obs.job_id != 0)
        state["published"].update(chunk)
    state["next_publish_page"] = completed_page_count
    if pages and not state["backing"].all_present(pages):
        raise RuntimeError("Q2D CPU backing misses newly published pages")
    return len(pages)


def _copy_current_pages(
    state: dict[str, Any],
    current_pages: list[int],
    full_cache: torch.Tensor,
    full_table: torch.Tensor,
    reload_cache: torch.Tensor,
    page_tokens: int,
) -> None:
    protected = set(current_pages)
    slots = _assign_many(state, current_pages, protected)
    source = _source_pages(current_pages, full_cache, full_table, page_tokens)
    slot_tensor = torch.tensor(slots, dtype=torch.int64, device=reload_cache.device)
    reload_cache.index_copy_(0, slot_tensor, source)
    _touch(state, current_pages)


def _stage_history(
    state: dict[str, Any],
    plan: dict[str, Any],
    history_pages: list[int],
    current_pages: list[int],
    full_cache: torch.Tensor,
    full_table: torch.Tensor,
    reload_cache: torch.Tensor,
) -> tuple[int, int]:
    protected = set(history_pages).union(current_pages)
    misses = [page for page in history_pages if page not in state["logical_to_slot"]]
    if any(page not in state["published"] for page in misses):
        missing = [page for page in misses if page not in state["published"]][:8]
        raise RuntimeError(f"Q2D selected history missing from CPU backing: {missing}")
    miss_slots = _assign_many(state, misses, protected)
    if misses:
        obs = state["backing"].stage_in(misses, miss_slots)
        state["h2d_bytes"] += int(obs.transfer_bytes)
        state["h2d_jobs"] += int(obs.job_id != 0)
        expected = _source_pages(
            misses, full_cache, full_table, int(plan["page_tokens"])
        )
        slot_tensor = torch.tensor(
            miss_slots, dtype=torch.int64, device=reload_cache.device
        )
        actual = reload_cache.index_select(0, slot_tensor)
        same = bool(torch.equal(
            expected.contiguous().view(torch.int16),
            actual.contiguous().view(torch.int16),
        ))
        state["stage_compare_pages"] += len(misses)
        state["stage_compare_exact"] = bool(state["stage_compare_exact"] and same)
        if not same:
            raise RuntimeError("Q2D CPU staged K/V differs from full-source page")
    _touch(state, history_pages)
    return len(misses), len(history_pages)


def run_reload_shadow(
    layer: Any,
    impl: Any,
    plan: dict[str, Any],
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    selected: torch.Tensor,
    positions: torch.Tensor,
    main_metadata: Any,
    side_metadata: Any,
) -> None:
    if query.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Q2D reload shadow requires eager execution")
    transfer_cache = layer._q2d_transfer_cache
    cap = int(plan["physical_page_count"])
    page_tokens = int(plan["page_tokens"])
    reload_cache = transfer_cache[:cap]
    impl.do_kv_cache_update(
        layer, key, value, layer.kv_cache, main_metadata.slot_mapping
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
    num_tokens = int(main_metadata.num_actual_tokens)
    if num_tokens <= 0:
        return
    reference = output[:num_tokens].clone()
    from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_sparse_paged_attention
    from vllm.utils.torch_utils import canonicalize_singleton_dim_strides

    # A second stock-cache reference uses the exact same 64-row call geometry
    # as the reload path. Comparing reload against this tensor separates
    # already-qualified row-split BF16 drift from any extra error introduced
    # by 16-token cache addressing, CPU reload, or dynamic READ mapping.
    stock_key, stock_value = layer.kv_cache.transpose(1, 2).split(
        impl.head_size, dim=-1
    )
    stock_key = canonicalize_singleton_dim_strides(stock_key)
    stock_value = canonicalize_singleton_dim_strides(stock_value)
    stock_split = torch.zeros_like(output[:num_tokens])
    batch_target = int(plan["query_row_batch"])
    request_ids = side_metadata.token_to_req[:num_tokens]
    for stock_start in range(0, num_tokens, batch_target):
        stock_end = min(stock_start + batch_target, num_tokens)
        qsa_sparse_paged_attention(
            query[stock_start:stock_end],
            stock_key,
            stock_value,
            selected[stock_start:stock_end],
            main_metadata.block_table,
            request_ids[stock_start:stock_end],
            stock_split[stock_start:stock_end],
        )
    output.zero_()
    pos = positions.to(device=selected.device, dtype=torch.int64).reshape(-1)
    if pos.numel() != num_tokens:
        raise RuntimeError("Q2D position count mismatch")
    state = _state(layer, plan, transfer_cache)
    # vLLM executes QSA warmups before health. A real single-request prefill
    # starts again at logical position zero. Reset all warmup cache/backing
    # state so fake warmup K/V can never satisfy a real historical lookup.
    if int(pos.min().item()) == 0 and bool(state["saw_forward"]):
        state = _new_state(layer, plan, transfer_cache)
    state["saw_forward"] = True
    published = _publish_completed(
        state,
        plan,
        pos,
        layer.kv_cache,
        main_metadata.block_table,
        transfer_cache,
    )
    current_pages = sorted(
        int(x) for x in torch.unique(
            torch.div(pos, page_tokens, rounding_mode="floor")
        ).cpu().tolist()
    )
    _copy_current_pages(
        state,
        current_pages,
        layer.kv_cache,
        main_metadata.block_table,
        reload_cache,
        page_tokens,
    )

    key_cache, value_cache = reload_cache.transpose(1, 2).split(
        impl.head_size, dim=-1
    )
    key_cache = canonicalize_singleton_dim_strides(key_cache)
    value_cache = canonicalize_singleton_dim_strides(value_cache)
    start = 0
    split_calls = 0
    min_batch = batch_target
    max_working = 0
    misses_total = 0
    selected_history_total = 0
    while start < num_tokens:
        size = min(batch_target, num_tokens - start)
        while True:
            end = start + size
            batch_selected = selected[start:end]
            valid = batch_selected >= 0
            pages = torch.div(
                batch_selected.clamp_min(0).to(torch.int64),
                page_tokens,
                rounding_mode="floor",
            )[valid]
            selected_pages = sorted(
                int(x) for x in torch.unique(pages).cpu().tolist()
            )
            working = len(set(selected_pages).union(current_pages))
            if working <= cap:
                break
            if size == 1:
                raise RuntimeError(
                    f"Q2D one query row needs {working} pages above cap {cap}"
                )
            size = max(1, size // 2)
        current_set = set(current_pages)
        history_pages = [page for page in selected_pages if page not in current_set]
        misses, selected_history = _stage_history(
            state,
            plan,
            history_pages,
            current_pages,
            layer.kv_cache,
            main_metadata.block_table,
            reload_cache,
        )
        table = torch.full(
            (1, int(plan["cpu_page_count"])),
            -1,
            dtype=main_metadata.block_table.dtype,
            device=main_metadata.block_table.device,
        )
        mapped_pages = sorted(set(selected_pages).union(current_pages))
        logical_tensor = torch.tensor(
            mapped_pages, dtype=torch.int64, device=table.device
        )
        physical_tensor = torch.tensor(
            [state["logical_to_slot"][page] for page in mapped_pages],
            dtype=table.dtype,
            device=table.device,
        )
        table[0].index_copy_(0, logical_tensor, physical_tensor)
        end = start + size
        qsa_sparse_paged_attention(
            query[start:end],
            key_cache,
            value_cache,
            selected[start:end],
            table,
            request_ids[start:end],
            output[start:end],
        )
        split_calls += 1
        min_batch = min(min_batch, size)
        max_working = max(max_working, working)
        misses_total += misses
        selected_history_total += selected_history
        start = end

    actual = output[:num_tokens]
    absolute = (reference.float() - actual.float()).abs()
    split_absolute = (stock_split.float() - actual.float()).abs()
    exact = bool(torch.equal(reference, actual))
    split_exact = bool(torch.equal(stock_split, actual))
    allclose = bool(
        torch.allclose(
            reference,
            actual,
            atol=float(plan["allclose_atol"]),
            rtol=float(plan["allclose_rtol"]),
        )
    )
    split_allclose = bool(
        torch.allclose(
            stock_split,
            actual,
            atol=float(plan["allclose_atol"]),
            rtol=float(plan["allclose_rtol"]),
        )
    )
    _write_event({
        "event": "q2d_reload_shadow",
        "layer": layer.layer_name,
        "first_pos": int(pos.min().item()) if pos.numel() else -1,
        "last_pos": int(pos.max().item()) if pos.numel() else -1,
        "query_rows": num_tokens,
        "split_calls": split_calls,
        "min_query_row_batch": min_batch,
        "current_write_pages": len(current_pages),
        "max_working_pages": max_working,
        "physical_page_cap": cap,
        "published_pages_this_forward": published,
        "published_pages_total": len(state["published"]),
        "reload_miss_pages": misses_total,
        "selected_history_pages": selected_history_total,
        "resident_slots": len(state["logical_to_slot"]),
        "peak_resident_slots": int(state["peak_slots"]),
        "d2h_bytes_total": int(state["d2h_bytes"]),
        "d2h_jobs_total": int(state["d2h_jobs"]),
        "h2d_bytes_total": int(state["h2d_bytes"]),
        "h2d_jobs_total": int(state["h2d_jobs"]),
        "stage_compare_pages_total": int(state["stage_compare_pages"]),
        "stage_compare_exact": bool(state["stage_compare_exact"]),
        "attention_exact": exact,
        "attention_allclose": allclose,
        "attention_mismatch_elements": int(torch.ne(reference, actual).sum().item()),
        "attention_elements": int(reference.numel()),
        "attention_max_abs": float(absolute.max().item()),
        "attention_mean_abs": float(absolute.mean().item()),
        "attention_rmse": float(torch.sqrt((absolute * absolute).mean()).item()),
        "reference_max_abs": float(reference.float().abs().max().item()),
        "reload_vs_stock_split_exact": split_exact,
        "reload_vs_stock_split_allclose": split_allclose,
        "reload_vs_stock_split_mismatch_elements": int(
            torch.ne(stock_split, actual).sum().item()
        ),
        "reload_vs_stock_split_max_abs": float(split_absolute.max().item()),
        "reload_vs_stock_split_mean_abs": float(split_absolute.mean().item()),
        "reload_vs_stock_split_rmse": float(
            torch.sqrt((split_absolute * split_absolute).mean()).item()
        ),
    })
    if not allclose or not split_allclose:
        raise RuntimeError("Q2D reload attention exceeds frozen BF16 tolerance")
