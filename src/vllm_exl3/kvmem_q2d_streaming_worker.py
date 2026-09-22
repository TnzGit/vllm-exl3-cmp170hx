"""CPU-authoritative QSA history with partitioned WRITE and READ slots."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

import torch

from vllm_exl3.kvmem_q2d_reload_shadow import _assign_many, _touch


def _write_event(payload: dict[str, Any]) -> None:
    path = os.environ.get("VLLM_QWEN_KVMEM_Q2D_WORKER_STATS_PATH")
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, separators=(",", ":")) + "\n")


def _new_state(layer: Any, plan: dict[str, Any]) -> dict[str, Any]:
    old = getattr(layer, "_q2d_streaming_state", None)
    if old is not None:
        old["backing"].close()
    staging = layer._q2d_staging
    page_bytes = int(staging[0].numel() * staging.element_size())
    if page_bytes != 32768:
        raise RuntimeError(f"Q2D staging page geometry mismatch: {page_bytes}")
    from vllm_exl3.kvmem_vllm_offload import single_tensor_cpu_backing

    backing = single_tensor_cpu_backing(
        tensor=staging,
        page_size_bytes=page_bytes,
        num_cpu_blocks=int(plan["cpu_page_count"]),
        lineage=f"q2d-runtime:{layer.layer_name}",
    )
    read_pages = int(plan["read_cache_page_count"])
    state = {
        "backing": backing,
        "logical_to_slot": {},
        "slot_to_logical": [None] * read_pages,
        "last_use": {},
        "clock": 0,
        "published": set(),
        "next_publish_page": 0,
        "saw_forward": False,
        "peak_read_slots": 0,
        "d2h_bytes": 0,
        "d2h_jobs": 0,
        "h2d_bytes": 0,
        "h2d_jobs": 0,
        "roundtrip_pages": 0,
        "roundtrip_exact": True,
    }
    layer._q2d_streaming_state = state
    return state


def _state(layer: Any, plan: dict[str, Any]) -> dict[str, Any]:
    state = getattr(layer, "_q2d_streaming_state", None)
    return state if state is not None else _new_state(layer, plan)


def _bits_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return bool(torch.equal(
        left.contiguous().view(torch.int16),
        right.contiguous().view(torch.int16),
    ))


def _logical_write_ids(
    logical_pages: Sequence[int], block_table: torch.Tensor, write_cap: int
) -> torch.Tensor:
    idx = torch.tensor(logical_pages, dtype=torch.int64, device=block_table.device)
    physical = block_table[0].index_select(0, idx).to(torch.int64)
    if physical.numel() and not bool(
        ((physical >= 0) & (physical < write_cap)).all().item()
    ):
        raise RuntimeError("Q2D current logical page is outside WRITE partition")
    return physical


def _validate_write_mapping(
    plan: dict[str, Any], positions: torch.Tensor, metadata: Any
) -> tuple[list[int], torch.Tensor]:
    page_tokens = int(plan["page_tokens"])
    pos = positions.to(device=metadata.block_table.device, dtype=torch.int64)
    logical = torch.div(pos, page_tokens, rounding_mode="floor")
    current_pages = sorted(int(x) for x in torch.unique(logical).cpu().tolist())
    physical = _logical_write_ids(
        current_pages, metadata.block_table, int(plan["write_page_count"])
    )
    per_token_physical = metadata.block_table[0].index_select(0, logical).to(torch.int64)
    expected = per_token_physical * page_tokens + torch.remainder(pos, page_tokens)
    actual = metadata.slot_mapping[:pos.numel()].to(
        device=expected.device, dtype=torch.int64
    )
    if not torch.equal(actual, expected):
        raise RuntimeError("Q2D WRITE slot_mapping disagrees with current block table")
    if len(current_pages) > int(plan["write_page_count"]):
        raise RuntimeError("Q2D current WRITE pages exceed frozen partition")
    return current_pages, physical


def _publish_completed(
    layer: Any,
    state: dict[str, Any],
    plan: dict[str, Any],
    positions: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
) -> int:
    page_tokens = int(plan["page_tokens"])
    reached = int(positions.max().item()) + 1
    completed = reached // page_tokens
    next_page = int(state["next_publish_page"])
    if completed < next_page:
        raise RuntimeError("Q2D publication high-water mark moved backwards")
    pages = list(range(next_page, completed))
    staging = layer._q2d_staging
    chunk_size = int(staging.shape[0])
    read_base = int(plan["write_page_count"])
    for start in range(0, len(pages), chunk_size):
        chunk = pages[start:start + chunk_size]
        write_ids = _logical_write_ids(
            chunk, block_table, int(plan["write_page_count"])
        )
        source = kv_cache.index_select(0, write_ids)
        n = len(chunk)
        staging[:n].copy_(source)
        torch.cuda.synchronize()
        obs = state["backing"].publish(chunk, list(range(n)))
        state["d2h_bytes"] += int(obs.transfer_bytes)
        state["d2h_jobs"] += int(obs.job_id != 0)

        # Every page is immediately round-tripped while its scheduler-owned
        # WRITE source is still intact. After this exact check the CPU copy is
        # authoritative and the scheduler may safely recycle the WRITE ID.
        restore = state["backing"].stage_in(chunk, list(range(n)))
        state["h2d_bytes"] += int(restore.transfer_bytes)
        state["h2d_jobs"] += int(restore.job_id != 0)
        exact = _bits_equal(staging[:n], source)
        state["roundtrip_pages"] += n
        state["roundtrip_exact"] = bool(state["roundtrip_exact"] and exact)
        if not exact:
            raise RuntimeError("Q2D CPU publication roundtrip differs from WRITE source")

        local_slots = _assign_many(state, chunk, set(chunk))
        read_ids = torch.tensor(
            [read_base + slot for slot in local_slots],
            dtype=torch.int64,
            device=kv_cache.device,
        )
        kv_cache.index_copy_(0, read_ids, staging[:n])
        _touch(state, chunk)
        state["published"].update(chunk)
    state["next_publish_page"] = completed
    state["peak_read_slots"] = max(
        int(state["peak_read_slots"]), len(state["logical_to_slot"])
    )
    if pages and not state["backing"].all_present(pages):
        raise RuntimeError("Q2D CPU backing misses newly published history")
    return len(pages)


def _stage_history(
    layer: Any,
    state: dict[str, Any],
    plan: dict[str, Any],
    history_pages: list[int],
    kv_cache: torch.Tensor,
) -> int:
    missing = [page for page in history_pages if page not in state["logical_to_slot"]]
    absent = [page for page in missing if page not in state["published"]]
    if absent:
        raise RuntimeError(f"Q2D selected history is not CPU-authoritative: {absent[:8]}")
    local_slots = _assign_many(state, missing, set(history_pages))
    staging = layer._q2d_staging
    chunk_size = int(staging.shape[0])
    read_base = int(plan["write_page_count"])
    for start in range(0, len(missing), chunk_size):
        pages = missing[start:start + chunk_size]
        slots = local_slots[start:start + chunk_size]
        n = len(pages)
        obs = state["backing"].stage_in(pages, list(range(n)))
        state["h2d_bytes"] += int(obs.transfer_bytes)
        state["h2d_jobs"] += int(obs.job_id != 0)
        read_ids = torch.tensor(
            [read_base + slot for slot in slots],
            dtype=torch.int64,
            device=kv_cache.device,
        )
        kv_cache.index_copy_(0, read_ids, staging[:n])
    _touch(state, history_pages)
    state["peak_read_slots"] = max(
        int(state["peak_read_slots"]), len(state["logical_to_slot"])
    )
    return len(missing)


def run_streaming_runtime(
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
        raise RuntimeError("Q2D streaming runtime requires eager execution")
    num_tokens = int(main_metadata.num_actual_tokens)
    if num_tokens <= 0:
        output.zero_()
        return
    pos = positions.to(device=selected.device, dtype=torch.int64).reshape(-1)
    if pos.numel() != num_tokens:
        raise RuntimeError("Q2D position count mismatch")
    state = _state(layer, plan)
    if int(pos.min().item()) == 0 and bool(state["saw_forward"]):
        state = _new_state(layer, plan)
    state["saw_forward"] = True
    current_pages, current_write_ids = _validate_write_mapping(plan, pos, main_metadata)

    impl.do_kv_cache_update(
        layer, key, value, layer.kv_cache, main_metadata.slot_mapping
    )
    published = _publish_completed(
        layer, state, plan, pos, layer.kv_cache, main_metadata.block_table
    )

    from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_sparse_paged_attention
    from vllm.utils.torch_utils import canonicalize_singleton_dim_strides

    key_cache, value_cache = layer.kv_cache.transpose(1, 2).split(
        impl.head_size, dim=-1
    )
    key_cache = canonicalize_singleton_dim_strides(key_cache)
    value_cache = canonicalize_singleton_dim_strides(value_cache)
    request_ids = side_metadata.token_to_req[:num_tokens]
    output.zero_()
    page_tokens = int(plan["page_tokens"])
    cap = int(plan["physical_page_count"])
    read_cap = int(plan["read_cache_page_count"])
    read_base = int(plan["write_page_count"])
    current_map = dict(zip(current_pages, current_write_ids.cpu().tolist(), strict=True))
    batch_target = int(plan["query_row_batch"])
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
            pages_tensor = torch.div(
                batch_selected.clamp_min(0).to(torch.int64),
                page_tokens,
                rounding_mode="floor",
            )[valid]
            selected_pages = sorted(
                int(x) for x in torch.unique(pages_tensor).cpu().tolist()
            )
            history_pages = [page for page in selected_pages if page not in current_map]
            working = len(history_pages) + len(current_pages)
            if working <= cap and len(history_pages) <= read_cap:
                break
            if size == 1:
                raise RuntimeError(
                    f"Q2D one query row needs history={len(history_pages)} "
                    f"working={working} above partition/cap"
                )
            size = max(1, size // 2)
        misses = _stage_history(
            layer, state, plan, history_pages, layer.kv_cache
        )
        table = torch.full(
            (1, int(plan["cpu_page_count"])),
            -1,
            dtype=main_metadata.block_table.dtype,
            device=main_metadata.block_table.device,
        )
        mapped_pages = sorted(set(history_pages).union(current_pages))
        physical = [
            current_map[page]
            if page in current_map
            else read_base + int(state["logical_to_slot"][page])
            for page in mapped_pages
        ]
        table[0].index_copy_(
            0,
            torch.tensor(mapped_pages, dtype=torch.int64, device=table.device),
            torch.tensor(physical, dtype=table.dtype, device=table.device),
        )
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
        selected_history_total += len(history_pages)
        start = end

    real_pages = len(current_pages) + len(state["logical_to_slot"])
    if real_pages > cap:
        raise RuntimeError("Q2D combined READ/WRITE ownership exceeds physical cap")
    _write_event({
        "event": "q2d_streaming_runtime",
        "layer": layer.layer_name,
        "first_pos": int(pos.min().item()),
        "last_pos": int(pos.max().item()),
        "query_rows": num_tokens,
        "split_calls": split_calls,
        "min_query_row_batch": min_batch,
        "current_write_pages": len(current_pages),
        "resident_read_pages": len(state["logical_to_slot"]),
        "combined_real_pages": real_pages,
        "peak_read_pages": int(state["peak_read_slots"]),
        "max_working_pages": max_working,
        "physical_page_cap": cap,
        "published_pages_this_forward": published,
        "published_pages_total": len(state["published"]),
        "reload_miss_pages": misses_total,
        "selected_history_pages": selected_history_total,
        "cpu_roundtrip_pages_total": int(state["roundtrip_pages"]),
        "cpu_roundtrip_exact": bool(state["roundtrip_exact"]),
        "d2h_bytes_total": int(state["d2h_bytes"]),
        "d2h_jobs_total": int(state["d2h_jobs"]),
        "h2d_bytes_total": int(state["h2d_bytes"]),
        "h2d_jobs_total": int(state["h2d_jobs"]),
        "read_table_mode": "dynamic_cpu_history_plus_scheduler_writes",
        "write_partition": [0, read_base - 1],
        "read_partition": [read_base, cap - 1],
    })
