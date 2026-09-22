"""CPU-authoritative QSA history with partitioned WRITE and READ slots."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import time
from typing import Any, Sequence

import numpy as np
import torch

from vllm_exl3.kvmem_q2d_reload_shadow import _assign_many, _touch


_LAYER_INDEX_RE = re.compile(r"\.layers\.(\d+)\.")
_TRACE_RESET_PATHS: set[Path] = set()
_TRANSFER_TIMING_FIELDS = (
    "event_seconds",
    "wall_seconds",
    "prepare_seconds",
    "submit_seconds",
    "wait_seconds",
    "finish_seconds",
)
_CUMULATIVE_TIMING_FIELDS = tuple(
    f"{direction}_{field}_total"
    for direction in ("d2h", "h2d")
    for field in _TRANSFER_TIMING_FIELDS
) + (
    "d2d_copy_submit_seconds_total",
    "direct_consumer_sync_seconds_total",
    "publication_oracle_gather_submit_seconds_total",
    "stage_residency_lookup_seconds_total",
    "stage_slot_assignment_seconds_total",
    "stage_transfer_call_seconds_total",
    "stage_table_delta_seconds_total",
    "stage_touch_seconds_total",
    "trace_wall_seconds_total",
)


def _write_event(payload: dict[str, Any]) -> None:
    path = os.environ.get("VLLM_QWEN_KVMEM_Q2D_WORKER_STATS_PATH")
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, separators=(",", ":")) + "\n")


def _layer_id(layer_name: str) -> int:
    match = _LAYER_INDEX_RE.search(layer_name)
    if match is None:
        raise RuntimeError(f"Q2E cannot derive layer id from {layer_name!r}")
    return int(match.group(1))


def _accumulate_transfer(
    state: dict[str, Any], direction: str, observation: Any
) -> None:
    for field in _TRANSFER_TIMING_FIELDS:
        key = f"{direction}_{field}_total"
        state[key] = float(state[key]) + float(getattr(observation, field))


def _trace_access(
    *,
    layer_name: str,
    first_pos: int,
    subbatch_start: int,
    query_rows: int,
    history_pages: list[int],
    missing_pages: list[int],
    victim_pages: list[int],
    assigned_slots: list[int],
    assigned_generations: list[int],
) -> tuple[bool, bool, int]:
    path = os.environ.get("VLLM_QWEN_KVMEM_Q2E_TRACE_PATH")
    if not path:
        return False, False, 0
    from vllm_exl3.kvmem_q2e_trace import write_trace_record

    result = write_trace_record(
        path,
        int(os.environ.get("VLLM_QWEN_KVMEM_Q2E_TRACE_MAX_BYTES", 268435456)),
        layer_id=_layer_id(layer_name),
        first_pos=first_pos,
        subbatch_start=subbatch_start,
        query_rows=query_rows,
        history_pages=history_pages,
        missing_pages=missing_pages,
        victim_pages=victim_pages,
        assigned_slots=assigned_slots,
        assigned_generations=assigned_generations,
    )
    return (
        bool(result["written"]),
        bool(result["truncated"]),
        int(result["record_bytes"]),
    )


def _reset_trace_after_warmup_once() -> None:
    """Discard warmup records before the one frozen qualification request."""
    raw_path = os.environ.get("VLLM_QWEN_KVMEM_Q2E_TRACE_PATH")
    if not raw_path:
        return
    path = Path(raw_path).expanduser().resolve(strict=False)
    if path in _TRACE_RESET_PATHS:
        return
    from vllm_exl3.kvmem_q2e_trace import close_trace_writers

    close_trace_writers()
    path.unlink(missing_ok=True)
    _TRACE_RESET_PATHS.add(path)


def _direct_io_enabled() -> bool:
    raw = os.environ.get("VLLM_QWEN_KVMEM_Q2E_DIRECT_IO", "0")
    if raw not in ("0", "1"):
        raise RuntimeError("VLLM_QWEN_KVMEM_Q2E_DIRECT_IO must be 0 or 1")
    return raw == "1"


def _dynamic_table_oracle_enabled() -> bool:
    raw = os.environ.get("VLLM_QWEN_KVMEM_Q2E_VERIFY_DYNAMIC_TABLE", "0")
    if raw not in ("0", "1"):
        raise RuntimeError(
            "VLLM_QWEN_KVMEM_Q2E_VERIFY_DYNAMIC_TABLE must be 0 or 1"
        )
    return raw == "1"


def _new_state(layer: Any, plan: dict[str, Any]) -> dict[str, Any]:
    old = getattr(layer, "_q2d_streaming_state", None)
    if old is not None:
        old["backing"].close()
    staging = layer._q2d_staging
    page_bytes = int(staging[0].numel() * staging.element_size())
    if page_bytes != 32768:
        raise RuntimeError(f"Q2D staging page geometry mismatch: {page_bytes}")
    from vllm_exl3.kvmem_vllm_offload import single_tensor_cpu_backing

    dedicated = layer.kv_cache
    direct_io = _direct_io_enabled()
    backing = single_tensor_cpu_backing(
        tensor=dedicated if direct_io else staging,
        page_size_bytes=page_bytes,
        num_cpu_blocks=int(plan["cpu_page_count"]),
        lineage=f"q2d-runtime:{layer.layer_name}",
    )
    read_pages = int(plan["read_cache_page_count"])
    timing_totals = {field: 0.0 for field in _CUMULATIVE_TIMING_FIELDS}
    state = {
        "backing": backing,
        "direct_io": direct_io,
        "logical_to_slot": {},
        "slot_to_logical": [None] * read_pages,
        "slot_generations": [0] * read_pages,
        "dynamic_table": torch.full(
            (1, int(plan["cpu_page_count"])),
            -1,
            dtype=torch.int32,
            device=dedicated.device,
        ),
        "selection_page_bitmap": torch.zeros(
            int(plan["cpu_page_count"]), dtype=torch.bool, device=dedicated.device
        ),
        "current_write_pages": [],
        "verify_dynamic_table": _dynamic_table_oracle_enabled(),
        "dynamic_table_oracle_checks": 0,
        "dynamic_table_oracle_pages": 0,
        "last_use": {},
        "last_use_array": np.full(
            int(plan["cpu_page_count"]), -1, dtype=np.int64
        ),
        "clock": 0,
        "peak_slots": 0,
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
        "direct_load_verified_pages": 0,
        "direct_consumer_syncs": 0,
        "trace_records": 0,
        "trace_bytes": 0,
        "trace_truncated": False,
        **timing_totals,
    }
    layer._q2d_streaming_state = state
    page_bytes = int(staging[0].numel() * staging.element_size())
    _write_event({
        "event": "q2e_memory_census",
        "layer": layer.layer_name,
        "layer_id": _layer_id(layer.layer_name),
        "io_mode": "direct_dedicated_slots" if direct_io else "staged_copy",
        "backing_tensor": "dedicated_qsa_cache" if direct_io else "staging",
        "staging_role": "allocated_unused" if direct_io else "transfer_bounce",
        "page_bytes": page_bytes,
        "addressable_pages": int(plan["physical_page_count"]),
        "write_pages": int(plan["write_page_count"]),
        "read_pages": read_pages,
        "null_pages": int(dedicated.shape[0]) - int(plan["physical_page_count"]),
        "dedicated_tensor_bytes": int(dedicated.numel() * dedicated.element_size()),
        "staging_pages": int(staging.shape[0]),
        "staging_tensor_bytes": int(staging.numel() * staging.element_size()),
        "cpu_backing_pages": int(plan["cpu_page_count"]),
        "cpu_backing_logical_bytes": int(plan["cpu_page_count"]) * page_bytes,
        "dynamic_table_bytes": int(plan["cpu_page_count"]) * 4,
        "selection_page_bitmap_bytes": int(plan["cpu_page_count"]),
        "cuda_memory_allocated_bytes": int(torch.cuda.memory_allocated()),
        "cuda_memory_reserved_bytes": int(torch.cuda.memory_reserved()),
        "cuda_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    })
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


def _update_dynamic_table(
    table: torch.Tensor,
    *,
    clear_pages: Sequence[int] = (),
    mapped_pages: Sequence[int] = (),
    physical_pages: Sequence[int] = (),
) -> None:
    """Apply a bounded mapping delta to the persistent logical block table."""
    if len(mapped_pages) != len(physical_pages):
        raise ValueError("Q2E dynamic table mapping lengths disagree")
    row = table[0]
    if clear_pages:
        clear = torch.tensor(clear_pages, dtype=torch.int64, device=table.device)
        row.index_fill_(0, clear, -1)
    if mapped_pages:
        logical = torch.tensor(mapped_pages, dtype=torch.int64, device=table.device)
        physical = torch.tensor(
            physical_pages, dtype=table.dtype, device=table.device
        )
        row.index_copy_(0, logical, physical)


def _prepare_forward_table(
    state: dict[str, Any],
    current_pages: Sequence[int],
    current_write_ids: Sequence[int],
    read_base: int,
) -> torch.Tensor:
    """Retire the previous WRITE view and install this forward's WRITE view."""
    previous = [int(page) for page in state["current_write_pages"]]
    resident = state["logical_to_slot"]
    restore_pages = [page for page in previous if page in resident]
    clear_pages = [page for page in previous if page not in resident]
    _update_dynamic_table(
        state["dynamic_table"],
        clear_pages=clear_pages,
        mapped_pages=restore_pages,
        physical_pages=[read_base + int(resident[page]) for page in restore_pages],
    )
    _update_dynamic_table(
        state["dynamic_table"],
        mapped_pages=current_pages,
        physical_pages=current_write_ids,
    )
    state["current_write_pages"] = [int(page) for page in current_pages]
    return state["dynamic_table"]


def _verify_dynamic_table(
    table: torch.Tensor,
    history_pages: Sequence[int],
    current_pages: Sequence[int],
    current_map: dict[int, int],
    logical_to_slot: dict[int, int],
    read_base: int,
) -> int:
    """Synchronously prove selected READ and WRITE entries are exact."""
    mapped_pages = sorted(set(history_pages).union(current_pages))
    expected = [
        current_map[page]
        if page in current_map
        else read_base + int(logical_to_slot[page])
        for page in mapped_pages
    ]
    logical = torch.tensor(mapped_pages, dtype=torch.int64, device=table.device)
    actual = table[0].index_select(0, logical)
    wanted = torch.tensor(expected, dtype=table.dtype, device=table.device)
    if not torch.equal(actual, wanted):
        raise RuntimeError("Q2E persistent dynamic table differs from exact mapping")
    return len(mapped_pages)


def _selected_history_page_tensor(
    selected_rows: torch.Tensor,
    page_tokens: int,
    current_pages: torch.Tensor,
    bitmap: torch.Tensor,
) -> torch.Tensor:
    """Return the sorted unique selected history pages via a reusable bitmap."""
    if bitmap.ndim != 1 or bitmap.dtype != torch.bool:
        raise RuntimeError("Q2E selection bitmap must be one-dimensional bool")
    if selected_rows.device != bitmap.device or current_pages.device != bitmap.device:
        raise RuntimeError("Q2E selection tensors must share one device")
    bitmap.zero_()
    valid = selected_rows >= 0
    pages = torch.div(
        selected_rows.clamp_min(0).to(torch.int64),
        page_tokens,
        rounding_mode="floor",
    )[valid]
    if pages.numel():
        bitmap.index_fill_(0, pages, True)
    if current_pages.numel():
        bitmap.index_fill_(0, current_pages, False)
    return torch.nonzero(bitmap, as_tuple=False).flatten()


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
        gather_start = time.perf_counter()
        source = kv_cache.index_select(0, write_ids)
        state["publication_oracle_gather_submit_seconds_total"] += (
            time.perf_counter() - gather_start
        )
        n = len(chunk)
        direct_io = bool(state["direct_io"])
        if direct_io:
            publish_source_ids = [int(page) for page in write_ids.cpu().tolist()]
        else:
            staging[:n].copy_(source)
            torch.cuda.synchronize()
            publish_source_ids = list(range(n))
        obs = state["backing"].publish(chunk, publish_source_ids)
        state["d2h_bytes"] += int(obs.transfer_bytes)
        state["d2h_jobs"] += int(obs.job_id != 0)
        _accumulate_transfer(state, "d2h", obs)

        before_slots = list(state["slot_to_logical"])
        before_pages = set(state["logical_to_slot"])
        local_slots = _assign_many(state, chunk, set(chunk))
        new_pages: list[int] = []
        new_slots: list[int] = []
        new_generations: list[int] = []
        victims: list[int] = []
        for page, slot in zip(chunk, local_slots, strict=True):
            if page in before_pages:
                continue
            previous = before_slots[slot]
            state["slot_generations"][slot] += 1
            if previous is not None:
                victims.append(int(previous))
            new_pages.append(int(page))
            new_slots.append(int(slot))
            new_generations.append(int(state["slot_generations"][slot]))
        read_id_list = [read_base + slot for slot in local_slots]
        read_ids = torch.tensor(
            read_id_list,
            dtype=torch.int64,
            device=kv_cache.device,
        )

        # Every page is immediately round-tripped while its scheduler-owned
        # WRITE source is still intact. After this exact check the CPU copy is
        # authoritative and the scheduler may safely recycle the WRITE ID.
        restore = state["backing"].stage_in(
            chunk, read_id_list if direct_io else list(range(n))
        )
        state["h2d_bytes"] += int(restore.transfer_bytes)
        state["h2d_jobs"] += int(restore.job_id != 0)
        _accumulate_transfer(state, "h2d", restore)
        state["direct_load_verified_pages"] += int(restore.verified_pages)
        state["direct_consumer_sync_seconds_total"] += float(
            restore.consumer_sync_seconds
        )
        state["direct_consumer_syncs"] += int(restore.consumer_syncs)
        restored = kv_cache.index_select(0, read_ids) if direct_io else staging[:n]
        exact = _bits_equal(restored, source)
        state["roundtrip_pages"] += n
        state["roundtrip_exact"] = bool(state["roundtrip_exact"] and exact)
        if not exact:
            raise RuntimeError("Q2D CPU publication roundtrip differs from WRITE source")

        if not direct_io:
            copy_start = time.perf_counter()
            kv_cache.index_copy_(0, read_ids, staging[:n])
            state["d2d_copy_submit_seconds_total"] += (
                time.perf_counter() - copy_start
            )
        _update_dynamic_table(
            state["dynamic_table"],
            clear_pages=victims,
            mapped_pages=new_pages,
            physical_pages=[read_base + slot for slot in new_slots],
        )
        _touch(state, chunk)
        state["published"].update(chunk)
        trace_start = time.perf_counter()
        written, truncated, record_bytes = _trace_access(
            layer_name=layer.layer_name,
            first_pos=int(positions.min().item()),
            subbatch_start=0,
            query_rows=0,
            history_pages=[],
            missing_pages=new_pages,
            victim_pages=victims,
            assigned_slots=new_slots,
            assigned_generations=new_generations,
        )
        state["trace_records"] += int(written)
        state["trace_bytes"] += record_bytes if written else 0
        state["trace_truncated"] = bool(state["trace_truncated"] or truncated)
        state["trace_wall_seconds_total"] = float(
            state.get("trace_wall_seconds_total", 0.0)
        ) + (time.perf_counter() - trace_start)
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
) -> tuple[int, list[int], list[int], list[int], list[int]]:
    lookup_start = time.perf_counter()
    missing = [page for page in history_pages if page not in state["logical_to_slot"]]
    absent = [page for page in missing if page not in state["published"]]
    state["stage_residency_lookup_seconds_total"] = float(
        state.get("stage_residency_lookup_seconds_total", 0.0)
    ) + (time.perf_counter() - lookup_start)
    if absent:
        raise RuntimeError(f"Q2D selected history is not CPU-authoritative: {absent[:8]}")
    assignment_start = time.perf_counter()
    generations: list[int] = []
    victims: list[int] = []
    if missing:
        before_slots = list(state["slot_to_logical"])
        local_slots = _assign_many(state, missing, set(history_pages))
        for page, slot in zip(missing, local_slots, strict=True):
            previous = before_slots[slot]
            if previous != page:
                state["slot_generations"][slot] += 1
                if previous is not None:
                    victims.append(int(previous))
            generations.append(int(state["slot_generations"][slot]))
    else:
        local_slots = []
    state["stage_slot_assignment_seconds_total"] = float(
        state.get("stage_slot_assignment_seconds_total", 0.0)
    ) + (time.perf_counter() - assignment_start)
    staging = layer._q2d_staging
    chunk_size = int(staging.shape[0])
    read_base = int(plan["write_page_count"])
    transfer_start = time.perf_counter()
    if missing and bool(state["direct_io"]):
        direct_read_ids = [read_base + slot for slot in local_slots]
        obs = state["backing"].stage_in(missing, direct_read_ids)
        state["h2d_bytes"] += int(obs.transfer_bytes)
        state["h2d_jobs"] += int(obs.job_id != 0)
        _accumulate_transfer(state, "h2d", obs)
        state["direct_load_verified_pages"] += int(obs.verified_pages)
        state["direct_consumer_sync_seconds_total"] += float(
            obs.consumer_sync_seconds
        )
        state["direct_consumer_syncs"] += int(obs.consumer_syncs)
    else:
        for start in range(0, len(missing), chunk_size):
            pages = missing[start:start + chunk_size]
            slots = local_slots[start:start + chunk_size]
            n = len(pages)
            obs = state["backing"].stage_in(pages, list(range(n)))
            state["h2d_bytes"] += int(obs.transfer_bytes)
            state["h2d_jobs"] += int(obs.job_id != 0)
            _accumulate_transfer(state, "h2d", obs)
            state["direct_load_verified_pages"] += int(obs.verified_pages)
            state["direct_consumer_sync_seconds_total"] += float(
                obs.consumer_sync_seconds
            )
            state["direct_consumer_syncs"] += int(obs.consumer_syncs)
            read_ids = torch.tensor(
                [read_base + slot for slot in slots],
                dtype=torch.int64,
                device=kv_cache.device,
            )
            copy_start = time.perf_counter()
            kv_cache.index_copy_(0, read_ids, staging[:n])
            state["d2d_copy_submit_seconds_total"] += (
                time.perf_counter() - copy_start
            )
    state["stage_transfer_call_seconds_total"] = float(
        state.get("stage_transfer_call_seconds_total", 0.0)
    ) + (time.perf_counter() - transfer_start)
    current_write_pages = set(int(page) for page in state["current_write_pages"])
    table_delta_start = time.perf_counter()
    if missing or victims:
        _update_dynamic_table(
            state["dynamic_table"],
            clear_pages=[page for page in victims if page not in current_write_pages],
            mapped_pages=missing,
            physical_pages=[read_base + slot for slot in local_slots],
        )
    state["stage_table_delta_seconds_total"] = float(
        state.get("stage_table_delta_seconds_total", 0.0)
    ) + (time.perf_counter() - table_delta_start)
    touch_start = time.perf_counter()
    _touch(state, history_pages)
    state["stage_touch_seconds_total"] = float(
        state.get("stage_touch_seconds_total", 0.0)
    ) + (time.perf_counter() - touch_start)
    state["peak_read_slots"] = max(
        int(state["peak_read_slots"]), len(state["logical_to_slot"])
    )
    return len(missing), missing, victims, local_slots, generations


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
    forward_start = time.perf_counter()
    pos = positions.to(device=selected.device, dtype=torch.int64).reshape(-1)
    if pos.numel() != num_tokens:
        raise RuntimeError("Q2D position count mismatch")
    state = _state(layer, plan)
    if int(pos.min().item()) == 0 and bool(state["saw_forward"]):
        _reset_trace_after_warmup_once()
        state = _new_state(layer, plan)
    state["saw_forward"] = True
    cumulative_timing_fields = _CUMULATIVE_TIMING_FIELDS
    timing_before = {
        field: float(state[field]) for field in cumulative_timing_fields
    }
    mapping_start = time.perf_counter()
    current_pages, current_write_ids = _validate_write_mapping(plan, pos, main_metadata)
    write_mapping_wall = time.perf_counter() - mapping_start

    kv_update_start = time.perf_counter()
    impl.do_kv_cache_update(
        layer, key, value, layer.kv_cache, main_metadata.slot_mapping
    )
    kv_update_submit_wall = time.perf_counter() - kv_update_start
    publish_start = time.perf_counter()
    published = _publish_completed(
        layer, state, plan, pos, layer.kv_cache, main_metadata.block_table
    )
    publish_wall = time.perf_counter() - publish_start

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
    current_write_id_list = [int(page) for page in current_write_ids.cpu().tolist()]
    current_map = dict(zip(current_pages, current_write_id_list, strict=True))
    current_pages_tensor = torch.tensor(
        current_pages, dtype=torch.int64, device=selected.device
    )
    if state["dynamic_table"].dtype != main_metadata.block_table.dtype:
        raise RuntimeError("Q2E dynamic table dtype differs from scheduler block table")
    table_refresh_start = time.perf_counter()
    table = _prepare_forward_table(
        state,
        current_pages,
        current_write_id_list,
        read_base,
    )
    table_forward_refresh_wall = time.perf_counter() - table_refresh_start
    batch_target = int(plan["query_row_batch"])
    start = 0
    split_calls = 0
    min_batch = batch_target
    max_working = 0
    misses_total = 0
    selected_history_total = 0
    selection_plan_wall = 0.0
    selection_tensor_wall = 0.0
    selection_cpu_filter_wall = 0.0
    stage_history_wall = 0.0
    table_build_wall = table_forward_refresh_wall
    attention_submit_wall = 0.0
    while start < num_tokens:
        plan_start = time.perf_counter()
        size = min(batch_target, num_tokens - start)
        while True:
            end = start + size
            batch_selected = selected[start:end]
            selection_tensor_start = time.perf_counter()
            history_page_tensor = _selected_history_page_tensor(
                batch_selected,
                page_tokens,
                current_pages_tensor,
                state["selection_page_bitmap"],
            )
            selection_tensor_wall += time.perf_counter() - selection_tensor_start
            selection_cpu_start = time.perf_counter()
            history_pages = [int(page) for page in history_page_tensor.cpu().tolist()]
            selection_cpu_filter_wall += time.perf_counter() - selection_cpu_start
            working = len(history_pages) + len(current_pages)
            if working <= cap and len(history_pages) <= read_cap:
                break
            if size == 1:
                raise RuntimeError(
                    f"Q2D one query row needs history={len(history_pages)} "
                    f"working={working} above partition/cap"
                )
            size = max(1, size // 2)
        selection_plan_wall += time.perf_counter() - plan_start
        stage_start = time.perf_counter()
        misses, missing, victims, assigned_slots, assigned_generations = _stage_history(
            layer, state, plan, history_pages, layer.kv_cache
        )
        stage_history_wall += time.perf_counter() - stage_start
        trace_start = time.perf_counter()
        written, truncated, record_bytes = _trace_access(
            layer_name=layer.layer_name,
            first_pos=int(pos.min().item()),
            subbatch_start=start,
            query_rows=size,
            history_pages=history_pages,
            missing_pages=missing,
            victim_pages=victims,
            assigned_slots=assigned_slots,
            assigned_generations=assigned_generations,
        )
        state["trace_records"] += int(written)
        state["trace_bytes"] += record_bytes if written else 0
        state["trace_truncated"] = bool(state["trace_truncated"] or truncated)
        state["trace_wall_seconds_total"] += time.perf_counter() - trace_start
        if bool(state["verify_dynamic_table"]):
            verified = _verify_dynamic_table(
                table,
                history_pages,
                current_pages,
                current_map,
                state["logical_to_slot"],
                read_base,
            )
            state["dynamic_table_oracle_checks"] += 1
            state["dynamic_table_oracle_pages"] += verified
        end = start + size
        attention_start = time.perf_counter()
        qsa_sparse_paged_attention(
            query[start:end],
            key_cache,
            value_cache,
            selected[start:end],
            table,
            request_ids[start:end],
            output[start:end],
        )
        attention_submit_wall += time.perf_counter() - attention_start
        split_calls += 1
        min_batch = min(min_batch, size)
        max_working = max(max_working, working)
        misses_total += misses
        selected_history_total += len(history_pages)
        start = end

    # Scheduler may pipeline the next 1024-token chunk before acknowledging
    # the previous worker result, so conservatively charge the complete frozen
    # WRITE partition rather than only this forward's current pages.
    real_pages = int(plan["write_page_count"]) + len(state["logical_to_slot"])
    if real_pages > cap:
        raise RuntimeError("Q2D combined READ/WRITE ownership exceeds physical cap")
    timing_delta = {
        field.removesuffix("_total"): float(state[field]) - timing_before[field]
        for field in cumulative_timing_fields
    }
    _write_event({
        "event": "q2d_streaming_runtime",
        "layer": layer.layer_name,
        "io_mode": (
            "direct_dedicated_slots" if state["direct_io"] else "staged_copy"
        ),
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
        "direct_load_verified_pages_total": int(
            state["direct_load_verified_pages"]
        ),
        "direct_consumer_sync_seconds_total": float(
            state["direct_consumer_sync_seconds_total"]
        ),
        "direct_consumer_syncs_total": int(state["direct_consumer_syncs"]),
        "dynamic_table_oracle_enabled": bool(state["verify_dynamic_table"]),
        "dynamic_table_oracle_checks_total": int(
            state["dynamic_table_oracle_checks"]
        ),
        "dynamic_table_oracle_pages_total": int(
            state["dynamic_table_oracle_pages"]
        ),
        "d2h_bytes_total": int(state["d2h_bytes"]),
        "d2h_jobs_total": int(state["d2h_jobs"]),
        "h2d_bytes_total": int(state["h2d_bytes"]),
        "h2d_jobs_total": int(state["h2d_jobs"]),
        "forward_exposed_wall_seconds": time.perf_counter() - forward_start,
        "write_mapping_wall_seconds": write_mapping_wall,
        "kv_update_submit_wall_seconds": kv_update_submit_wall,
        "publish_wall_seconds": publish_wall,
        "selection_plan_wall_seconds": selection_plan_wall,
        "selection_tensor_wall_seconds": selection_tensor_wall,
        "selection_cpu_filter_wall_seconds": selection_cpu_filter_wall,
        "stage_history_wall_seconds": stage_history_wall,
        "table_build_wall_seconds": table_build_wall,
        "table_forward_refresh_wall_seconds": table_forward_refresh_wall,
        "attention_submit_wall_seconds": attention_submit_wall,
        **timing_delta,
        "trace_records_total": int(state["trace_records"]),
        "trace_bytes_total": int(state["trace_bytes"]),
        "trace_truncated": bool(state["trace_truncated"]),
        "read_table_mode": "dynamic_cpu_history_plus_scheduler_writes",
        "write_partition": [0, read_base - 1],
        "read_partition": [read_base, cap - 1],
    })
