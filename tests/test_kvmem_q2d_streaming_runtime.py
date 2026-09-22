import importlib.util
import json
from pathlib import Path

import pytest
import torch

from vllm_exl3.kvmem_q2d_scheduler_runtime import (
    QSAStreamingRuntimeManager,
    make_qsa_streaming_spec,
    validate_streaming_plan,
)
from vllm_exl3.kvmem_q2d_streaming_worker import (
    _CUMULATIVE_TIMING_FIELDS,
    _bits_equal,
    _direct_io_enabled,
    _dynamic_table_oracle_enabled,
    _layer_id,
    _logical_write_ids,
    _prepare_forward_table,
    _stage_history,
    _update_dynamic_table,
    _verify_dynamic_table,
)
from vllm_exl3.kvmem_q2d_reload_shadow import _assign_many
from vllm_exl3.kvmem_vllm_offload import TransferObservation


ROOT = Path(__file__).resolve().parents[1]
MAKER = ROOT / "tools" / "kvmem_qsa_make_q2d_runtime_plan.py"


class _Block:
    def __init__(self, block_id: int, *, is_null: bool = False):
        self.block_id = block_id
        self.is_null = is_null
        self.ref_cnt = 0 if is_null else 1
        self.block_hash = None


class _Pool:
    def __init__(self):
        self.null_block = _Block(0, is_null=True)


def _q2c():
    return {
        "schema": 1,
        "mode": "qsa_scheduler_owned_transition",
        "page_tokens": 16,
        "physical_page_count": 4160,
        "scheduler_chunk_tokens": 1024,
        "query_span": [159488, 159533],
        "target_facts": [],
    }


def _load_maker():
    spec = importlib.util.spec_from_file_location("q2d_runtime_plan", MAKER)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _plan():
    return _load_maker().promote(_q2c())


def _manager():
    spec = make_qsa_streaming_spec(_plan())
    manager = QSAStreamingRuntimeManager(
        spec,
        block_pool=_Pool(),
        enable_caching=False,
        kv_cache_group_id=0,
        scheduler_block_size=1568,
        dcp_world_size=1,
        pcp_world_size=1,
        needs_kv_cache_zeroing=False,
    )
    return spec, manager


def test_runtime_plan_freezes_partition_and_cpu_capacity():
    plan = _plan()
    assert validate_streaming_plan(plan) == plan
    assert plan["write_page_count"] == 128
    assert plan["read_cache_page_count"] == 4032
    assert plan["cpu_page_count"] == 10063
    assert plan["write_page_count"] + plan["read_cache_page_count"] == 4160
    bad = dict(plan, write_page_count=129)
    with pytest.raises(ValueError, match="write_page_count"):
        validate_streaming_plan(bad)


def test_scheduler_owns_bounded_two_chunk_write_pipeline(monkeypatch, tmp_path):
    stats = tmp_path / "scheduler.jsonl"
    monkeypatch.setenv("VLLM_QWEN_KVMEM_Q2D_SCHED_STATS_PATH", str(stats))
    spec, manager = _manager()
    request_id = "r"
    # The live scheduler allocates the second chunk before its first processed
    # callback, so model that pipeline explicitly.
    manager.allocate_new_blocks(request_id, 1024, 1024)
    manager.allocate_new_blocks(request_id, 2048, 2048)
    assert manager._real_count(request_id) == 128
    for processed in range(1024, 160000, 1024):
        manager.remove_skipped_blocks(request_id, processed)
        target = min(processed + 1024, 160000)
        assert manager.get_num_blocks_to_allocate(
            request_id, target, [], processed, processed, target
        ) == 0
        fresh = manager.allocate_new_blocks(request_id, target, target)
        assert 0 <= len(fresh) <= 64
        assert manager._real_count(request_id) <= 128
        assert all(0 <= block.block_id < 128 for block in fresh)
    manager.remove_skipped_blocks(request_id, 160000)
    assert manager._real_count(request_id) == 0
    assert manager._peak_real_pages[request_id] == 128
    assert all(block.is_null for block in manager.req_to_blocks[request_id])
    assert spec.physical_page_cap == 4160
    rows = [json.loads(line) for line in stats.read_text().splitlines()]
    assert sum(row.get("freed_pages", 0) for row in rows) == 10000
    assert sum(len(row.get("logical_pages", [])) for row in rows) == 10000
    assert max(row.get("peak_real_write_pages", 0) for row in rows) == 128


def test_scheduler_keeps_partial_decode_page_until_complete():
    _, manager = _manager()
    request_id = "r"
    manager.allocate_new_blocks(request_id, 15, 15)
    manager.remove_skipped_blocks(request_id, 15)
    assert manager._real_count(request_id) == 1
    manager.allocate_new_blocks(request_id, 16, 16)
    manager.remove_skipped_blocks(request_id, 16)
    assert manager._real_count(request_id) == 0


def test_worker_write_partition_and_bit_comparison_are_strict():
    assert _layer_id("model.layers.17.self_attn") == 17
    with pytest.raises(RuntimeError, match="derive layer id"):
        _layer_id("model.attn")
    table = torch.tensor([[0, 1, 63, 64]], dtype=torch.int32)
    assert _logical_write_ids([0, 1, 2], table, 64).tolist() == [0, 1, 63]
    with pytest.raises(RuntimeError, match="outside WRITE"):
        _logical_write_ids([3], table, 64)
    left = torch.tensor([1.0, -0.0], dtype=torch.bfloat16)
    right = left.clone()
    assert _bits_equal(left, right)
    right[0] = 2.0
    assert not _bits_equal(left, right)


def test_persistent_dynamic_table_applies_deltas_and_forward_write_view():
    table = torch.full((1, 16), -1, dtype=torch.int32)
    _update_dynamic_table(
        table,
        mapped_pages=[1, 2, 5],
        physical_pages=[129, 130, 133],
    )
    _update_dynamic_table(
        table,
        clear_pages=[2],
        mapped_pages=[3],
        physical_pages=[131],
    )
    assert table[0, [1, 2, 3, 5]].tolist() == [129, -1, 131, 133]

    state = {
        "dynamic_table": table,
        "current_write_pages": [1, 4],
        "logical_to_slot": {1: 7, 5: 5},
    }
    table[0, 1] = 9
    table[0, 4] = 10
    result = _prepare_forward_table(state, [6, 7], [11, 12], 128)
    assert result is table
    assert table[0, [1, 4, 6, 7]].tolist() == [135, -1, 11, 12]
    assert state["current_write_pages"] == [6, 7]

    verified = _verify_dynamic_table(
        table,
        history_pages=[1, 5],
        current_pages=[6, 7],
        current_map={6: 11, 7: 12},
        logical_to_slot={1: 7, 5: 5},
        read_base=128,
    )
    assert verified == 4
    table[0, 5] = 999
    with pytest.raises(RuntimeError, match="differs from exact mapping"):
        _verify_dynamic_table(
            table,
            history_pages=[1, 5],
            current_pages=[6, 7],
            current_map={6: 11, 7: 12},
            logical_to_slot={1: 7, 5: 5},
            read_base=128,
        )


def test_direct_io_flag_is_strict(monkeypatch):
    assert "direct_consumer_sync_seconds_total" in _CUMULATIVE_TIMING_FIELDS
    assert "stage_slot_assignment_seconds_total" in _CUMULATIVE_TIMING_FIELDS
    monkeypatch.delenv("VLLM_QWEN_KVMEM_Q2E_DIRECT_IO", raising=False)
    assert _direct_io_enabled() is False
    monkeypatch.setenv("VLLM_QWEN_KVMEM_Q2E_DIRECT_IO", "1")
    assert _direct_io_enabled() is True
    monkeypatch.setenv("VLLM_QWEN_KVMEM_Q2E_DIRECT_IO", "yes")
    with pytest.raises(RuntimeError, match="must be 0 or 1"):
        _direct_io_enabled()
    monkeypatch.delenv(
        "VLLM_QWEN_KVMEM_Q2E_VERIFY_DYNAMIC_TABLE", raising=False
    )
    assert _dynamic_table_oracle_enabled() is False
    monkeypatch.setenv("VLLM_QWEN_KVMEM_Q2E_VERIFY_DYNAMIC_TABLE", "1")
    assert _dynamic_table_oracle_enabled() is True


def test_direct_stage_history_uses_one_arbitrary_destination_job():
    class _Backing:
        def __init__(self):
            self.calls = []

        def stage_in(self, pages, destinations):
            self.calls.append((list(pages), list(destinations)))
            return TransferObservation(1, len(pages) * 32768, 0.1, 0.2)

    backing = _Backing()
    state = {
        "backing": backing,
        "direct_io": True,
        "logical_to_slot": {},
        "slot_to_logical": [None] * 4,
        "slot_generations": [0] * 4,
        "dynamic_table": torch.full((1, 32), -1, dtype=torch.int32),
        "current_write_pages": [],
        "last_use": {},
        "clock": 0,
        "peak_slots": 0,
        "peak_read_slots": 0,
        "published": {10, 11, 12},
        "h2d_bytes": 0,
        "h2d_jobs": 0,
        "direct_load_verified_pages": 0,
        "direct_consumer_sync_seconds_total": 0.0,
        "direct_consumer_syncs": 0,
        **{
            f"h2d_{field}_total": 0.0
            for field in (
                "event_seconds", "wall_seconds", "prepare_seconds",
                "submit_seconds", "wait_seconds", "finish_seconds",
            )
        },
        "d2d_copy_submit_seconds_total": 0.0,
    }
    layer = type("Layer", (), {"_q2d_staging": torch.empty((2, 1))})()
    result = _stage_history(
        layer,
        state,
        {"write_page_count": 128},
        [10, 11, 12],
        torch.empty((132, 1)),
    )
    assert result[0] == 3
    assert backing.calls == [([10, 11, 12], [128, 129, 130])]
    assert state["dynamic_table"][0, [10, 11, 12]].tolist() == [128, 129, 130]
    assert state["h2d_jobs"] == 1
    assert state["d2d_copy_submit_seconds_total"] == 0.0


def test_streaming_read_state_satisfies_shared_bulk_lru_contract():
    state = {
        "logical_to_slot": {},
        "slot_to_logical": [None] * 4,
        "last_use": {},
        "clock": 0,
        "peak_slots": 0,
    }
    assert _assign_many(state, [10, 11], {10, 11}) == [0, 1]
    assert state["peak_slots"] == 2
