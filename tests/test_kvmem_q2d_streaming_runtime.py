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
from vllm_exl3.kvmem_q2d_streaming_worker import _bits_equal, _logical_write_ids
from vllm_exl3.kvmem_q2d_reload_shadow import _assign_many


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
    table = torch.tensor([[0, 1, 63, 64]], dtype=torch.int32)
    assert _logical_write_ids([0, 1, 2], table, 64).tolist() == [0, 1, 63]
    with pytest.raises(RuntimeError, match="outside WRITE"):
        _logical_write_ids([3], table, 64)
    left = torch.tensor([1.0, -0.0], dtype=torch.bfloat16)
    right = left.clone()
    assert _bits_equal(left, right)
    right[0] = 2.0
    assert not _bits_equal(left, right)


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
