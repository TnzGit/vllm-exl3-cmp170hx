import json
from pathlib import Path

import pytest

from vllm_exl3.kvmem_qsa_scheduler_runtime import (
    QSAResidentRuntimeManager,
    QSAResidentRuntimeSpec,
    make_qsa_runtime_spec,
    validate_runtime_plan,
)


class _StockPoolSentinel:
    pass


def _plan():
    resident = list(range(4096))
    return {
        "schema": 1,
        "mode": "qsa_scheduler_owned_transition",
        "page_tokens": 16,
        "resident_pages": resident,
        "resident_page_count": len(resident),
        "apply_min_pos": 160000,
        "active_from_pos": 160000,
        "active_page0": 10000,
        "active_reserve_pages": 64,
        "active_reserve_tokens": 1024,
        "scheduler_chunk_tokens": 1024,
        "physical_page_count": 4160,
    }


def _manager():
    spec = make_qsa_runtime_spec(_plan())
    stock = _StockPoolSentinel()
    mgr = QSAResidentRuntimeManager(
        spec, block_pool=stock, enable_caching=False, kv_cache_group_id=0,
        scheduler_block_size=322000, dcp_world_size=1, pcp_world_size=1,
        needs_kv_cache_zeroing=True,
    )
    return spec, stock, mgr


def test_runtime_spec_separates_logical_width_from_bounded_accounting():
    spec = make_qsa_runtime_spec(_plan())
    assert isinstance(spec, QSAResidentRuntimeSpec)
    assert spec.block_size == 16
    assert spec.page_size_bytes == 32768
    assert spec.physical_page_cap == 4160
    assert spec.max_num_blocks_per_req(None, 161000) == 10063
    assert spec.max_memory_usage_bytes(None) == 4160 * 32768
    assert spec.prefix_cacheable is False
    assert spec.q2c_private_pool is True
    assert spec.private_pool_num_blocks == 4161


def test_full_sequence_admission_is_private_and_invisible_to_stock_pool():
    _, stock, mgr = _manager()
    assert mgr._stock_block_pool is stock
    assert mgr.block_pool.num_gpu_blocks == 4161
    assert mgr.block_pool.null_block.block_id == 0
    assert mgr.block_pool.get_num_free_blocks() == 4160
    assert mgr._private_num_blocks_to_allocate(
        "r", 161000, [], True
    ) == 4160
    assert mgr.get_num_blocks_to_allocate(
        "r", 161000, [], 0, 0, 161000, apply_admission_cap=True
    ) == 0
    assert mgr.take_new_block_ids() == []


def test_progressive_prefill_never_exceeds_4160_and_keeps_logical_holes(
    monkeypatch, tmp_path
):
    stats = tmp_path / "sched.jsonl"
    monkeypatch.setenv("VLLM_QWEN_KVMEM_Q2C_SCHED_STATS_PATH", str(stats))
    _, _, mgr = _manager()
    req = "r"

    # The hardware runner pins max_num_batched_tokens=1024 = 64 QSA pages.
    for processed in range(0, 160000, 1024):
        mgr.remove_skipped_blocks(req, processed)
        target = min(processed + 1024, 160000)
        predicted = mgr._private_num_blocks_to_allocate(
            req, target, [], False
        )
        assert mgr.get_num_blocks_to_allocate(
            req, target, [], processed, processed, target
        ) == 0
        fresh = mgr.allocate_new_blocks(req, target, target)
        assert len(fresh) == predicted
        assert mgr._real_count(req) <= 4160

    # Commit the final historical chunk: all non-sticky history is now null.
    mgr.remove_skipped_blocks(req, 160000)
    blocks = mgr.req_to_blocks[req]
    assert len(blocks) == 10000
    assert mgr._real_count(req) == 4096
    assert all(not blocks[i].is_null for i in range(4096))
    assert all(blocks[i].is_null for i in range(4096, 10000))
    assert mgr.block_pool.get_num_free_blocks() == 64
    assert mgr._peak_real_pages[req] == 4160

    rows = [json.loads(x) for x in stats.read_text().splitlines()]
    reclaim = [x for x in rows if x["event"] == "q2c_scheduler_reclaim"]
    boundary = [x for x in rows if x["event"] == "q2c_scheduler_boundary"]
    assert sum(x["freed_pages"] for x in reclaim) == 5904
    assert len(boundary) == 1
    assert boundary[0]["logical_row_pages"] == 10000
    assert boundary[0]["real_pages_at_boundary"] == 4096
    assert boundary[0]["physical_page_cap"] == 4160
    assert boundary[0]["peak_real_pages"] == 4160


def test_post_boundary_allocation_only_grows_active_reserve():
    _, _, mgr = _manager()
    req = "r"
    for processed in range(0, 160000, 1024):
        mgr.remove_skipped_blocks(req, processed)
        target = min(processed + 1024, 160000)
        mgr.allocate_new_blocks(req, target, target)
    mgr.remove_skipped_blocks(req, 160000)

    assert mgr._private_num_blocks_to_allocate(
        req, 161000, [], False
    ) == 63
    assert mgr.get_num_blocks_to_allocate(
        req, 161000, [], 160000, 160000, 161000
    ) == 0
    fresh = mgr.allocate_new_blocks(req, 161000, 161000)
    assert len(fresh) == 63
    assert len(mgr.req_to_blocks[req]) == 10063
    assert mgr._real_count(req) == 4159

    assert mgr._private_num_blocks_to_allocate(
        req, 161024, [], False
    ) == 1
    assert mgr.get_num_blocks_to_allocate(
        req, 161024, [], 161000, 161000, 161024
    ) == 0
    fresh = mgr.allocate_new_blocks(req, 161024, 161024)
    assert len(fresh) == 1
    assert mgr._real_count(req) == 4160
    with pytest.raises(RuntimeError, match="active suffix exceeded reserve"):
        mgr.get_num_blocks_to_allocate(
            req, 161025, [], 161024, 161024, 161025
        )
    assert mgr.block_pool.get_num_free_blocks() == 0
    mgr.free(req)
    assert mgr.block_pool.get_num_free_blocks() == 4160


def test_runtime_plan_is_strict():
    p = validate_runtime_plan(_plan())
    assert p["resident_pages"] == tuple(range(4096))
    bad = dict(_plan(), mode="qsa_cpu_backed_shadow")
    with pytest.raises(ValueError, match="mode mismatch"):
        validate_runtime_plan(bad)
    bad_chunk = dict(_plan(), scheduler_chunk_tokens=2048)
    with pytest.raises(ValueError, match="chunk must equal"):
        validate_runtime_plan(bad_chunk)
