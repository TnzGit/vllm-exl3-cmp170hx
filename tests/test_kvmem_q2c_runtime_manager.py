import json
from pathlib import Path

import pytest

from vllm_exl3.kvmem_qsa_scheduler_runtime import (
    QSAResidentRuntimeManager,
    QSAResidentRuntimeSpec,
    make_qsa_runtime_spec,
    validate_runtime_plan,
)


class _Block:
    def __init__(self, block_id: int, *, is_null: bool = False):
        self.block_id = block_id
        self.is_null = is_null
        self.ref_cnt = 0 if is_null else 1
        self.block_hash = None


class _Pool:
    def __init__(self, n: int = 20000):
        self.null_block = _Block(0, is_null=True)
        self.free = [_Block(i) for i in range(1, n + 1)]
        self.freed = []

    def get_new_blocks(self, n: int):
        assert n <= len(self.free)
        out = self.free[:n]
        del self.free[:n]
        return out

    def free_blocks(self, blocks):
        rows = list(blocks)
        assert all(not b.is_null for b in rows)
        self.freed.extend(rows)
        self.free.extend(rows)


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
    pool = _Pool()
    mgr = QSAResidentRuntimeManager(
        spec, block_pool=pool, enable_caching=False, kv_cache_group_id=0,
        scheduler_block_size=1568, dcp_world_size=1, pcp_world_size=1,
        needs_kv_cache_zeroing=False,
    )
    return spec, pool, mgr


def test_runtime_spec_separates_logical_width_from_bounded_accounting():
    spec = make_qsa_runtime_spec(_plan())
    assert isinstance(spec, QSAResidentRuntimeSpec)
    assert spec.block_size == 16
    assert spec.page_size_bytes == 32768
    assert spec.physical_page_cap == 4160
    assert spec.max_num_blocks_per_req(None, 161000) == 10063
    # Generic shared-HMA config keeps one normal-geometry placeholder page.
    # The real 4160+null QSA pages live in the model-owned dedicated tensor.
    assert spec.max_memory_usage_bytes(None) == 32768
    assert spec.dedicated_page_count == 4161
    assert spec.virtual_null_block_id == 4160
    assert spec.dedicated_memory_bytes_per_layer == 4161 * 32768
    assert spec.prefix_cacheable is False


def test_full_sequence_admission_costs_zero_shared_blocks():
    _, pool, mgr = _manager()
    assert mgr.get_num_blocks_to_allocate(
        "r", 161000, [], 0, 0, 161000, apply_admission_cap=True
    ) == 0
    assert mgr.virtual_free_pages == 4160
    assert len(pool.free) == 20000


def test_progressive_prefill_never_exceeds_4160_and_keeps_logical_holes(
    monkeypatch, tmp_path
):
    stats = tmp_path / "sched.jsonl"
    monkeypatch.setenv("VLLM_QWEN_KVMEM_Q2C_SCHED_STATS_PATH", str(stats))
    _, pool, mgr = _manager()
    req = "r"

    # The hardware runner pins max_num_batched_tokens=1024 = 64 QSA pages.
    for processed in range(0, 160000, 1024):
        mgr.remove_skipped_blocks(req, processed)
        target = min(processed + 1024, 160000)
        shared_cost = mgr.get_num_blocks_to_allocate(
            req, target, [], processed, processed, target
        )
        fresh = mgr.allocate_new_blocks(req, target, target)
        assert shared_cost == 0
        assert len(fresh) <= 64
        assert mgr._real_count(req) <= 4160
        assert mgr.virtual_free_pages == 4160 - mgr._real_count(req)

    # Commit the final historical chunk: all non-sticky history is now null.
    mgr.remove_skipped_blocks(req, 160000)
    blocks = mgr.req_to_blocks[req]
    assert len(blocks) == 10000
    assert mgr._real_count(req) == 4096
    assert all(not blocks[i].is_null for i in range(4096))
    assert all(blocks[i].is_null for i in range(4096, 10000))
    assert len(pool.freed) == 0
    assert mgr._peak_real_pages[req] == 4160
    assert mgr.virtual_free_pages == 64
    assert all(0 <= b.block_id < 4160 for b in blocks if not b.is_null)
    assert all(b.block_id == 4160 for b in blocks if b.is_null)

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

    assert mgr.get_num_blocks_to_allocate(
        req, 161000, [], 160000, 160000, 161000
    ) == 0
    fresh = mgr.allocate_new_blocks(req, 161000, 161000)
    assert len(fresh) == 63
    assert len(mgr.req_to_blocks[req]) == 10063
    assert mgr._real_count(req) == 4159

    assert mgr.get_num_blocks_to_allocate(
        req, 161024, [], 161000, 161000, 161024
    ) == 0
    fresh = mgr.allocate_new_blocks(req, 161024, 161024)
    assert len(fresh) == 1
    assert mgr._real_count(req) == 4160
    assert mgr.get_num_blocks_to_allocate(
        req, 161025, [], 161024, 161024, 161025
    ) == 0
    with pytest.raises(RuntimeError, match="active suffix exceeded reserve"):
        mgr.allocate_new_blocks(req, 161025, 161025)


def test_runtime_plan_is_strict():
    p = validate_runtime_plan(_plan())
    assert p["resident_pages"] == tuple(range(4096))
    bad = dict(_plan(), mode="qsa_cpu_backed_shadow")
    with pytest.raises(ValueError, match="mode mismatch"):
        validate_runtime_plan(bad)
    bad_chunk = dict(_plan(), scheduler_chunk_tokens=2048)
    with pytest.raises(ValueError, match="chunk must equal"):
        validate_runtime_plan(bad_chunk)


def test_virtual_blocks_never_return_to_shared_pool():
    _, pool, mgr = _manager()
    req = "r"
    mgr.allocate_new_blocks(req, 1024, 1024)
    assert mgr._real_count(req) == 64
    ids = [b.block_id for b in mgr.req_to_blocks[req] if not b.is_null]
    assert ids == list(range(64))
    assert mgr.take_new_block_ids() == []
    returned = mgr.pop_blocks_for_free(req)
    assert returned == []
    assert mgr.virtual_free_pages == 4160
    assert len(pool.free) == 20000
    assert pool.freed == []


def test_append_only_worker_ids_keep_logical_alignment_under_virtual_reuse():
    _, _, mgr = _manager()
    req = "r"
    worker_ids: list[int] = []

    for processed in range(0, 160000, 1024):
        mgr.remove_skipped_blocks(req, processed)
        target = min(processed + 1024, 160000)
        before_len = len(worker_ids)
        fresh = mgr.allocate_new_blocks(req, target, target)
        worker_ids.extend(int(b.block_id) for b in fresh)

        # Every new logical page in this frozen prefill gets one fresh virtual
        # ID; reclaim only changes old scheduler entries, so the vLLM 0.29
        # worker's append-only table stays position-aligned.
        logical_len = target // 16
        assert len(worker_ids) == logical_len
        assert len(worker_ids) - before_len == (target - processed) // 16
        assert all(0 <= x < 4160 for x in worker_ids)

    mgr.remove_skipped_blocks(req, 160000)
    scheduler_blocks = mgr.req_to_blocks[req]
    assert len(worker_ids) == len(scheduler_blocks) == 10000

    # Scheduler has real null holes, while the worker intentionally retains
    # stale virtual IDs at those old positions. Those positions must therefore
    # be masked by frozen policy before attention rather than interpreted as
    # current physical ownership.
    assert any(b.is_null for b in scheduler_blocks)
    assert all(0 <= x < 4160 for x in worker_ids)
    assert len(set(worker_ids)) <= 4160


def test_variable_mamba_aligned_prefill_chunks_stay_within_virtual_cap():
    _, _, mgr = _manager()
    req = "mamba-aligned"
    worker_ids: list[int] = []
    processed = 0

    # Emulate a 1024-token budget that is periodically shortened to land on
    # 1568-token hybrid/Mamba boundaries. 1568 == 98 QSA pages, so every
    # resulting boundary still lands exactly on a 16-token QSA page.
    while processed < 160000:
        mgr.remove_skipped_blocks(req, processed)
        nominal_end = min(processed + 1024, 160000)
        next_mamba_boundary = ((processed // 1568) + 1) * 1568
        target = min(nominal_end, next_mamba_boundary)
        if target <= processed:
            target = nominal_end
        assert target % 16 == 0

        fresh = mgr.allocate_new_blocks(req, target, target)
        worker_ids.extend(int(b.block_id) for b in fresh)
        assert len(fresh) <= 64
        assert mgr._real_count(req) <= 4160
        assert mgr.virtual_free_pages == 4160 - mgr._real_count(req)
        assert len(worker_ids) == target // 16
        processed = target

    mgr.remove_skipped_blocks(req, 160000)
    assert len(worker_ids) == 10000
    assert mgr._real_count(req) == 4096
    assert mgr._peak_real_pages[req] <= 4160
    assert all(0 <= x < 4160 for x in worker_ids)
