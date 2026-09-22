import json
from pathlib import Path

import pytest
import torch

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
        "physical_page_count": 4160,
    }


def _manager(tmp_path: Path | None = None):
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
    assert spec.max_memory_usage_bytes(None) == 4160 * 32768
    assert spec.prefix_cacheable is False


def test_transition_releases_nonresident_history_but_keeps_full_logical_row(monkeypatch, tmp_path):
    stats = tmp_path / "sched.jsonl"
    monkeypatch.setenv("VLLM_QWEN_KVMEM_Q2C_SCHED_STATS_PATH", str(stats))
    spec, pool, mgr = _manager()
    req = "r"
    num_tokens = 161000
    predicted = mgr.get_num_blocks_to_allocate(req, num_tokens, [], 0, 0, num_tokens)
    assert predicted == 10063
    new = mgr.allocate_new_blocks(req, num_tokens, num_tokens)
    assert len(new) == 10063
    assert len(mgr.req_to_blocks[req]) == 10063
    assert mgr._real_count(req) == 10063

    mgr.remove_skipped_blocks(req, 159999)
    assert mgr._real_count(req) == 10063

    mgr.remove_skipped_blocks(req, 160000)
    blocks = mgr.req_to_blocks[req]
    assert len(blocks) == 10063
    assert mgr._real_count(req) == 4159
    assert len(pool.freed) == 5904
    assert all(not blocks[i].is_null for i in range(4096))
    assert all(blocks[i].is_null for i in range(4096, 10000))
    assert all(not blocks[i].is_null for i in range(10000, 10063))
    row = json.loads(stats.read_text().strip())
    assert row["event"] == "q2c_scheduler_shrink"
    assert row["logical_row_pages"] == 10063
    assert row["real_pages_before"] == 10063
    assert row["real_pages_after"] == 4159
    assert row["freed_pages"] == 5904
    assert row["physical_page_cap"] == 4160


def test_post_transition_allocation_only_grows_active_reserve():
    spec, pool, mgr = _manager()
    req = "r"
    mgr.allocate_new_blocks(req, 161000, 161000)
    mgr.remove_skipped_blocks(req, 160000)
    assert mgr.get_num_blocks_to_allocate(req, 161024, [], 160000, 160000, 161024) == 1
    fresh = mgr.allocate_new_blocks(req, 161024, 161024)
    assert len(fresh) == 1
    assert len(mgr.req_to_blocks[req]) == 10064
    assert mgr._real_count(req) == 4160
    with pytest.raises(RuntimeError, match="active suffix exceeded reserve"):
        mgr.get_num_blocks_to_allocate(req, 161025, [], 160000, 160000, 161025)


def test_runtime_plan_is_strict():
    p = validate_runtime_plan(_plan())
    assert p["resident_pages"] == tuple(range(4096))
    bad = dict(_plan(), mode="qsa_cpu_backed_shadow")
    with pytest.raises(ValueError, match="mode mismatch"):
        validate_runtime_plan(bad)
