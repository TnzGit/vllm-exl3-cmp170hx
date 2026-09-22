import importlib.util
from pathlib import Path

import pytest

from vllm_exl3.kvmem_q2d_reload_shadow import (
    _allocate_slot,
    _assign_many,
    _assign_slot,
    _touch,
)
from vllm_exl3.kvmem_q2d_plan import validate_reload_plan


ROOT = Path(__file__).resolve().parents[1]
MAKER = ROOT / "tools" / "kvmem_qsa_make_q2d_reload_plan.py"


def _load_maker():
    spec = importlib.util.spec_from_file_location("q2d_plan", MAKER)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _q2c():
    return {
        "schema": 1,
        "mode": "qsa_scheduler_owned_transition",
        "page_tokens": 16,
        "physical_page_count": 4160,
        "scheduler_chunk_tokens": 1024,
        "expected_qsa_layers": 12,
        "query_span": [159488, 159533],
        "target_facts": [{"code": "violet-harbor-31"}],
    }


def test_promote_builds_frozen_reload_geometry():
    plan = _load_maker().promote(_q2c())
    assert plan["mode"] == "qsa_cpu_reload_shadow"
    assert plan["diagnostic_full_source_shadow"] is True
    assert plan["physical_page_count"] == 4160
    assert plan["query_row_batch"] == 64
    assert plan["staging_pages"] == 128
    assert plan["cpu_page_count"] == 10063
    assert validate_reload_plan(plan) == plan


def test_plan_rejects_weakened_capacity_or_tolerance():
    plan = _load_maker().promote(_q2c())
    plan["physical_page_count"] = 4161
    with pytest.raises(ValueError, match="physical_page_count"):
        validate_reload_plan(plan)
    plan = _load_maker().promote(_q2c())
    plan["allclose_atol"] = 0.03
    with pytest.raises(ValueError, match="atol"):
        validate_reload_plan(plan)


def _lru_state(capacity=3):
    return {
        "logical_to_slot": {},
        "slot_to_logical": [None] * capacity,
        "last_use": {},
        "clock": 0,
        "peak_slots": 0,
    }


def test_lru_never_evicts_protected_read_or_write_pages():
    state = _lru_state()
    assert [_assign_slot(state, page, set()) for page in (10, 11, 12)] == [0, 1, 2]
    _touch(state, (10, 11, 12))
    _touch(state, (10,))
    slot = _assign_slot(state, 13, {10, 12, 13})
    assert slot == 1
    assert set(state["logical_to_slot"]) == {10, 12, 13}
    assert state["peak_slots"] == 3


def test_lru_refuses_when_entire_pool_is_protected():
    state = _lru_state(2)
    _assign_slot(state, 1, set())
    _assign_slot(state, 2, set())
    with pytest.raises(RuntimeError, match="no evictable"):
        _allocate_slot(state, {1, 2})


def test_bulk_lru_assigns_all_misses_with_one_victim_selection():
    state = _lru_state(4)
    _assign_many(state, [1, 2, 3, 4], set())
    _touch(state, [1, 2, 3, 4])
    slots = _assign_many(state, [2, 5, 6], {2, 5, 6})
    assert len(set(slots)) == 3
    assert set(state["logical_to_slot"]) == {2, 4, 5, 6}


def test_bulk_assignment_zero_miss_and_free_slots_preserve_exact_order():
    state = _lru_state(4)
    assert _assign_many(state, [10, 11], {10, 11}) == [0, 1]
    _touch(state, [10, 11])
    before = (
        dict(state["logical_to_slot"]),
        list(state["slot_to_logical"]),
        dict(state["last_use"]),
    )
    assert _assign_many(state, [11, 10], {10, 11}) == [1, 0]
    assert before == (
        state["logical_to_slot"],
        state["slot_to_logical"],
        state["last_use"],
    )
    assert _assign_many(state, [12, 13], {12, 13}) == [2, 3]
