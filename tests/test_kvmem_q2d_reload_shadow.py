import importlib.util
from pathlib import Path

import numpy as np
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


def test_bulk_touch_preserves_exact_sequential_clock_values():
    state = _lru_state()
    _assign_many(state, [10, 11, 12], set())
    _touch(state, [10, 11, 12])
    _touch(state, [12, 10])
    assert state["clock"] == 5
    assert state["last_use"] == {10: 5, 11: 2, 12: 4}


def test_array_lru_matches_stable_dict_ties_and_touch_order():
    state = _lru_state(3)
    state["last_use_array"] = np.full(32, -1, dtype=np.int64)
    _assign_many(state, [10, 11, 12], set())
    _touch(state, [10, 11, 12])
    _touch(state, [12, 10])
    assert state["clock"] == 5
    assert state["last_use_array"][[10, 11, 12]].tolist() == [5, 2, 4]
    assert _assign_many(state, [13], {10, 12, 13}) == [1]
    assert int(state["last_use_array"][11]) == -1

    tied = _lru_state(2)
    tied["last_use_array"] = np.full(32, -1, dtype=np.int64)
    _assign_many(tied, [20, 21], set())
    tied["last_use_array"][[20, 21]] = 7
    assert _assign_many(tied, [22], {22}) == [0]


def test_array_lru_victim_selection_matches_stable_full_sort():
    rng = np.random.default_rng(423)
    for capacity in (4, 31, 4032):
        for missing_count in (1, 2, min(23, capacity - 1)):
            resident = [int(page) for page in rng.permutation(capacity)]
            timestamps = np.full(capacity + missing_count, -1, dtype=np.int64)
            timestamps[:capacity] = rng.integers(0, 9, size=capacity)
            state = _lru_state(capacity)
            state["logical_to_slot"] = {
                page: slot for slot, page in enumerate(resident)
            }
            state["slot_to_logical"] = list(resident)
            state["last_use_array"] = timestamps
            protected = set(resident[:1])
            candidates = [page for page in resident if page not in protected]
            victims = sorted(candidates, key=lambda page: timestamps[page])[
                :missing_count
            ]
            expected_slots = [state["logical_to_slot"][page] for page in victims]
            new_pages = list(range(capacity, capacity + missing_count))
            assert _assign_many(state, new_pages, protected | set(new_pages)) == (
                expected_slots
            )


def test_array_lru_replays_stable_lru_events():
    def reference_assign(state, pages):
        missing = [page for page in pages if page not in state["logical_to_slot"]]
        free = [
            slot for slot, logical in enumerate(state["slot_to_logical"])
            if logical is None
        ][:len(missing)]
        need = max(0, len(missing) - len(free))
        candidates = [
            page for page in state["logical_to_slot"] if page not in pages
        ]
        victims = sorted(
            candidates, key=lambda page: state["last_use_array"][page]
        )[:need]
        available = list(free)
        for victim in victims:
            slot = state["logical_to_slot"].pop(victim)
            state["last_use_array"][victim] = -1
            state["slot_to_logical"][slot] = None
            available.append(slot)
        for page, slot in zip(missing, available, strict=True):
            state["logical_to_slot"][page] = slot
            state["slot_to_logical"][slot] = page
        state["peak_slots"] = max(state["peak_slots"], len(state["logical_to_slot"]))
        return [state["logical_to_slot"][page] for page in pages]

    rng = np.random.default_rng(424)
    actual = _lru_state(64)
    actual["last_use_array"] = np.full(256, -1, dtype=np.int64)
    expected = _lru_state(64)
    expected["last_use_array"] = np.full(256, -1, dtype=np.int64)
    for _ in range(500):
        pages = sorted(
            int(page) for page in rng.choice(256, size=int(rng.integers(1, 25)), replace=False)
        )
        assert _assign_many(actual, pages, set(pages)) == reference_assign(
            expected, pages
        )
        _touch(actual, pages)
        _touch(expected, pages)
        assert actual["logical_to_slot"] == expected["logical_to_slot"]
        assert actual["slot_to_logical"] == expected["slot_to_logical"]
        assert actual["clock"] == expected["clock"]
        assert np.array_equal(actual["last_use_array"], expected["last_use_array"])


def test_array_lru_rejects_range_and_clock_overflow_and_preserves_duplicates():
    state = _lru_state(2)
    state["last_use_array"] = np.full(4, -1, dtype=np.int64)
    with pytest.raises(RuntimeError, match="outside LRU timestamp"):
        _assign_many(state, [-1], set())
    with pytest.raises(RuntimeError, match="outside LRU timestamp"):
        _assign_many(state, [4], set())

    _assign_many(state, [1, 2], set())
    _touch(state, [1, 1, 2])
    assert state["clock"] == 3
    assert state["last_use_array"][[1, 2]].tolist() == [2, 3]
    state["clock"] = int(np.iinfo(np.int64).max)
    with pytest.raises(RuntimeError, match="exceeds array dtype"):
        _touch(state, [1])


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
