from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from kvmem_q2e_lru_replay import (  # noqa: E402
    ReplayError,
    ReplayState,
    replay_records,
    replay_trace,
)
from vllm_exl3.kvmem_q2e_trace import (  # noqa: E402
    HEADER_STRUCT,
    Q2ETraceRecord,
    Q2ETraceWriter,
    TraceFormatError,
)


def _publication(
    pages: tuple[int, ...],
    *,
    victims: tuple[int, ...] = (),
    slots: tuple[int, ...] | None = None,
    generations: tuple[int, ...] | None = None,
) -> Q2ETraceRecord:
    if slots is None:
        slots = tuple(range(len(pages)))
    if generations is None:
        generations = (1,) * len(pages)
    return Q2ETraceRecord(
        layer_id=0,
        first_pos=0,
        subbatch_start=0,
        query_rows=0,
        missing_pages=pages,
        victim_pages=victims,
        assigned_slots=slots,
        assigned_generations=generations,
    )


def _query(
    history: tuple[int, ...],
    *,
    missing: tuple[int, ...] = (),
    victims: tuple[int, ...] = (),
    slots: tuple[int, ...] = (),
    generations: tuple[int, ...] = (),
) -> Q2ETraceRecord:
    return Q2ETraceRecord(
        layer_id=0,
        first_pos=1024,
        subbatch_start=0,
        query_rows=1,
        history_pages=history,
        missing_pages=missing,
        victim_pages=victims,
        assigned_slots=slots,
        assigned_generations=generations,
    )


def test_replays_zero_miss_and_free_slot_assignments() -> None:
    records = (
        _publication((10, 11)),
        _query((10, 11)),
        _query((11, 10)),
    )
    result = replay_records(records, capacity=4)
    assert result.records == 3
    assert result.publication_records == 1
    assert result.query_records == 2
    assert result.assignment_records == 2
    assert result.max_resident_pages == 2


def test_replays_eviction_and_does_not_evict_protected_pages() -> None:
    records = (
        _publication((1, 2)),
        _query((1, 2)),
        _publication((3,), victims=(1,), slots=(0,), generations=(2,)),
        _query((2, 3)),
        _publication((4,), victims=(2,), slots=(1,), generations=(2,)),
        # Page 2 was published before it was evicted.  This is the real
        # publication-before-reload shape produced by the runtime.
        _query((2, 3), missing=(2,), victims=(4,), slots=(1,), generations=(3,)),
    )
    result = replay_records(records, capacity=2)
    assert result.assignment_records == 5
    assert result.max_resident_pages == 2


def test_equal_lru_timestamps_use_stable_logical_insertion_order() -> None:
    state = ReplayState(2)
    state.logical_to_slot.update({10: 0, 11: 1})
    state.slot_to_logical[:] = [10, 11]
    state.last_use.update({10: 7, 11: 7})
    state.clock = 7

    slots, victims, generations = state.assign_many((12,), set())
    assert slots == [0]
    assert victims == [10]
    assert generations == [1]


def test_protected_tie_candidate_is_skipped() -> None:
    state = ReplayState(2)
    state.logical_to_slot.update({10: 0, 11: 1})
    state.slot_to_logical[:] = [10, 11]
    state.last_use.update({10: 7, 11: 7})
    slots, victims, _ = state.assign_many((12,), {10, 12})
    assert slots == [1]
    assert victims == [11]


def test_rejects_protected_pool_exhaustion_and_mismatch() -> None:
    records = (
        _publication((1, 2)),
        _publication((3,), victims=(1,), slots=(0,), generations=(2,)),
        _query((1, 2, 3), missing=(1,), victims=(), slots=(), generations=()),
    )
    with pytest.raises(ReplayError, match="insufficient evictable"):
        replay_records(records, capacity=2)

    bad = (
        _publication((1, 2)),
        _publication((3,), victims=(1,), slots=(0,), generations=(2,)),
        _query((1, 3), missing=(1,), victims=(2,), slots=(0,), generations=(2,)),
    )
    with pytest.raises(ReplayError, match="assigned_slots mismatch"):
        replay_records(bad, capacity=2)


def test_rejects_unpublished_history_and_duplicate_history() -> None:
    with pytest.raises(ReplayError, match="non-CPU-authoritative"):
        replay_records((_query((99,), missing=()),), capacity=2)
    with pytest.raises(ReplayError, match="duplicate pages"):
        replay_records((_query((1, 1)),), capacity=2)


def test_truncated_trace_is_not_accepted(tmp_path: Path) -> None:
    trace = tmp_path / "truncated.q2e"
    first = _publication((1,))
    with Q2ETraceWriter(
        trace,
        max_bytes=HEADER_STRUCT.size + len(first.to_bytes()) + 1,
    ) as writer:
        assert writer.append(first)
        assert writer.append(first) is False
    with pytest.raises(ReplayError, match="truncated"):
        replay_trace(trace, capacity=2)


def test_malformed_trace_surfaces_trace_format_error(tmp_path: Path) -> None:
    trace = tmp_path / "corrupt.q2e"
    with Q2ETraceWriter(trace) as writer:
        assert writer.append(_publication((1,)))
    payload = bytearray(trace.read_bytes())
    payload[-1] ^= 1
    trace.write_bytes(payload)
    with pytest.raises(TraceFormatError, match="CRC"):
        replay_trace(trace, capacity=2)
