#!/usr/bin/env python3
"""Replay and validate the bounded Q2E LRU assignment trace.

This is deliberately a standalone validator.  It consumes the public Q2E
trace reader, but does not import the runtime allocator, worker, summarizer,
or trace writer.  The assignment model mirrors the current Q2E semantics:

* free slots are considered in ascending slot order;
* when free slots are insufficient, least-recently-used unprotected pages are
  evicted, with Python's stable insertion order resolving equal timestamps;
* a slot generation is incremented whenever a new logical page takes it;
* publication records make pages CPU-authoritative and query records touch the
  complete selected history after any reload.

The validator is intentionally strict.  A truncated trace, malformed trace,
unreplayable protected set, or any mismatch in missing pages, victims, slots,
or generations is an error rather than a partial success.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
from typing import Iterable, Sequence


_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vllm_exl3.kvmem_q2e_trace import (  # noqa: E402
    Q2ETraceRecord,
    TraceFormatError,
    read_trace,
    read_trace_records,
)


class ReplayError(ValueError):
    """Raised when a complete trace cannot be replayed exactly."""


@dataclass
class ReplayState:
    """Independent model of one layer's bounded READ-page LRU."""

    capacity: int
    logical_to_slot: dict[int, int] = field(default_factory=dict)
    slot_to_logical: list[int | None] = field(default_factory=list)
    slot_generations: list[int] = field(default_factory=list)
    last_use: dict[int, int] = field(default_factory=dict)
    published: set[int] = field(default_factory=set)
    clock: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.capacity, bool) or not isinstance(self.capacity, int):
            raise ValueError("capacity must be an integer")
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if not self.slot_to_logical:
            self.slot_to_logical = [None] * self.capacity
        if not self.slot_generations:
            self.slot_generations = [0] * self.capacity
        if len(self.slot_to_logical) != self.capacity:
            raise ValueError("slot_to_logical length must equal capacity")
        if len(self.slot_generations) != self.capacity:
            raise ValueError("slot_generations length must equal capacity")

    def assign_many(
        self, logical_pages: Sequence[int], protected: set[int]
    ) -> tuple[list[int], list[int], list[int]]:
        """Assign pages using the current bulk LRU semantics.

        Returns ``(slots, victims, generations)`` for the pages that were
        missing on entry.  The caller must pass only the missing pages.  This
        keeps the returned arrays aligned with Q2E's trace fields.
        """

        pages = [int(page) for page in logical_pages]
        if len(set(pages)) != len(pages):
            raise ReplayError("bulk assignment contains duplicate logical pages")
        if any(page in self.logical_to_slot for page in pages):
            raise ReplayError("bulk assignment contains an already resident page")

        before_slots = list(self.slot_to_logical)
        free = [
            slot for slot, logical in enumerate(self.slot_to_logical)
            if logical is None
        ]
        need_victims = max(0, len(pages) - len(free))
        candidates = sorted(
            (
                int(logical)
                for logical in self.logical_to_slot
                if int(logical) not in protected
            ),
            key=lambda logical: self.last_use.get(logical, -1),
        )
        victims = candidates[:need_victims]
        if len(free) + len(victims) < len(pages):
            raise ReplayError(
                "reload pool has insufficient evictable slots for protected pages"
            )

        available = list(free)
        for victim in victims:
            slot = int(self.logical_to_slot.pop(victim))
            self.last_use.pop(victim, None)
            self.slot_to_logical[slot] = None
            available.append(slot)

        assigned: list[int] = []
        generations: list[int] = []
        recorded_victims: list[int] = []
        for page, slot in zip(pages, available[: len(pages)], strict=True):
            previous = before_slots[slot]
            if previous is not None:
                # The selected victim must be the page that was in this slot.
                # Keeping this check here catches an inconsistent model before
                # it can contaminate all later generation comparisons.
                if int(previous) not in victims:
                    raise ReplayError("LRU victim/slot state is inconsistent")
                recorded_victims.append(int(previous))
            self.slot_generations[slot] += 1
            generation = int(self.slot_generations[slot])
            self.logical_to_slot[page] = slot
            self.slot_to_logical[slot] = page
            assigned.append(int(slot))
            generations.append(generation)
        return assigned, recorded_victims, generations

    def touch(self, pages: Sequence[int]) -> None:
        """Apply the runtime's ordered recency updates."""

        for page_value in pages:
            page = int(page_value)
            if page not in self.logical_to_slot:
                raise ReplayError(f"cannot touch non-resident page {page}")
            self.clock += 1
            self.last_use[page] = self.clock


@dataclass(frozen=True)
class ReplayResult:
    records: int
    layers: int
    publication_records: int
    query_records: int
    assignment_records: int
    max_resident_pages: int
    per_layer_records: dict[int, int]
    truncated: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "records": self.records,
            "layers": self.layers,
            "publication_records": self.publication_records,
            "query_records": self.query_records,
            "assignment_records": self.assignment_records,
            "max_resident_pages": self.max_resident_pages,
            "per_layer_records": {
                str(layer): count
                for layer, count in sorted(self.per_layer_records.items())
            },
            "truncated": self.truncated,
            "complete": not self.truncated,
        }


def _mismatch(
    index: int,
    record: Q2ETraceRecord,
    field: str,
    recorded: Sequence[int],
    expected: Sequence[int],
) -> ReplayError:
    return ReplayError(
        f"record {index} layer {record.layer_id} {field} mismatch: "
        f"recorded={tuple(recorded)!r}, expected={tuple(expected)!r}"
    )


def _compare_assignment(
    index: int,
    record: Q2ETraceRecord,
    expected_missing: Sequence[int],
    expected_victims: Sequence[int],
    expected_slots: Sequence[int],
    expected_generations: Sequence[int],
) -> None:
    checks = (
        ("missing_pages", record.missing_pages, expected_missing),
        ("victim_pages", record.victim_pages, expected_victims),
        ("assigned_slots", record.assigned_slots, expected_slots),
        ("assigned_generations", record.assigned_generations, expected_generations),
    )
    for field, recorded, expected in checks:
        if tuple(recorded) != tuple(expected):
            raise _mismatch(index, record, field, recorded, expected)


def replay_records(
    records: Iterable[Q2ETraceRecord],
    *,
    capacity: int = 4032,
    truncated: bool = False,
) -> ReplayResult:
    """Replay records in order and compare every assignment tuple exactly."""

    states: dict[int, ReplayState] = {}
    counts: defaultdict[int, int] = defaultdict(int)
    publication_records = 0
    query_records = 0
    assignment_records = 0
    max_resident = 0
    record_count = 0

    for index, record in enumerate(records):
        if not isinstance(record, Q2ETraceRecord):
            raise ReplayError(f"record {index} is not a Q2ETraceRecord")
        state = states.setdefault(record.layer_id, ReplayState(capacity))
        counts[record.layer_id] += 1
        record_count += 1

        if record.query_rows == 0:
            publication_records += 1
            if record.history_pages:
                raise ReplayError(
                    f"record {index} publication contains history_pages"
                )
            if any(page in state.published for page in record.missing_pages):
                raise ReplayError(
                    f"record {index} republishes a previously published page"
                )
            missing = list(record.missing_pages)
            protected = set(missing)
            assigned, victims, generations = state.assign_many(missing, protected)
            _compare_assignment(
                index, record, missing, victims, assigned, generations
            )
            state.touch(missing)
            state.published.update(missing)
            assignment_records += len(missing)
        else:
            query_records += 1
            history = list(record.history_pages)
            if len(set(history)) != len(history):
                raise ReplayError(
                    f"record {index} query history contains duplicate pages"
                )
            missing = [page for page in history if page not in state.logical_to_slot]
            absent = [page for page in missing if page not in state.published]
            if absent:
                raise ReplayError(
                    f"record {index} selects non-CPU-authoritative pages: {absent[:8]}"
                )
            assigned, victims, generations = state.assign_many(
                missing, set(history)
            )
            _compare_assignment(
                index, record, missing, victims, assigned, generations
            )
            state.touch(history)
            assignment_records += len(missing)

        max_resident = max(max_resident, len(state.logical_to_slot))

    if truncated:
        raise ReplayError("trace is truncated; exact replay requires a complete trace")
    return ReplayResult(
        records=record_count,
        layers=len(states),
        publication_records=publication_records,
        query_records=query_records,
        assignment_records=assignment_records,
        max_resident_pages=max_resident,
        per_layer_records=dict(counts),
    )


def replay_trace(
    source: str | Path,
    *,
    capacity: int = 4032,
) -> ReplayResult:
    """Read, structurally validate, and exactly replay one Q2E trace."""

    summary = read_trace(source)
    records = read_trace_records(source)
    if summary.truncated:
        raise ReplayError("trace is truncated; exact replay requires a complete trace")
    return replay_records(
        records,
        capacity=capacity,
        truncated=False,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--capacity", type=int, default=4032)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        result = replay_trace(
            args.trace,
            capacity=args.capacity,
        )
    except (OSError, ReplayError, TraceFormatError, ValueError) as exc:
        parser.error(str(exc))
    payload = json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
