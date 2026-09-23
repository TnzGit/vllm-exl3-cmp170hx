"""Replay a complete Q2E access trace against full-sort and partial LRU."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vllm_exl3.kvmem_q2d_reload_shadow import _assign_many, _touch  # noqa: E402
from vllm_exl3.kvmem_q2e_trace import read_trace_records  # noqa: E402


def _full_sort_assign(state, logical_pages, protected):
    """Frozen pre-optimization array-LRU branch, for matched trace timing."""
    pages = [int(page) for page in logical_pages]
    missing = [page for page in pages if page not in state["logical_to_slot"]]
    if not missing:
        return [int(state["logical_to_slot"][page]) for page in pages]
    free = []
    for slot, logical in enumerate(state["slot_to_logical"]):
        if logical is None:
            free.append(slot)
            if len(free) == len(missing):
                break
    need_victims = max(0, len(missing) - len(free))
    candidates = [
        int(logical) for logical in state["logical_to_slot"]
        if int(logical) not in protected
    ] if need_victims else []
    candidate_pages = np.asarray(candidates, dtype=np.int64)
    candidate_times = state["last_use_array"][candidate_pages]
    insertion_order = np.arange(len(candidates), dtype=np.int64)
    order = np.lexsort((insertion_order, candidate_times))
    victims = candidate_pages[order[:need_victims]].tolist()
    if len(free) + len(victims) < len(missing):
        raise RuntimeError("insufficient evictable slots in trace replay")
    available = list(free)
    for victim in victims:
        slot = int(state["logical_to_slot"].pop(victim))
        state["last_use"].pop(victim, None)
        state["last_use_array"][victim] = -1
        state["slot_to_logical"][slot] = None
        available.append(slot)
    for page, slot in zip(missing, available[:len(missing)], strict=True):
        state["logical_to_slot"][page] = slot
        state["slot_to_logical"][slot] = page
    state["peak_slots"] = max(
        int(state.get("peak_slots", 0)), len(state["logical_to_slot"])
    )
    return [int(state["logical_to_slot"][page]) for page in pages]


def _new_state(cpu_pages: int, read_pages: int):
    return {
        "logical_to_slot": {},
        "slot_to_logical": [None] * read_pages,
        "slot_generations": [0] * read_pages,
        "last_use": {},
        "last_use_array": np.full(cpu_pages, -1, dtype=np.int64),
        "clock": 0,
        "peak_slots": 0,
    }


def _replay(records, *, cpu_pages: int, read_pages: int, assign):
    states = {}
    assignment_seconds = 0.0
    calls = 0
    total_missing = 0
    for index, record in enumerate(records):
        state = states.setdefault(record.layer_id, _new_state(cpu_pages, read_pages))
        publication = record.query_rows == 0
        history = [int(page) for page in record.history_pages]
        recorded_missing = [int(page) for page in record.missing_pages]
        if publication:
            pages = recorded_missing
            protected = set(pages)
        else:
            pages = [page for page in history if page not in state["logical_to_slot"]]
            protected = set(history)
            if pages != recorded_missing:
                raise RuntimeError(f"record {index}: missing-page sequence diverged")
        before = list(state["slot_to_logical"])
        slots = []
        if pages:
            start = time.perf_counter()
            slots = assign(state, pages, protected)
            assignment_seconds += time.perf_counter() - start
            calls += 1
        victims = []
        generations = []
        for page, slot in zip(pages, slots, strict=True):
            previous = before[slot]
            if previous != page:
                state["slot_generations"][slot] += 1
                if previous is not None:
                    victims.append(int(previous))
            generations.append(int(state["slot_generations"][slot]))
        if (slots != list(record.assigned_slots)
                or victims != list(record.victim_pages)
                or generations != list(record.assigned_generations)):
            raise RuntimeError(f"record {index}: slot/victim/generation diverged")
        _touch(state, pages if publication else history)
        total_missing += len(pages)
    return {
        "records": len(records),
        "layers": sorted(states),
        "assignment_calls": calls,
        "missing_pages": total_missing,
        "assignment_seconds": assignment_seconds,
        "assignment_ms_per_call": assignment_seconds * 1000 / calls if calls else 0.0,
        "all_records_exact": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--cpu-pages", type=int, default=10063)
    parser.add_argument("--read-pages", type=int, default=4032)
    parser.add_argument("--max-records", type=int)
    args = parser.parse_args()
    records = read_trace_records(args.trace)
    if args.max_records is not None:
        records = records[:args.max_records]
    results = {
        "schema": 1,
        "trace": str(args.trace),
        "full_sort": _replay(
            records, cpu_pages=args.cpu_pages, read_pages=args.read_pages,
            assign=_full_sort_assign,
        ),
        "partial": _replay(
            records, cpu_pages=args.cpu_pages, read_pages=args.read_pages,
            assign=_assign_many,
        ),
    }
    print(json.dumps(results))


if __name__ == "__main__":
    main()
