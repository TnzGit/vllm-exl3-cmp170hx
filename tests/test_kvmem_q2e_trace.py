from __future__ import annotations

import json
from pathlib import Path
import struct
import subprocess
import sys

import pytest

from vllm_exl3.kvmem_q2e_trace import (
    HEADER_STRUCT,
    Q2ETraceReader,
    Q2ETraceRecord,
    Q2ETraceWriter,
    TraceFormatError,
    close_trace_writers,
    read_trace,
    read_trace_records,
    write_trace_record,
)


def _record(layer: int = 3) -> Q2ETraceRecord:
    return Q2ETraceRecord(
        layer_id=layer,
        first_pos=1000,
        subbatch_start=64,
        query_rows=64,
        history_pages=(1, 65535, 9),
        missing_pages=(20, 21),
        victim_pages=(7,),
        assigned_slots=(4, 65535),
        assigned_generations=(12, 13),
    )


def test_roundtrip_and_aggregate(tmp_path: Path) -> None:
    trace = tmp_path / "access.q2e"
    with Q2ETraceWriter(trace, max_bytes=4096) as writer:
        assert writer.append(_record())
        assert writer.append(_record(layer=4))
        assert writer.records_written == 2

    records = read_trace_records(trace)
    assert records == (_record(), _record(layer=4))
    summary = read_trace(trace)
    assert summary.records == 2
    assert summary.complete is True
    assert summary.truncated is False
    assert summary.history_pages == 6
    assert summary.missing_pages == 4
    assert summary.victim_pages == 2
    assert summary.max_assigned_slots == 2
    assert summary.per_layer_records == {3: 1, 4: 1}
    assert len(summary.sha256) == 64
    assert summary.bytes == trace.stat().st_size


def test_worker_api_writes_identical_bytes_and_encodes_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    direct_trace = tmp_path / "direct.q2e"
    worker_trace = tmp_path / "worker.q2e"
    record = Q2ETraceRecord(
        layer_id=3,
        first_pos=1000,
        subbatch_start=64,
        query_rows=64,
        history_pages=(1, 2, 9),
        missing_pages=(20, 21),
        victim_pages=(7,),
        assigned_slots=(4, 5),
        assigned_generations=(12, 13),
    )
    original_to_bytes = Q2ETraceRecord.to_bytes
    expected_record_bytes = len(original_to_bytes(record))
    calls = 0

    def counted_to_bytes(self: Q2ETraceRecord) -> bytes:
        nonlocal calls
        calls += 1
        return original_to_bytes(self)

    monkeypatch.setattr(Q2ETraceRecord, "to_bytes", counted_to_bytes)
    with Q2ETraceWriter(direct_trace, max_bytes=4096) as writer:
        assert writer.append(record)
    direct_bytes = direct_trace.read_bytes()
    assert calls == 1

    calls = 0
    result = write_trace_record(
        worker_trace,
        4096,
        layer_id=record.layer_id,
        first_pos=record.first_pos,
        subbatch_start=record.subbatch_start,
        query_rows=record.query_rows,
        history_pages=record.history_pages,
        missing_pages=record.missing_pages,
        victim_pages=record.victim_pages,
        assigned_slots=record.assigned_slots,
        assigned_generations=record.assigned_generations,
    )
    close_trace_writers()

    assert result["written"] is True
    assert result["record_bytes"] == expected_record_bytes
    assert calls == 1
    assert worker_trace.read_bytes() == direct_bytes
    assert read_trace_records(worker_trace) == (record,)
    assert read_trace(worker_trace) == read_trace(direct_trace)


def test_corruption_and_trailing_junk_are_rejected(tmp_path: Path) -> None:
    trace = tmp_path / "access.q2e"
    with Q2ETraceWriter(trace) as writer:
        assert writer.append(_record())
    original = trace.read_bytes()

    corrupted = bytearray(original)
    corrupted[-1] ^= 0x01
    trace.write_bytes(corrupted)
    with pytest.raises(TraceFormatError, match="CRC"):
        Q2ETraceReader(trace).read()

    trace.write_bytes(original + b"junk")
    with pytest.raises(TraceFormatError):
        read_trace(trace)


def test_cap_stops_before_partial_record_and_marks_truncated(tmp_path: Path) -> None:
    trace = tmp_path / "access.q2e"
    first_size = len(_record().to_bytes())
    with Q2ETraceWriter(trace, max_bytes=HEADER_STRUCT.size + first_size + 1) as writer:
        assert writer.append(_record())
        assert writer.append(_record(layer=5)) is False
        assert writer.truncated is True
        assert writer.records_written == 1

    summary = read_trace(trace)
    assert summary.truncated is True
    assert summary.records == 1
    assert summary.bytes <= HEADER_STRUCT.size + first_size + 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("layer_id", 65536),
        ("history_pages", (65536,)),
        ("assigned_slots", (65536,)),
        ("assigned_generations", (1 << 32,)),
    ],
)
def test_invalid_values_are_rejected(field: str, value: object) -> None:
    kwargs = {
        "layer_id": 0,
        "first_pos": 0,
        "subbatch_start": 0,
        "query_rows": 1,
        "history_pages": (),
        "missing_pages": (),
        "victim_pages": (),
        "assigned_slots": (),
        "assigned_generations": (),
    }
    kwargs[field] = value
    with pytest.raises(ValueError):
        Q2ETraceRecord(**kwargs)


def test_mismatched_assignment_arrays_are_rejected() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        Q2ETraceRecord(
            layer_id=0,
            first_pos=0,
            subbatch_start=0,
            query_rows=1,
            assigned_slots=(1,),
            assigned_generations=(),
        )


def test_cli_writes_json(tmp_path: Path) -> None:
    trace = tmp_path / "access.q2e"
    output = tmp_path / "summary.json"
    with Q2ETraceWriter(trace) as writer:
        assert writer.append(_record())
    script = Path(__file__).parents[1] / "tools" / "kvmem_q2e_trace_summarize.py"
    subprocess.run(
        [sys.executable, str(script), "--trace", str(trace), "--out", str(output)],
        check=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["records"] == 1
    assert result["complete"] is True
    assert result["per_layer_records"] == {"3": 1}


def test_worker_api_caches_writer_flushes_and_reports_truncation(tmp_path: Path) -> None:
    trace = tmp_path / "worker.q2e"
    record_bytes = len(_record().to_bytes())
    cap = HEADER_STRUCT.size + record_bytes + 1
    first = write_trace_record(
        trace,
        cap,
        layer_id=3,
        first_pos=1000,
        subbatch_start=64,
        query_rows=64,
        history_pages=(1, 2),
        missing_pages=(3,),
        victim_pages=(),
        assigned_slots=(4,),
        assigned_generations=(8,),
    )
    assert first["written"] is True
    assert first["truncated"] is False
    assert first["record_bytes"] > 0
    # The first record is visible before close because append flushes it.
    assert read_trace(trace).records == 1

    second = write_trace_record(
        trace,
        cap,
        layer_id=4,
        first_pos=2000,
        subbatch_start=0,
        query_rows=1,
    )
    assert second["written"] is False
    assert second["truncated"] is True
    assert read_trace(trace).truncated is True
    close_trace_writers()


def test_worker_api_truncated_bytes_keep_validation_behavior(tmp_path: Path) -> None:
    trace = tmp_path / "worker-malformed.q2e"
    record_bytes = len(_record().to_bytes())
    cap = HEADER_STRUCT.size + record_bytes + 1
    write_trace_record(
        trace,
        cap,
        layer_id=3,
        first_pos=1000,
        subbatch_start=64,
        query_rows=64,
        history_pages=(1, 2, 9),
        missing_pages=(20, 21),
        victim_pages=(7,),
        assigned_slots=(4, 5),
        assigned_generations=(12, 13),
    )
    write_trace_record(
        trace,
        cap,
        layer_id=4,
        first_pos=2000,
        subbatch_start=0,
        query_rows=1,
    )
    close_trace_writers()

    original = trace.read_bytes()
    assert read_trace(trace).truncated is True
    corrupted = bytearray(original)
    corrupted[-1] ^= 0x01
    trace.write_bytes(corrupted)
    with pytest.raises(TraceFormatError, match="CRC"):
        read_trace(trace)

    trace.write_bytes(original[:-1])
    with pytest.raises(TraceFormatError, match="record length"):
        read_trace(trace)


def test_worker_api_rejects_cap_change_for_open_path(tmp_path: Path) -> None:
    trace = tmp_path / "worker.q2e"
    kwargs = dict(
        layer_id=0,
        first_pos=0,
        subbatch_start=0,
        query_rows=1,
    )
    write_trace_record(trace, 4096, **kwargs)
    with pytest.raises(ValueError, match="max_bytes"):
        write_trace_record(trace, 4097, **kwargs)
    close_trace_writers()
