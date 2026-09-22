"""Bounded binary access traces for the Q2E page-access experiments.

The trace is a sequence of self-contained records.  Records are never
partially written: when the configured byte budget cannot fit the next
record, the writer stops accepting records and marks the file truncated.
The record area is append-only; the small header flag is finalized on close.
"""

from __future__ import annotations

from array import array
from collections import Counter
from dataclasses import dataclass
import hashlib
import os
import struct
import sys
from pathlib import Path
from typing import BinaryIO, Sequence
import zlib
import atexit


MAGIC = b"Q2ETRACE"
RECORD_MAGIC = b"Q2ER"
VERSION = 1
DEFAULT_MAX_BYTES = 64 * 1024 * 1024
FLAG_TRUNCATED = 1

# magic, version, header size, flags, hard byte limit
HEADER_STRUCT = struct.Struct("<8sHHIQ")
# magic, record size, layer, reserved, first position, subbatch start,
# query rows, and the five payload-array lengths
RECORD_HEADER_STRUCT = struct.Struct("<4sIHHIIIIIIII")
CRC_STRUCT = struct.Struct("<I")

_UINT16_MAX = (1 << 16) - 1
_UINT32_MAX = (1 << 32) - 1


class TraceFormatError(ValueError):
    """Raised when a trace fails structural, length, or CRC validation."""


def _check_uint(name: str, value: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < 0 or value > maximum:
        raise ValueError(f"{name} must be in [0, {maximum}], got {value}")
    return value


def _checked_values(name: str, values: Sequence[int], maximum: int) -> tuple[int, ...]:
    result = tuple(values)
    for index, value in enumerate(result):
        _check_uint(f"{name}[{index}]", value, maximum)
    if len(result) > _UINT32_MAX:
        raise ValueError(f"{name} contains too many values")
    return result


def _array_bytes(typecode: str, values: Sequence[int]) -> bytes:
    encoded = array(typecode, values)
    expected_itemsize = 2 if typecode == "H" else 4
    if encoded.itemsize != expected_itemsize:
        raise RuntimeError(f"unexpected array item size for {typecode!r}")
    if sys.byteorder != "little":
        encoded.byteswap()
    return encoded.tobytes()


def _array_values(typecode: str, payload: bytes) -> tuple[int, ...]:
    expected_itemsize = 2 if typecode == "H" else 4
    if len(payload) % expected_itemsize:
        raise TraceFormatError("array payload is not aligned to its item size")
    decoded = array(typecode)
    decoded.frombytes(payload)
    if sys.byteorder != "little":
        decoded.byteswap()
    return tuple(decoded)


@dataclass(frozen=True)
class Q2ETraceRecord:
    """One page-access event recorded without per-token expansion."""

    layer_id: int
    first_pos: int
    subbatch_start: int
    query_rows: int
    history_pages: Sequence[int] = ()
    missing_pages: Sequence[int] = ()
    victim_pages: Sequence[int] = ()
    assigned_slots: Sequence[int] = ()
    assigned_generations: Sequence[int] = ()

    def __post_init__(self) -> None:
        _check_uint("layer_id", self.layer_id, _UINT16_MAX)
        _check_uint("first_pos", self.first_pos, _UINT32_MAX)
        _check_uint("subbatch_start", self.subbatch_start, _UINT32_MAX)
        _check_uint("query_rows", self.query_rows, _UINT32_MAX)
        for name in ("history_pages", "missing_pages", "victim_pages", "assigned_slots"):
            values = _checked_values(name, getattr(self, name), _UINT16_MAX)
            object.__setattr__(self, name, values)
        generations = _checked_values(
            "assigned_generations", self.assigned_generations, _UINT32_MAX
        )
        object.__setattr__(self, "assigned_generations", generations)
        if len(self.assigned_slots) != len(self.assigned_generations):
            raise ValueError(
                "assigned_slots and assigned_generations must have equal lengths"
            )

    @property
    def history_count(self) -> int:
        return len(self.history_pages)

    @property
    def missing_count(self) -> int:
        return len(self.missing_pages)

    @property
    def victim_count(self) -> int:
        return len(self.victim_pages)

    @property
    def assigned_count(self) -> int:
        return len(self.assigned_slots)

    def to_bytes(self) -> bytes:
        arrays = (
            _array_bytes("H", self.history_pages),
            _array_bytes("H", self.missing_pages),
            _array_bytes("H", self.victim_pages),
            _array_bytes("H", self.assigned_slots),
            _array_bytes("I", self.assigned_generations),
        )
        payload = b"".join(arrays)
        record_size = RECORD_HEADER_STRUCT.size + len(payload) + CRC_STRUCT.size
        _check_uint("record_size", record_size, _UINT32_MAX)
        header = RECORD_HEADER_STRUCT.pack(
            RECORD_MAGIC,
            record_size,
            self.layer_id,
            0,
            self.first_pos,
            self.subbatch_start,
            self.query_rows,
            self.history_count,
            self.missing_count,
            self.victim_count,
            self.assigned_count,
            self.assigned_count,
        )
        body = header + payload
        return body + CRC_STRUCT.pack(zlib.crc32(body) & _UINT32_MAX)


@dataclass(frozen=True)
class Q2ETraceSummary:
    records: int
    bytes: int
    truncated: bool
    history_pages: int
    missing_pages: int
    victim_pages: int
    assigned_slots: int
    max_history_pages: int
    max_missing_pages: int
    max_victim_pages: int
    max_assigned_slots: int
    max_query_rows: int
    per_layer_records: dict[int, int]
    sha256: str

    @property
    def complete(self) -> bool:
        return not self.truncated

    def to_dict(self) -> dict[str, object]:
        return {
            "records": self.records,
            "bytes": self.bytes,
            "complete": self.complete,
            "truncated": self.truncated,
            "history_pages": self.history_pages,
            "missing_pages": self.missing_pages,
            "victim_pages": self.victim_pages,
            "assigned_slots": self.assigned_slots,
            "max_history_pages": self.max_history_pages,
            "max_missing_pages": self.max_missing_pages,
            "max_victim_pages": self.max_victim_pages,
            "max_assigned_slots": self.max_assigned_slots,
            "max_query_rows": self.max_query_rows,
            "per_layer_records": {
                str(layer): count for layer, count in sorted(self.per_layer_records.items())
            },
            "sha256": self.sha256,
        }


def _pack_header(flags: int, max_bytes: int) -> bytes:
    return HEADER_STRUCT.pack(MAGIC, VERSION, HEADER_STRUCT.size, flags, max_bytes)


class Q2ETraceWriter:
    """Write a bounded Q2E trace to a new path or writable binary stream."""

    def __init__(
        self,
        target: str | os.PathLike[str] | BinaryIO,
        *,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> None:
        _check_uint("max_bytes", max_bytes, (1 << 63) - 1)
        if max_bytes < HEADER_STRUCT.size:
            raise ValueError(f"max_bytes must be at least {HEADER_STRUCT.size}")
        self.max_bytes = max_bytes
        self.truncated = False
        self.records_written = 0
        self.bytes_written = HEADER_STRUCT.size
        self._owns_file = not hasattr(target, "write")
        self._file: BinaryIO = open(target, "wb") if self._owns_file else target  # type: ignore[arg-type]
        self._closed = False
        self._file.write(_pack_header(0, max_bytes))
        self._file.flush()

    def _persist_truncated_flag(self) -> None:
        if not self._file.seekable():
            raise ValueError("truncated traces require a seekable output")
        self._file.seek(0)
        self._file.write(_pack_header(FLAG_TRUNCATED, self.max_bytes))
        self._file.flush()
        self._file.seek(0, os.SEEK_END)

    def append(self, record: Q2ETraceRecord) -> bool:
        """Append a record, returning false when the hard cap rejects it."""
        if self._closed:
            raise ValueError("cannot append to a closed trace")
        if not isinstance(record, Q2ETraceRecord):
            raise TypeError("record must be a Q2ETraceRecord")
        if self.truncated:
            return False
        encoded = record.to_bytes()
        if self.bytes_written + len(encoded) > self.max_bytes:
            self.truncated = True
            self._persist_truncated_flag()
            return False
        self._file.write(encoded)
        self.bytes_written += len(encoded)
        self.records_written += 1
        self._file.flush()
        return True

    write = append

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._file.flush()
            # Finalize only the status flag. The record region remains append-only.
            if self.truncated:
                self._persist_truncated_flag()
        finally:
            if self._owns_file:
                self._file.close()
            self._closed = True

    def __enter__(self) -> "Q2ETraceWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def _decode_record(data: bytes, offset: int) -> tuple[Q2ETraceRecord, int]:
    remaining = len(data) - offset
    if remaining < RECORD_HEADER_STRUCT.size:
        raise TraceFormatError("trailing bytes are shorter than a record header")
    fields = RECORD_HEADER_STRUCT.unpack_from(data, offset)
    (
        magic,
        record_size,
        layer_id,
        reserved,
        first_pos,
        subbatch_start,
        query_rows,
        history_count,
        missing_count,
        victim_count,
        assigned_count,
        generation_count,
    ) = fields
    if magic != RECORD_MAGIC:
        raise TraceFormatError(f"invalid record magic at offset {offset}")
    if reserved != 0:
        raise TraceFormatError("record reserved field is nonzero")
    if record_size < RECORD_HEADER_STRUCT.size + CRC_STRUCT.size:
        raise TraceFormatError("record length is too small")
    if record_size > remaining:
        raise TraceFormatError("record length exceeds remaining trace bytes")
    expected_payload = 2 * (history_count + missing_count + victim_count + assigned_count)
    expected_payload += 4 * generation_count
    expected_size = RECORD_HEADER_STRUCT.size + expected_payload + CRC_STRUCT.size
    if record_size != expected_size:
        raise TraceFormatError("record length does not match its array counts")
    end = offset + record_size
    stored_crc = CRC_STRUCT.unpack_from(data, end - CRC_STRUCT.size)[0]
    actual_crc = zlib.crc32(data[offset : end - CRC_STRUCT.size]) & _UINT32_MAX
    if stored_crc != actual_crc:
        raise TraceFormatError(f"record CRC mismatch at offset {offset}")
    cursor = offset + RECORD_HEADER_STRUCT.size

    def take(count: int, typecode: str) -> tuple[int, ...]:
        nonlocal cursor
        itemsize = 2 if typecode == "H" else 4
        chunk = data[cursor : cursor + count * itemsize]
        cursor += len(chunk)
        return _array_values(typecode, chunk)

    record = Q2ETraceRecord(
        layer_id=layer_id,
        first_pos=first_pos,
        subbatch_start=subbatch_start,
        query_rows=query_rows,
        history_pages=take(history_count, "H"),
        missing_pages=take(missing_count, "H"),
        victim_pages=take(victim_count, "H"),
        assigned_slots=take(assigned_count, "H"),
        assigned_generations=take(generation_count, "I"),
    )
    if cursor != end - CRC_STRUCT.size:
        raise TraceFormatError("record payload cursor does not reach CRC")
    return record, end


class Q2ETraceReader:
    """Validate and summarize a Q2E trace."""

    def __init__(self, source: str | os.PathLike[str] | BinaryIO) -> None:
        self._source = source

    def _read_all(self) -> tuple[tuple[Q2ETraceRecord, ...], Q2ETraceSummary]:
        if hasattr(self._source, "read"):
            file_obj = self._source  # type: ignore[assignment]
            if file_obj.seekable():
                file_obj.seek(0)
            data = file_obj.read()
        else:
            data = Path(self._source).read_bytes()
        if len(data) < HEADER_STRUCT.size:
            raise TraceFormatError("trace is shorter than its header")
        magic, version, header_size, flags, max_bytes = HEADER_STRUCT.unpack_from(data)
        if magic != MAGIC:
            raise TraceFormatError("invalid trace magic")
        if version != VERSION:
            raise TraceFormatError(f"unsupported trace version {version}")
        if header_size != HEADER_STRUCT.size:
            raise TraceFormatError("invalid trace header size")
        if flags & ~FLAG_TRUNCATED:
            raise TraceFormatError("unknown trace header flags")
        if max_bytes and len(data) > max_bytes:
            raise TraceFormatError("trace exceeds the declared hard byte limit")

        records: list[Q2ETraceRecord] = []
        cursor = HEADER_STRUCT.size
        while cursor < len(data):
            record, cursor = _decode_record(data, cursor)
            records.append(record)

        layer_counts = Counter(record.layer_id for record in records)
        summary = Q2ETraceSummary(
            records=len(records),
            bytes=len(data),
            truncated=bool(flags & FLAG_TRUNCATED),
            history_pages=sum(record.history_count for record in records),
            missing_pages=sum(record.missing_count for record in records),
            victim_pages=sum(record.victim_count for record in records),
            assigned_slots=sum(record.assigned_count for record in records),
            max_history_pages=max((record.history_count for record in records), default=0),
            max_missing_pages=max((record.missing_count for record in records), default=0),
            max_victim_pages=max((record.victim_count for record in records), default=0),
            max_assigned_slots=max((record.assigned_count for record in records), default=0),
            max_query_rows=max((record.query_rows for record in records), default=0),
            per_layer_records=dict(layer_counts),
            sha256=hashlib.sha256(data).hexdigest(),
        )
        return tuple(records), summary

    def read_records(self) -> tuple[Q2ETraceRecord, ...]:
        return self._read_all()[0]

    def read(self) -> Q2ETraceSummary:
        return self._read_all()[1]


def read_trace(source: str | os.PathLike[str] | BinaryIO) -> Q2ETraceSummary:
    """Validate and summarize a trace source."""
    return Q2ETraceReader(source).read()


def read_trace_records(
    source: str | os.PathLike[str] | BinaryIO,
) -> tuple[Q2ETraceRecord, ...]:
    """Validate and return all records from a trace source."""
    return Q2ETraceReader(source).read_records()


_TRACE_WRITERS: dict[Path, tuple[int, Q2ETraceWriter]] = {}


def write_trace_record(
    path: str | Path,
    max_bytes: int,
    *,
    layer_id: int,
    first_pos: int,
    subbatch_start: int,
    query_rows: int,
    history_pages: Sequence[int] = (),
    missing_pages: Sequence[int] = (),
    victim_pages: Sequence[int] = (),
    assigned_slots: Sequence[int] = (),
    assigned_generations: Sequence[int] = (),
) -> dict[str, bool | int]:
    """Append one worker event using a cached writer for ``path``.

    The cache keeps the file open across worker callbacks.  Each successful
    record is flushed immediately, and a rejected record persists the
    truncated header flag before returning.
    """
    record = Q2ETraceRecord(
        layer_id=layer_id,
        first_pos=first_pos,
        subbatch_start=subbatch_start,
        query_rows=query_rows,
        history_pages=history_pages,
        missing_pages=missing_pages,
        victim_pages=victim_pages,
        assigned_slots=assigned_slots,
        assigned_generations=assigned_generations,
    )
    record_bytes = len(record.to_bytes())
    key = Path(path).expanduser().resolve(strict=False)
    cached = _TRACE_WRITERS.get(key)
    if cached is None:
        writer = Q2ETraceWriter(key, max_bytes=max_bytes)
        _TRACE_WRITERS[key] = (max_bytes, writer)
    else:
        cached_max_bytes, writer = cached
        if cached_max_bytes != max_bytes:
            raise ValueError(
                f"trace {key} already uses max_bytes={cached_max_bytes}, "
                f"cannot change it to {max_bytes} while open"
            )
    written = writer.append(record)
    return {
        "written": written,
        "truncated": writer.truncated,
        "record_bytes": record_bytes,
    }


def close_trace_writers() -> None:
    """Flush and close all writers held by :func:`write_trace_record`."""
    cached = list(_TRACE_WRITERS.values())
    _TRACE_WRITERS.clear()
    first_error: BaseException | None = None
    for _, writer in cached:
        try:
            writer.close()
        except BaseException as exc:  # pragma: no cover - unusual I/O failure
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error


atexit.register(close_trace_writers)
