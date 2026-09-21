#!/usr/bin/env python3
"""Watch one EngineCore process and capture kernel I/O/fault counters.

Read-only diagnostic. It discovers the process by cmdline substring, samples
/proc, and records samples at model-load start, first "Loading weights took"
(main-model completion), and runner stop.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time


def _find_pid(match: str) -> int | None:
    me = os.getpid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == me:
            continue
        try:
            cmd = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                errors="replace"
            )
            comm = (entry / "comm").read_text(errors="replace").strip()
        except (OSError, ProcessLookupError):
            continue
        if match in cmd or match in comm:
            return pid
    return None


def _proc_io(pid: int) -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        text = Path(f"/proc/{pid}/io").read_text()
    except OSError:
        return out
    for line in text.splitlines():
        key, sep, raw = line.partition(":")
        if not sep:
            continue
        try:
            out[key.strip()] = int(raw.strip())
        except ValueError:
            pass
    return out


def _proc_stat(pid: int) -> dict[str, int | float]:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return {}
    end = raw.rfind(")")
    if end < 0:
        return {}
    fields = raw[end + 2 :].split()
    if len(fields) < 22:
        return {}
    hz = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    page = os.sysconf("SC_PAGE_SIZE")
    return {
        "minflt": int(fields[7]),
        "majflt": int(fields[9]),
        "utime_s": int(fields[11]) / hz,
        "stime_s": int(fields[12]) / hz,
        "rss_bytes": int(fields[21]) * page,
    }


def _sample(pid: int, tag: str) -> dict:
    return {
        "tag": tag,
        "pid": pid,
        "time_s": time.time(),
        "monotonic_s": time.monotonic(),
        "io": _proc_io(pid),
        "stat": _proc_stat(pid),
    }


def _delta(a: dict | None, b: dict | None) -> dict | None:
    if not a or not b:
        return None
    out: dict[str, dict[str, float | int]] = {"io": {}, "stat": {}}
    for section in ("io", "stat"):
        av = a.get(section) or {}
        bv = b.get(section) or {}
        for key in sorted(set(av) & set(bv)):
            if isinstance(av[key], (int, float)) and isinstance(
                bv[key], (int, float)
            ):
                out[section][key] = bv[key] - av[key]
    out["elapsed_s"] = b["monotonic_s"] - a["monotonic_s"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--match", default="VLLM::EngineCore")
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--stop-file", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--poll-ms", type=int, default=100)
    ap.add_argument("--timeout-s", type=int, default=1800)
    args = ap.parse_args()

    deadline = time.monotonic() + args.timeout_s
    pid = None
    while time.monotonic() < deadline:
        pid = _find_pid(args.match)
        if pid is not None:
            break
        time.sleep(args.poll_ms / 1000)
    if pid is None:
        raise SystemExit("EngineCore process not found before timeout")

    process_start = _sample(pid, "ENGINECORE_DISCOVERED")
    model_start = None
    main_weights_done = None
    last = process_start

    while time.monotonic() < deadline:
        if not Path(f"/proc/{pid}").exists():
            break
        current = _sample(pid, "PERIODIC")
        last = current
        try:
            log_text = args.log.read_text(errors="replace")
        except OSError:
            log_text = ""
        if model_start is None and "Starting to load model" in log_text:
            model_start = dict(current, tag="MODEL_LOAD_START")
        if main_weights_done is None and "Loading weights took" in log_text:
            main_weights_done = dict(current, tag="MAIN_WEIGHTS_DONE")
        if args.stop_file.exists():
            last = dict(current, tag="RUNNER_STOP")
            break
        time.sleep(args.poll_ms / 1000)

    result = {
        "schema": 1,
        "match": args.match,
        "pid": pid,
        "process_start": process_start,
        "model_start": model_start,
        "main_weights_done": main_weights_done,
        "last": last,
        "deltas": {
            "enginecore_to_main_weights_done": _delta(
                process_start, main_weights_done
            ),
            "model_start_to_main_weights_done": _delta(
                model_start, main_weights_done
            ),
            "enginecore_to_stop": _delta(process_start, last),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
