#!/usr/bin/env python3
"""Patch vLLM 0.29 safetensors iterator for bounded per-shard prefetch A/B.

Opt-in only with VLLM_EXL3_SHARD_PREFETCH_AB=1. Odd natural-order checkpoint
shards launch vLLM's existing _prefetch_checkpoint() in one background thread
immediately before safe_open/consumption. Even shards remain unchanged.

The thread is joined before leaving that same shard, so prefetch I/O cannot
leak into the next shard's measurement.
"""

from __future__ import annotations

import argparse
import py_compile
import shutil
import sys
from pathlib import Path


MARKER = "# EXL3_SHARD_PREFETCH_AB_V1"
TARGET = Path("model_executor/model_loader/weight_utils.py")

HELPER_ANCHOR = "\ndef safetensors_weights_iterator(\n"
HELPER = r'''
# EXL3_SHARD_PREFETCH_AB_V1
def _exl3_shard_ab_snapshot() -> dict[str, object]:
    import resource

    io: dict[str, int] = {}
    try:
        with open("/proc/self/io", encoding="utf-8") as fh:
            for line in fh:
                key, sep, raw = line.partition(":")
                if sep:
                    try:
                        io[key.strip()] = int(raw.strip())
                    except ValueError:
                        pass
    except OSError:
        pass
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "monotonic_s": time.monotonic(),
        "io": io,
        "minflt": int(usage.ru_minflt),
        "majflt": int(usage.ru_majflt),
        "inblock": int(usage.ru_inblock),
        "utime_s": float(usage.ru_utime),
        "stime_s": float(usage.ru_stime),
    }


def _exl3_shard_ab_delta(before: dict, after: dict) -> dict[str, object]:
    out: dict[str, object] = {
        "elapsed_s": float(after["monotonic_s"]) - float(before["monotonic_s"]),
        "minflt": int(after["minflt"]) - int(before["minflt"]),
        "majflt": int(after["majflt"]) - int(before["majflt"]),
        "inblock": int(after["inblock"]) - int(before["inblock"]),
        "utime_s": float(after["utime_s"]) - float(before["utime_s"]),
        "stime_s": float(after["stime_s"]) - float(before["stime_s"]),
        "io": {},
    }
    b_io = before.get("io") or {}
    a_io = after.get("io") or {}
    io_out = out["io"]
    assert isinstance(io_out, dict)
    for key in sorted(set(b_io) & set(a_io)):
        io_out[key] = int(a_io[key]) - int(b_io[key])
    return out


def _exl3_shard_ab_write(payload: dict[str, object]) -> None:
    path = os.environ.get("VLLM_EXL3_SHARD_PREFETCH_AB_STATS_PATH", "").strip()
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, separators=(",", ":")) + "\n")


def _exl3_start_shard_prefetch(
    path: str,
    block_size: int,
) -> tuple[threading.Thread, dict[str, object]]:
    state: dict[str, object] = {
        "path": path,
        "started_s": time.monotonic(),
        "finished_s": None,
        "error": None,
    }

    def _run() -> None:
        t0 = time.perf_counter()
        try:
            _prefetch_checkpoint(path, block_size)
        except Exception as exc:
            state["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            state["wall_s"] = time.perf_counter() - t0
            state["finished_s"] = time.monotonic()

    thread = threading.Thread(
        target=_run,
        name=f"exl3-shard-prefetch:{os.path.basename(path)}",
        daemon=True,
    )
    thread.start()
    return thread, state


'''

INIT_ANCHOR = """    leftover_state_dict: dict[str, torch.Tensor] = {}
    for st_file in tqdm(
"""
INIT_BLOCK = """    leftover_state_dict: dict[str, torch.Tensor] = {}
    _exl3_shard_ab_enabled = (
        os.environ.get("VLLM_EXL3_SHARD_PREFETCH_AB", "0") == "1"
    )
    if _exl3_shard_ab_enabled and should_prefetch:
        raise RuntimeError(
            "EXL3 shard-prefetch A/B cannot run with full-checkpoint prefetch"
        )
    _exl3_model_shards = [
        path
        for path in sorted_files
        if os.path.basename(path).startswith("model-")
        and os.path.basename(path).endswith(".safetensors")
    ]
    _exl3_shard_ab_index = {
        path: idx for idx, path in enumerate(_exl3_model_shards)
    }
    for st_file in tqdm(
"""

LAZY_ANCHOR = """        else:
            with safe_open(st_file, framework="pt") as f:
                for name in f.keys():  # noqa: SIM118
                    if should_skip_weight(name, local_expert_ids):
                        continue
                    param = f.get_tensor(name)
                    yield name, param
"""

LAZY_BLOCK = """        else:
            _exl3_ab_eligible = st_file in _exl3_shard_ab_index
            _exl3_ab_idx = (
                _exl3_shard_ab_index[st_file]
                if _exl3_ab_eligible
                else -1
            )
            _exl3_ab_arm = (
                "excluded"
                if _exl3_shard_ab_enabled and not _exl3_ab_eligible
                else (
                    "prefetch"
                    if _exl3_shard_ab_enabled and (_exl3_ab_idx & 1)
                    else "control"
                )
            )
            _exl3_ab_before = (
                _exl3_shard_ab_snapshot()
                if _exl3_shard_ab_enabled
                else None
            )
            _exl3_prefetch_thread = None
            _exl3_prefetch_state = None
            if _exl3_ab_arm == "prefetch":
                _exl3_prefetch_thread, _exl3_prefetch_state = (
                    _exl3_start_shard_prefetch(
                        st_file,
                        safetensors_prefetch_block_size,
                    )
                )

            _exl3_consume_t0 = time.perf_counter()
            with safe_open(st_file, framework="pt") as f:
                for name in f.keys():  # noqa: SIM118
                    if should_skip_weight(name, local_expert_ids):
                        continue
                    param = f.get_tensor(name)
                    yield name, param
            _exl3_consume_wall = time.perf_counter() - _exl3_consume_t0

            _exl3_prefetch_join_wall = 0.0
            if _exl3_prefetch_thread is not None:
                _exl3_join_t0 = time.perf_counter()
                _exl3_prefetch_thread.join()
                _exl3_prefetch_join_wall = (
                    time.perf_counter() - _exl3_join_t0
                )
                if _exl3_prefetch_state["error"] is not None:
                    raise RuntimeError(
                        "EXL3 shard prefetch failed: "
                        + str(_exl3_prefetch_state["error"])
                    )

            if _exl3_shard_ab_enabled:
                assert _exl3_ab_before is not None
                _exl3_ab_after = _exl3_shard_ab_snapshot()
                _exl3_delta = _exl3_shard_ab_delta(
                    _exl3_ab_before,
                    _exl3_ab_after,
                )
                _exl3_prefetch_wall = (
                    float(_exl3_prefetch_state.get("wall_s", 0.0))
                    if _exl3_prefetch_state is not None
                    else 0.0
                )
                _exl3_shard_ab_write(
                    {
                        "schema": 1,
                        "index": int(_exl3_ab_idx),
                        "arm": _exl3_ab_arm,
                        "eligible": bool(_exl3_ab_eligible),
                        "file": os.path.basename(st_file),
                        "file_bytes": int(os.path.getsize(st_file)),
                        "consume_wall_s": float(_exl3_consume_wall),
                        "prefetch_wall_s": _exl3_prefetch_wall,
                        "prefetch_join_wall_s": float(
                            _exl3_prefetch_join_wall
                        ),
                        "total_shard_wall_s": float(
                            _exl3_delta["elapsed_s"]
                        ),
                        "delta": _exl3_delta,
                    }
                )
"""


def patch(path: Path, *, check_only: bool = False) -> str:
    src = path.read_text(encoding="utf-8")
    if MARKER in src:
        return "already patched"

    for name, anchor in (
        ("helper", HELPER_ANCHOR),
        ("init", INIT_ANCHOR),
        ("lazy", LAZY_ANCHOR),
    ):
        count = src.count(anchor)
        if count != 1:
            raise RuntimeError(f"{name} anchor count != 1: {count}")

    out = src.replace(HELPER_ANCHOR, "\n" + HELPER + HELPER_ANCHOR, 1)
    out = out.replace(INIT_ANCHOR, INIT_BLOCK, 1)
    out = out.replace(LAZY_ANCHOR, LAZY_BLOCK, 1)

    required = (
        MARKER,
        "VLLM_EXL3_SHARD_PREFETCH_AB",
        "VLLM_EXL3_SHARD_PREFETCH_AB_STATS_PATH",
        "_prefetch_checkpoint(",
        "_exl3_start_shard_prefetch",
        '"prefetch_join_wall_s"',
        '"total_shard_wall_s"',
        '"eligible"',
        '"excluded"',
        "_exl3_model_shards",
        "full-checkpoint prefetch",
    )
    missing = [item for item in required if item not in out]
    if missing:
        raise RuntimeError(f"postcondition missing: {missing}")

    compile(out, str(path), "exec")
    if check_only:
        return "anchors/postconditions validated; no files changed"

    backup = path.with_suffix(path.suffix + ".exl3_shard_prefetch_ab.orig")
    if backup.exists():
        raise RuntimeError(f"stale backup exists: {backup}")
    shutil.copyfile(path, backup)
    path.write_text(out, encoding="utf-8")
    py_compile.compile(str(path), doraise=True)
    return f"patched (backup {backup})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("vllm_root", type=Path)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    target = args.vllm_root.resolve() / TARGET
    if not target.is_file():
        print(f"ERROR: missing weight_utils source: {target}", file=sys.stderr)
        return 2
    try:
        status = patch(target, check_only=args.check_only)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"PASS: EXL3 shard-prefetch A/B {status}: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
