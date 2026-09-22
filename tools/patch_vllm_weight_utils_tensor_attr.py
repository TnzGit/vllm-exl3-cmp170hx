#!/usr/bin/env python3
"""Patch vLLM lazy safetensors iterator for tensor-consumer attribution.

Diagnostic only. When VLLM_EXL3_TENSOR_ATTR_PATH is set, the lazy iterator
measures for every yielded tensor:

- safe_open.get_tensor() wall + process CPU;
- yield->resume wall + process CPU, which covers downstream mapper/dispatch/
  weight-loader work before the iterator asks for the next tensor.

Only compact per-shard/category aggregates and top consumers are written after
the iterator is exhausted. Loader semantics and tensor order are unchanged.
"""

from __future__ import annotations

import argparse
import py_compile
import shutil
import sys
from pathlib import Path


MARKER = "# EXL3_TENSOR_CONSUMER_ATTR_V1"
TARGET = Path("model_executor/model_loader/weight_utils.py")

HELPER_ANCHOR = "\ndef safetensors_weights_iterator(\n"
HELPER = r'''
# EXL3_TENSOR_CONSUMER_ATTR_V1
def _exl3_tensor_attr_scope(name: str) -> str:
    if ".experts." in name:
        return "routed_expert"
    if ".shared_experts." in name or ".shared_expert." in name:
        return "shared_expert"
    if "ngram" in name:
        return "ngram"
    if ".layers." in name:
        return "layer_nonexpert"
    return "other"


def _exl3_tensor_attr_suffix(name: str) -> str:
    suffix = name.rsplit(".", 1)[-1]
    if suffix in {"trellis", "suh", "svh", "mcg", "mul1", "weight", "bias"}:
        return suffix
    return "other"


def _exl3_tensor_attr_size_bin(nbytes: int) -> str:
    if nbytes < 4 * 1024:
        return "lt_4k"
    if nbytes < 64 * 1024:
        return "4k_64k"
    if nbytes < 1024 * 1024:
        return "64k_1m"
    if nbytes < 16 * 1024 * 1024:
        return "1m_16m"
    return "ge_16m"


def _exl3_tensor_attr_add(
    table: dict[str, dict[str, float | int]],
    key: str,
    *,
    nbytes: int,
    get_wall_s: float,
    get_cpu_s: float,
    consumer_wall_s: float,
    consumer_cpu_s: float,
) -> None:
    row = table.setdefault(
        key,
        {
            "count": 0,
            "bytes": 0,
            "get_tensor_wall_s": 0.0,
            "get_tensor_cpu_s": 0.0,
            "consumer_wall_s": 0.0,
            "consumer_cpu_s": 0.0,
        },
    )
    row["count"] = int(row["count"]) + 1
    row["bytes"] = int(row["bytes"]) + nbytes
    row["get_tensor_wall_s"] = float(row["get_tensor_wall_s"]) + get_wall_s
    row["get_tensor_cpu_s"] = float(row["get_tensor_cpu_s"]) + get_cpu_s
    row["consumer_wall_s"] = (
        float(row["consumer_wall_s"]) + consumer_wall_s
    )
    row["consumer_cpu_s"] = (
        float(row["consumer_cpu_s"]) + consumer_cpu_s
    )


def _exl3_tensor_attr_top_push(
    heap: list[tuple[float, str, int, float]],
    *,
    name: str,
    nbytes: int,
    consumer_wall_s: float,
    get_wall_s: float,
    limit: int = 24,
) -> None:
    item = (consumer_wall_s, name, nbytes, get_wall_s)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item[0] > heap[0][0]:
        heapq.heapreplace(heap, item)


def _exl3_tensor_attr_flush(
    path: str,
    rows: list[dict[str, object]],
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": 1,
        "mode": "lazy_safetensors_tensor_consumer",
        "rows": rows,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


'''

INIT_ANCHOR = """    sorted_files = sorted(hf_weights_files, key=_natural_sort_key)

    fs_type = _get_fs_type(sorted_files)
"""
INIT_BLOCK = """    sorted_files = sorted(hf_weights_files, key=_natural_sort_key)

    _exl3_tensor_attr_path = os.environ.get(
        "VLLM_EXL3_TENSOR_ATTR_PATH", ""
    ).strip()
    _exl3_tensor_attr_enabled = bool(_exl3_tensor_attr_path)
    if _exl3_tensor_attr_enabled and safetensors_load_strategy is not None:
        raise RuntimeError(
            "EXL3 tensor attribution requires default lazy safetensors strategy"
        )
    _exl3_tensor_attr_rows: list[dict[str, object]] = []

    fs_type = _get_fs_type(sorted_files)
"""

LOOP_ANCHOR = """    for st_file in tqdm(
        sorted_files,
        desc=loading_desc,
        disable=not enable_tqdm(use_tqdm_on_load),
        bar_format=_BAR_FORMAT,
    ):
"""
LOOP_BLOCK = """    for st_file in tqdm(
        sorted_files,
        desc=loading_desc,
        disable=not enable_tqdm(use_tqdm_on_load),
        bar_format=_BAR_FORMAT,
    ):
        _exl3_shard_attr = None
        _exl3_shard_t0 = 0.0
        if _exl3_tensor_attr_enabled:
            _exl3_shard_t0 = time.perf_counter()
            _exl3_shard_attr = {
                "file": os.path.basename(st_file),
                "file_bytes": int(os.path.getsize(st_file)),
                "tensor_count": 0,
                "tensor_bytes": 0,
                "get_tensor_wall_s": 0.0,
                "get_tensor_cpu_s": 0.0,
                "consumer_wall_s": 0.0,
                "consumer_cpu_s": 0.0,
                "by_suffix": {},
                "by_scope": {},
                "by_size_bin": {},
                "by_scope_suffix": {},
                "_top_heap": [],
            }
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
            with safe_open(st_file, framework="pt") as f:
                for name in f.keys():  # noqa: SIM118
                    if should_skip_weight(name, local_expert_ids):
                        continue

                    if not _exl3_tensor_attr_enabled:
                        param = f.get_tensor(name)
                        yield name, param
                        continue

                    assert _exl3_shard_attr is not None
                    _exl3_get_cpu0 = time.process_time()
                    _exl3_get_t0 = time.perf_counter()
                    param = f.get_tensor(name)
                    _exl3_get_wall = time.perf_counter() - _exl3_get_t0
                    _exl3_get_cpu = time.process_time() - _exl3_get_cpu0

                    _exl3_nbytes = int(param.numel()) * int(
                        param.element_size()
                    )
                    _exl3_cons_cpu0 = time.process_time()
                    _exl3_cons_t0 = time.perf_counter()
                    yield name, param
                    _exl3_cons_wall = (
                        time.perf_counter() - _exl3_cons_t0
                    )
                    _exl3_cons_cpu = (
                        time.process_time() - _exl3_cons_cpu0
                    )

                    _exl3_shard_attr["tensor_count"] = (
                        int(_exl3_shard_attr["tensor_count"]) + 1
                    )
                    _exl3_shard_attr["tensor_bytes"] = (
                        int(_exl3_shard_attr["tensor_bytes"]) + _exl3_nbytes
                    )
                    _exl3_shard_attr["get_tensor_wall_s"] = (
                        float(_exl3_shard_attr["get_tensor_wall_s"])
                        + _exl3_get_wall
                    )
                    _exl3_shard_attr["get_tensor_cpu_s"] = (
                        float(_exl3_shard_attr["get_tensor_cpu_s"])
                        + _exl3_get_cpu
                    )
                    _exl3_shard_attr["consumer_wall_s"] = (
                        float(_exl3_shard_attr["consumer_wall_s"])
                        + _exl3_cons_wall
                    )
                    _exl3_shard_attr["consumer_cpu_s"] = (
                        float(_exl3_shard_attr["consumer_cpu_s"])
                        + _exl3_cons_cpu
                    )

                    _exl3_scope = _exl3_tensor_attr_scope(name)
                    _exl3_suffix = _exl3_tensor_attr_suffix(name)
                    _exl3_size_bin = _exl3_tensor_attr_size_bin(
                        _exl3_nbytes
                    )
                    _exl3_values = {
                        "nbytes": _exl3_nbytes,
                        "get_wall_s": _exl3_get_wall,
                        "get_cpu_s": _exl3_get_cpu,
                        "consumer_wall_s": _exl3_cons_wall,
                        "consumer_cpu_s": _exl3_cons_cpu,
                    }
                    _exl3_tensor_attr_add(
                        _exl3_shard_attr["by_suffix"],
                        _exl3_suffix,
                        **_exl3_values,
                    )
                    _exl3_tensor_attr_add(
                        _exl3_shard_attr["by_scope"],
                        _exl3_scope,
                        **_exl3_values,
                    )
                    _exl3_tensor_attr_add(
                        _exl3_shard_attr["by_size_bin"],
                        _exl3_size_bin,
                        **_exl3_values,
                    )
                    _exl3_tensor_attr_add(
                        _exl3_shard_attr["by_scope_suffix"],
                        f"{_exl3_scope}|{_exl3_suffix}",
                        **_exl3_values,
                    )
                    _exl3_tensor_attr_top_push(
                        _exl3_shard_attr["_top_heap"],
                        name=name,
                        nbytes=_exl3_nbytes,
                        consumer_wall_s=_exl3_cons_wall,
                        get_wall_s=_exl3_get_wall,
                    )

        if _exl3_tensor_attr_enabled:
            assert _exl3_shard_attr is not None
            _exl3_shard_attr["shard_wall_s"] = (
                time.perf_counter() - _exl3_shard_t0
            )
            _exl3_shard_attr["iterator_other_wall_s"] = max(
                0.0,
                float(_exl3_shard_attr["shard_wall_s"])
                - float(_exl3_shard_attr["get_tensor_wall_s"])
                - float(_exl3_shard_attr["consumer_wall_s"]),
            )
            _exl3_heap = _exl3_shard_attr.pop("_top_heap")
            _exl3_shard_attr["top_consumers"] = [
                {
                    "consumer_wall_s": float(item[0]),
                    "name": item[1],
                    "bytes": int(item[2]),
                    "get_tensor_wall_s": float(item[3]),
                }
                for item in sorted(_exl3_heap, reverse=True)
            ]
            _exl3_tensor_attr_rows.append(_exl3_shard_attr)
"""

FLUSH_ANCHOR = """def multi_thread_safetensors_weights_iterator(
"""
FLUSH_PREFIX = """    if _exl3_tensor_attr_enabled:
        _exl3_tensor_attr_flush(
            _exl3_tensor_attr_path,
            _exl3_tensor_attr_rows,
        )


"""


def patch(path: Path, *, check_only: bool = False) -> str:
    src = path.read_text(encoding="utf-8")
    if MARKER in src:
        return "already patched"

    anchors = (
        ("helper", HELPER_ANCHOR),
        ("init", INIT_ANCHOR),
        ("loop", LOOP_ANCHOR),
        ("lazy", LAZY_ANCHOR),
        ("flush", FLUSH_ANCHOR),
    )
    for name, anchor in anchors:
        count = src.count(anchor)
        if count != 1:
            raise RuntimeError(f"{name} anchor count != 1: {count}")

    out = src.replace("import json\n", "import heapq\nimport json\n", 1)
    out = out.replace(HELPER_ANCHOR, "\n" + HELPER + HELPER_ANCHOR, 1)
    out = out.replace(INIT_ANCHOR, INIT_BLOCK, 1)
    out = out.replace(LOOP_ANCHOR, LOOP_BLOCK, 1)
    out = out.replace(LAZY_ANCHOR, LAZY_BLOCK, 1)
    out = out.replace(FLUSH_ANCHOR, FLUSH_PREFIX + FLUSH_ANCHOR, 1)

    required = (
        MARKER,
        "VLLM_EXL3_TENSOR_ATTR_PATH",
        "time.process_time()",
        '"by_suffix"',
        '"by_scope"',
        '"by_size_bin"',
        '"by_scope_suffix"',
        '"top_consumers"',
        '"iterator_other_wall_s"',
        "_exl3_tensor_attr_flush",
        "default lazy safetensors strategy",
    )
    missing = [item for item in required if item not in out]
    if missing:
        raise RuntimeError(f"postcondition missing: {missing}")

    compile(out, str(path), "exec")
    if check_only:
        return "anchors/postconditions validated; no files changed"

    backup = path.with_suffix(path.suffix + ".exl3_tensor_attr.orig")
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
    print(f"PASS: EXL3 tensor-consumer attribution {status}: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
