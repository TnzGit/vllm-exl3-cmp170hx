#!/usr/bin/env python3
"""Exact-pack SM80 microbench for dense EXL3 C1 GEMV dispatch.

The parent process only reads safetensors headers/index metadata and launches one
fresh child process per dispatch variant. Fresh processes are required because
ExLlamaV3 caches several GEMV environment variables on first C++ use.

The child loads one real dense EXL3 tensor family from the prepared pack and
calls exllamav3_ext.exl3_gemm directly with preallocated input/output/scratch.
This isolates the same C1 dispatch used by LinearEXL3 without model startup,
HTTP, vLLM scheduling, output allocation, or CUDA-graph replay.

No production kernel is modified by this tool.

Examples:

  # Inspect exact dense tensor families/K values without touching CUDA
  python tools/r0_bench_dense_exl3.py --model-dir /path/to/pack --list

  # Default representative K=4/K=5 families and dispatch matrix
  python tools/r0_bench_dense_exl3.py --model-dir /path/to/pack

  # Restrict to a known family
  python tools/r0_bench_dense_exl3.py --model-dir /path/to/pack \
      --selector linear_attn.in_proj_qkv
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import statistics
import struct
import subprocess
import sys
from dataclasses import dataclass
from typing import Any


DEFAULT_SELECTORS = (
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.out_proj",
    "lm_head",
    "self_attn.q_proj",
)

# Keep this matrix intentionally small. "current" is the production ExLlamaV3
# Ampere dispatch. The fp16 variants test the existing QTIP GEMV alternative;
# k4_cap tests whether K=5 should cross over earlier on this exact CMP170HX.
VARIANTS: dict[str, dict[str, str]] = {
    "current": {
        "EXL3_INT8_GEMV": "2",
        "EXL3_INT8_GEMV_MAX_K": "5",
        "EXL3_GEMV": "1",
        "EXL3_GEMV_SMEM": "-1",
    },
    "int8_k4_cap": {
        "EXL3_INT8_GEMV": "2",
        "EXL3_INT8_GEMV_MAX_K": "4",
        "EXL3_GEMV": "1",
        "EXL3_GEMV_SMEM": "-1",
    },
    "fp16_default": {
        "EXL3_INT8_GEMV": "0",
        "EXL3_GEMV": "1",
        "EXL3_GEMV_SMEM": "-1",
    },
    "fp16_force_auto": {
        "EXL3_INT8_GEMV": "0",
        "EXL3_GEMV": "2",
        "EXL3_GEMV_SMEM": "-1",
    },
    "fp16_force_shuffle": {
        "EXL3_INT8_GEMV": "0",
        "EXL3_GEMV": "2",
        "EXL3_GEMV_SMEM": "0",
    },
    "fp16_force_smem": {
        "EXL3_INT8_GEMV": "0",
        "EXL3_GEMV": "2",
        "EXL3_GEMV_SMEM": "1",
    },
}


@dataclass(frozen=True)
class TensorFamily:
    base: str
    shard: str
    k: int
    in_features: int | None
    out_features: int | None
    has_mcg: bool
    has_mul1: bool


def _header(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise RuntimeError(f"short safetensors header prefix: {path}")
        n = struct.unpack("<Q", raw)[0]
        return json.loads(fh.read(n))


def _numel(shape: Any) -> int | None:
    if not isinstance(shape, list):
        return None
    out = 1
    for dim in shape:
        if not isinstance(dim, int):
            return None
        out *= dim
    return out


def _weight_map(model_dir: Path) -> dict[str, str]:
    candidates = [
        model_dir / "model.safetensors.index.json",
        *sorted(model_dir.glob("*.safetensors.index.json")),
    ]
    for path in candidates:
        if not path.is_file():
            continue
        obj = json.loads(path.read_text(encoding="utf-8"))
        wm = obj.get("weight_map")
        if isinstance(wm, dict):
            return {str(k): str(v) for k, v in wm.items()}

    # Small/single-shard fallback.
    wm: dict[str, str] = {}
    for shard in sorted(model_dir.glob("*.safetensors")):
        for key in _header(shard):
            if key != "__metadata__":
                wm[key] = shard.name
    if not wm:
        raise RuntimeError(f"no safetensors weights found in {model_dir}")
    return wm


def _metadata_cache(
    model_dir: Path, weight_map: dict[str, str]
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for shard in sorted(set(weight_map.values())):
        out[shard] = _header(model_dir / shard)
    return out


def catalog(model_dir: Path) -> list[TensorFamily]:
    wm = _weight_map(model_dir)
    headers = _metadata_cache(model_dir, wm)
    families: list[TensorFamily] = []

    for key, shard in sorted(wm.items()):
        if not key.endswith(".trellis"):
            continue
        base = key[: -len(".trellis")]
        if ".experts." in base or "ngram" in base.lower():
            continue
        meta = headers[shard].get(key) or {}
        shape = meta.get("shape")
        if not isinstance(shape, list) or len(shape) != 3:
            continue
        k_words = shape[2]
        if not isinstance(k_words, int) or k_words % 16:
            continue

        def tensor_numel(suffix: str) -> int | None:
            name = base + suffix
            owner = wm.get(name)
            if owner is None:
                return None
            m = headers[owner].get(name) or {}
            return _numel(m.get("shape"))

        families.append(
            TensorFamily(
                base=base,
                shard=shard,
                k=k_words // 16,
                in_features=tensor_numel(".suh"),
                out_features=tensor_numel(".svh"),
                has_mcg=(base + ".mcg") in wm,
                has_mul1=(base + ".mul1") in wm,
            )
        )
    return families


def choose_families(
    families: list[TensorFamily], selectors: list[str]
) -> list[TensorFamily]:
    chosen: list[TensorFamily] = []
    seen: set[str] = set()
    for selector in selectors:
        matches = [f for f in families if selector in f.base]
        if not matches:
            continue
        # Prefer the first main-model layer; exact shape/bitrate is reported.
        fam = matches[0]
        if fam.base not in seen:
            seen.add(fam.base)
            chosen.append(fam)
    return chosen


def _load_one_tensor(
    model_dir: Path,
    weight_map: dict[str, str],
    name: str,
    *,
    optional: bool = False,
):
    owner = weight_map.get(name)
    if owner is None:
        if optional:
            return None
        raise KeyError(name)
    from safetensors import safe_open

    with safe_open(str(model_dir / owner), framework="pt", device="cpu") as fh:
        return fh.get_tensor(name)


def _kernel_names_from_one_call(callable_obj) -> list[str]:
    import torch

    names: list[str] = []
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            callable_obj()
            torch.cuda.synchronize()
        for ev in prof.events():
            dev = str(getattr(ev, "device_type", ""))
            if "CUDA" in dev.upper():
                name = str(getattr(ev, "name", ""))
                if name and name not in names:
                    names.append(name)
    except Exception as exc:  # diagnostic only; timing still remains valid
        names.append(f"<profiler-unavailable:{type(exc).__name__}>")
    return names[:12]


def child_main(args: argparse.Namespace) -> int:
    # Imports below happen only after the parent has set the variant env and
    # spawned a fresh interpreter.
    import torch

    model_dir = args.model_dir.resolve()
    wm = _weight_map(model_dir)
    base = args.child_base

    trellis = _load_one_tensor(model_dir, wm, base + ".trellis").contiguous()
    suh = _load_one_tensor(model_dir, wm, base + ".suh").contiguous()
    svh = _load_one_tensor(model_dir, wm, base + ".svh").contiguous()
    mcg = _load_one_tensor(model_dir, wm, base + ".mcg", optional=True)
    mul1 = _load_one_tensor(model_dir, wm, base + ".mul1", optional=True)

    if trellis.dtype != torch.int16:
        raise RuntimeError(f"{base}: trellis dtype {trellis.dtype}, expected int16")
    suh = suh.to(torch.float16)
    svh = svh.to(torch.float16)

    in_features = int(suh.numel())
    out_features = int(svh.numel())
    k = int(trellis.shape[2]) // 16

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    trellis = trellis.to(device, non_blocking=False)
    suh = suh.to(device, non_blocking=False)
    svh = svh.to(device, non_blocking=False)

    # Import extension only after env has been fixed for this child process.
    from exllamav3.ext import exllamav3_ext as ext

    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)
    a = torch.randn(
        (1, in_features),
        dtype=torch.float16,
        device=device,
        generator=gen,
    )
    c = torch.empty((1, out_features), dtype=torch.float32, device=device)
    a_had = torch.empty_like(a)

    mcg_flag = mcg is not None
    mul1_flag = mul1 is not None

    def op() -> None:
        ext.exl3_gemm(
            a,
            trellis,
            c,
            suh,
            a_had,
            svh,
            -1,
            mcg_flag,
            mul1_flag,
            0,
        )

    for _ in range(args.warmup):
        op()
    torch.cuda.synchronize()

    kernel_names = _kernel_names_from_one_call(op)
    torch.cuda.synchronize()

    samples_ms: list[float] = []
    for _ in range(args.repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            op()
        end.record()
        end.synchronize()
        samples_ms.append(float(start.elapsed_time(end)))

    us_per_call = [
        ms * 1000.0 / args.iters for ms in samples_ms
    ]
    median_us = statistics.median(us_per_call)

    # A stable output fingerprint catches catastrophic dispatch mistakes without
    # pretending fp16/int8 variants must be token-exact.
    digest = hashlib.sha256(
        c.detach().float().cpu().numpy().tobytes()
    ).hexdigest()

    result = {
        "kind": "dense_exl3_microbench",
        "base": base,
        "k": k,
        "in_features": in_features,
        "out_features": out_features,
        "mcg": mcg_flag,
        "mul1": mul1_flag,
        "variant": args.child_variant,
        "env": {
            key: os.environ.get(key)
            for key in (
                "EXL3_INT8_GEMV",
                "EXL3_INT8_GEMV_MAX_K",
                "EXL3_GEMV",
                "EXL3_GEMV_SMEM",
            )
        },
        "warmup": args.warmup,
        "iters": args.iters,
        "repeat": args.repeat,
        "samples_us": us_per_call,
        "median_us": median_us,
        "spread_pct": (
            0.0
            if median_us == 0
            else (max(us_per_call) - min(us_per_call)) / median_us * 100.0
        ),
        "kernel_names": kernel_names,
        "output_sha256": digest,
        "device_name": torch.cuda.get_device_name(device),
        "cc": list(torch.cuda.get_device_capability(device)),
    }
    print(json.dumps(result, sort_keys=True))
    return 0


def _run_child(
    args: argparse.Namespace,
    family: TensorFamily,
    variant: str,
) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(VARIANTS[variant])
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model-dir",
        str(args.model_dir.resolve()),
        "--child-base",
        family.base,
        "--child-variant",
        variant,
        "--device",
        args.device,
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--repeat",
        str(args.repeat),
        "--seed",
        str(args.seed),
    ]
    proc = subprocess.run(
        cmd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{variant} / {family.base} failed rc={proc.returncode}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("kind") == "dense_exl3_microbench":
            return obj
    raise RuntimeError(
        f"child produced no result JSON for {variant} / {family.base}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


def _print_catalog(families: list[TensorFamily]) -> None:
    print(
        f"{'K':>2} {'in':>7} {'out':>7} {'mul1':>5} {'mcg':>4}  base"
    )
    for f in families:
        print(
            f"{f.k:2d} {str(f.in_features):>7} {str(f.out_features):>7} "
            f"{str(f.has_mul1):>5} {str(f.has_mcg):>4}  {f.base}"
        )


def _print_summary(results: list[dict[str, Any]]) -> None:
    by_base: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        by_base.setdefault(str(row["base"]), []).append(row)

    for base, rows in by_base.items():
        baseline = next((r for r in rows if r["variant"] == "current"), None)
        print(f"\n## {base}")
        print(
            f"K={rows[0]['k']} in={rows[0]['in_features']} "
            f"out={rows[0]['out_features']} mul1={rows[0]['mul1']}"
        )
        print(f"{'variant':22s} {'median us':>10s} {'vs current':>11s}  kernels")
        for row in rows:
            rel = ""
            if baseline is not None:
                rel = f"{row['median_us'] / baseline['median_us']:.3f}x"
            kernels = "; ".join(row.get("kernel_names") or [])
            print(
                f"{row['variant']:22s} {row['median_us']:10.3f} "
                f"{rel:>11s}  {kernels}"
            )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--selector", action="append", default=[])
    ap.add_argument(
        "--variants",
        nargs="+",
        default=["current", "int8_k4_cap", "fp16_default",
                 "fp16_force_auto", "fp16_force_shuffle", "fp16_force_smem"],
        choices=sorted(VARIANTS),
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--output-json", type=Path, default=None)

    # Internal child-only args.
    ap.add_argument("--child-base", default=None)
    ap.add_argument("--child-variant", choices=sorted(VARIANTS), default=None)
    args = ap.parse_args()

    if args.child_base is not None:
        if args.child_variant is None:
            raise SystemExit("--child-base requires --child-variant")
        return child_main(args)

    families = catalog(args.model_dir.resolve())
    if args.list:
        _print_catalog(families)
        return 0

    selectors = args.selector or list(DEFAULT_SELECTORS)
    chosen = choose_families(families, selectors)
    if not chosen:
        raise SystemExit(
            "no dense EXL3 families matched selectors; run with --list first"
        )

    print("Selected exact-pack families:")
    _print_catalog(chosen)

    results: list[dict[str, Any]] = []
    for family in chosen:
        for variant in args.variants:
            # K=4-cap is identical to current for K<=4, so skip redundant work.
            if variant == "int8_k4_cap" and family.k <= 4:
                continue
            # QTIP fp16 GEMV is hard-limited to K<=4; forced variants cannot
            # make K=5 eligible and would only duplicate the regular fallback.
            if family.k > 4 and variant in {
                "fp16_force_auto",
                "fp16_force_shuffle",
                "fp16_force_smem",
            }:
                continue
            row = _run_child(args, family, variant)
            results.append(row)
            print(
                f"{variant:22s} K={row['k']} "
                f"{row['median_us']:.3f} us  {family.base}",
                flush=True,
            )

    _print_summary(results)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
