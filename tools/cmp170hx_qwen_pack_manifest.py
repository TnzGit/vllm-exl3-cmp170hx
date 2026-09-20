#!/usr/bin/env python3
"""Produce a read-only CMP170HX/Qwen EXL3 pack manifest.

The tool reads safetensors headers through the existing qwen_pack_scan.py and
runs qwen_pack_config.py in --dry-run mode. It never rewrites config.json,
weight files, or the safetensors index.

usage:
  python3 tools/cmp170hx_qwen_pack_manifest.py PACK [--out manifest.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCAN = ROOT / "tools" / "exl3_pack_tools" / "qwen_pack_scan.py"
CONFIG = ROOT / "tools" / "exl3_pack_tools" / "qwen_pack_config.py"


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, check=False)


def _qconfig(cfg: dict[str, Any]) -> dict[str, Any]:
    holder = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    q = holder.get("quantization_config") or cfg.get("quantization_config") or {}
    return q if isinstance(q, dict) else {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pack", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    pack = args.pack.resolve()
    config_path = pack / "config.json"
    index_path = pack / "model.safetensors.index.json"
    if not config_path.is_file():
        print(f"ERROR: missing {config_path}", file=sys.stderr)
        return 2

    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"ERROR: cannot parse {config_path}: {exc}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="cmp170hx-qwen-pack-") as td:
        scan_path = Path(td) / "pack_scan.json"
        scan_run = _run(
            [sys.executable, str(SCAN), str(pack), "--out", str(scan_path)]
        )
        if scan_run.returncode != 0:
            print(scan_run.stdout, end="")
            print(scan_run.stderr, end="", file=sys.stderr)
            return scan_run.returncode or 1
        scan = json.loads(scan_path.read_text(encoding="utf-8"))

        dry = _run(
            [
                sys.executable,
                str(CONFIG),
                str(pack),
                "--scan",
                str(scan_path),
                "--dry-run",
            ]
        )
        if dry.returncode != 0:
            print(scan_run.stdout, end="")
            print(dry.stdout, end="")
            print(dry.stderr, end="", file=sys.stderr)
            return dry.returncode or 1

    q = _qconfig(cfg)
    bits = q.get("bits")
    weight_files = sorted(pack.glob("*.safetensors"))
    weight_bytes = sum(p.stat().st_size for p in weight_files)

    index: dict[str, Any] | None = None
    index_error: str | None = None
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception as exc:
            index_error = f"{type(exc).__name__}: {exc}"

    ngram = scan.get("ngram_tables") or {}
    expert_values: set[int] = set()
    for row in (scan.get("expert_k_per_layer") or {}).values():
        for values in row.values():
            expert_values.update(int(v) for v in values)

    manifest = {
        "pack": str(pack),
        "config": {
            "sha256": _sha256(config_path),
            "architecture": cfg.get("architectures"),
            "model_type": cfg.get("model_type"),
            "source_quant_method": q.get("quant_method"),
            "source_bits": bits,
            "source_bits_python_type": type(bits).__name__,
            "source_codebook": q.get("codebook"),
            "source_head_bits": q.get("head_bits"),
        },
        "index": {
            "present": index_path.is_file(),
            "sha256": _sha256(index_path),
            "parse_error": index_error,
            "weight_map_entries": (
                len(index.get("weight_map") or {}) if isinstance(index, dict) else None
            ),
        },
        "files": {
            "safetensors_count": len(weight_files),
            "safetensors_bytes": weight_bytes,
            "safetensors_gib": weight_bytes / 2**30,
        },
        "scan": {
            "tensor_count": scan.get("tensors"),
            "moe_layers": scan.get("moe_layers"),
            "expert_k_values": sorted(expert_values),
            "expert_k_nonuniform": scan.get("expert_k_nonuniform"),
            "dense_k_by_family": scan.get("dense_k_by_family"),
            "ngram_tables": ngram,
            "ngram_problems": scan.get("ngram_problems"),
        },
        "config_dry_run_stdout": dry.stdout,
        "warnings": [],
    }

    if not isinstance(bits, int):
        manifest["warnings"].append(
            "source quantization_config.bits is not an integer; trust tensor-header "
            "geometry and the qwen_pack_config dry-run rather than schema parsing"
        )
    if not index_path.is_file():
        manifest["warnings"].append("model.safetensors.index.json is absent")
    if index_error:
        manifest["warnings"].append(
            f"model.safetensors.index.json could not be parsed: {index_error}"
        )
    if scan.get("ngram_problems"):
        manifest["warnings"].append("pack scan reported n-gram problems")

    rendered = json.dumps(manifest, indent=2, sort_keys=True)
    print(rendered)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
        print(f"written: {args.out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
