#!/usr/bin/env python3
"""Patch vLLM 0.29 Qwen4Exp QSA with K1-Q1 resident visibility replay.

Default-off: when VLLM_QWEN_KVMEM_RESIDENT_PLAN is unset, QSA behavior is
unchanged.

When enabled in eager mode, only rows whose logical position is at/after the
plan's apply_min_pos are filtered. Historical selected tokens remain visible
only if their 256-token planner region is in the precomputed sticky resident
set. Current query/decode tokens at/after active_from_pos always remain visible.

The patch modifies QSA's already-computed selected logical indices in place.
It does not alter QSA scoring/top-k math, KV writes, the physical cache, or the
scheduler block table.
"""

from __future__ import annotations

import argparse
import py_compile
import sys
from pathlib import Path

MARKER = "# KVMEM_QSA_RESIDENT_VISIBILITY_V1"
TARGET = Path("models/qwen4_exp/nvidia/qsa.py")

IMPORT_ANCHOR = "from __future__ import annotations\n\n"
IMPORT_BLOCK = """from __future__ import annotations

import json
import os
from pathlib import Path

"""

HELPER_ANCHOR = "from .indexer_qsa import QSAIndexer\n\n"
HELPER = r'''from .indexer_qsa import QSAIndexer

# KVMEM_QSA_RESIDENT_VISIBILITY_V1
_KVMEM_QSA_RESIDENT_PLAN_CACHE = None
_KVMEM_QSA_RESIDENT_TENSOR_CACHE = {}


def _kvmem_qsa_resident_plan():
    global _KVMEM_QSA_RESIDENT_PLAN_CACHE
    path = os.environ.get("VLLM_QWEN_KVMEM_RESIDENT_PLAN")
    if not path:
        return None
    cached = _KVMEM_QSA_RESIDENT_PLAN_CACHE
    if cached is not None and cached[0] == path:
        return cached[1]

    plan = json.loads(Path(path).read_text())
    if int(plan.get("schema", 0)) != 1:
        raise RuntimeError("KVMem resident plan schema mismatch")
    if plan.get("mode") != "qsa_selected_visibility":
        raise RuntimeError("KVMem resident plan mode mismatch")
    region_tokens = int(plan["region_tokens"])
    page_tokens = int(plan["page_tokens"])
    resident = [int(x) for x in plan["resident_regions"]]
    if region_tokens <= 0 or page_tokens <= 0 or region_tokens % page_tokens:
        raise RuntimeError("KVMem resident plan has invalid region/page geometry")
    if len(resident) != int(plan["resident_region_count"]):
        raise RuntimeError("KVMem resident region count mismatch")
    if len(set(resident)) != len(resident):
        raise RuntimeError("KVMem resident plan contains duplicate regions")
    if any(x < 0 for x in resident):
        raise RuntimeError("KVMem resident plan contains negative region ids")

    normalized = dict(plan)
    normalized["resident_regions"] = resident
    _KVMEM_QSA_RESIDENT_PLAN_CACHE = (path, normalized)
    _KVMEM_QSA_RESIDENT_TENSOR_CACHE.clear()
    return normalized


def _kvmem_qsa_resident_tensor(plan, device):
    key = (str(device), tuple(plan["resident_regions"]))
    cached = _KVMEM_QSA_RESIDENT_TENSOR_CACHE.get(key)
    if cached is None:
        cached = torch.tensor(
            plan["resident_regions"],
            dtype=torch.int64,
            device=device,
        )
        _KVMEM_QSA_RESIDENT_TENSOR_CACHE[key] = cached
    return cached


def _kvmem_qsa_visibility_record(payload):
    path = os.environ.get("VLLM_QWEN_KVMEM_VISIBILITY_STATS_PATH")
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, separators=(",", ":")) + "\n")


def _kvmem_qsa_apply_resident_visibility(
    *,
    layer_name: str,
    selected: torch.Tensor,
    logical_positions: torch.Tensor,
) -> torch.Tensor:
    """Mask nonresident historical selections for replay rows only."""

    plan = _kvmem_qsa_resident_plan()
    if plan is None:
        return selected
    if selected.ndim != 2 or logical_positions.ndim != 1:
        raise RuntimeError("KVMem resident visibility received unexpected ranks")
    if selected.shape[0] != logical_positions.shape[0]:
        raise RuntimeError("KVMem resident visibility row counts disagree")
    if selected.shape[0] == 0:
        return selected
    if selected.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "KVMem resident visibility requires eager execution; "
            "disable CUDA graph capture for the diagnostic run"
        )

    apply_min_pos = int(plan["apply_min_pos"])
    active_from_pos = int(plan["active_from_pos"])
    region_tokens = int(plan["region_tokens"])
    if active_from_pos < apply_min_pos:
        raise RuntimeError("KVMem resident plan active boundary precedes apply boundary")

    positions_device = logical_positions.to(
        device=selected.device,
        dtype=torch.int64,
    )
    apply_rows = positions_device >= apply_min_pos
    if not bool(apply_rows.any().item()):
        return selected

    valid = selected >= 0
    historical = selected < active_from_pos
    nonnegative = selected.clamp_min(0).to(torch.int64)
    selected_regions = torch.div(
        nonnegative,
        region_tokens,
        rounding_mode="floor",
    )
    resident_tensor = _kvmem_qsa_resident_tensor(plan, selected.device)
    historical_resident = torch.isin(selected_regions, resident_tensor)
    row_mask = apply_rows.unsqueeze(1)

    drop = row_mask & valid & historical & ~historical_resident

    total_valid = int((row_mask & valid).sum().item())
    historical_total = int((row_mask & valid & historical).sum().item())
    historical_kept = int(
        (row_mask & valid & historical & historical_resident).sum().item()
    )
    active_kept = int((row_mask & valid & ~historical).sum().item())
    dropped = int(drop.sum().item())

    selected.masked_fill_(drop, -1)

    pos = positions_device[apply_rows]
    _kvmem_qsa_visibility_record(
        {
            "schema": 1,
            "layer_name": str(layer_name),
            "rows_applied": int(apply_rows.sum().item()),
            "first_applied_pos": int(pos[0].detach().item()),
            "last_applied_pos": int(pos[-1].detach().item()),
            "resident_region_count": int(plan["resident_region_count"]),
            "selected_valid_before": total_valid,
            "historical_selected": historical_total,
            "historical_resident_kept": historical_kept,
            "active_selected_kept": active_kept,
            "historical_selected_dropped": dropped,
        }
    )
    return selected


'''

CALL_ANCHOR = """        if selected.shape != (
            num_tokens,
            self.indexer.output_width,
        ):
            raise RuntimeError("QSA indexer returned an invalid selection shape")
"""
CALL_BLOCK = CALL_ANCHOR + """        _kvmem_qsa_apply_resident_visibility(
            layer_name=self.layer_name,
            selected=selected,
            logical_positions=side_metadata.logical_positions[:num_tokens],
        )
"""


def patch(path: Path, *, check_only: bool = False) -> str:
    src = path.read_text(encoding="utf-8")
    if MARKER in src:
        return "already patched"

    if src.count(IMPORT_ANCHOR) != 1:
        raise RuntimeError("QSA import anchor count != 1")
    if src.count(HELPER_ANCHOR) != 1:
        raise RuntimeError("QSA helper anchor count != 1")
    if src.count(CALL_ANCHOR) != 1:
        raise RuntimeError("QSA selection call anchor count != 1")

    patched = src.replace(IMPORT_ANCHOR, IMPORT_BLOCK, 1)
    patched = patched.replace(HELPER_ANCHOR, HELPER, 1)
    patched = patched.replace(CALL_ANCHOR, CALL_BLOCK, 1)

    required = (
        MARKER,
        "VLLM_QWEN_KVMEM_RESIDENT_PLAN",
        "VLLM_QWEN_KVMEM_VISIBILITY_STATS_PATH",
        "torch.cuda.is_current_stream_capturing()",
        "torch.isin",
        "selected.masked_fill_(drop, -1)",
        "layer_name=self.layer_name",
        "logical_positions=side_metadata.logical_positions[:num_tokens]",
    )
    missing = [token for token in required if token not in patched]
    if missing:
        raise RuntimeError(
            f"QSA resident visibility postcondition missing: {missing}"
        )

    if check_only:
        compile(patched, str(path), "exec")
        return "anchors/postconditions validated; no files changed"

    original = path.with_suffix(path.suffix + ".kvmem_qsa_visibility.orig")
    if not original.exists():
        original.write_text(src, encoding="utf-8")
    path.write_text(patched, encoding="utf-8")
    py_compile.compile(str(path), doraise=True)
    return "patched"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("vllm_root", type=Path)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    target = args.vllm_root.resolve() / TARGET
    if not target.is_file():
        print(f"ERROR: missing vLLM QSA source: {target}", file=sys.stderr)
        return 2
    try:
        status = patch(target, check_only=args.check_only)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"PASS: KVMem QSA resident visibility {status}: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
