#!/usr/bin/env python3
"""Patch vLLM 0.29 Qwen4Exp NVIDIA QSA with research-only shadow export.

The patch is deliberately diagnostic-only. When
VLLM_QWEN_KVMEM_SHADOW_PATH is unset, the helper returns immediately and
QSA behavior is unchanged.

When enabled, run the engine in eager mode. The helper copies only the tail
rows of QSA's already-computed logical top-k indices to a JSONL file. It never
changes the indices consumed by sparse attention.
"""

from __future__ import annotations

import argparse
import py_compile
import sys
from pathlib import Path

MARKER = "# KVMEM_QSA_SHADOW_V1"
TARGET = Path("models/qwen4_exp/nvidia/qsa.py")

IMPORT_ANCHOR = "from __future__ import annotations\n\n"
IMPORT_BLOCK = """from __future__ import annotations

import json
import os
from pathlib import Path

"""

HELPER_ANCHOR = "from .indexer_qsa import QSAIndexer\n\n"
HELPER = r'''from .indexer_qsa import QSAIndexer

# KVMEM_QSA_SHADOW_V1
def _kvmem_qsa_shadow_record(
    *,
    layer_name: str,
    selected: torch.Tensor,
    logical_positions: torch.Tensor,
    skip_topk: bool,
) -> None:
    """Export already-computed QSA selections without changing model math."""

    path = os.environ.get("VLLM_QWEN_KVMEM_SHADOW_PATH")
    if not path:
        return
    if selected.ndim != 2 or logical_positions.ndim != 1:
        raise RuntimeError("KVMem QSA shadow received unexpected tensor ranks")
    if selected.shape[0] != logical_positions.shape[0]:
        raise RuntimeError("KVMem QSA shadow row counts disagree")
    if selected.shape[0] == 0:
        return
    if selected.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "KVMem QSA shadow export requires eager execution; "
            "disable CUDA graph capture for the diagnostic run"
        )

    try:
        tail_rows = int(os.environ.get("VLLM_QWEN_KVMEM_SHADOW_ROWS", "128"))
        min_pos = int(os.environ.get("VLLM_QWEN_KVMEM_SHADOW_MIN_POS", "-1"))
    except ValueError as exc:
        raise RuntimeError("invalid KVMem QSA shadow integer setting") from exc
    tail_rows = max(1, tail_rows)

    if min_pos >= 0:
        last_pos = int(logical_positions[-1].detach().item())
        if last_pos < min_pos:
            return

    start = max(0, selected.shape[0] - tail_rows)

    pos_cpu = (
        logical_positions[start:]
        .detach()
        .to(device="cpu", dtype=torch.int64)
        .tolist()
    )
    sel_cpu = (
        selected[start:]
        .detach()
        .to(device="cpu", dtype=torch.int32)
        .tolist()
    )
    rows = []
    for pos, indices in zip(pos_cpu, sel_cpu):
        rows.append(
            {
                "pos": int(pos),
                "selected": [int(x) for x in indices if int(x) >= 0],
            }
        )

    payload = {
        "schema": 1,
        "pid": os.getpid(),
        "layer_name": str(layer_name),
        "skip_topk": bool(skip_topk),
        "num_rows_total": int(selected.shape[0]),
        "output_width": int(selected.shape[1]),
        "first_exported_pos": int(rows[0]["pos"]),
        "last_exported_pos": int(rows[-1]["pos"]),
        "rows": rows,
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, separators=(",", ":")) + "\n")


'''

CALL_ANCHOR = """        if selected.shape != (
            num_tokens,
            self.indexer.output_width,
        ):
            raise RuntimeError("QSA indexer returned an invalid selection shape")
"""
CALL_BLOCK = CALL_ANCHOR + """        _kvmem_qsa_shadow_record(
            layer_name=self.layer_name,
            selected=selected,
            logical_positions=side_metadata.logical_positions[:num_tokens],
            skip_topk=self.indexer.skip_topk,
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
        "VLLM_QWEN_KVMEM_SHADOW_PATH",
        "VLLM_QWEN_KVMEM_SHADOW_MIN_POS",
        "torch.cuda.is_current_stream_capturing()",
        "layer_name=self.layer_name",
        "selected=selected",
        "logical_positions=side_metadata.logical_positions[:num_tokens]",
    )
    missing = [token for token in required if token not in patched]
    if missing:
        raise RuntimeError(f"QSA shadow postcondition missing: {missing}")
    if "layer_id=self.layer_id" in patched:
        raise RuntimeError(
            "QSA shadow patch must not depend on Qwen4ExpQSAAttention.self.layer_id"
        )

    if check_only:
        compile(patched, str(path), "exec")
        return "anchors/postconditions validated; no files changed"

    original = path.with_suffix(path.suffix + ".kvmem_qsa_shadow.orig")
    if not original.exists():
        original.write_text(src, encoding="utf-8")

    src = patched
    path.write_text(src, encoding="utf-8")
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
    print(f"PASS: KVMem QSA shadow {status}: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
