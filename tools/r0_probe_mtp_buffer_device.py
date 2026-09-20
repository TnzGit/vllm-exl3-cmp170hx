#!/usr/bin/env python3
"""Diagnostic: log the real device of Qwen4Exp `_mtp_hidden_buffer`.

vLLM 0.29.0 constructs the buffer as

    self._mtp_hidden_buffer = torch.empty(
        max_num_batched_tokens, config.hc_count * config.hidden_size,
        dtype=...,
    )

with no explicit `device=` (upstream #56742 tracks this class of bug). The old
recipe served MTP, so a wrong device cannot be assumed from source alone.

This script inserts a one-line diagnostic immediately after the allocation in
the *installed* vLLM source, so the next MTP boot reports the device the
running engine actually uses. It is diagnostic only and never changes the
allocation. `--revert` restores the original file from the `.mtpdev.bak` copy.

usage:
  python3 r0_probe_mtp_buffer_device.py <site-packages/vllm> [--revert]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ANCHOR = """        if needs_mtp_hidden:
            self._mtp_hidden_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                config.hc_count * config.hidden_size,
                dtype=vllm_config.model_config.dtype,
            )
"""

MARKER = "EXL3_MTP_HIDDEN_BUFFER_DEVICE"

PATCHED = """        if needs_mtp_hidden:
            self._mtp_hidden_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                config.hc_count * config.hidden_size,
                dtype=vllm_config.model_config.dtype,
            )
            import logging as _mtp_logging
            _mtp_logging.getLogger(__name__).warning(
                "%s device=%s dtype=%s shape=%s default_device=%s cuda_current=%s",
                "EXL3_MTP_HIDDEN_BUFFER_DEVICE",
                self._mtp_hidden_buffer.device,
                self._mtp_hidden_buffer.dtype,
                tuple(self._mtp_hidden_buffer.shape),
                torch.empty(0).device,
                (torch.cuda.current_device() if torch.cuda.is_available() else None),
            )
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("vllm_root", type=Path)
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    path = args.vllm_root.resolve() / "models" / "qwen4_exp" / "nvidia" / "model.py"
    if not path.is_file():
        print(f"ERROR: missing {path}", file=sys.stderr)
        return 2
    bak = path.with_suffix(path.suffix + ".mtpdev.bak")

    if args.revert:
        if bak.is_file():
            shutil.copyfile(bak, path)
            print(f"reverted {path} from {bak}")
        else:
            print(f"no backup at {bak}; nothing to revert")
        return 0

    src = path.read_text(encoding="utf-8")
    if MARKER in src:
        print("already instrumented")
        return 0
    if src.count(ANCHOR) != 1:
        print(f"ERROR: allocation anchor found {src.count(ANCHOR)} times", file=sys.stderr)
        return 1
    if not bak.is_file():
        shutil.copyfile(path, bak)
    out = src.replace(ANCHOR, PATCHED)
    compile(out, str(path), "exec")
    path.write_text(out, encoding="utf-8")
    print(f"instrumented {path} (backup {bak})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
