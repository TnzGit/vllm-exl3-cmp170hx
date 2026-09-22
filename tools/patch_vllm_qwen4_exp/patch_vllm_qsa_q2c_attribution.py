#!/usr/bin/env python3
"""Patch stock vLLM QSA for full-KV K1-Q2C attribution controls.

The patch does not change KV ownership or addressing. In ``baseline`` mode it
only records the indexer selection. In ``progressive_mask`` mode it applies the
same shared visibility function as bounded Q2C before stock full-KV attention.
In ``split_reference`` mode it compares full and row-split QSA attention in the
same forward call with identical inputs.
"""

from __future__ import annotations

import argparse
import py_compile
import sys
from pathlib import Path


MARKER = "# KVMEM_QSA_Q2C_ATTRIBUTION_V1"
TARGET = Path("models/qwen4_exp/nvidia/qsa.py")

IMPORT_ANCHOR = "from __future__ import annotations\n\n"
IMPORT_BLOCK = """from __future__ import annotations

import os

"""

HELPER_ANCHOR = "from .indexer_qsa import QSAIndexer\n\n"
HELPER = r'''from .indexer_qsa import QSAIndexer

# KVMEM_QSA_Q2C_ATTRIBUTION_V1
_Q2C_ATTRIB_PLAN_CACHE = None
_Q2C_ATTRIB_RESIDENT_CACHE = {}


def _q2c_attrib_config():
    global _Q2C_ATTRIB_PLAN_CACHE
    mode = os.environ.get("VLLM_QWEN_KVMEM_Q2C_ATTRIB_MODE")
    if not mode:
        return None, None
    if mode not in {"baseline", "progressive_mask", "split_reference"}:
        raise RuntimeError(f"invalid Q2C attribution mode: {mode}")
    path = os.environ.get("VLLM_QWEN_KVMEM_Q2C_PLAN")
    if not path:
        raise RuntimeError("Q2C attribution mode requires a frozen plan")
    cached = _Q2C_ATTRIB_PLAN_CACHE
    if cached is None or cached[0] != path:
        from vllm_exl3.kvmem_qsa_scheduler_runtime import load_runtime_plan
        _Q2C_ATTRIB_PLAN_CACHE = (path, load_runtime_plan(path))
        _Q2C_ATTRIB_RESIDENT_CACHE.clear()
    return mode, _Q2C_ATTRIB_PLAN_CACHE[1]


def _q2c_attrib_selection(layer, mode, plan, selected, positions):
    from vllm_exl3.kvmem_q2c_attribution import apply_progressive_visibility
    key = str(selected.device)
    resident = _Q2C_ATTRIB_RESIDENT_CACHE.get(key)
    if resident is None:
        resident = torch.tensor(
            plan["resident_pages"], dtype=torch.int64, device=selected.device
        )
        _Q2C_ATTRIB_RESIDENT_CACHE[key] = resident
    observation = apply_progressive_visibility(
        plan,
        selected,
        positions,
        apply_mask=mode == "progressive_mask",
        resident_pages_tensor=resident,
    )
    return observation


def _q2c_split_reference(
    layer, impl, query, key, value, kv_cache, metadata, output, token_to_req
):
    rows = int(os.environ.get("VLLM_QWEN_KVMEM_Q2C_SPLIT_ROWS", "0"))
    if rows <= 0:
        raise RuntimeError("Q2C split-reference mode requires positive split rows")
    num_tokens = int(metadata.num_actual_tokens)

    # Full and split calls share the exact same already-written K/V, selected
    # indices, block table, Q, and request mapping in one forward invocation.
    impl.forward_qsa(
        layer,
        query,
        key,
        value,
        kv_cache,
        metadata,
        output,
        token_to_req=token_to_req,
    )
    reference = output[:num_tokens].clone()
    output.zero_()
    if num_tokens == 0:
        return {
            "split_rows": rows,
            "split_calls": 0,
            "split_exact": True,
            "split_elements": 0,
            "split_mismatch_elements": 0,
            "split_max_abs": 0.0,
        }

    logical_indices = layer.topk_indices_buffer[:num_tokens]
    request_ids = token_to_req[:num_tokens]
    key_cache, value_cache = kv_cache.transpose(1, 2).split(
        impl.head_size, dim=-1
    )
    key_cache = canonicalize_singleton_dim_strides(key_cache)
    value_cache = canonicalize_singleton_dim_strides(value_cache)
    from .ops.qsa import qsa_sparse_paged_attention

    calls = 0
    for start in range(0, num_tokens, rows):
        end = min(start + rows, num_tokens)
        qsa_sparse_paged_attention(
            query[start:end],
            key_cache,
            value_cache,
            logical_indices[start:end],
            metadata.block_table,
            request_ids[start:end],
            output[start:end],
        )
        calls += 1
    actual = output[:num_tokens]
    exact = bool(torch.equal(reference, actual))
    mismatch = int(torch.ne(reference, actual).sum().item())
    max_abs = float((reference.float() - actual.float()).abs().max().item())
    return {
        "split_rows": rows,
        "split_calls": calls,
        "split_exact": exact,
        "split_elements": int(reference.numel()),
        "split_mismatch_elements": mismatch,
        "split_max_abs": max_abs,
    }

'''

RUN_ANCHOR = """        impl = cast(Qwen4ExpQSAFlashAttentionImpl, self.impl)
        impl.do_kv_cache_update(
            self,
            key,
            value,
            self.kv_cache,
            main_metadata.slot_mapping,
        )
        impl.forward_qsa(
            self,
            query,
            key,
            value,
            self.kv_cache,
            main_metadata,
            output,
            token_to_req=side_metadata.token_to_req,
        )
"""

RUN_BLOCK = """        impl = cast(Qwen4ExpQSAFlashAttentionImpl, self.impl)
        _q2c_attrib_mode, _q2c_attrib_plan = _q2c_attrib_config()
        _q2c_attrib_observation = None
        if _q2c_attrib_mode is not None:
            _q2c_attrib_observation = _q2c_attrib_selection(
                self,
                _q2c_attrib_mode,
                _q2c_attrib_plan,
                selected,
                side_metadata.logical_positions[:num_tokens],
            )
        impl.do_kv_cache_update(
            self,
            key,
            value,
            self.kv_cache,
            main_metadata.slot_mapping,
        )
        _q2c_split_observation = None
        if _q2c_attrib_mode == "split_reference":
            _q2c_split_observation = _q2c_split_reference(
                self,
                impl,
                query,
                key,
                value,
                self.kv_cache,
                main_metadata,
                output,
                side_metadata.token_to_req,
            )
        else:
            impl.forward_qsa(
                self,
                query,
                key,
                value,
                self.kv_cache,
                main_metadata,
                output,
                token_to_req=side_metadata.token_to_req,
            )
        if _q2c_attrib_observation is not None:
            from vllm_exl3.kvmem_q2c_attribution import (
                tensor_bit_fingerprint,
                write_attribution_event,
            )
            write_attribution_event({
                "event": "q2c_selection",
                "mode": (
                    "a_full_original"
                    if _q2c_attrib_mode == "baseline"
                    else (
                        "b_full_masked"
                        if _q2c_attrib_mode == "progressive_mask"
                        else "d_full_split"
                    )
                ),
                "layer": self.layer_name,
                **_q2c_attrib_observation,
                **(_q2c_split_observation or {}),
                **tensor_bit_fingerprint(output),
            })
            if (
                _q2c_split_observation is not None
                and not _q2c_split_observation["split_exact"]
            ):
                raise RuntimeError(
                    "Q2C full-KV row-split attention differs from same-forward reference"
                )
"""


def patch(path: Path, *, check_only: bool = False) -> str:
    src = path.read_text(encoding="utf-8")
    if MARKER in src:
        return "already patched"
    for name, anchor in (
        ("import", IMPORT_ANCHOR),
        ("helper", HELPER_ANCHOR),
        ("run", RUN_ANCHOR),
    ):
        if src.count(anchor) != 1:
            raise RuntimeError(f"{name} anchor count != 1")

    out = src.replace(IMPORT_ANCHOR, IMPORT_BLOCK, 1)
    out = out.replace(HELPER_ANCHOR, HELPER, 1)
    out = out.replace(RUN_ANCHOR, RUN_BLOCK, 1)
    required = (
        MARKER,
        "VLLM_QWEN_KVMEM_Q2C_ATTRIB_MODE",
        "apply_progressive_visibility",
        'mode == "progressive_mask"',
        '"split_reference"',
        "_q2c_split_reference",
        '"d_full_split"',
        '"split_exact"',
        '"a_full_original"',
        '"b_full_masked"',
        "tensor_bit_fingerprint",
        "main_metadata.slot_mapping",
    )
    missing = [item for item in required if item not in out]
    if missing:
        raise RuntimeError(f"Q2C attribution postcondition missing: {missing}")

    if check_only:
        compile(out, str(path), "exec")
        return "anchors/postconditions validated; no files changed"

    original = path.with_suffix(path.suffix + ".kvmem_qsa_q2c_attribution.orig")
    if not original.exists():
        original.write_text(src, encoding="utf-8")
    path.write_text(out, encoding="utf-8")
    py_compile.compile(str(path), doraise=True)
    return "patched"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("vllm_root", type=Path)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    target = args.vllm_root.resolve() / TARGET
    if not target.is_file():
        print(f"ERROR: missing QSA source: {target}", file=sys.stderr)
        return 2
    try:
        status = patch(target, check_only=args.check_only)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"PASS: K1-Q2C attribution QSA {status}: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
