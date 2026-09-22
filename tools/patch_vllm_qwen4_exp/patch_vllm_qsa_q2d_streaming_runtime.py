#!/usr/bin/env python3
"""Patch stock vLLM QSA for Q2D scheduler-owned CPU reload runtime."""

from __future__ import annotations

import argparse
import py_compile
import sys
from pathlib import Path


MARKER = "# KVMEM_QSA_Q2D_STREAMING_RUNTIME_V1"
TARGET = Path("models/qwen4_exp/nvidia/qsa.py")
IMPORT_ANCHOR = "from __future__ import annotations\n\n"
IMPORT_BLOCK = """from __future__ import annotations

import os

"""
HELPER_ANCHOR = "from .indexer_qsa import QSAIndexer\n\n"
HELPER = '''from .indexer_qsa import QSAIndexer

# KVMEM_QSA_Q2D_STREAMING_RUNTIME_V1
_Q2D_RUNTIME_PLAN_CACHE = None


def _q2d_runtime_plan():
    global _Q2D_RUNTIME_PLAN_CACHE
    path = os.environ.get("VLLM_QWEN_KVMEM_Q2D_RUNTIME_PLAN")
    if not path:
        return None
    cached = _Q2D_RUNTIME_PLAN_CACHE
    if cached is None or cached[0] != path:
        from vllm_exl3.kvmem_q2d_scheduler_runtime import load_streaming_plan
        _Q2D_RUNTIME_PLAN_CACHE = (path, load_streaming_plan(path))
    return _Q2D_RUNTIME_PLAN_CACHE[1]


def _q2d_runtime_spec(plan):
    from vllm_exl3.kvmem_q2d_scheduler_runtime import make_qsa_streaming_spec
    return make_qsa_streaming_spec(plan)


'''

INIT_ANCHOR = """        self.kv_sharing_target_layer_name = None
        self.kv_cache = torch.tensor([])
        set_default_quant_scales(self, register_buffer=True)
"""
INIT_BLOCK = """        self.kv_sharing_target_layer_name = None
        self.kv_cache = torch.tensor([])
        _q2d_plan_obj = _q2d_runtime_plan()
        if _q2d_plan_obj is not None:
            _q2d_page_tokens = int(_q2d_plan_obj["page_tokens"])
            self.register_buffer(
                "_q2d_dedicated_kv_cache",
                torch.empty(
                    int(_q2d_plan_obj["physical_page_count"]) + 1,
                    self.num_kv_heads,
                    _q2d_page_tokens,
                    2 * self.head_dim,
                    dtype=self.kv_cache_torch_dtype,
                ),
                persistent=False,
            )
            self.register_buffer(
                "_q2d_staging",
                torch.empty(
                    int(_q2d_plan_obj["staging_pages"]),
                    self.num_kv_heads,
                    _q2d_page_tokens,
                    2 * self.head_dim,
                    dtype=self.kv_cache_torch_dtype,
                ),
                persistent=False,
            )
        set_default_quant_scales(self, register_buffer=True)
"""

BIND_ANCHOR = """    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend
"""
BIND_BLOCK = """    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        _q2d_plan_obj = _q2d_runtime_plan()
        if _q2d_plan_obj is None:
            self.kv_cache = kv_cache
            return
        dedicated = getattr(self, "_q2d_dedicated_kv_cache", None)
        expected = (
            int(_q2d_plan_obj["physical_page_count"]) + 1,
            self.num_kv_heads,
            int(_q2d_plan_obj["page_tokens"]),
            2 * self.head_dim,
        )
        if dedicated is None or tuple(dedicated.shape) != expected:
            raise RuntimeError("Q2D dedicated KV shape mismatch")
        if dedicated.dtype != torch.bfloat16 or not dedicated.is_cuda:
            raise RuntimeError("Q2D dedicated KV must be CUDA BF16")
        self._q2d_placeholder_shape = tuple(int(x) for x in kv_cache.shape)
        self._q2d_placeholder_bytes = kv_cache.numel() * kv_cache.element_size()
        self.kv_cache = dedicated
        self.kv_cache[int(_q2d_plan_obj["physical_page_count"])].zero_()
        self._q2d_dedicated_bound = True

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend
"""

SPEC_ANCHOR = """    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )
"""
SPEC_BLOCK = """    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        _q2d_plan_obj = _q2d_runtime_plan()
        if _q2d_plan_obj is not None:
            return _q2d_runtime_spec(_q2d_plan_obj)
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )
"""

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
        _q2d_plan_obj = _q2d_runtime_plan()
        if _q2d_plan_obj is None:
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
        else:
            from vllm_exl3.kvmem_q2d_streaming_worker import run_streaming_runtime
            run_streaming_runtime(
                self,
                impl,
                _q2d_plan_obj,
                query,
                key,
                value,
                output,
                selected,
                side_metadata.logical_positions[:num_tokens],
                main_metadata,
                side_metadata,
            )
"""


def patch(path: Path, *, check_only: bool = False) -> str:
    src = path.read_text(encoding="utf-8")
    if MARKER in src:
        return "already patched"
    for name, anchor in (
        ("import", IMPORT_ANCHOR),
        ("helper", HELPER_ANCHOR),
        ("init", INIT_ANCHOR),
        ("bind", BIND_ANCHOR),
        ("spec", SPEC_ANCHOR),
        ("run", RUN_ANCHOR),
    ):
        if src.count(anchor) != 1:
            raise RuntimeError(f"{name} anchor count != 1")
    out = src.replace(IMPORT_ANCHOR, IMPORT_BLOCK, 1)
    out = out.replace(HELPER_ANCHOR, HELPER, 1)
    out = out.replace(INIT_ANCHOR, INIT_BLOCK, 1)
    out = out.replace(BIND_ANCHOR, BIND_BLOCK, 1)
    out = out.replace(SPEC_ANCHOR, SPEC_BLOCK, 1)
    out = out.replace(RUN_ANCHOR, RUN_BLOCK, 1)
    required = (
        MARKER,
        "VLLM_QWEN_KVMEM_Q2D_RUNTIME_PLAN",
        "make_qsa_streaming_spec",
        "_q2d_dedicated_kv_cache",
        "_q2d_staging",
        "run_streaming_runtime",
        "main_metadata.slot_mapping",
        "side_metadata.logical_positions[:num_tokens]",
    )
    missing = [item for item in required if item not in out]
    if missing:
        raise RuntimeError(f"Q2D runtime postcondition missing: {missing}")
    if check_only:
        compile(out, str(path), "exec")
        return "anchors/postconditions validated; no files changed"
    original = path.with_suffix(path.suffix + ".kvmem_qsa_q2d_streaming.orig")
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
    print(f"PASS: K1-Q2D streaming runtime QSA {status}: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
