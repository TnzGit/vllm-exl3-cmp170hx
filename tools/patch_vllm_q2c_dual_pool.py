#!/usr/bin/env python3
"""Patch vLLM 0.29 for the Qwen4Exp Q2C private-QSA-pool proof.

The patch is strictly gated by VLLM_QWEN_KVMEM_Q2C_PLAN at runtime.

It changes three installed vLLM files:
- platforms/interface.py: do not force QSA+Mamba into one 1568-token page.
- v1/core/kv_cache_utils.py: size QSA and regular/Mamba pools independently.
- v1/worker/utils.py: allocate a dedicated QSA backing arena and bind normal
  per-layer views into it.

Stock behavior is unchanged when Q2C is not active.
"""

from __future__ import annotations

import argparse
import py_compile
import sys
from pathlib import Path


MARKER_PLATFORM = "# KVMEM_Q2C_DUAL_POOL_PLATFORM_V1"
MARKER_CORE = "# KVMEM_Q2C_DUAL_POOL_CORE_V1"
MARKER_WORKER = "# KVMEM_Q2C_DUAL_POOL_WORKER_V1"

TARGETS = {
    "platform": Path("platforms/interface.py"),
    "core": Path("v1/core/kv_cache_utils.py"),
    "worker": Path("v1/worker/utils.py"),
}

PLATFORM_OLD = """        # Phase 2: Align block/mamba sizes for hybrid models
        # (may override user settings).
        if model_config.is_hybrid:
            cls._align_hybrid_block_size(vllm_config, backend_cls)
"""

PLATFORM_NEW = """        # Phase 2: Align block/mamba sizes for hybrid models
        # (may override user settings).
        if model_config.is_hybrid:
            # KVMEM_Q2C_DUAL_POOL_PLATFORM_V1
            if os.environ.get("VLLM_QWEN_KVMEM_Q2C_PLAN"):
                model_type = getattr(model_config.hf_config, "model_type", None)
                backend_name = backend_cls.get_name()
                layout = os.environ.get("VLLM_KV_CACHE_LAYOUT")
                if (
                    model_type != "qwen4_exp"
                    or backend_name != "QWEN4_EXP_QSA_TRITON"
                    or cache_config.enable_prefix_caching
                    or cache_config.mamba_cache_mode != "none"
                    or cache_config.block_size != 16
                    or layout != "BLHNC"
                ):
                    raise RuntimeError(
                        "Q2C dual-pool platform gate mismatch: "
                        f"model_type={model_type} backend={backend_name} "
                        f"prefix={cache_config.enable_prefix_caching} "
                        f"mamba_mode={cache_config.mamba_cache_mode} "
                        f"block_size={cache_config.block_size} layout={layout}"
                    )
                logger.info(
                    "Q2C dual-pool: preserving 16-token QSA pages; "
                    "skipping legacy QSA/Mamba shared-page alignment."
                )
            else:
                cls._align_hybrid_block_size(vllm_config, backend_cls)
"""

CORE_ANCHOR = """def get_kv_cache_config_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
"""

CORE_HELPERS = r'''# KVMEM_Q2C_DUAL_POOL_CORE_V1
def _q2c_private_layer_specs(spec: KVCacheSpec) -> list[KVCacheSpec]:
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return list(spec.kv_cache_specs.values())
    return [spec]


def _q2c_is_private_group(group: KVCacheGroupSpec) -> bool:
    specs = _q2c_private_layer_specs(group.kv_cache_spec)
    flags = [bool(getattr(spec, "q2c_private_pool", False)) for spec in specs]
    if any(flags) and not all(flags):
        raise RuntimeError("Q2C private and stock specs cannot share one group")
    return bool(flags) and all(flags)


def _q2c_dual_pool_enabled(groups: list[KVCacheGroupSpec]) -> bool:
    if not os.environ.get("VLLM_QWEN_KVMEM_Q2C_PLAN"):
        return False
    private = [g for g in groups if _q2c_is_private_group(g)]
    if len(private) != 1:
        raise RuntimeError(
            f"Q2C requires exactly one private QSA group, got {len(private)}"
        )
    return True


def _q2c_split_groups(
    groups: list[KVCacheGroupSpec],
) -> tuple[KVCacheGroupSpec, list[KVCacheGroupSpec]]:
    private = [g for g in groups if _q2c_is_private_group(g)]
    regular = [g for g in groups if not _q2c_is_private_group(g)]
    if len(private) != 1 or not regular:
        raise RuntimeError(
            "Q2C dual-pool requires one private QSA group plus regular groups"
        )
    return private[0], regular


def _q2c_private_num_blocks(group: KVCacheGroupSpec) -> int:
    values = {
        int(getattr(spec, "private_pool_num_blocks"))
        for spec in _q2c_private_layer_specs(group.kv_cache_spec)
    }
    if len(values) != 1:
        raise RuntimeError(f"Q2C private block-count mismatch: {sorted(values)}")
    value = values.pop()
    if value != 4161:
        raise RuntimeError(f"Q2C private pool must be 4161 blocks, got {value}")
    return value


def _q2c_private_bytes_per_block(group: KVCacheGroupSpec) -> int:
    return int(group.kv_cache_spec.page_size_bytes)


def _q2c_regular_request_blocks(
    vllm_config: VllmConfig, groups: list[KVCacheGroupSpec]
) -> int:
    total = 0
    for group in groups:
        spec = group.kv_cache_spec
        total += cdiv(
            spec.max_memory_usage_bytes(vllm_config),
            spec.page_size_bytes,
        )
    return total


def _q2c_request_memory_bytes(
    vllm_config: VllmConfig, groups: list[KVCacheGroupSpec]
) -> int:
    private, regular = _q2c_split_groups(groups)
    regular_stride = _pool_bytes_per_block(regular)
    regular_needed = regular_stride * _q2c_regular_request_blocks(
        vllm_config, regular
    )
    private_needed = private.kv_cache_spec.max_memory_usage_bytes(vllm_config)
    return int(regular_needed + private_needed)


def _q2c_null_overhead_bytes(groups: list[KVCacheGroupSpec]) -> int:
    private, regular = _q2c_split_groups(groups)
    return _pool_bytes_per_block(regular) + _q2c_private_bytes_per_block(private)


def _q2c_build_private_tensors(
    group: KVCacheGroupSpec,
    layout: KVCacheLayout,
) -> tuple[list[KVCacheTensor], int]:
    if not layout.is_block_outermost or layout.name != "BLHNC":
        raise RuntimeError(f"Q2C private QSA pool requires BLHNC, got {layout.name}")
    num_blocks = _q2c_private_num_blocks(group)
    bytes_per_block = _q2c_private_bytes_per_block(group)
    size = num_blocks * bytes_per_block
    group_spec = group.kv_cache_spec
    layers_by_spec: defaultdict[KVCacheSpec, list[str]] = defaultdict(list)
    if isinstance(group_spec, UniformTypeKVCacheSpecs):
        for layer_name, spec in group_spec.kv_cache_specs.items():
            layers_by_spec[spec].append(layer_name)
    elif group.layer_names:
        layers_by_spec[group_spec].extend(group.layer_names)

    tensors: list[KVCacheTensor] = []
    byte_offset = 0
    for spec, layer_names in layers_by_spec.items():
        layer_stride, block_stride, _, _, _ = compute_layout_strides(
            spec,
            num_blocks,
            len(layer_names),
            layout,
            fixed_strides=(None, bytes_per_block, None, None, None),
        )
        offset = (
            byte_offset
            * max(layer_stride, spec.page_size_bytes)
            // spec.page_size_bytes
        )
        tensors.append(
            KVCacheTensor(
                size=size,
                layers=layer_names,
                layer_stride=layer_stride,
                block_stride=block_stride,
                offset=offset,
            )
        )
        byte_offset += len(layer_names) * spec.page_size_bytes
    if byte_offset != bytes_per_block:
        raise RuntimeError(
            f"Q2C private packed bytes mismatch {byte_offset} != {bytes_per_block}"
        )
    return tensors, size


def _q2c_get_kv_cache_config_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    private, regular = _q2c_split_groups(kv_cache_groups)
    layout = vllm_config.cache_config.get_resolved_kv_cache_layout()
    validate_kv_cache_layout(layout, kv_cache_groups)

    private_tensors, private_size = _q2c_build_private_tensors(private, layout)
    if private_size >= available_memory:
        raise ValueError(
            "Q2C private QSA arena alone exceeds available KV cache memory: "
            f"{private_size} >= {available_memory}"
        )
    regular_available = available_memory - private_size
    regular_cfg = get_kv_cache_config_from_groups(
        vllm_config, regular, regular_available
    )
    logger.info(
        "Q2C dual-pool KV config: private_qsa_blocks=%d "
        "private_qsa_bytes=%d regular_blocks=%d regular_bytes=%d",
        _q2c_private_num_blocks(private),
        private_size,
        regular_cfg.num_blocks,
        max((t.size for t in regular_cfg.kv_cache_tensors), default=0),
    )
    return KVCacheConfig(
        num_blocks=regular_cfg.num_blocks,
        kv_cache_tensors=[*regular_cfg.kv_cache_tensors, *private_tensors],
        kv_cache_groups=kv_cache_groups,
        prefix_cache_retention_interval=(
            vllm_config.cache_config.prefix_cache_retention_interval
        ),
    )


'''

CORE_FUNCTION_ENTRY_OLD = CORE_ANCHOR + '''    """
    Generate the KV cache configuration from the KV cache groups and spec
'''

CORE_FUNCTION_ENTRY_NEW = CORE_ANCHOR + '''    if _q2c_dual_pool_enabled(kv_cache_groups):
        return _q2c_get_kv_cache_config_from_groups(
            vllm_config, kv_cache_groups, available_memory
        )
    """
    Generate the KV cache configuration from the KV cache groups and spec
'''

CORE_MEMORY_OLD = """    if not kv_cache_groups:
        return 0

    bytes_per_block = _pool_bytes_per_block(kv_cache_groups)
"""

CORE_MEMORY_NEW = """    if not kv_cache_groups:
        return 0
    if _q2c_dual_pool_enabled(kv_cache_groups):
        return _q2c_request_memory_bytes(vllm_config, kv_cache_groups)

    bytes_per_block = _pool_bytes_per_block(kv_cache_groups)
"""

CORE_CHECK_MEMORY_OLD = """    check_memory = [
        avail_mem - _pool_bytes_per_block(groups) if groups else avail_mem
        for groups, avail_mem in zip(projected_groups_per_worker, available_memory)
    ]
"""

CORE_CHECK_MEMORY_NEW = """    check_memory = [
        (
            avail_mem - _q2c_null_overhead_bytes(groups)
            if groups and _q2c_dual_pool_enabled(groups)
            else avail_mem - _pool_bytes_per_block(groups)
            if groups
            else avail_mem
        )
        for groups, avail_mem in zip(projected_groups_per_worker, available_memory)
    ]
"""

CORE_CAPACITY_OLD = """    num_blocks_per_request = sum(
        cdiv(
            group.kv_cache_spec.max_memory_usage_bytes(vllm_config),
            group.kv_cache_spec.page_size_bytes,
        )
        for group in kv_cache_config.kv_cache_groups
    )
    max_concurrency = kv_cache_config.num_blocks / num_blocks_per_request
    return max_concurrency
"""

CORE_CAPACITY_NEW = """    if _q2c_dual_pool_enabled(kv_cache_config.kv_cache_groups):
        private, regular = _q2c_split_groups(kv_cache_config.kv_cache_groups)
        regular_req = _q2c_regular_request_blocks(vllm_config, regular)
        regular_concurrency = kv_cache_config.num_blocks / regular_req
        private_usable = _q2c_private_num_blocks(private) - 1
        private_req = cdiv(
            private.kv_cache_spec.max_memory_usage_bytes(vllm_config),
            private.kv_cache_spec.page_size_bytes,
        )
        private_concurrency = private_usable / private_req
        return min(regular_concurrency, private_concurrency)

    num_blocks_per_request = sum(
        cdiv(
            group.kv_cache_spec.max_memory_usage_bytes(vllm_config),
            group.kv_cache_spec.page_size_bytes,
        )
        for group in kv_cache_config.kv_cache_groups
    )
    max_concurrency = kv_cache_config.num_blocks / num_blocks_per_request
    return max_concurrency
"""

CORE_SINGLE_WORKER_OLD = """    # Change the num_blocks of each rank to the smallest among all ranks.
    # We also need to shrink the tensor size proportionally to avoid
    # allocating unused memory.
    min_num_blocks = min(
"""

CORE_SINGLE_WORKER_NEW = """    if any(
        _q2c_dual_pool_enabled(cfg.kv_cache_groups) for cfg in kv_cache_configs
    ) and len(kv_cache_configs) != 1:
        raise RuntimeError("Q2C dual-pool proof currently requires one worker")

    # Change the num_blocks of each rank to the smallest among all ranks.
    # We also need to shrink the tensor size proportionally to avoid
    # allocating unused memory.
    min_num_blocks = min(
"""

WORKER_ALLOC_OLD = '''    sizes = {tensor.size for tensor in kv_cache_config.kv_cache_tensors}
    assert len(sizes) == 1, "KV cache tensors must share one backing allocation."
    raw_size = sizes.pop()
    # wvSplitKrc's process-lifetime static workspaces (csrc/rocm/skinny_gemms.cu)
    # are created lazily on the first qualifying GEMM. Force that now, before
    # the giant backing allocation below: if one landed in this segment's
    # rounding tail it would pin the whole segment at engine shutdown.
    if current_platform.is_rocm():
        warmup_rocm_skinny_gemm_workspaces(device)
        # Pad to the page granularity MoRIIO needs to register the shared
        # backing as a single RDMA memory region. Other platforms keep the
        # exact-size allocation: NIXL and SimpleCPUOffload rely on
        # storage.nbytes() matching the logical KV size (see #53974).
        page_size = 4096
        buf_size = ((raw_size + page_size - 1) // page_size) * page_size
    else:
        buf_size = raw_size
    buf = torch.zeros(buf_size, dtype=torch.int8, device=device)

    kv_caches: dict[str, torch.Tensor] = {}
    for tensor in kv_cache_config.kv_cache_tensors:
        layer_name = tensor.layers[0]
        group_id, group = next(
            (group_id, group)
            for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
            if layer_name in group.layer_names
        )
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            spec = spec.kv_cache_specs[layer_name]

        num_blocks = kv_cache_config.num_blocks
        kernel_block_size = None
        if kernel_block_sizes is not None and group_id < len(kernel_block_sizes):
            kernel_block_size = kernel_block_sizes[group_id]

        views = create_kv_cache_views(
            buf,
            spec,
            num_blocks,
            layout,
            tensor,
            kernel_block_size=kernel_block_size,
        )
        kv_caches.update(zip(tensor.layers, views))
    return kv_caches
'''

WORKER_ALLOC_NEW = '''    # KVMEM_Q2C_DUAL_POOL_WORKER_V1
    def private_spec(spec):
        if isinstance(spec, UniformTypeKVCacheSpecs):
            values = list(spec.kv_cache_specs.values())
            flags = [bool(getattr(x, "q2c_private_pool", False)) for x in values]
            if any(flags) and not all(flags):
                raise RuntimeError("mixed Q2C private/stock worker group")
            return all(flags)
        return bool(getattr(spec, "q2c_private_pool", False))

    q2c_active = bool(os.environ.get("VLLM_QWEN_KVMEM_Q2C_PLAN"))
    group_is_private = {
        i: private_spec(group.kv_cache_spec)
        for i, group in enumerate(kv_cache_config.kv_cache_groups)
    }
    private_group_ids = [i for i, flag in group_is_private.items() if flag]

    if q2c_active:
        if len(private_group_ids) != 1 or layout.name != "BLHNC":
            raise RuntimeError(
                f"Q2C worker dual-pool gate mismatch: "
                f"private_groups={private_group_ids} layout={layout.name}"
            )
        stock_sizes = {
            tensor.size
            for tensor in kv_cache_config.kv_cache_tensors
            if not any(
                group_is_private[i]
                for i, group in enumerate(kv_cache_config.kv_cache_groups)
                if tensor.layers[0] in group.layer_names
            )
        }
        private_sizes = {
            tensor.size
            for tensor in kv_cache_config.kv_cache_tensors
            if any(
                group_is_private[i]
                for i, group in enumerate(kv_cache_config.kv_cache_groups)
                if tensor.layers[0] in group.layer_names
            )
        }
        if len(stock_sizes) != 1 or len(private_sizes) != 1:
            raise RuntimeError(
                f"Q2C worker arena sizes invalid: "
                f"stock={stock_sizes} private={private_sizes}"
            )
        buffers = {
            False: torch.zeros(stock_sizes.pop(), dtype=torch.int8, device=device),
            True: torch.zeros(private_sizes.pop(), dtype=torch.int8, device=device),
        }
    else:
        sizes = {tensor.size for tensor in kv_cache_config.kv_cache_tensors}
        assert len(sizes) == 1, "KV cache tensors must share one backing allocation."
        raw_size = sizes.pop()
        # wvSplitKrc's process-lifetime static workspaces
        # (csrc/rocm/skinny_gemms.cu) are created lazily on the first qualifying
        # GEMM. Force that before the giant backing allocation.
        if current_platform.is_rocm():
            warmup_rocm_skinny_gemm_workspaces(device)
            page_size = 4096
            buf_size = ((raw_size + page_size - 1) // page_size) * page_size
        else:
            buf_size = raw_size
        buffers = {False: torch.zeros(buf_size, dtype=torch.int8, device=device)}

    kv_caches: dict[str, torch.Tensor] = {}
    for tensor in kv_cache_config.kv_cache_tensors:
        layer_name = tensor.layers[0]
        group_id, group = next(
            (group_id, group)
            for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
            if layer_name in group.layer_names
        )
        group_spec = group.kv_cache_spec
        is_private = group_is_private[group_id] if q2c_active else False
        spec = group_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            spec = spec.kv_cache_specs[layer_name]

        num_blocks = (
            int(getattr(spec, "private_pool_num_blocks"))
            if is_private
            else kv_cache_config.num_blocks
        )
        if is_private and num_blocks != 4161:
            raise RuntimeError(
                f"Q2C worker private pool must be 4161 blocks, got {num_blocks}"
            )
        kernel_block_size = None
        if kernel_block_sizes is not None and group_id < len(kernel_block_sizes):
            kernel_block_size = kernel_block_sizes[group_id]

        views = create_kv_cache_views(
            buffers[is_private],
            spec,
            num_blocks,
            layout,
            tensor,
            kernel_block_size=kernel_block_size,
        )
        kv_caches.update(zip(tensor.layers, views))
    return kv_caches
'''


def replace_once(src: str, old: str, new: str, name: str) -> str:
    count = src.count(old)
    if count != 1:
        raise RuntimeError(f"{name} anchor count={count}, expected 1")
    return src.replace(old, new, 1)


def patch_platform(src: str) -> str:
    if MARKER_PLATFORM in src:
        return src
    return replace_once(src, PLATFORM_OLD, PLATFORM_NEW, "platform")


def patch_core(src: str) -> str:
    if MARKER_CORE in src:
        return src
    out = replace_once(src, CORE_ANCHOR, CORE_HELPERS + CORE_ANCHOR, "core helpers")
    out = replace_once(
        out, CORE_FUNCTION_ENTRY_OLD, CORE_FUNCTION_ENTRY_NEW, "core config entry"
    )
    out = replace_once(out, CORE_MEMORY_OLD, CORE_MEMORY_NEW, "core memory")
    out = replace_once(
        out, CORE_CHECK_MEMORY_OLD, CORE_CHECK_MEMORY_NEW, "core check memory"
    )
    out = replace_once(out, CORE_CAPACITY_OLD, CORE_CAPACITY_NEW, "core capacity")
    out = replace_once(
        out, CORE_SINGLE_WORKER_OLD, CORE_SINGLE_WORKER_NEW, "core single worker"
    )
    return out


def patch_worker(src: str) -> str:
    if MARKER_WORKER in src:
        return src
    return replace_once(src, WORKER_ALLOC_OLD, WORKER_ALLOC_NEW, "worker allocate")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("vllm_root", type=Path)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    root = args.vllm_root.resolve()

    funcs = {
        "platform": patch_platform,
        "core": patch_core,
        "worker": patch_worker,
    }
    outputs: dict[str, tuple[Path, str]] = {}
    try:
        for key, rel in TARGETS.items():
            path = root / rel
            if not path.is_file():
                raise RuntimeError(f"missing installed source: {path}")
            src = path.read_text(encoding="utf-8")
            out = funcs[key](src)
            compile(out, str(path), "exec")
            outputs[key] = (path, out)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if args.check_only:
        print("PASS: Q2C dual-pool installed-source anchors/postconditions validated")
        return 0

    for key, (path, out) in outputs.items():
        path.write_text(out, encoding="utf-8")
        py_compile.compile(str(path), doraise=True)
        print(f"PATCHED {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
