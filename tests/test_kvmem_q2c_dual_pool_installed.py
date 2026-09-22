import os
from types import SimpleNamespace

import torch

from vllm.config.cache import CacheConfig
from vllm.v1.core import kv_cache_utils
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheLayout,
    KVCacheTensor,
    MambaSpec,
)
from vllm.v1.worker.utils import allocate_kv_cache

from vllm_exl3.kvmem_qsa_scheduler_runtime import make_qsa_runtime_spec


def _plan():
    return {
        "schema": 1,
        "mode": "qsa_scheduler_owned_transition",
        "page_tokens": 16,
        "resident_pages": list(range(4096)),
        "resident_page_count": 4096,
        "apply_min_pos": 159488,
        "active_from_pos": 159488,
        "active_page0": 9968,
        "active_reserve_tokens": 1024,
        "active_reserve_pages": 64,
        "scheduler_chunk_tokens": 1024,
        "physical_page_count": 4160,
    }


def _config():
    cache = CacheConfig()
    cache.kv_cache_layout = "BLHNC"
    cache.mamba_cache_mode = "none"
    cache.enable_prefix_caching = False
    return SimpleNamespace(
        cache_config=cache,
        model_config=SimpleNamespace(max_model_len=161000),
    )


def test_q2c_dual_pool_config_charges_private_qsa_at_32k_stride(monkeypatch):
    monkeypatch.setenv("VLLM_QWEN_KVMEM_Q2C_PLAN", "/tmp/frozen-plan.json")
    qsa = make_qsa_runtime_spec(_plan())
    mamba = MambaSpec(
        shapes=((2_097_152,),),
        dtypes=(torch.bfloat16,),
        block_size=161000,
        mamba_cache_mode="none",
    )
    groups = [
        KVCacheGroupSpec(["qsa"], qsa),
        KVCacheGroupSpec(["mamba"], mamba),
    ]
    private_size = 4161 * qsa.page_size_bytes
    regular_blocks = 10
    available = private_size + regular_blocks * mamba.page_size_bytes
    cfg = kv_cache_utils.get_kv_cache_config_from_groups(
        _config(), groups, available
    )

    assert cfg.num_blocks == regular_blocks
    assert len(cfg.kv_cache_tensors) == 2
    by_layer = {t.layers[0]: t for t in cfg.kv_cache_tensors}
    assert by_layer["qsa"].size == private_size
    assert by_layer["qsa"].block_stride == qsa.page_size_bytes
    assert by_layer["mamba"].size == regular_blocks * mamba.page_size_bytes

    needed = kv_cache_utils._max_memory_usage_bytes_from_groups(
        _config(), groups
    )
    assert needed == 4160 * qsa.page_size_bytes + mamba.page_size_bytes
    assert kv_cache_utils._q2c_null_overhead_bytes(groups) == (
        qsa.page_size_bytes + mamba.page_size_bytes
    )
    assert kv_cache_utils.get_max_concurrency_for_kv_cache_config(
        _config(), cfg
    ) == 1.0


def test_q2c_worker_allocates_private_qsa_storage(monkeypatch):
    monkeypatch.setenv("VLLM_QWEN_KVMEM_Q2C_PLAN", "/tmp/frozen-plan.json")
    qsa = make_qsa_runtime_spec(_plan())
    regular = FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=256,
        head_size_v=256,
        dtype=torch.bfloat16,
    )
    qsa_size = 4161 * qsa.page_size_bytes
    regular_blocks = 4
    regular_size = regular_blocks * regular.page_size_bytes
    cfg = KVCacheConfig(
        num_blocks=regular_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=regular_size,
                layers=["regular"],
                layer_stride=regular.page_size_bytes,
                block_stride=regular.page_size_bytes,
            ),
            KVCacheTensor(
                size=qsa_size,
                layers=["qsa"],
                layer_stride=qsa.page_size_bytes,
                block_stride=qsa.page_size_bytes,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["regular"], regular),
            KVCacheGroupSpec(["qsa"], qsa),
        ],
    )
    out = allocate_kv_cache(
        cfg,
        torch.device("cpu"),
        KVCacheLayout.BLHNC,
        kernel_block_sizes=[16, 16],
    )
    assert out["regular"].shape[0] == regular_blocks
    assert out["qsa"].shape[0] == 4161
    assert out["regular"].untyped_storage().data_ptr() != (
        out["qsa"].untyped_storage().data_ptr()
    )


def test_stock_worker_still_requires_one_backing_without_q2c(monkeypatch):
    monkeypatch.delenv("VLLM_QWEN_KVMEM_Q2C_PLAN", raising=False)
    regular = FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=256,
        head_size_v=256,
        dtype=torch.bfloat16,
    )
    cfg = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[
            KVCacheTensor(
                size=4 * regular.page_size_bytes,
                layers=["a"],
                layer_stride=regular.page_size_bytes,
                block_stride=regular.page_size_bytes,
            ),
            KVCacheTensor(
                size=5 * regular.page_size_bytes,
                layers=["b"],
                layer_stride=regular.page_size_bytes,
                block_stride=regular.page_size_bytes,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["a"], regular),
            KVCacheGroupSpec(["b"], regular),
        ],
    )
    try:
        allocate_kv_cache(
            cfg,
            torch.device("cpu"),
            KVCacheLayout.BLHNC,
            kernel_block_sizes=[16, 16],
        )
    except AssertionError as exc:
        assert "share one backing allocation" in str(exc)
    else:
        raise AssertionError("stock allocate_kv_cache accepted split backing sizes")
