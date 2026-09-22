from math import ceil
from types import SimpleNamespace

import torch

from vllm.v1.core.kv_cache_utils import _max_memory_usage_bytes_from_groups
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_exl3.kvmem_qsa_scheduler_runtime import make_qsa_runtime_spec


def _config():
    return SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=161000),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )


def test_q2c_shared_sizer_charges_one_placeholder_not_4160_wide_blocks():
    cfg = _config()
    qsa_specs = {
        f"qsa.{i}": make_qsa_runtime_spec(
            {
                "schema": 1,
                "mode": "qsa_scheduler_owned_transition",
                "page_tokens": 16,
                "resident_pages": tuple(range(4096)),
                "resident_page_count": 4096,
                "apply_min_pos": 160000,
                "active_from_pos": 160000,
                "active_page0": 10000,
                "active_reserve_pages": 64,
                "active_reserve_tokens": 1024,
                "scheduler_chunk_tokens": 1024,
                "physical_page_count": 4160,
            }
        )
        for i in range(12)
    }
    qsa_group = UniformTypeKVCacheSpecs.from_specs(qsa_specs)
    assert qsa_group is not None
    assert qsa_group.block_size == 16
    assert qsa_group.page_size_bytes == 12 * 32768
    assert qsa_group.max_memory_usage_pages(cfg) == 1
    assert qsa_group.max_num_blocks_per_req(cfg, 161000) == 10063

    # Reproduce the shared-HMA failure mechanism with a stock 1568-token
    # full-attention group. Its page is wider than the whole 12-layer QSA
    # placeholder group, so every global BlockPool ID is charged at this stride.
    wide = FullAttentionSpec(
        block_size=1568,
        num_kv_heads=2,
        head_size=256,
        head_size_v=256,
        dtype=torch.bfloat16,
    )
    wide_pages = ceil(161000 / 1568)
    assert wide_pages == 103
    assert wide.page_size_bytes > qsa_group.page_size_bytes

    groups = [
        KVCacheGroupSpec(list(qsa_specs), qsa_group),
        KVCacheGroupSpec(["wide.0"], wide),
    ]
    actual = _max_memory_usage_bytes_from_groups(cfg, groups)
    expected = (1 + wide_pages) * wide.page_size_bytes
    assert actual == expected

    # This is exactly the architectural regression from the first live run:
    # charging 4160 QSA IDs in the wide shared stride explodes into >12 GiB.
    old_shared_charge = (4160 + wide_pages) * wide.page_size_bytes
    assert old_shared_charge > 12 * 2**30
    assert actual < 1 * 2**30


def test_q2c_dedicated_storage_is_explicit_and_outside_shared_placeholder():
    spec = next(iter({
        "qsa": make_qsa_runtime_spec(
            {
                "schema": 1,
                "mode": "qsa_scheduler_owned_transition",
                "page_tokens": 16,
                "resident_pages": tuple(range(4096)),
                "resident_page_count": 4096,
                "apply_min_pos": 160000,
                "active_from_pos": 160000,
                "active_page0": 10000,
                "active_reserve_pages": 64,
                "active_reserve_tokens": 1024,
                "scheduler_chunk_tokens": 1024,
                "physical_page_count": 4160,
            }
        )
    }.values()))
    assert spec.max_memory_usage_bytes(_config()) == 32768
    assert spec.dedicated_page_count == 4161
    assert spec.dedicated_memory_bytes_per_layer == 4161 * 32768
    assert spec.dedicated_memory_bytes_per_layer * 12 / 2**30 == 1.5238037109375
