"""Q2C scheduler-visible shrink contract for Qwen4Exp QSA.

This module is intentionally CPU/static only. It defines the cache-spec
geometry and a registry probe manager needed to prove that vLLM 0.29 can
separate a full logical QSA block table from a bounded physical GPU budget.

It is not the runtime Q2C resident manager and must not be used to claim that
scheduler-visible shrink is already implemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHashList, KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheSpec
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry


QSA_RESIDENT_PAGE_TOKENS = 16
QSA_NUM_KV_HEADS = 2
QSA_HEAD_DIM = 256
QSA_DTYPE = torch.bfloat16
QSA_HISTORY_BUDGET_TOKENS = 65_536
QSA_ACTIVE_RESERVE_TOKENS = 1_024
QSA_HISTORY_PAGES = QSA_HISTORY_BUDGET_TOKENS // QSA_RESIDENT_PAGE_TOKENS
QSA_ACTIVE_PAGES = QSA_ACTIVE_RESERVE_TOKENS // QSA_RESIDENT_PAGE_TOKENS
QSA_PHYSICAL_PAGE_CAP = QSA_HISTORY_PAGES + QSA_ACTIVE_PAGES
QSA_LAYERS = 12


@dataclass(frozen=True, kw_only=True)
class QSAResidentContractSpec(AttentionSpec):
    """Full logical addressing with bounded physical accounting.

    max_num_blocks_per_req sizes the logical block-table row for complete
    history. max_memory_usage_bytes separately advertises only the bounded
    resident+active GPU pages. Runtime Q2C still needs a coordinator/manager
    that fills the logical row with resident block IDs and host-backed holes.
    """

    physical_page_cap: int = QSA_PHYSICAL_PAGE_CAP

    @property
    def prefix_cacheable(self) -> bool:
        return False

    def max_memory_usage_bytes(self, vllm_config: Any) -> int:
        del vllm_config
        return int(self.physical_page_cap) * int(self.page_size_bytes)

    def max_num_blocks_per_req(self, vllm_config: Any, max_len: int) -> int:
        del vllm_config
        return cdiv(int(max_len), int(self.block_size))


class QSAResidentRegistryProbeManager(SingleTypeKVCacheManager):
    """Registry-shape probe only; not the runtime residency policy."""

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        del running_request_id
        return 0

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        drop_eagle_block: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        del (
            block_hashes,
            max_length,
            block_pool,
            kv_cache_spec,
            drop_eagle_block,
            alignment_tokens,
            dcp_world_size,
            pcp_world_size,
        )
        return tuple([] for _ in kv_cache_group_ids), 0


def make_qsa_resident_contract_spec(
    *, physical_page_cap: int = QSA_PHYSICAL_PAGE_CAP
) -> QSAResidentContractSpec:
    return QSAResidentContractSpec(
        block_size=QSA_RESIDENT_PAGE_TOKENS,
        num_kv_heads=QSA_NUM_KV_HEADS,
        head_size=QSA_HEAD_DIM,
        head_size_v=QSA_HEAD_DIM,
        dtype=QSA_DTYPE,
        physical_page_cap=int(physical_page_cap),
    )


def register_qsa_resident_contract() -> None:
    KVCacheSpecRegistry.register(
        QSAResidentContractSpec,
        QSAResidentRegistryProbeManager,
        uniform_type_base_spec=QSAResidentContractSpec,
    )


def q2c_geometry(max_len: int = 161_000) -> dict[str, int | float]:
    spec = make_qsa_resident_contract_spec()
    logical_pages = spec.max_num_blocks_per_req(None, max_len)
    full_physical_bytes = logical_pages * spec.page_size_bytes
    bounded_physical_bytes = spec.max_memory_usage_bytes(None)
    return {
        "max_len": int(max_len),
        "resident_page_tokens": QSA_RESIDENT_PAGE_TOKENS,
        "page_size_bytes": int(spec.page_size_bytes),
        "logical_table_pages": int(logical_pages),
        "history_pages": QSA_HISTORY_PAGES,
        "active_pages": QSA_ACTIVE_PAGES,
        "physical_page_cap": QSA_PHYSICAL_PAGE_CAP,
        "bounded_bytes_per_layer": int(bounded_physical_bytes),
        "bounded_mib_per_layer": bounded_physical_bytes / 2**20,
        "bounded_gib_all_qsa_layers": bounded_physical_bytes * QSA_LAYERS / 2**30,
        "full_bytes_per_layer_at_max_len": int(full_physical_bytes),
        "full_to_bounded_ratio": full_physical_bytes / bounded_physical_bytes,
        "logical_minus_physical_pages": int(logical_pages - QSA_PHYSICAL_PAGE_CAP),
    }


__all__ = [
    "QSAResidentContractSpec",
    "QSAResidentRegistryProbeManager",
    "make_qsa_resident_contract_spec",
    "register_qsa_resident_contract",
    "q2c_geometry",
]
