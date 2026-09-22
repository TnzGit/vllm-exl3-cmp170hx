"""K1-Q2C live scheduler ownership for bounded QSA history.

Research-only transition manager for vLLM 0.29. Before the configured
transition boundary it behaves like full attention at 16-token page geometry.
Once the scheduler has safely processed that boundary it releases all
non-resident historical pages, preserves a full logical block-table row with
null holes, and keeps only the frozen sticky history plus bounded active pages.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Sequence

import torch

from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHashList, KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheSpec
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry


@dataclass(frozen=True, kw_only=True)
class QSAResidentRuntimeSpec(AttentionSpec):
    resident_pages: tuple[int, ...]
    shrink_from_pos: int
    active_page0: int
    active_reserve_pages: int

    @property
    def prefix_cacheable(self) -> bool:
        return False

    @property
    def physical_page_cap(self) -> int:
        return len(self.resident_pages) + self.active_reserve_pages

    def max_memory_usage_bytes(self, vllm_config: Any) -> int:
        del vllm_config
        return self.physical_page_cap * self.page_size_bytes

    def max_num_blocks_per_req(self, vllm_config: Any, max_len: int) -> int:
        del vllm_config
        return cdiv(max_len, self.block_size)


class QSAResidentRuntimeManager(SingleTypeKVCacheManager):
    """Transition from full logical ownership to bounded real GPU pages."""

    def __init__(self, kv_cache_spec: QSAResidentRuntimeSpec, **kwargs) -> None:
        kwargs["enable_caching"] = False
        super().__init__(kv_cache_spec, **kwargs)
        self.spec = kv_cache_spec
        self._resident_set = frozenset(int(x) for x in kv_cache_spec.resident_pages)
        self._shrunk: set[str] = set()

    def _required_pages(self, num_tokens: int) -> int:
        return cdiv(int(num_tokens), self.block_size)

    def _active_end_page(self, required_pages: int) -> int:
        limit = self.spec.active_page0 + self.spec.active_reserve_pages
        if required_pages > limit:
            raise RuntimeError(
                "Q2C active suffix exceeded reserve: "
                f"required_pages={required_pages} limit={limit}"
            )
        return required_pages

    def _desired_after_shrink(self, required_pages: int) -> list[int]:
        active_end = self._active_end_page(required_pages)
        hist = [p for p in self.spec.resident_pages if p < required_pages]
        active = list(range(self.spec.active_page0, active_end))
        return hist + active

    def _real_count(self, request_id: str) -> int:
        return sum(
            1 for b in self.req_to_blocks.get(request_id, ()) if not b.is_null
        )

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        if request_id not in self._shrunk:
            return super().get_num_blocks_to_allocate(
                request_id, num_tokens, new_computed_blocks,
                total_computed_tokens, num_local_computed_tokens,
                num_tokens_main_model, apply_admission_cap=apply_admission_cap,
            )
        assert not new_computed_blocks
        required = self._required_pages(num_tokens)
        desired = self._desired_after_shrink(required)
        blocks = self.req_to_blocks.get(request_id, ())
        missing = 0
        for idx in desired:
            if idx >= len(blocks) or blocks[idx].is_null:
                missing += 1
        return missing

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int, num_tokens_main_model: int
    ) -> list[KVCacheBlock]:
        del num_tokens_main_model
        if request_id not in self._shrunk:
            return super().allocate_new_blocks(request_id, num_tokens, num_tokens)
        required = self._required_pages(num_tokens)
        desired = self._desired_after_shrink(required)
        blocks = self.req_to_blocks[request_id]
        if len(blocks) < required:
            blocks.extend([self._null_block] * (required - len(blocks)))
        missing = [idx for idx in desired if blocks[idx].is_null]
        fresh = self.block_pool.get_new_blocks(len(missing)) if missing else []
        for idx, block in zip(missing, fresh, strict=True):
            blocks[idx] = block
        if self._record_new_block_ids:
            self.new_block_ids.extend(b.block_id for b in fresh)
        if self._real_count(request_id) > self.spec.physical_page_cap:
            raise RuntimeError("Q2C physical page cap exceeded after shrink")
        return fresh

    def remove_skipped_blocks(
        self,
        request_id: str,
        processed_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        del num_prompt_tokens
        if request_id in self._shrunk:
            return
        if processed_computed_tokens < self.spec.shrink_from_pos:
            return
        blocks = self.req_to_blocks.get(request_id)
        if not blocks:
            return
        before = self._real_count(request_id)
        hist_end = min(self.spec.active_page0, len(blocks))
        freed: list[KVCacheBlock] = []
        for idx in range(hist_end):
            block = blocks[idx]
            if idx in self._resident_set or block.is_null:
                continue
            freed.append(block)
            blocks[idx] = self._null_block
        if freed:
            self.block_pool.free_blocks(reversed(freed))
        self._shrunk.add(request_id)
        after = self._real_count(request_id)
        if after > self.spec.physical_page_cap:
            raise RuntimeError(
                f"Q2C transition left {after} real pages above cap "
                f"{self.spec.physical_page_cap}"
            )
        _write_scheduler_event({
            "event": "q2c_scheduler_shrink",
            "request_id": request_id,
            "processed_computed_tokens": int(processed_computed_tokens),
            "logical_row_pages": len(blocks),
            "real_pages_before": before,
            "real_pages_after": after,
            "freed_pages": len(freed),
            "physical_page_cap": self.spec.physical_page_cap,
            "resident_history_pages": len(self.spec.resident_pages),
            "active_reserve_pages": self.spec.active_reserve_pages,
        })

    def add_local_computed_blocks(
        self, request_id: str, new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int, num_external_computed_tokens: int
    ) -> None:
        assert not new_computed_blocks
        if num_local_computed_tokens or num_external_computed_tokens:
            raise RuntimeError("Q2C runtime prototype requires prefix cache OFF")

    def allocate_external_computed_blocks(
        self, request_id: str, num_local_computed_tokens: int,
        num_external_computed_tokens: int
    ) -> None:
        if num_external_computed_tokens:
            raise RuntimeError("Q2C runtime prototype does not accept connector hits")

    def cache_blocks(self, request, num_tokens: int, retention_interval=None) -> None:
        return None

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        return 0

    @classmethod
    def find_longest_cache_hit(
        cls, block_hashes: BlockHashList, max_length: int,
        kv_cache_group_ids: list[int], block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec, drop_eagle_block: bool,
        alignment_tokens: int, dcp_world_size: int = 1, pcp_world_size: int = 1
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        return tuple([] for _ in kv_cache_group_ids), 0

    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
        blocks = self.req_to_blocks.pop(request_id, [])
        self.num_cached_block.pop(request_id, None)
        self._partial_hit_reqs.pop(request_id, None)
        self._shrunk.discard(request_id)
        return [b for b in blocks if not b.is_null]


def _write_scheduler_event(payload: dict[str, Any]) -> None:
    path = os.environ.get("VLLM_QWEN_KVMEM_Q2C_SCHED_STATS_PATH")
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, separators=(",", ":")) + "\n")


def validate_runtime_plan(plan: dict[str, Any]) -> dict[str, Any]:
    if int(plan.get("schema", 0)) != 1:
        raise ValueError("Q2C plan schema mismatch")
    if plan.get("mode") != "qsa_scheduler_owned_transition":
        raise ValueError("Q2C runtime plan mode mismatch")
    page_tokens = int(plan["page_tokens"])
    if page_tokens != 16:
        raise ValueError("Q2C runtime requires 16-token pages")
    resident = tuple(int(x) for x in plan["resident_pages"])
    if len(resident) != int(plan["resident_page_count"]) or len(set(resident)) != len(resident):
        raise ValueError("Q2C resident page list mismatch")
    active_page0 = int(plan["active_page0"])
    reserve = int(plan["active_reserve_pages"])
    if any(p < 0 or p >= active_page0 for p in resident):
        raise ValueError("Q2C historical resident pages must precede active suffix")
    if len(resident) + reserve != int(plan["physical_page_count"]):
        raise ValueError("Q2C physical page count mismatch")
    return dict(plan, resident_pages=resident)


def load_runtime_plan(path: str | os.PathLike[str]) -> dict[str, Any]:
    return validate_runtime_plan(json.loads(Path(path).read_text()))


def make_qsa_runtime_spec(plan: dict[str, Any]) -> QSAResidentRuntimeSpec:
    p = validate_runtime_plan(plan)
    return QSAResidentRuntimeSpec(
        block_size=int(p["page_tokens"]),
        num_kv_heads=2, head_size=256, head_size_v=256,
        dtype=torch.bfloat16,
        resident_pages=tuple(p["resident_pages"]),
        shrink_from_pos=int(p["apply_min_pos"]),
        active_page0=int(p["active_page0"]),
        active_reserve_pages=int(p["active_reserve_pages"]),
    )


def register_qsa_runtime_spec() -> None:
    KVCacheSpecRegistry._ensure_registered()
    KVCacheSpecRegistry.register(
        QSAResidentRuntimeSpec, QSAResidentRuntimeManager,
        uniform_type_base_spec=QSAResidentRuntimeSpec,
    )


register_qsa_runtime_spec()
