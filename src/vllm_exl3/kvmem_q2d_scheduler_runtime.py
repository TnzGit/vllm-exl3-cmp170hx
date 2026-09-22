"""Scheduler-owned QSA write window for Q2D CPU-authoritative history."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
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


def _write_event(payload: dict[str, Any]) -> None:
    path = os.environ.get("VLLM_QWEN_KVMEM_Q2D_SCHED_STATS_PATH")
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, separators=(",", ":")) + "\n")


def validate_streaming_plan(plan: dict[str, Any]) -> dict[str, Any]:
    if int(plan.get("schema", 0)) != 1:
        raise ValueError("Q2D runtime plan schema mismatch")
    if plan.get("mode") != "qsa_cpu_reload_runtime":
        raise ValueError("Q2D runtime plan mode mismatch")
    frozen = {
        "page_tokens": 16,
        "physical_page_count": 4160,
        "write_page_count": 64,
        "read_cache_page_count": 4096,
        "scheduler_chunk_tokens": 1024,
        "query_row_batch": 64,
        "staging_pages": 128,
        "max_model_len": 161000,
    }
    for field, expected in frozen.items():
        if int(plan.get(field, 0)) != expected:
            raise ValueError(f"Q2D runtime {field} must equal {expected}")
    if int(plan["write_page_count"]) + int(plan["read_cache_page_count"]) != int(
        plan["physical_page_count"]
    ):
        raise ValueError("Q2D read/write partition does not equal physical cap")
    if int(plan.get("cpu_page_count", 0)) < math.ceil(
        int(plan["max_model_len"]) / int(plan["page_tokens"])
    ):
        raise ValueError("Q2D CPU history cannot cover max model length")
    return dict(plan)


def load_streaming_plan(path: str | os.PathLike[str]) -> dict[str, Any]:
    return validate_streaming_plan(json.loads(Path(path).read_text()))


@dataclass(frozen=True, kw_only=True)
class QSAStreamingRuntimeSpec(AttentionSpec):
    physical_page_cap: int
    write_page_count: int

    @property
    def prefix_cacheable(self) -> bool:
        return False

    @property
    def virtual_null_block_id(self) -> int:
        return self.physical_page_cap

    @property
    def dedicated_page_count(self) -> int:
        return self.physical_page_cap + 1

    def max_memory_usage_bytes(self, vllm_config: Any) -> int:
        del vllm_config
        # The model-owned 4160+null tensor is already included in profiling.
        # Generic hybrid metadata retains only one placeholder page.
        return self.page_size_bytes

    def max_num_blocks_per_req(self, vllm_config: Any, max_len: int) -> int:
        del vllm_config
        return cdiv(max_len, self.block_size)


class QSAStreamingRuntimeManager(SingleTypeKVCacheManager):
    """Own only the current <=64 QSA write pages on behalf of scheduler."""

    def __init__(self, kv_cache_spec: QSAStreamingRuntimeSpec, **kwargs) -> None:
        kwargs["enable_caching"] = False
        super().__init__(kv_cache_spec, **kwargs)
        self.spec = kv_cache_spec
        self._processed_tokens: dict[str, int] = {}
        self._peak_real_pages: dict[str, int] = {}
        self._virtual_blocks = [
            KVCacheBlock(block_id=i) for i in range(kv_cache_spec.write_page_count)
        ]
        self._virtual_null_block = KVCacheBlock(
            block_id=kv_cache_spec.virtual_null_block_id, is_null=True
        )
        self._virtual_free_ids = list(reversed(range(kv_cache_spec.write_page_count)))
        self._virtual_free_set = set(range(kv_cache_spec.write_page_count))
        self._virtual_generation = [0] * kv_cache_spec.write_page_count
        self._null_block = self._virtual_null_block
        self._record_new_block_ids = False
        self.new_block_ids = []

    def _required_pages(self, num_tokens: int) -> int:
        return cdiv(int(num_tokens), self.block_size)

    def _real_count(self, request_id: str) -> int:
        return sum(1 for block in self.req_to_blocks.get(request_id, ()) if not block.is_null)

    def _alloc_virtual(self, count: int) -> list[KVCacheBlock]:
        if count > len(self._virtual_free_ids):
            raise RuntimeError(
                f"Q2D write pool exhausted need={count} free={len(self._virtual_free_ids)}"
            )
        out = []
        for _ in range(count):
            block_id = self._virtual_free_ids.pop()
            if block_id not in self._virtual_free_set:
                raise RuntimeError("Q2D write free-list corruption")
            self._virtual_free_set.remove(block_id)
            self._virtual_generation[block_id] += 1
            block = self._virtual_blocks[block_id]
            if block.ref_cnt != 0:
                raise RuntimeError("Q2D write block was not free")
            block.ref_cnt = 1
            block.reset_hash()
            out.append(block)
        return out

    def _free_virtual(self, blocks: Sequence[KVCacheBlock]) -> None:
        for block in blocks:
            if block.is_null:
                continue
            block_id = int(block.block_id)
            if not 0 <= block_id < self.spec.write_page_count:
                raise RuntimeError(f"Q2D invalid write block id {block_id}")
            if block_id in self._virtual_free_set or block.ref_cnt != 1:
                raise RuntimeError(f"Q2D write double-free/corruption id={block_id}")
            block.ref_cnt = 0
            block.reset_hash()
            self._virtual_free_set.add(block_id)
            self._virtual_free_ids.append(block_id)

    def get_num_blocks_to_allocate(
        self, request_id: str, num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock], total_computed_tokens: int,
        num_local_computed_tokens: int, num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        del request_id, num_tokens, total_computed_tokens, num_local_computed_tokens
        del num_tokens_main_model, apply_admission_cap
        assert not new_computed_blocks
        return 0

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int, num_tokens_main_model: int
    ) -> list[KVCacheBlock]:
        del num_tokens_main_model
        required = self._required_pages(num_tokens)
        processed = int(self._processed_tokens.get(request_id, 0))
        work_start = min(processed // self.block_size, required)
        desired = list(range(work_start, required))
        if len(desired) > self.spec.write_page_count:
            raise RuntimeError(
                f"Q2D current write window {len(desired)} exceeds "
                f"{self.spec.write_page_count} pages"
            )
        blocks = self.req_to_blocks[request_id]
        if len(blocks) < required:
            blocks.extend([self._null_block] * (required - len(blocks)))
        missing = [idx for idx in desired if blocks[idx].is_null]
        fresh = self._alloc_virtual(len(missing))
        for idx, block in zip(missing, fresh, strict=True):
            blocks[idx] = block
        real = self._real_count(request_id)
        self._peak_real_pages[request_id] = max(
            real, self._peak_real_pages.get(request_id, 0)
        )
        if real > self.spec.write_page_count:
            raise RuntimeError("Q2D scheduler write ownership exceeded partition")
        if fresh:
            _write_event({
                "event": "q2d_scheduler_assign",
                "request_id": request_id,
                "logical_pages": missing,
                "write_ids": [int(block.block_id) for block in fresh],
                "generations": [
                    int(self._virtual_generation[int(block.block_id)]) for block in fresh
                ],
                "real_write_pages": real,
                "peak_real_write_pages": self._peak_real_pages[request_id],
                "write_page_cap": self.spec.write_page_count,
                "physical_page_cap": self.spec.physical_page_cap,
            })
        return fresh

    def remove_skipped_blocks(
        self, request_id: str, processed_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        del num_prompt_tokens
        processed = max(0, int(processed_computed_tokens))
        self._processed_tokens[request_id] = processed
        blocks = self.req_to_blocks.get(request_id)
        if not blocks:
            return
        fully_processed = min(processed // self.block_size, len(blocks))
        freed = []
        logical = []
        for idx in range(fully_processed):
            block = blocks[idx]
            if block.is_null:
                continue
            freed.append(block)
            logical.append(idx)
            blocks[idx] = self._null_block
        if freed:
            self._free_virtual(freed)
            _write_event({
                "event": "q2d_scheduler_reclaim",
                "request_id": request_id,
                "processed_computed_tokens": processed,
                "freed_logical_pages": logical,
                "freed_write_ids": [int(block.block_id) for block in freed],
                "freed_pages": len(freed),
                "real_write_pages": self._real_count(request_id),
                "peak_real_write_pages": self._peak_real_pages.get(request_id, 0),
                "write_page_cap": self.spec.write_page_count,
                "physical_page_cap": self.spec.physical_page_cap,
            })

    def add_local_computed_blocks(
        self, request_id: str, new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int, num_external_computed_tokens: int,
    ) -> None:
        assert not new_computed_blocks
        if num_local_computed_tokens or num_external_computed_tokens:
            raise RuntimeError("Q2D requires prefix cache OFF")

    def allocate_external_computed_blocks(
        self, request_id: str, num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        if num_external_computed_tokens:
            raise RuntimeError("Q2D does not accept connector hits")

    def cache_blocks(self, request, num_tokens: int, retention_interval=None) -> None:
        return None

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        return 0

    @classmethod
    def find_longest_cache_hit(
        cls, block_hashes: BlockHashList, max_length: int,
        kv_cache_group_ids: list[int], block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec, drop_eagle_block: bool,
        alignment_tokens: int, dcp_world_size: int = 1, pcp_world_size: int = 1,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        return tuple([] for _ in kv_cache_group_ids), 0

    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
        blocks = self.req_to_blocks.pop(request_id, [])
        self._free_virtual([block for block in blocks if not block.is_null])
        self.num_cached_block.pop(request_id, None)
        self._partial_hit_reqs.pop(request_id, None)
        self._processed_tokens.pop(request_id, None)
        self._peak_real_pages.pop(request_id, None)
        return []

    def take_new_block_ids(self) -> list[int]:
        return []


def make_qsa_streaming_spec(plan: dict[str, Any]) -> QSAStreamingRuntimeSpec:
    p = validate_streaming_plan(plan)
    return QSAStreamingRuntimeSpec(
        block_size=int(p["page_tokens"]),
        num_kv_heads=2,
        head_size=256,
        head_size_v=256,
        dtype=torch.bfloat16,
        physical_page_cap=int(p["physical_page_count"]),
        write_page_count=int(p["write_page_count"]),
    )


def register_qsa_streaming_spec() -> None:
    KVCacheSpecRegistry._ensure_registered()
    KVCacheSpecRegistry.register(
        QSAStreamingRuntimeSpec,
        QSAStreamingRuntimeManager,
        uniform_type_base_spec=QSAStreamingRuntimeSpec,
    )


register_qsa_streaming_spec()
