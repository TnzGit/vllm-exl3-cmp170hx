"""K1-Q2C live scheduler ownership for bounded QSA history.

Research-only frozen-plan manager for vLLM 0.29. QSA keeps full logical history
at 16-token page geometry while processed nonresident historical pages are
progressively reclaimed. Real QSA page IDs live in a dedicated virtual pool,
not the shared hybrid/Mamba BlockPool; the worker uses model-owned dedicated KV
storage and masks reclaimed logical history before attention.
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

    @property
    def virtual_null_block_id(self) -> int:
        return self.physical_page_cap

    @property
    def dedicated_page_count(self) -> int:
        # One extra zero page backs all logical null holes on the worker.
        return self.physical_page_cap + 1

    @property
    def dedicated_memory_bytes_per_layer(self) -> int:
        return self.dedicated_page_count * self.page_size_bytes

    def max_memory_usage_bytes(self, vllm_config: Any) -> int:
        # Generic vLLM 0.29 uses one shared BlockPool and charges every block
        # in units of the widest hybrid/Mamba group. Q2C's 4,160 physical pages
        # live in a dedicated per-layer tensor instead, allocated during model
        # construction and therefore already included in the worker memory
        # profile. Keep exactly one normal-geometry placeholder page in the
        # generic KV config so metadata/binding machinery still sees this group.
        del vllm_config
        return self.page_size_bytes

    def max_num_blocks_per_req(self, vllm_config: Any, max_len: int) -> int:
        del vllm_config
        return cdiv(max_len, self.block_size)


class QSAResidentRuntimeManager(SingleTypeKVCacheManager):
    """Progressively reclaim processed nonresident QSA pages.

    The full logical row grows with sequence length. Real GPU ownership is the
    frozen sticky resident history plus the current unprocessed prefill window;
    after the transition boundary it is sticky history plus the bounded active
    suffix. This avoids ever requiring the full 160K history to remain real.
    """

    def __init__(self, kv_cache_spec: QSAResidentRuntimeSpec, **kwargs) -> None:
        kwargs["enable_caching"] = False
        super().__init__(kv_cache_spec, **kwargs)
        self.spec = kv_cache_spec
        self._resident_set = frozenset(int(x) for x in kv_cache_spec.resident_pages)
        self._processed_tokens: dict[str, int] = {}
        self._boundary_emitted: set[str] = set()
        self._peak_real_pages: dict[str, int] = {}

        # QSA virtual page IDs index the dedicated QSA tensor directly. They
        # must never enter vLLM's shared hybrid/Mamba BlockPool accounting,
        # zeroing, copying, or free queue.
        self._virtual_blocks = [
            KVCacheBlock(block_id=i) for i in range(kv_cache_spec.physical_page_cap)
        ]
        self._virtual_null_block = KVCacheBlock(
            block_id=kv_cache_spec.virtual_null_block_id,
            is_null=True,
        )
        self._virtual_free_ids = list(
            reversed(range(kv_cache_spec.physical_page_cap))
        )
        self._virtual_free_set = set(range(kv_cache_spec.physical_page_cap))
        self._null_block = self._virtual_null_block
        self._record_new_block_ids = False
        self.new_block_ids = []

    def _required_pages(self, num_tokens: int) -> int:
        return cdiv(int(num_tokens), self.block_size)

    def _work_start_page(self, request_id: str) -> int:
        processed = int(self._processed_tokens.get(request_id, 0))
        if processed >= self.spec.shrink_from_pos:
            return self.spec.active_page0
        return processed // self.block_size

    def _desired_pages(self, request_id: str, required_pages: int) -> list[int]:
        active_limit = self.spec.active_page0 + self.spec.active_reserve_pages
        if required_pages > active_limit and self._processed_tokens.get(request_id, 0) >= self.spec.shrink_from_pos:
            raise RuntimeError(
                "Q2C active suffix exceeded reserve: "
                f"required_pages={required_pages} limit={active_limit}"
            )

        # Frozen sticky pages are retained once they have been allocated.
        hist = [p for p in self.spec.resident_pages if p < required_pages]

        # Before the frozen query boundary, all not-yet-processed pages in the
        # current scheduler chunk remain real so causal prefill can read/write
        # them. Once at the boundary, the active suffix is the only work range.
        work_start = min(self._work_start_page(request_id), required_pages)
        work = list(range(work_start, required_pages))
        return sorted(set(hist).union(work))

    def _real_count(self, request_id: str) -> int:
        return sum(
            1 for b in self.req_to_blocks.get(request_id, ()) if not b.is_null
        )

    def _record_peak(self, request_id: str) -> int:
        value = self._real_count(request_id)
        self._peak_real_pages[request_id] = max(
            value, self._peak_real_pages.get(request_id, 0)
        )
        return value

    def _alloc_virtual(self, count: int) -> list[KVCacheBlock]:
        if count < 0:
            raise ValueError("virtual allocation count must be non-negative")
        if count > len(self._virtual_free_ids):
            raise RuntimeError(
                "Q2C dedicated virtual page pool exhausted: "
                f"need={count} free={len(self._virtual_free_ids)} "
                f"cap={self.spec.physical_page_cap}"
            )
        out: list[KVCacheBlock] = []
        for _ in range(count):
            block_id = self._virtual_free_ids.pop()
            if block_id not in self._virtual_free_set:
                raise RuntimeError("Q2C virtual free-list corruption")
            self._virtual_free_set.remove(block_id)
            block = self._virtual_blocks[block_id]
            if block.ref_cnt != 0 or block.is_null:
                raise RuntimeError("Q2C virtual block was not free")
            block.ref_cnt = 1
            block.reset_hash()
            out.append(block)
        return out

    def _free_virtual(self, blocks: Sequence[KVCacheBlock]) -> None:
        for block in blocks:
            if block.is_null:
                continue
            block_id = int(block.block_id)
            if not 0 <= block_id < self.spec.physical_page_cap:
                raise RuntimeError(f"Q2C invalid virtual block id {block_id}")
            if block_id in self._virtual_free_set or block.ref_cnt != 1:
                raise RuntimeError(f"Q2C virtual double-free/corruption id={block_id}")
            block.ref_cnt = 0
            block.reset_hash()
            self._virtual_free_set.add(block_id)
            self._virtual_free_ids.append(block_id)

    @property
    def virtual_free_pages(self) -> int:
        return len(self._virtual_free_ids)

    def _maybe_emit_boundary(
        self, request_id: str, *, reached_tokens: int, trigger: str
    ) -> None:
        if (
            reached_tokens < self.spec.shrink_from_pos
            or request_id in self._boundary_emitted
        ):
            return
        blocks = self.req_to_blocks.get(request_id, ())
        real = self._real_count(request_id)
        if real > self.spec.physical_page_cap:
            raise RuntimeError(
                f"Q2C boundary has {real} real pages above cap "
                f"{self.spec.physical_page_cap}"
            )
        self._boundary_emitted.add(request_id)
        _write_scheduler_event({
            "event": "q2c_scheduler_boundary",
            "request_id": request_id,
            "boundary_trigger": trigger,
            "boundary_reached_tokens": int(reached_tokens),
            "processed_computed_tokens": int(
                self._processed_tokens.get(request_id, 0)
            ),
            "logical_row_pages": len(blocks),
            "real_pages_at_boundary": real,
            "physical_page_cap": self.spec.physical_page_cap,
            "resident_history_pages": len(self.spec.resident_pages),
            "active_reserve_pages": self.spec.active_reserve_pages,
            "peak_real_pages": self._peak_real_pages.get(request_id, real),
        })

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
        # This return value is deliberately the number of *shared BlockPool*
        # blocks required. QSA pages come from the dedicated virtual pool, so
        # they cost zero shared blocks. Capacity is hard-enforced by
        # _alloc_virtual and the 4,160-page peak invariant.
        del (
            request_id,
            num_tokens,
            total_computed_tokens,
            num_local_computed_tokens,
            num_tokens_main_model,
            apply_admission_cap,
        )
        assert not new_computed_blocks
        return 0

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int, num_tokens_main_model: int
    ) -> list[KVCacheBlock]:
        del num_tokens_main_model
        required = self._required_pages(num_tokens)
        desired = self._desired_pages(request_id, required)
        blocks = self.req_to_blocks[request_id]
        if len(blocks) < required:
            blocks.extend([self._null_block] * (required - len(blocks)))
        missing = [idx for idx in desired if blocks[idx].is_null]
        fresh = self._alloc_virtual(len(missing))
        for idx, block in zip(missing, fresh, strict=True):
            blocks[idx] = block

        previous_peak = self._peak_real_pages.get(request_id, 0)
        real = self._record_peak(request_id)
        if real > previous_peak:
            _write_scheduler_event({
                "event": "q2c_scheduler_peak",
                "request_id": request_id,
                "logical_row_pages": len(blocks),
                "real_pages": real,
                "peak_real_pages": real,
                "physical_page_cap": self.spec.physical_page_cap,
                "virtual_free_pages": self.virtual_free_pages,
            })
        hard_peak = self.spec.physical_page_cap
        if real > hard_peak:
            raise RuntimeError(
                f"Q2C scheduler real-page peak {real} exceeds guarded peak {hard_peak}"
            )
        # A request can finish in the same scheduler step that first reaches
        # the frozen query boundary. In that case vLLM does not call
        # remove_skipped_blocks() again, so allocation is the last authoritative
        # scheduler lifecycle point at which boundary ownership can be emitted.
        self._maybe_emit_boundary(
            request_id,
            reached_tokens=int(num_tokens),
            trigger="allocation_reaches_boundary",
        )

        # These virtual blocks MUST be returned: worker block-table state is
        # updated from per-group new_block_ids. They are intentionally absent
        # from take_new_block_ids(), so global KV zeroing never sees them.
        return fresh

    def remove_skipped_blocks(
        self,
        request_id: str,
        processed_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        del num_prompt_tokens
        processed = max(0, int(processed_computed_tokens))
        self._processed_tokens[request_id] = processed

        blocks = self.req_to_blocks.get(request_id)
        if not blocks:
            return

        before = self._real_count(request_id)
        fully_processed_pages = min(processed // self.block_size, len(blocks))
        historical_end = min(
            fully_processed_pages,
            self.spec.active_page0,
        )
        freed: list[KVCacheBlock] = []
        for idx in range(historical_end):
            block = blocks[idx]
            if idx in self._resident_set or block.is_null:
                continue
            freed.append(block)
            blocks[idx] = self._null_block
        if freed:
            self._free_virtual(freed)
        after = self._record_peak(request_id)

        if freed:
            _write_scheduler_event({
                "event": "q2c_scheduler_reclaim",
                "request_id": request_id,
                "processed_computed_tokens": processed,
                "logical_row_pages": len(blocks),
                "real_pages_before": before,
                "real_pages_after": after,
                "freed_pages": len(freed),
                "physical_page_cap": self.spec.physical_page_cap,
                "peak_real_pages": self._peak_real_pages.get(request_id, after),
            })

        self._maybe_emit_boundary(
            request_id,
            reached_tokens=processed,
            trigger="processed_reaches_boundary",
        )

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
        real = [b for b in blocks if not b.is_null]
        self._free_virtual(real)
        self.num_cached_block.pop(request_id, None)
        self._partial_hit_reqs.pop(request_id, None)
        self._processed_tokens.pop(request_id, None)
        self._boundary_emitted.discard(request_id)
        self._peak_real_pages.pop(request_id, None)
        # Virtual QSA blocks are already recycled internally. Returning them
        # would incorrectly hand them to the shared hybrid/Mamba BlockPool.
        return []

    def take_new_block_ids(self) -> list[int]:
        # Global KV zeroing must never index the shared backing with virtual
        # QSA IDs. The dedicated cache is written explicitly by QSA.
        return []



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
    chunk_tokens = int(plan.get("scheduler_chunk_tokens", 0))
    if chunk_tokens != reserve * page_tokens or chunk_tokens != 1024:
        raise ValueError(
            "Q2C scheduler chunk must equal the 1024-token active reserve"
        )
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
