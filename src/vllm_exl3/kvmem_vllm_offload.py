"""Thin vLLM-0.29 CPU backing adapter for K1 logical KV pages.

The adapter deliberately reuses the installed generic CPU offload manager and
worker, while leaving sparse residency policy to kvmem_resident.py.

Current K1-T scope is one KV cache group and blocks_per_chunk == 1.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time
from typing import Any, Sequence


@dataclass(frozen=True, slots=True)
class TransferObservation:
    job_id: int
    transfer_bytes: int
    event_seconds: float
    wall_seconds: float

    @property
    def event_gib_s(self) -> float:
        if self.event_seconds <= 0:
            return float("inf")
        return (self.transfer_bytes / (1024**3)) / self.event_seconds

    @property
    def wall_gib_s(self) -> float:
        if self.wall_seconds <= 0:
            return float("inf")
        return (self.transfer_bytes / (1024**3)) / self.wall_seconds


def stable_page_digest(lineage: str, logical_page: int) -> bytes:
    if not lineage:
        raise ValueError("lineage must be non-empty")
    if logical_page < 0:
        raise ValueError("logical_page must be non-negative")
    return hashlib.sha256(
        f"k1:{lineage}:logical-page:{logical_page}".encode()
    ).digest()


class VllmCPUPageBacking:
    """Single-group K1-T adapter over vLLM generic CPU offload primitives."""

    def __init__(
        self,
        *,
        kv_caches: Any,
        num_cpu_blocks: int,
        lineage: str,
        group_idx: int = 0,
    ) -> None:
        if num_cpu_blocks <= 0:
            raise ValueError("num_cpu_blocks must be positive")
        if group_idx < 0:
            raise ValueError("group_idx must be non-negative")

        from vllm.v1.kv_offload.base import ReqContext
        from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
        from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

        self.lineage = lineage
        self.group_idx = group_idx
        self.manager = CPUOffloadingManager(
            num_blocks=num_cpu_blocks,
            cache_policy="lru",
        )
        self.worker = CPUOffloadingWorker(
            kv_caches=kv_caches,
            blocks_per_chunk=1,
            num_cpu_blocks=num_cpu_blocks,
            mmap_region=None,
            canonical_layout=False,
        )
        self.req_context = ReqContext(req_id=f"k1:{lineage}")
        self._next_job_id = 1
        self._closed = False

    def _key(self, logical_page: int):
        from vllm.v1.kv_offload.base import make_offload_key

        return make_offload_key(
            stable_page_digest(self.lineage, logical_page),
            self.group_idx,
        )

    def _job_id(self) -> int:
        job_id = self._next_job_id
        self._next_job_id += 1
        return job_id

    def _finished(self, job_id: int) -> Any:
        rows = self.worker.get_finished()
        match = [row for row in rows if row.job_id == job_id]
        if len(match) != 1:
            raise RuntimeError(
                f"expected one completed transfer for job {job_id}; "
                f"got {[row.job_id for row in rows]}"
            )
        result = match[0]
        if not result.success:
            raise RuntimeError(f"offload transfer job {job_id} failed")
        if result.transfer_size is None or result.transfer_time is None:
            raise RuntimeError(f"offload transfer job {job_id} lacks timing metadata")
        return result

    @staticmethod
    def _gpu_spec(block_ids: Sequence[int]):
        from vllm.v1.kv_offload.base import GPULoadStoreSpec

        return GPULoadStoreSpec(
            list(block_ids),
            group_sizes=[len(block_ids)],
            block_indices=[0],
        )

    def publish(
        self,
        logical_pages: Sequence[int],
        source_gpu_pages: Sequence[int],
    ) -> TransferObservation:
        if self._closed:
            raise RuntimeError("backing adapter is closed")
        if len(logical_pages) != len(source_gpu_pages):
            raise ValueError("logical/source page counts disagree")
        if not logical_pages:
            raise ValueError("publish requires at least one page")

        keys = [self._key(int(page)) for page in logical_pages]
        source_by_key = {
            key: int(gpu_page)
            for key, gpu_page in zip(keys, source_gpu_pages, strict=True)
        }
        prepared = self.manager.prepare_store(keys, self.req_context)
        if prepared is None:
            raise RuntimeError("CPU offload manager refused page publication")

        store_keys = list(prepared.keys_to_store)
        if not store_keys:
            return TransferObservation(
                job_id=0,
                transfer_bytes=0,
                event_seconds=0.0,
                wall_seconds=0.0,
            )
        try:
            src_pages = [source_by_key[key] for key in store_keys]
        except KeyError as exc:
            raise RuntimeError("manager returned an unknown store key") from exc

        job_id = self._job_id()
        t0 = time.perf_counter()
        submitted = self.worker.submit_store(
            job_id,
            self._gpu_spec(src_pages),
            prepared.store_spec,
        )
        if not submitted:
            self.manager.complete_store(
                store_keys,
                self.req_context,
                success=False,
            )
            raise RuntimeError("CPU offload worker rejected store transfer")

        self.worker.wait({job_id})
        wall = time.perf_counter() - t0
        result = self._finished(job_id)
        self.manager.complete_store(
            store_keys,
            self.req_context,
            success=True,
        )
        return TransferObservation(
            job_id=job_id,
            transfer_bytes=int(result.transfer_size),
            event_seconds=float(result.transfer_time),
            wall_seconds=wall,
        )

    def all_present(self, logical_pages: Sequence[int]) -> bool:
        from vllm.v1.kv_offload.base import LookupResult

        return all(
            self.manager.lookup(self._key(int(page)), self.req_context)
            is LookupResult.HIT
            for page in logical_pages
        )

    def stage_in(
        self,
        logical_pages: Sequence[int],
        destination_gpu_pages: Sequence[int],
    ) -> TransferObservation:
        if self._closed:
            raise RuntimeError("backing adapter is closed")
        if len(logical_pages) != len(destination_gpu_pages):
            raise ValueError("logical/destination page counts disagree")
        if not logical_pages:
            raise ValueError("stage_in requires at least one page")

        keys = [self._key(int(page)) for page in logical_pages]
        if not self.all_present(logical_pages):
            raise RuntimeError("stage-in requested a page missing from CPU backing")

        cpu_spec = self.manager.prepare_load(keys, self.req_context)
        job_id = self._job_id()
        t0 = time.perf_counter()
        submitted = self.worker.submit_load(
            job_id,
            cpu_spec,
            self._gpu_spec(destination_gpu_pages),
        )
        if not submitted:
            self.manager.complete_load(keys, self.req_context)
            raise RuntimeError("CPU offload worker rejected load transfer")

        self.worker.wait({job_id})
        wall = time.perf_counter() - t0
        result = self._finished(job_id)
        self.manager.complete_load(keys, self.req_context)
        return TransferObservation(
            job_id=job_id,
            transfer_bytes=int(result.transfer_size),
            event_seconds=float(result.transfer_time),
            wall_seconds=wall,
        )

    def close(self) -> None:
        if not self._closed:
            self.worker.shutdown()
            self._closed = True

    def __enter__(self) -> "VllmCPUPageBacking":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
