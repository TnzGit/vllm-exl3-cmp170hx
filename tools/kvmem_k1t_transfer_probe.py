#!/usr/bin/env python3
"""K1-T real transfer-plane probe using vLLM 0.29 CPU offload primitives.

No model engine is started. The probe allocates a synthetic main-KV layout with
the measured Qwen geometry, publishes arbitrary noncontiguous logical history
pages through CPUOffloadingManager/CPUOffloadingWorker, then stages them into
sticky resident GPU slots and verifies every byte.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vllm_exl3.kvmem_resident import (
    ResidentGeometry,
    StickyResidentCoordinator,
)
from vllm_exl3.kvmem_transfer import (
    expand_resident_transition,
    page_bytes_per_layer,
    qsa_main_kv_bytes_per_token,
)


def _offload_key(logical_page: int):
    from vllm.v1.kv_offload.base import make_offload_key

    digest = hashlib.sha256(
        f"k1t-logical-page:{logical_page}".encode()
    ).digest()
    return make_offload_key(digest, 0)


def _take_result(worker, job_id: int):
    results = worker.get_finished()
    matches = [row for row in results if row.job_id == job_id]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one completed transfer for job {job_id}; "
            f"got {[row.job_id for row in results]}"
        )
    result = matches[0]
    if not result.success:
        raise RuntimeError(f"transfer job {job_id} reported failure")
    return result


def _fill_pattern(
    tensor: torch.Tensor,
    gpu_pages: list[int],
    logical_pages: list[int],
    *,
    layer_idx: int,
) -> None:
    if len(gpu_pages) != len(logical_pages):
        raise ValueError("page-id arrays disagree")
    if not gpu_pages:
        return
    width = tensor.shape[1]
    offsets = torch.arange(width, dtype=torch.int64, device=tensor.device)
    logical = torch.tensor(
        logical_pages,
        dtype=torch.int64,
        device=tensor.device,
    ).view(-1, 1)
    values = (logical * 17 + layer_idx * 31 + offsets.view(1, -1)) % 251
    values = values - 125
    tensor[gpu_pages] = values.to(torch.int8)


def _verify_pattern(
    tensor: torch.Tensor,
    gpu_pages: list[int],
    logical_pages: list[int],
    *,
    layer_idx: int,
) -> tuple[bool, int | None]:
    width = tensor.shape[1]
    offsets = torch.arange(width, dtype=torch.int64, device=tensor.device)
    logical = torch.tensor(
        logical_pages,
        dtype=torch.int64,
        device=tensor.device,
    ).view(-1, 1)
    expected = (
        (logical * 17 + layer_idx * 31 + offsets.view(1, -1)) % 251 - 125
    ).to(torch.int8)
    actual = tensor[gpu_pages]
    diff = actual != expected
    if not bool(diff.any().item()):
        return True, None
    flat = int(torch.nonzero(diff, as_tuple=False)[0, 0].item())
    return False, flat


def _primary_transition(
    *,
    capacity_regions: int,
    replacement_fraction: float,
):
    coordinator = StickyResidentCoordinator(
        capacity_blocks=capacity_regions,
        replacement_fraction=replacement_fraction,
    )
    coordinator.bootstrap(set(range(capacity_regions)))

    # Deliberately noncontiguous regions spread through a 240K history.
    candidates = (
        300,
        337,
        411,
        468,
        512,
        577,
        633,
        701,
        744,
        812,
        871,
        930,
    )
    replacement_cap = int(capacity_regions * replacement_fraction)
    if replacement_cap != len(candidates):
        raise ValueError(
            "probe currently expects the measured 64K/5% geometry "
            f"(replacement cap {len(candidates)}), got {replacement_cap}"
        )

    desired = set(range(capacity_regions - replacement_cap))
    desired.update(candidates)

    scores = {block: 1000.0 - block for block in range(capacity_regions)}
    for rank, block in enumerate(candidates):
        scores[block] = 10000.0 - rank
    scores[0] = 100000.0

    transition = coordinator.update(
        desired_blocks=desired,
        mandatory_blocks={0},
        scores=scores,
    )
    if transition.query_replacements != replacement_cap:
        raise RuntimeError("sticky coordinator did not use the expected 5% cap")
    return coordinator, transition


def run_probe(args: argparse.Namespace) -> dict:
    from vllm.v1.kv_offload.base import (
        CanonicalKVCaches,
        CanonicalKVCacheRef,
        CanonicalKVCacheTensor,
        GPULoadStoreSpec,
        LookupResult,
        ReqContext,
    )
    from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
    from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    geometry = ResidentGeometry(
        region_tokens=args.region_tokens,
        page_tokens=args.page_tokens,
    )
    if args.capacity_tokens % args.region_tokens:
        raise ValueError("capacity_tokens must be divisible by region_tokens")
    capacity_regions = args.capacity_tokens // args.region_tokens
    coordinator, transition = _primary_transition(
        capacity_regions=capacity_regions,
        replacement_fraction=args.replacement_fraction,
    )
    expanded = expand_resident_transition(transition, geometry)

    logical_pages = [move.logical_page for move in expanded.stage_in_pages]
    dst_gpu_pages = [move.physical_page for move in expanded.stage_in_pages]
    num_transfer_pages = len(logical_pages)
    resident_gpu_pages = capacity_regions * geometry.pages_per_region
    staging_gpu_pages = list(
        range(resident_gpu_pages, resident_gpu_pages + num_transfer_pages)
    )
    total_gpu_pages = resident_gpu_pages + num_transfer_pages

    per_layer_page_bytes = page_bytes_per_layer(
        page_tokens=args.page_tokens,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        dtype_bytes=args.dtype_bytes,
    )
    main_bytes_per_token = qsa_main_kv_bytes_per_token(
        qsa_layers=args.qsa_layers,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        dtype_bytes=args.dtype_bytes,
    )
    expected_transfer_bytes = (
        num_transfer_pages * per_layer_page_bytes * args.qsa_layers
    )

    gpu_tensors = [
        torch.empty(
            (total_gpu_pages, per_layer_page_bytes),
            dtype=torch.int8,
            device="cuda",
        )
        for _ in range(args.qsa_layers)
    ]

    kv_caches = CanonicalKVCaches(
        tensors=[
            CanonicalKVCacheTensor(
                tensor=tensor,
                page_size_bytes=per_layer_page_bytes,
            )
            for tensor in gpu_tensors
        ],
        group_data_refs=[
            [
                CanonicalKVCacheRef(
                    tensor_idx=layer,
                    page_size_bytes=per_layer_page_bytes,
                )
                for layer in range(args.qsa_layers)
            ]
        ],
    )

    manager = CPUOffloadingManager(
        num_blocks=num_transfer_pages,
        cache_policy="lru",
    )
    worker = CPUOffloadingWorker(
        kv_caches=kv_caches,
        blocks_per_chunk=1,
        num_cpu_blocks=num_transfer_pages,
        mmap_region=None,
        canonical_layout=False,
    )
    req = ReqContext(req_id="k1t-transfer-probe")
    keys = [_offload_key(page) for page in logical_pages]

    store_row = None
    load_rows = []
    verify_rows = []
    page_table_checks = []
    try:
        for layer, tensor in enumerate(gpu_tensors):
            _fill_pattern(
                tensor,
                staging_gpu_pages,
                logical_pages,
                layer_idx=layer,
            )

        prepared_store = manager.prepare_store(keys, req)
        if prepared_store is None:
            raise RuntimeError("CPU manager refused the initial KV publication")
        if prepared_store.keys_to_store != keys:
            raise RuntimeError("CPU manager changed initial key order/content")

        store_gpu_spec = GPULoadStoreSpec(
            staging_gpu_pages,
            group_sizes=[num_transfer_pages],
            block_indices=[0],
        )
        t0 = time.perf_counter()
        if not worker.submit_store(
            1,
            store_gpu_spec,
            prepared_store.store_spec,
        ):
            raise RuntimeError("CPUOffloadingWorker.submit_store returned false")
        worker.wait({1})
        store_wall = time.perf_counter() - t0
        store_result = _take_result(worker, 1)
        manager.complete_store(keys, req, success=True)
        store_row = {
            "bytes": store_result.transfer_size,
            "event_seconds": store_result.transfer_time,
            "wall_seconds": store_wall,
        }

        if any(manager.lookup(key, req) is not LookupResult.HIT for key in keys):
            raise RuntimeError("published CPU backing keys are not all HIT")

        cpu_spec = manager.prepare_load(keys, req)
        dst_gpu_spec = GPULoadStoreSpec(
            dst_gpu_pages,
            group_sizes=[num_transfer_pages],
            block_indices=[0],
        )

        for repeat in range(args.load_repeats):
            for tensor in gpu_tensors:
                tensor[dst_gpu_pages].zero_()
            torch.cuda.synchronize()

            job_id = 100 + repeat
            t0 = time.perf_counter()
            if not worker.submit_load(job_id, cpu_spec, dst_gpu_spec):
                raise RuntimeError(
                    f"CPUOffloadingWorker.submit_load failed at repeat {repeat}"
                )
            worker.wait({job_id})
            wall = time.perf_counter() - t0
            result = _take_result(worker, job_id)
            load_rows.append(
                {
                    "repeat": repeat,
                    "bytes": result.transfer_size,
                    "event_seconds": result.transfer_time,
                    "wall_seconds": wall,
                }
            )

            layer_ok = []
            for layer, tensor in enumerate(gpu_tensors):
                ok, bad_page_row = _verify_pattern(
                    tensor,
                    dst_gpu_pages,
                    logical_pages,
                    layer_idx=layer,
                )
                layer_ok.append(
                    {
                        "layer": layer,
                        "byte_exact": ok,
                        "first_bad_transfer_page_row": bad_page_row,
                    }
                )
            verify_rows.append(
                {
                    "repeat": repeat,
                    "all_layers_byte_exact": all(
                        row["byte_exact"] for row in layer_ok
                    ),
                    "layers": layer_ok,
                }
            )

        manager.complete_load(keys, req)

        table = coordinator.materialize_qsa_page_table(
            logical_tokens=args.logical_tokens,
            geometry=geometry,
        )
        for move in expanded.stage_in_pages:
            mapped = (
                table[move.logical_page]
                if move.logical_page < len(table)
                else None
            )
            page_table_checks.append(
                {
                    "logical_page": move.logical_page,
                    "expected_physical_page": move.physical_page,
                    "mapped_physical_page": mapped,
                    "match": mapped == move.physical_page,
                }
            )

        for move in expanded.stage_out_pages:
            if move.logical_page < len(table):
                if table[move.logical_page] != -1:
                    raise RuntimeError(
                        "evicted logical page remained mapped in QSA page table"
                    )

    finally:
        worker.shutdown()

    if store_row is None:
        raise RuntimeError("store transfer did not complete")
    if store_row["bytes"] != expected_transfer_bytes:
        raise RuntimeError(
            f"D2H byte count mismatch: {store_row['bytes']} "
            f"!= {expected_transfer_bytes}"
        )
    if any(row["bytes"] != expected_transfer_bytes for row in load_rows):
        raise RuntimeError("H2D byte count mismatch")
    if not all(row["all_layers_byte_exact"] for row in verify_rows):
        raise RuntimeError("byte-for-byte H2D verification failed")
    if not all(row["match"] for row in page_table_checks):
        raise RuntimeError("QSA logical->physical page table validation failed")

    event_times = [float(row["event_seconds"]) for row in load_rows]
    wall_times = [float(row["wall_seconds"]) for row in load_rows]
    transfer_gib = expected_transfer_bytes / (1024**3)
    median_event = statistics.median(event_times)
    median_wall = statistics.median(wall_times)

    return {
        "schema": 1,
        "device": {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
        },
        "geometry": {
            "logical_tokens": args.logical_tokens,
            "capacity_tokens": args.capacity_tokens,
            "capacity_regions": capacity_regions,
            "region_tokens": args.region_tokens,
            "page_tokens": args.page_tokens,
            "pages_per_region": geometry.pages_per_region,
            "resident_gpu_pages": resident_gpu_pages,
            "qsa_layers": args.qsa_layers,
            "num_kv_heads": args.num_kv_heads,
            "head_dim": args.head_dim,
            "dtype_bytes": args.dtype_bytes,
            "main_kv_bytes_per_token": main_bytes_per_token,
            "per_layer_page_bytes": per_layer_page_bytes,
        },
        "transition": {
            "query_replacement_cap_regions": transition.query_replacement_cap,
            "query_replacements_regions": transition.query_replacements,
            "stage_in_regions": list(transition.stage_in_blocks),
            "stage_in_pages": num_transfer_pages,
            "stage_in_bytes": expected_transfer_bytes,
            "stage_in_mib": expected_transfer_bytes / (1024**2),
            "stage_in_gib": transfer_gib,
        },
        "cpu_backing": {
            "num_cpu_blocks": num_transfer_pages,
            "all_keys_hit_after_store": True,
        },
        "d2h_publication": store_row,
        "h2d_stage_in": {
            "repeats": load_rows,
            "median_event_seconds": median_event,
            "median_wall_seconds": median_wall,
            "median_event_ms": median_event * 1000.0,
            "median_wall_ms": median_wall * 1000.0,
            "effective_event_gib_s": transfer_gib / median_event,
            "effective_wall_gib_s": transfer_gib / median_wall,
        },
        "byte_verification": {
            "all_repeats_exact": True,
            "repeats": verify_rows,
        },
        "qsa_page_table": {
            "all_stage_in_mappings_exact": True,
            "all_evicted_pages_are_negative": True,
            "checked_stage_in_pages": len(page_table_checks),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logical-tokens", type=int, default=240000)
    ap.add_argument("--capacity-tokens", type=int, default=65536)
    ap.add_argument("--region-tokens", type=int, default=256)
    ap.add_argument("--page-tokens", type=int, default=16)
    ap.add_argument("--replacement-fraction", type=float, default=0.05)
    ap.add_argument("--qsa-layers", type=int, default=12)
    ap.add_argument("--num-kv-heads", type=int, default=2)
    ap.add_argument("--head-dim", type=int, default=256)
    ap.add_argument("--dtype-bytes", type=int, default=2)
    ap.add_argument("--load-repeats", type=int, default=5)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    result = run_probe(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
