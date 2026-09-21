# R1 K1-T: real generic-offload transfer plane

## Status

K1 target runtime core is GO.

K1-T is the first experiment in this lane that performs real KV-like GPU ->
CPU backing -> GPU transfers through the installed vLLM 0.29 generic offload
implementation. It does not start the Qwen model.

## What is being proven

K1-T tests the transfer plane independently from QSA/GDN model semantics:

- stable logical-page identity;
- CPUOffloadingManager allocation/readiness/refcount behavior;
- CPUOffloadingWorker real pinned D2H/H2D transfers;
- arbitrary noncontiguous logical history pages;
- explicit destination GPU physical pages;
- sticky resident-slot reuse;
- byte-for-byte page correctness;
- logical-page -> physical-page QSA table materialization;
- real worker event time and submit/wait wall time.

## Primary geometry

The synthetic cache matches the measured Qwen main-KV geometry:

- QSA layers: 12
- K/V heads: 2
- head dimension: 256
- dtype: BF16-equivalent 2 bytes
- main K/V bytes/token: 24,576
- planner region: 256 tokens
- resident budget: 65,536 tokens
- resident regions: 256
- replacement cap: 5% = 12 regions

The runner reads vLLM CacheConfig.DEFAULT_BLOCK_SIZE from the installed wheel
and uses it as page_tokens. For vLLM 0.29 this is expected to be 16.

At 16 tokens/page:

- pages/region: 16
- physical resident pages: 4096
- stage-in pages: 12 * 16 = 192
- bytes/layer/page: 32 KiB
- aggregate stage-in: 72 MiB

This is exactly the K1B primary 5% transition volume.

## GPU/CPU footprint

The probe allocates one int8 byte-view tensor per QSA layer. The bytes match the
BF16 K+V payload size even though the test tensor dtype is int8, because the
vLLM worker itself canonicalizes transfers as byte pages.

Approximate GPU allocation at the primary geometry:

- 4096 resident pages plus 192 staging pages
- 12 layers
- 32 KiB/layer/page
- about 1.57 GiB total

CPU backing stores only the 192 incoming pages across 12 layers:

- about 72 MiB pinned host memory

No full 240K CPU KV archive is allocated in K1-T.

## Arbitrary sparse history

The 12 incoming planner regions are deliberately noncontiguous and spread
through the 240K logical history:

300, 337, 411, 468, 512, 577, 633, 701, 744, 812, 871, 930

They are not selected because of semantic facts here. K1-T is a transfer/mapping
test, so these positions stress arbitrary logical-page identity and destination
mapping.

## vLLM adapter

src/vllm_exl3/kvmem_vllm_offload.py defines VllmCPUPageBacking.

It directly reuses:

- CPUOffloadingManager
- CPUOffloadingWorker
- GPULoadStoreSpec
- ReqContext / LookupResult
- generic manager prepare_store / complete_store
- generic manager prepare_load / complete_load
- worker submit_store / submit_load / wait / get_finished

It does not call tensor.copy_ for the transfer path.

Logical page keys are stable SHA-256 identities derived from:

- request/lineage identity
- logical page position
- KV group index through vLLM make_offload_key

Physical GPU block IDs remain transient destinations and are not part of
logical identity.

## Real publication and stage-in sequence

1. Allocate bounded GPU resident pages plus a small staging tail.
2. Fill staging pages with deterministic per-byte patterns keyed by logical
   page and QSA layer.
3. Publish those pages with CPUOffloadingManager.prepare_store and
   CPUOffloadingWorker.submit_store.
4. Verify every logical page is a CPU-backing HIT.
5. Apply the real StickyResidentCoordinator 5% transition.
6. Expand 12 region moves into 192 page moves.
7. Zero destination resident pages.
8. prepare_load + submit_load into the exact physical pages assigned by the
   coordinator.
9. wait/get_finished and complete_load.
10. Verify every transferred byte for all 12 layers.
11. Repeat H2D load five times for timing.
12. Materialize the QSA page table and verify:
    - every staged logical page maps to its assigned physical page;
    - every evicted page maps to -1.

## Timing outputs

vLLM transfer results expose CUDA-event transfer_time. The adapter also measures
submit -> wait -> completion wall time.

K1-T reports both:

- median_worker_event_ms
- median_submit_wait_wall_ms

The measured K1B pinned memcpy reference remains 6.3494 GiB/s.

For 72 MiB the old raw floor is about 11.1 ms.

The summary reports:

- event_over_raw_floor
- wall_over_raw_floor

Performance classification:

- <= 1.5x raw floor: CLOSE_TO_RAW_FLOOR
- <= 2.5x: MODERATE_TRANSFER_OVERHEAD
- > 2.5x: HIGH_TRANSFER_OVERHEAD

This classification is diagnostic, not the K1-T hard gate.

## Hard K1-T gate

Correctness requires all of the following:

- exactly 12 resident-region replacements;
- exactly 192 page stage-ins;
- exactly 72 MiB transfer size;
- all published CPU keys are HIT;
- all 5 H2D repeats byte-exact across all 12 layers;
- all staged logical pages map to assigned physical pages;
- all evicted logical pages are -1;
- no new NVIDIA Xid.

If these pass, report K1-T TRANSFER CORRECTNESS GO even if transfer overhead is
higher than desired. Performance class then determines what to optimize before
full model integration.

## Why performance is not a hard gate yet

K1-T measures generic-worker overhead in isolation. A high ratio can come from
descriptor construction, the selected swap backend, synchronization, or small
per-layer fragmentation. Those are optimizable without invalidating the
residency architecture.

By contrast, any byte or mapping error invalidates the runtime architecture.

## Next phase after GO

K1-Q1 will attach this transfer plane to real Qwen canonical main-KV storage in
a target-only diagnostic path.

It still will not immediately claim multi-turn TTFT savings. Current QSA
selection is generated layer-by-layer during forward, so the common all-layer
resident proposal requires a selection/replay boundary. GDN/recurrent
checkpoint semantics remain a separate K2 dependency.

## Non-goals

K1-T does not:

- start vLLM model inference;
- patch installed QSA;
- change persistent_topk;
- change attention math;
- implement GDN replay;
- enable MTP;
- allocate NVMe backing;
- merge to production.
