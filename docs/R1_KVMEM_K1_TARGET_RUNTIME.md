# R1 K1 target-only bounded-KV runtime design

## Evidence gate

The research phase is complete enough to start runtime engineering.

K0 established that QSA-native selection recalled all 12 deterministic needles
at 160K/240K across 128/256/512-token planner regions and both 32K/64K
budgets.

K1A showed that fresh 64K/256 desired sets require roughly 18-22% stage-in
per semantic query transition, while target recall remains 1.0. Exact
same-query fresh-set Jaccard was only about 0.67-0.69.

K1B showed that a 5% sticky replacement cap preserves target recall = 1.0
at both 160K and 240K while reducing query-specific stage-in to 4.6875%,
about 3072 tokens / 0.0703 GiB main KV per transition. Local pinned H2D
measured about 6.35 GiB/s, so the raw copy floor is about 11 ms.

The persistent-topk microdiagnostic then reproduced the instability mechanism
on CMP170HX / SM80: unique-score sets were stable, tied-cutoff sets produced
7-10 distinct sets per row with minimum pairwise Jaccard 0.684, while a
torch.topk tied reference remained stable.

Sticky residency is therefore both a transfer-volume policy and the stability
boundary that prevents equal-score top-k tail noise from becoming physical
KV churn.

## Installed vLLM 0.29 route

HiSparse is not present in the installed wheel.

Available generic primitives are OffloadingConnector, CPUOffloadingSpec,
CPUOffloadingManager, CPUOffloadingWorker / OffloadingWorker transfer
contracts, GPULoadStoreSpec, and canonical KV cache registration.

Recommended route: extend_generic_offloading_contract.

This does NOT mean using OffloadingConnector unchanged.

The stock scheduler is still contiguous prefix/chunk cache semantics:
- lookup asks how many additional prefix tokens are externally available;
- allocation/load builds contiguous GPU block ranges;
- offload keys are derived from prefix block hashes/group identity.

A KVMem resident set is different:
- logical history remains large;
- arbitrary historical regions may be GPU-resident or absent;
- the resident set is query-conditioned and stateful;
- a logical page may have no current GPU physical page.

K1 therefore reuses generic CPU backing and transfer primitives while owning
a KVMem-specific resident coordinator and logical-page mapping.

## Existing QSA virtualization boundary

Qwen4Exp QSA sparse attention already receives selected logical token indices,
a request block table, and physical paged K/V caches.

For each selected token the kernel computes its logical page and reads:

physical_page = block_table[request, logical_page]

Negative/out-of-range physical pages are rejected by the current kernel.

Therefore attention math does not need to change. K1 can virtualize residency
by presenting a block table where resident logical pages map to physical
resident pages and non-resident logical pages are -1.

## Planner region vs KV page

K1B's 256-token 'block' is a planner region, not necessarily one vLLM KV page.

Runtime definitions:
- planner region: 256 logical tokens by default;
- KV page: cache_config.block_size tokens at runtime;
- pages per region: region_tokens / page_tokens.

The first implementation requires exact divisibility. If installed page size
does not divide 256, planner region size must be aligned before GPU
integration.

One resident planner region owns a contiguous physical-page span.

Example for 16-token KV pages:
- 256-token planner region;
- 16 pages per region;
- 64K budget = 256 resident regions;
- 4096 physical GPU pages.

The runtime coordinator maps logical_region -> resident_region_slot and expands
this into logical_page -> physical_gpu_page for QSA.

## Runtime core

src/vllm_exl3/kvmem_resident.py provides:
- deterministic fresh-set construction;
- 5% sticky/hysteretic replacement;
- mandatory sink/recent admission outside the semantic replacement cap;
- stable physical resident-slot reuse;
- deterministic score-tie handling;
- explicit stage-in/stage-out moves;
- planner-region to QSA-page-table expansion;
- -1 holes for non-resident logical pages.

Important tie rule: a fresh candidate must have a strictly greater current
score than the weakest evictable resident region. Equal-score candidates never
churn residency.

## Research-to-runtime contract

tools/kvmem_runtime_replay_check.py replays the captured K1A shadows through
the runtime coordinator and compares them with the K1B primary reference:
5% replacement, 256-token regions, 64K budget.

Required exact agreement:
- every turn's logical resident set;
- every transition's stage-in count;
- every transition's stage-out count.

Runtime integration must not proceed if this contract diverges.

## What generic offload can be reused for

CPUOffloadingManager already provides logical offload-key -> CPU block
identity, ref-counting, eviction protection while loads are in flight, CPU
block allocation/free-list, readiness tracking, and metrics/events.

CPUOffloadingWorker already provides canonicalized KV cache layout, pinned or
shared CPU backing, async CPU<->GPU transfer contracts, explicit destination
GPU block IDs, and transfer completion/fencing.

GPULoadStoreSpec is useful because it already carries explicit GPU block IDs.

The stock scheduler is the component that needs extension/replacement, not the
byte-copy machinery.

## Offload identity

Generic OffloadKey is block_hash || group_idx.

K1 also requires stable logical-history identity. The target-only prototype
should keep a separate mapping:

(request lineage, logical region/page) -> OffloadKey

The authoritative CPU backing remains compatible with generic offload manager
and worker semantics, while residency is planned by logical position.

Never key resident identity only by transient physical GPU block ID.

## Critical timing constraint

Current QSA computes, in one layer forward:
1. current-layer hidden state;
2. QSA index projection;
3. selected logical token indices;
4. main K/V sparse attention.

A common resident set aggregated across all QSA layers cannot be derived from
all current-layer selections before the first attention layer runs.

This prevents a naive 'compute all current selections, stage once, then run'
architecture without a replay/planning phase.

## K1-T — transfer-plane prototype

This is the next executable implementation.

Goal: prove real CPU backing + resident physical slots + QSA page-table
virtualization before changing model semantics.

K1-T will:
- reuse generic CPU offload backing/worker primitives;
- allocate a bounded physical GPU resident page pool;
- publish synthetic or captured KV pages to CPU;
- request arbitrary non-contiguous logical regions;
- use StickyResidentCoordinator for destination slots;
- stage only missing regions H2D;
- verify copied page bytes;
- materialize the sparse QSA page table;
- verify holes remain -1;
- measure real gather/load/fence overhead on top of the memcpy floor.

K1-T does not yet run the full Qwen model with sparse history. This isolates
storage/mapping correctness from query replay and recurrent state.

## K1-Q — query-selection / replay integration

Only after K1-T is correct.

Because current-query QSA selection is produced layer-by-layer, the final
common sticky proposal cannot be known before first attention without replay.

Likely architecture:
1. begin from previous sticky resident state;
2. run a selection/replay pass for the new suffix;
3. aggregate QSA proposals across the 12 full-attention layers;
4. apply the 5% sticky transition;
5. stage missing main-KV regions;
6. replay the current suffix using the updated common resident page table.

The exact replay boundary must be designed together with recurrent/GDN state.

## K2 — GDN/recurrent checkpoint and replay

Qwen4Exp is hybrid. Attention KV virtualization alone is insufficient for
cheap multi-turn replay.

Before claiming repeated-turn TTFT gains we must define authoritative
recurrent state boundaries, checkpoint frequency, suffix replay start,
speculative rollback, and state reuse across turns.

Do not treat GDN state as ordinary KV pages.

## K3 — MTP follower

Only after target correctness.

MTP must consume the target resident-set version, never independently retrieve
historical regions, share logical-position identity, roll back rejected
speculative state, and measure acceptance-rate impact.

## Product expectations

Measured main K/V footprint:
- 24 KiB/token;
- 160K full: about 3.66 GiB/sequence;
- 240K full: about 5.49 GiB/sequence;
- 64K resident: 1.50 GiB/sequence;
- 32K resident: 0.75 GiB/sequence.

KV-component-only reduction from 240K is 3.66x at 64K and 7.32x at 32K.
Actual service concurrency gain is lower because weights, recurrent state,
QSA side caches, workspaces and allocator reserve remain.

A reasonable first end-to-end target is roughly 2-3x more simultaneous long
sessions at 64K residency, to be measured under C2/C4/C8.

Cold prefill is not expected to become faster inherently. The large possible
user-visible gain is repeated-turn TTFT once both attention K/V reuse and
recurrent-state replay exist.

Single-stream decode TPOT should remain close to the current production
baseline. Resident transitions should occur at semantic turn/replay boundaries,
not once per emitted token.

Required future metrics include resident hit rate, CPU backing hit rate, H2D
bytes per turn and per output token, transfer wait time, selected-token misses
caused by nonresident pages, target quality, preemption, and eventually MTP
acceptance.

## Production safety

All K1 work remains opt-in, target-only, no MTP, no production default, and no
merge without explicit approval. Frozen production remains unchanged.
