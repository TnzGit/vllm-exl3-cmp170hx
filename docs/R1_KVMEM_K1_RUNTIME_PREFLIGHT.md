# R1 K1 runtime preflight

## Why this exists

K1B already crossed the engineering threshold for a real bounded-KV
prototype:

- K0 QSA-native retrieval: 12/12 deterministic needles recalled;
- K1A fresh desired-set stage-in: ~18-22% of a 64K resident budget;
- K1B sticky planner at 5% replacement cap:
  - target recall = 1.0 at 160K and 240K;
  - median/max stage-in fraction = 0.046875;
  - ~0.0703 GiB main-KV stage-in per semantic query transition;
- local pinned H2D bandwidth = ~6.35 GiB/s;
- raw 0.0703 GiB copy floor ~= 11 ms.

The remaining K1B Section A failure was only an extension-package import
mismatch. The installed vLLM 0.29 wheel registers
`torch.ops._C.persistent_topk` from `vllm._C_stable_libtorch`, not the
legacy `vllm._C` module.

The K1 runtime should not be delayed by that diagnostic.

Before writing a block manager, however, we need to know which upstream vLLM
ownership/transfer contracts are actually present in the installed wheel.

## Important upstream discovery

Current upstream vLLM has an experimental **HiSparse local KV offload**
architecture with responsibilities very close to the intended K1 design:

- normal KV cache manager owns GPU block pools/tables;
- a HiSparse coordinator owns logical host blocks and residency transitions;
- a connector carries scheduler/worker transfer commands;
- worker-side runtime owns pinned host/hot storage and GPU replacement state;
- resident hits can bypass host/hot lookup inside the fused resolver;
- logical host identity remains separate from physical GPU residency.

That architecture is stronger than generic "copy full prefix from CPU" KV
offload because it explicitly models host-backed sparse history and a bounded
GPU resident/hot footprint.

It was not assumed to exist when this project's KVMem feasibility work began.

## What the preflight checks

`tools/kvmem_k1_runtime_preflight.py` is read-only.

It reports installed vLLM version/source path and probes these interfaces:

### HiSparse

- `HiSparseCoordinator`
- `HiSparseConnector`
- `HiSparseResidentManager`
- `HiSparseHotManager`
- `HiSparseSourceManager`
- `HiSparseResidentSpec`
- `HiSparseHotSpec`

### Generic CPU offloading

- `OffloadingConnector`
- `CPUOffloadingSpec`
- `CPUOffloadingManager`

It also scans the installed Qwen4Exp Python package for references to:

- HiSparse
- OffloadingConnector
- CPUOffloading
- QSAIndexer / qsa_select
- KV cache spec hooks

The scan is evidence only. A string hit does not prove compatibility.

## Route selection

### adapt_hisparse_contract

Chosen when the installed wheel exposes the full HiSparse coordinator,
connector, manager and spec surface.

This is the preferred route.

It means K1 should reuse upstream ownership/lifecycle boundaries and add the
Qwen4Exp-specific pieces:

1. derive block relevance from QSA selections;
2. apply the measured 5% sticky replacement policy;
3. map selected logical 256-token regions into the host/resident/hot page
   representation;
4. preserve Qwen4Exp GDN/recurrent state as a separate problem;
5. keep MTP disabled until target-only correctness is established.

A positive preflight does **not** mean the current HiSparse CUDA resolver already
supports Qwen4Exp's exact K/V layout. That must be checked separately.

### extend_generic_offloading_contract

Chosen if HiSparse is absent but generic CPU offloading exists.

In that case, reuse:
- scheduler/worker job lifecycle;
- pinned CPU backing allocation;
- transfer completion/fencing;

but add a KVMem-specific resident-set coordinator instead of trying to force
query-conditioned sparse history into whole-prefix offload semantics.

### custom_vllm_patch_required

Only if neither interface family is present.

This is the least desirable path and would justify a larger vLLM patch surface.

## Persistent-topk retry

The same runner also reruns only the cheap Section A diagnostic.

The diagnostic now registers the op by trying:

1. `vllm._C_stable_libtorch`
2. legacy `vllm._C`

and verifies `torch.ops._C.persistent_topk` exists.

It compares:
- unique-score repeated top-k sets;
- a deliberately tied cutoff with far more equal-score candidates than slots.

If unique inputs are stable but tied cutoffs produce multiple sets, that
supports the hypothesis that K1A fresh-set replay instability comes from
tie-boundary collection rather than relevance instability.

This diagnosis is useful for understanding QSA, but it is no longer a gate for
K1 because the stateful sticky policy already controls the replacement volume
without losing target recall.

## No model run

The preflight:
- does not start vLLM;
- does not patch QSA;
- does not modify installed source;
- does not rerun 160K/240K;
- does not allocate real CPU KV backing.

The only GPU work is the small top-k kernel microdiagnostic.

## Product expectations after K1B

The expected benefits should be ordered as follows.

### 1. Long-context concurrency / capacity

Strongest expected benefit.

Measured main K/V footprint:
- 160K: ~3.66 GiB / sequence
- 240K: ~5.49 GiB / sequence
- 64K resident: 1.50 GiB / sequence
- 32K resident: 0.75 GiB / sequence

KV-component-only scaling:
- 240K -> 64K: ~3.66x less resident main-KV memory
- 240K -> 32K: ~7.32x less

Actual service concurrency gain will be lower because weights, recurrent state,
QSA index/cache, workspaces and allocator reserve remain resident.

A realistic first engineering target is therefore roughly **2-3x more
simultaneous long sessions** at 64K resident budget, to be measured rather than
assumed.

### 2. Repeated-turn prefill / TTFT

Potentially the largest user-visible latency improvement for long-running
coding/agent sessions.

Cold ingestion of 160K/240K still processes the full history and may initially
be slightly slower due to host publication/bookkeeping.

The large potential gain comes later: when unchanged historical state can be
reused from CPU backing plus recurrent checkpoints, a new turn need not
recompute the entire logical history.

K1 target-only attention KV is necessary but not sufficient. Qwen4Exp
GDN/recurrent resume semantics are a separate K2 requirement before claiming
full multi-turn replay savings.

### 3. Aggregate decode throughput

Likely meaningful indirectly.

A bounded resident footprint permits more long requests to coexist, which can
increase useful decode batching and reduce capacity/preemption pressure.

### 4. Single-sequence decode TPOT

Expected to be the smallest benefit.

The current production Amdahl is dominated by MoE/dense work; QSA itself is a
small fraction of decode GPU time.

The design target is therefore:
- no material TPOT regression;
- resident-set transitions happen at semantic turn/prefill boundaries, not once
  per emitted token;
- any ~11 ms sticky H2D floor is amortized per turn, not paid per decode token.

## Next implementation gate

After this preflight:

- if HiSparse core is present: audit Qwen4Exp cache-layout compatibility and
  build a minimal target-only HiSparse-derived adapter;
- otherwise if generic offload exists: build a target-only KVMem coordinator on
  generic transfer/backing primitives;
- do not add GDN replay or MTP in the first executable K1 prototype.
