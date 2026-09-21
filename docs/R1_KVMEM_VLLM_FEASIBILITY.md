# R1 feasibility: KVMem ideas on vLLM / Qwen3.8-Flash-Next

## Status

Research lane only. It must not interrupt or contaminate the current
MTP-k3/COOP production qualification.

External source audited:

- repository: `kvmem/kvmem-llama.cpp`
- branch: `master`
- audited head: `485824c66b7e9a464e1d3bb1dd943d658c4aa103`
- date of audit: 2026-09-21
- upstream engine: llama.cpp
- target model family in that project includes Qwen3.x/Qwen3.8 hybrid + MTP

Our frozen comparison point:

- repo base: `f2c6a719a1709ba40d08b97a4cc8d64b3c2256d8`
- single CMP170HX / SM80
- vLLM 0.29 compatibility lane
- Qwen3.8-Flash-Next EXL3 3bpw
- MTP k=3
- COOP=1
- PIECEWISE
- prefix cache OFF
- healthy through 240K

## Executive conclusion

KVMem is **highly relevant to our next capability lane**, but it is not a
drop-in performance patch for the current 4K-240K production baseline.

The strongest transferable ideas are:

1. bounded GPU attention working set independent of logical conversation length;
2. query-conditioned block retrieval;
3. selection-diff / retain-resident / batched-D2D block movement;
4. MTP as a lockstep follower of the target attention working set;
5. hybrid recurrent-state checkpoint/replay kept separate from attention KV;
6. measurement discipline around stage-in/stage-out/replay instead of treating
   "KV offload" as one opaque operation.

The llama.cpp adapter code itself should **not** be ported mechanically. vLLM
already owns paged KV allocation, hybrid KV groups, speculative scheduling and
offload/connector abstractions. The right design is to reimplement the KVMem
semantics on top of vLLM's native block/cache manager.

Most importantly, Qwen4Exp in vLLM 0.29 already has a QSA side-cache/index
pipeline with raw keys, compressed keys, paged metadata and sparse index
selection. Before creating a second mean-K index, we should test whether the QSA
index/cache can provide the relevance signal for a KVMem-style block selector.

## What KVMem actually does

### 1. Bounded GPU working set

The project separates logical token position from physical GPU KV slot.

The GPU attention pool is bounded by a configured selection budget plus a
generation reserve. Historical KV may live outside that GPU pool, while selected
historical blocks are brought back into the active window.

This is fundamentally different from ordinary KV offload:
- ordinary offload generally preserves a full logical attention sequence and
  moves storage tiers;
- KVMem deliberately selects a sparse historical working set for attention.

That distinction matters for correctness: once blocks are omitted, output is no
longer exact full-context inference.

### 2. Mean-K retrieval index

The llama.cpp implementation maintains a compact per-block/per-layer K summary.

The audited `score_retrieval()`:
- loads `mean_k(block, layer)`;
- uses the captured current query Q sums/counts;
- groups Q heads to KV heads;
- computes a cosine-like Q/K relevance score per KV head;
- averages across heads/layers;
- assigns one score per historical block.

The store then selects top-scoring blocks subject to sink/mandatory/budget
constraints and finally restores chronological block order.

This is simple, cheap metadata compared with full KV.

### 3. Differential block plan

`KvMemStore::set_selection()` does not reload the entire active window.

It computes:
- stage_out blocks;
- stage_in blocks;
- resident overlap;
- blocks whose physical position can remain unchanged;
- remaps for moved blocks.

Resident selected KV can be retained. The newer llama.cpp adapter also batches
resident re-layout using device-to-device copies rather than round-tripping every
row through the host.

This "plan diff first, transfer only the delta" principle maps well to vLLM.

### 4. MTP follower slot-pool

The MTP cache does not independently retrieve history.

`llama_memory_kvmem_mtp`:
- asks the target cache for each original position's slot;
- uses the same block IDs / slot indices / original positions;
- follows target retrieval/stage-out/layout changes;
- restores only missing draft KV.

This is a strong design constraint for speculative decode: target and draft must
observe the same sparse historical view.

### 5. Hybrid / recurrent state

The project explicitly does **not** model GDN/recurrent state as ordinary KV.

For Qwen hybrid models it separates:
- attention-KV virtualization;
- recurrent-state checkpoints;
- query replay;
- speculative-reject rollback.

That separation is directly relevant to Qwen4Exp PLE/GDN in vLLM.

### 6. Query replay and proof-based fast paths

Later KVMem versions do more than retrieve blocks. They prove when a previous
query representation / selected view can be reused and avoid unnecessary replay.

The design distinguishes cases such as:
- all resident;
- unchanged selection;
- cached query + reselect;
- full query replay;
- legacy fallback.

This is interesting for multi-turn/tool-agent workloads where repeated long
prefill can dominate TTFT.

## What is *not* directly transferable

### llama.cpp memory adapter

Do not copy:
- `llama_memory_i` wrappers;
- ggml cache-cell layout code;
- llama-specific graph hooks;
- llama speculative-state machine integration.

vLLM has different ownership boundaries and a paged block manager.

### Host RoPE / GGML stage-in code

The project's CUDA stage-in path includes llama/GGML-specific:
- dequant;
- NeoX/MRoPE handling;
- Hadamard transforms;
- cache tensor layout.

The principles are useful, but our vLLM/EXL3 cache representation differs.

### Product NVMe claims need care

The research architecture and source contain NVMe tier abstractions and
experiments, but the current project README/release notes explicitly say NVMe
offload is not a supported product feature in this llama.cpp port.

Do not treat its current product as proof that NVMe KV paging is solved for our
runtime.

### Exact parity

Sparse historical selection is intentionally approximate relative to full
attention.

Once the active attention history is smaller than the true history, exact
full-context token parity is no longer a valid requirement. Quality must instead
be evaluated on retrieval/task outcomes.

## License/provenance note

The external repository currently has an unusual licensing presentation:
its README says KVMem-qw3 source is Apache-2.0 and that this port should be
treated the same unless a LICENSE file is added, while llama.cpp remains under
its upstream license.

Until a repository-level license file / provenance is unambiguous, prefer:
- ideas;
- algorithms;
- interfaces;
- independently written implementations;

over copying substantial source verbatim.

## Fit against our measured bottlenecks

### Current short/decode performance: LOW direct upside

Our current post-MTP data already shows:
- 4K decode ~10.3 ms/output-token;
- 160K slowdown is dominated by MTP acceptance loss;
- per-pass GPU cost changes only modestly with context.

Therefore KVMem is unlikely to be a large immediate 4K throughput optimization.

It can even reduce speculative acceptance if sparse history changes the target
distribution, so the MTP interaction must be measured.

### Current 160K-240K exact-context path: MEDIUM / uncertain performance upside

Reducing the attention working set may reduce some attention/QSA work and memory
pressure, but:
- Qwen4Exp already uses sparse QSA infrastructure;
- our measured QSA bucket is small in the current Amdahl;
- the dominant MoE compute is unaffected;
- retrieval/replay/stage-in introduce new overhead.

Do not forecast a large tok/s win before a shadow-mode experiment.

### >240K capacity: HIGH upside

Our current qualified exact path ends at 240K and 250K/256K is a memory-capacity
limit at the chosen GPU utilization.

A bounded KV working set changes the scaling law. It is therefore a strong
candidate for:
- 512K logical workspaces;
- 1M-class agent histories;
- long-running tool sessions;
- keeping model weights + MTP stable while history grows.

This is the most compelling reason to pursue the idea on a 65 GB CMP170HX.

### Multi-turn agent TTFT / history replay: HIGH potential upside

KVMem's query-state/replay work is particularly relevant to agent workloads:
large unchanged histories plus a relatively small new tool/user suffix.

The external project reports large reductions in repeated prefill/replay time on
its own engine/hardware. Those numbers are not transferable, but the mechanism
is applicable.

## Important bridge: Qwen4Exp QSA already gives us an index-like substrate

vLLM 0.29 Qwen4Exp has:
- `QSAIndexer`;
- paged raw-key side cache;
- compressed-key side cache;
- position metadata;
- paged prefill/decode selection kernels;
- sparse-attention block tables.

The NVIDIA implementation explicitly describes QSA projection weights, side
caches and paged weight-free selection.

Exact vLLM 0.29 source adds an important distinction:
- `QSAKeyStateCache` (raw BF16 key) is intentionally a small
  `CircularBufferSpec` that only retains the open compression group plus
  speculative rows;
- `QSACompressedKeyCache` is an `MLAAttentionSpec` with one normalized BF16
  key per complete compression group.

Therefore the raw-key ring is **not** a full historical retrieval index. The
compressed-key cache is the more promising substrate. A K0 prototype may need
to snapshot compressed group keys to CPU before their vLLM cache blocks are
recycled if we want relevance metadata to outlive GPU KV residency.

This means the first R1 experiment should **not** immediately add KVMem's
mean-K capture.

First ask:

> Can QSA's existing compressed/raw key representation produce a stable
> query-conditioned ranking of coarse historical KV blocks?

If yes, we can reuse already-computed model-native indexing information and
avoid:
- another K capture;
- another persistent index;
- extra reductions in the hot path.

If no, add a minimal mean-K side index as the fallback reference.

## K0 hardware result

K0 completed on the QSA-native selector with a strong GO result:

- 8 cases across 160K / 240K
- 12 deterministic needles
- direct QSA token hit: 12/12
- 128/256/512-token block recall: 12/12 at both 32K and 64K budgets
- primary 256/64K recall: 1.0
- primary 256/32K recall: 1.0
- `go_signal=true`

Therefore the native QSA signal is sufficiently strong to remain the preferred
retrieval signal for the next engineering stage. An independent mean-K selector
is not needed yet.

The next experiment is K1A selection-diff economics: fixed history, changing
queries, measuring resident-set overlap and estimated stage-in volume before any
real KV eviction/offload code is written.

## Proposed staged program

### K0 — shadow retrieval, no inference change

Goal: measure retrieval quality/overhead without changing attention or output.

No KV eviction. No stage-in. No output change.

For deterministic 160K and 240K workloads:
1. capture/query existing QSA relevance metadata;
2. aggregate/rank at coarse block sizes (candidate 128/256/512 tokens);
3. simulate 32K and 64K active budgets;
4. log selected block IDs only.

Reference selector:
- optionally implement KVMem-style mean-K in analysis/offline form;
- compare block overlap with QSA-derived ranking.

Required workloads:
- single needle at early/middle/late positions;
- multi-needle;
- long code/repository-like context;
- tool-turn transcript with old facts queried later;
- adversarial recency-vs-semantic conflict.

Metrics:
- required/needle block recall;
- selected-block precision where labels exist;
- selection stability across adjacent queries;
- QSA-vs-mean-K block overlap;
- index bytes;
- scoring latency;
- CPU/GPU synchronization introduced;
- no production TPOT regression when shadow logging is disabled.

GO:
- >=99% required-block recall on deterministic needle suite at 64K budget;
- >=95% at 32K is desirable, not initially mandatory;
- scoring/index overhead small enough to stay outside the decode hot path;
- no change to model outputs because shadow mode cannot affect attention.

NO-GO:
- native QSA signal cannot rank historical relevance at block granularity;
- or required-block recall is unstable enough that a larger semantic index is
  obviously necessary.

### K1 — bounded full-attention block working set

Only after K0 succeeds.

Implement a research-only virtual historical block set using vLLM's native block
manager/offload abstractions.

First tier should be CPU, not NVMe.

Key rules borrowed from KVMem:
- original logical positions stay authoritative;
- selection is a block-set diff;
- retained resident blocks should not move if unnecessary;
- stage-in/out only changed blocks;
- sink/recent/query-mandatory blocks consume the same finite budget;
- fail closed to full-context/legacy path when sparse invariants are not proven.

Initial targets:
- 512K logical prompt with 64K active attention working set;
- then 1M if CPU memory and replay cost are sane.

### K2 — hybrid PLE/GDN replay

Do not drop/recreate recurrent state as if it were attention KV.

Define:
- which PLE/GDN state is authoritative;
- checkpoint boundary;
- query replay boundary;
- rollback behavior;
- what can be reused across turns.

Use vLLM's existing hybrid cache/recurrent abstractions where possible.

### K3 — MTP lockstep follower

Only after target sparse attention works.

MTP must consume the same selected historical view.

Do not let draft independently choose blocks.

Required:
- target selection version/id;
- follower mapping from selected logical blocks to draft QSA/KV state;
- rejected draft tokens never become authoritative historical index state;
- acceptance-rate measurements become a first-class metric.

### K4 — quality and agent qualification

Once sparse execution changes attention, exact full-context parity is no longer
the acceptance criterion.

Use:
- needle/multi-needle recall;
- long-context retrieval suites;
- repo/code tasks;
- long tool transcripts;
- factual continuity across turns;
- completion quality;
- MTP acceptance;
- TTFT;
- decode TPOT;
- memory slope vs logical context length.

Always compare:
1. exact full-context production baseline where it fits;
2. recency-only sparse baseline;
3. QSA/KVMem sparse selector.

## Current stage

The exact-context R0 lane is frozen: PR #13 did not qualify.

K0 QSA-native shadow retrieval subsequently passed.

Current order:

1. K0 selector feasibility — **complete / GO**;
2. K1A resident-set churn / selection-diff economics — **current**;
3. only if K1A is viable, implement target-only CPU-tier bounded attention KV;
4. add recurrent/GDN replay semantics;
5. add MTP follower semantics;
6. consider NVMe only after CPU-tier behavior is proven.

## Expected payoff by objective

| Objective | Expected value from KVMem ideas |
|---|---|
| 4K decode tok/s | Low |
| 160K-240K exact-parity tok/s | Low-to-medium / uncertain |
| reduce MoE compute | None directly |
| reduce MTP acceptance loss | Uncertain; may hurt if retrieval is poor |
| break current 240K capacity ceiling | High |
| 512K/1M logical agent workspace | Very high if retrieval quality holds |
| repeated tool-turn TTFT | High potential |
| NVMe-first implementation | Low priority; CPU tier first |
| reuse literal llama.cpp code | Low; reimplement against vLLM |
| reuse design constraints/methodology | Very high |

## Decision

**Worth pursuing: YES.**

But classify it as:

**R1 capability/long-context virtualization program, not R0 decode optimization.**

The next hardware task remains PR #13 EARLY qualification.

After PR #13 closes, start K0 shadow retrieval with no model-output change.
