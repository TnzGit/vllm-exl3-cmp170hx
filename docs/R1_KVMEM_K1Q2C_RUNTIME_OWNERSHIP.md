# R1 K1-Q2C runtime ownership: bounded live scheduler QSA KV

## Scope

This branch is the first live ownership proof after the Q2C scheduler-shrink
preflight. It is deliberately narrower than the final multi-turn product.

The proof target is one frozen 160K `ask_d_e` request, eager, MTP off, prefix
cache off, with the exact K1B turn-specific sticky resident plan.

This is an **oracle/frozen-plan mechanical proof**, not a production residency
policy: the resident set was computed offline for this turn and is applied
throughout prefill. A production implementation must later choose/update
residency causally without future-turn knowledge.

## Runtime invariant

QSA uses 16-token / 32-KiB manager pages. Its 4,160 real page IDs live in a
dedicated virtual QSA pool rather than vLLM's shared hybrid/Mamba BlockPool.
Each QSA layer owns a model buffer of 4,161 pages (4,160 real + one physical
zero/null page), about 130.03 MiB/layer or 1.524 GiB across 12 layers. Because
that buffer is created during model construction, normal vLLM GPU memory
profiling accounts for it before the shared KV arena is sized.

The generic KV config keeps one normal-geometry QSA placeholder page solely for
metadata/binding. This avoids the first live attempt's failure mode where 4,160
QSA IDs were each charged at the global hybrid/Mamba block stride.

The runner fixes:

- sticky resident history: 4096 pages = 65,536 tokens
- active/current work reserve: 64 pages = 1,024 tokens
- max_num_batched_tokens: 1,024
- hard live real-page cap: 4,160 QSA pages/request
- layout: BLHNC
- one request, eager, no MTP, no prefix cache

During prefill, every completed 16-token historical page that is not in the
frozen sticky set is released on the scheduler side. The current unprocessed
chunk stays real for causal prefill. Therefore the worst-case live ownership
is sticky pages plus at most one 64-page scheduler chunk: <=4160.

The scheduler logical row is not truncated. Freed historical positions become
scheduler-side null holes, and released virtual IDs are recycled for later work.

vLLM 0.29 worker block tables are append-only for running requests, so those
old worker entries are not overwritten in place with the scheduler null ID.
The worker therefore masks every processed nonresident logical position using
the frozen resident policy before QSA attention. New work pages are appended
in exact logical order; stale old virtual IDs are never allowed to become
visible aliases after their IDs are recycled.

## Admission

vLLM 0.29 normally checks required blocks against its shared BlockPool before
admitting a long request. Q2C virtual QSA pages do not consume that shared pool,
so the custom manager reports zero *shared* blocks. Its independent 4,160-page
virtual capacity is allocated up front in the model-owned dedicated tensors and
hard-enforced by the manager on every QSA allocation. The other hybrid/Mamba
groups continue to use the normal full-sequence shared-pool admission checks.

## Worker ownership / CPU authority

The QSA worker no longer allocates the Q2B shadow in addition to a stock full
QSA cache. The model-owned ~130-MiB/layer dedicated tensor is now the actual
QSA cache. Writes use virtual IDs from scheduler-provided slot_mapping and
reads use the same virtual-ID block table.

At the frozen query boundary the worker:

1. gathers only the retained 4096 scheduler-owned historical pages through a
   reusable 128-page staging tensor (4 MiB/layer);
2. publishes those bytes through the existing generic vLLM CPU offload adapter;
3. destroys each retained GPU chunk;
4. stages the CPU copy back into the same 4-MiB staging tensor;
5. compares the restored bytes against a transient pre-destruction reference;
6. writes them back into the scheduler-owned physical pages;
7. masks processed nonresident historical selections according to the frozen
   policy before stale append-only worker entries can alias recycled IDs;
8. runs sparse attention against the scheduler-owned cache.

The transient reference is one staging chunk, not a persistent resident shadow.

## GO classification

`Q2C_FROZEN_PLAN_OWNERSHIP_SEMANTIC_GO` requires all of:

- scheduler emits progressive reclaim events;
- one boundary event reports logical row > physical cap;
- peak scheduler real pages <=4160 for the whole request;
- boundary real pages <=4160;
- scheduler proves real null-hole reclaim and peak ownership <=4160;
- worker sees all 12 QSA layers, keeps virtual IDs in [0, 4159], and validates
  all retained resident IDs;
- all 12 workers are bound to 4,161-page dedicated tensors, not the shared
  placeholder storage;
- 4096 retained pages/layer publish and restore through generic CPU backing;
- D2H and H2D each exactly 128 MiB/layer, 32 jobs at 128 pages/job;
- restored CPU bytes exactly match bytes destroyed from the scheduler cache;
- progressive prefill visibility is exercised (`prefill_historical_selected_dropped > 0`);
- boundary/query visibility accounting is exercised (`historical_selected_dropped > 0`);
- frozen target codes remain correct and request finishes with `stop`;
- Xid delta = 0 and installed QSA restores byte-for-byte.

## Non-claims after GO

A GO does not yet establish:

- 240K runtime;
- multi-request concurrency scaling;
- MTP follower correctness;
- causal resident selection without future-turn knowledge;
- exact hidden-state/token parity against full-history prefill;
- dynamic sticky replacement across multiple turns;
- production latency/TTFT;
- an engine-start arena allocation reduction (vLLM still preallocates its
  shared KV arena according to gpu_memory_utilization).

The claim is scheduler/block-pool ownership and per-request occupancy, which is
the prerequisite for useful long-session concurrency.

## Stop rule

Run exactly one 160K frozen semantic case. Stop after evidence collection.
Do not continue to 240K, MTP, multi-request, or dynamic replacement.
