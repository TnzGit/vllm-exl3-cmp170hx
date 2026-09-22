# R1 K1-Q2C runtime ownership: bounded live scheduler QSA KV

## Scope

This branch is the first live ownership proof after the Q2C scheduler-shrink
preflight. It is deliberately narrower than the final multi-turn product.

The proof target is one frozen 160K `ask_d_e` request, eager, MTP off, prefix
cache off, with the exact K1B sticky resident policy.

## Runtime invariant

QSA uses 16-token / 32-KiB manager pages. The request block-table row grows
with logical history, but processed nonresident historical pages are reclaimed
immediately into the shared null block.

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

The logical row is not truncated. Freed historical positions remain as null
holes so their logical indices are stable.

## Admission

vLLM 0.29 normally reserves the full input sequence before admitting a long
chunked-prefill request. The custom manager keeps this safety gate enabled, but
when `apply_admission_cap=True` reports the proven recycling peak (4160) rather
than the full logical row width. Actual per-step allocation is still checked
against the same hard cap.

## Worker ownership / CPU authority

The QSA worker no longer allocates the Q2B 130-MiB/layer resident shadow.
Writes use the scheduler-provided slot_mapping and reads use the same
scheduler block_table.

At the frozen query boundary the worker:

1. gathers only the retained 4096 scheduler-owned historical pages through a
   reusable 128-page staging tensor (4 MiB/layer);
2. publishes those bytes through the existing generic vLLM CPU offload adapter;
3. destroys each retained GPU chunk;
4. stages the CPU copy back into the same 4-MiB staging tensor;
5. compares the restored bytes against a transient pre-destruction reference;
6. writes them back into the scheduler-owned physical pages;
7. masks QSA selections targeting nonresident historical null holes;
8. runs sparse attention against the scheduler-owned cache.

The transient reference is one staging chunk, not a persistent resident shadow.

## GO classification

`Q2C_SCHEDULER_OWNERSHIP_SEMANTIC_GO` requires all of:

- scheduler emits progressive reclaim events;
- one boundary event reports logical row > physical cap;
- peak scheduler real pages <=4160 for the whole request;
- boundary real pages <=4160;
- worker sees all 12 QSA layers with null historical holes;
- worker scheduler real-page count <=4160;
- 4096 retained pages/layer publish and restore through generic CPU backing;
- D2H and H2D each exactly 128 MiB/layer, 32 jobs at 128 pages/job;
- restored CPU bytes exactly match bytes destroyed from the scheduler cache;
- visibility accounting is exercised (`historical_selected_dropped > 0`);
- frozen target codes remain correct and request finishes with `stop`;
- Xid delta = 0 and installed QSA restores byte-for-byte.

## Non-claims after GO

A GO does not yet establish:

- 240K runtime;
- multi-request concurrency scaling;
- MTP follower correctness;
- dynamic sticky replacement across multiple turns;
- production latency/TTFT;
- an engine-start arena allocation reduction (vLLM still preallocates its
  shared KV arena according to gpu_memory_utilization).

The claim is scheduler/block-pool ownership and per-request occupancy, which is
the prerequisite for useful long-session concurrency.

## Stop rule

Run exactly one 160K frozen semantic case. Stop after evidence collection.
Do not continue to 240K, MTP, multi-request, or dynamic replacement.
