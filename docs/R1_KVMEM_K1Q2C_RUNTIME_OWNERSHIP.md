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

## Hardware conclusion

The final CMP170HX qualification used code SHA
`ccec5bf05229bc1bde6331500f4541b0afc5dac8`. CPU/static gates were
`24 passed`, the engine reached `/health`, and the 159,533-token request
completed without OOM, crash, preemption, or Xid.

The ownership substrate passed every mechanical gate:

- one allocation-boundary event at a 9,971-page logical row;
- 4,099 real scheduler pages at the boundary and at peak, below the 4,160 cap;
- 5,872 historical pages reclaimed across 120 events;
- all 12 QSA layers bound to exact 4,161-page dedicated tensors;
- append-only worker virtual IDs remained in range and resident IDs stayed valid;
- 128 MiB D2H and 128 MiB H2D per layer, 32 jobs each;
- CPU restore was byte-exact on all 12 layers;
- installed QSA restored to its baseline SHA and the machine returned idle.

The frozen semantic gate failed reproducibly. The request emitted only
`<|im_end|>` and neither target code. The final classification is therefore:

```text
Q2C_FROZEN_PLAN_SEMANTIC_NO_GO
```

This result separates mechanism from policy: 16-token dedicated/virtual
ownership, bounded scheduler occupancy, null-hole reclaim, READ block-table / WRITE
slot-mapping coexistence, and CPU authority are live-proven. The offline K1B
resident set is not semantically sufficient when applied progressively during
prefill. Unlike Q2B, which kept the full physical source through prefill and
applied the frozen mask only to final query rows, Q2C must hide reclaimed
nonresident history from later prefill rows. The final run recorded
1,876,958,640 dropped prefill historical selections, and the target semantics
collapsed.

This run alone did not distinguish a policy failure from an implementation
error: exact CPU round-trip proves that stored bytes survive, but cannot prove
that the correct logical bytes were stored or addressed. That distinction was
resolved by the controlled attribution experiment below.

## Controlled semantic attribution

The attribution harness at exact SHA
`7ef81b931922bc9109b82e6a443d046f6bd3a950` ran three 159,533-token cases with
the same model, prompt, 1,024-token chunking, eager mode, MTP disabled, and
sampling settings:

| Case | Physical QSA source | Visibility | Result |
| --- | --- | --- | --- |
| A | stock full GPU KV | original selection | PASS, both target codes |
| B | stock full GPU KV | shared Q2C progressive mask | FAIL, immediate `<|im_end|>` |
| C | bounded Q2C ownership | same shared mask | FAIL, immediate `<|im_end|>` |

B retains the full stock GPU KV and changes only visibility. Its failure is
therefore direct evidence that the progressive frozen mask is semantically
causal; Q2C reclaim, virtual-ID reuse, CPU restore, or bounded storage are not
required to reproduce the failure. C simultaneously passed its mechanical
gates: scheduler peak 4,099 pages, 5,872 reclaims, exact READ block-table / WRITE
slot-mapping accounting, 12-layer CPU authority, and exact virtual-ID lifecycle.

Cross-process attention/selection bit fingerprints were not used as a gate.
A and B already diverged before any page became mask-eligible, demonstrating
that EXL3/CUDA run-to-run numerical variation makes cross-process bit equality
unsuitable here. The causal classification is:

```text
Q2C_ATTRIBUTION_PROGRESSIVE_POLICY_CAUSAL
```

Under the unchanged progressive frozen policy, another implementation-only
retry is not justified. A semantic successor must preserve selected history,
for example through CPU-authoritative history and bounded selection-driven
reload.

## Reload working-set feasibility

The original-selection A control was instrumented without changing its
semantics. Exact SHA `31016ab71eacd8fac8328123f8633eeef6f7cdd5` passed 39
CPU/static tests and the 160K semantic request, then measured all 1,896
layer/chunk events across 12 QSA layers.

- full 12-layer QSA history at 9,971 pages/layer is 3.6515 GiB in CPU backing;
- a whole 1,024-query chunk needs as many as 7,304 unique pages and exceeds
  the 4,160-page cap in 695 events;
- 128-query-row batches still peak at 4,723 pages and exceed the cap in 9
  events;
- 64-query-row batches peak at 3,973 pages and exceed the cap in 0 events.

The resulting classification is:

```text
Q2C_RELOAD_ROW_BATCH_64_FEASIBLE
```

This is a capacity result, not yet a semantic or performance qualification of
reload. The current `4096 sticky + 64 active` contract has no reload space and
cannot be retained. The next phase must make historical residency evictable
and define one hard pool of at most 4,160 pages containing the current writes
plus the selected working set for each <=64-row QSA sub-batch. It must save all
completed history to CPU before eviction, restore selected misses before READ,
and compare staged K/V plus attention output with the full-KV A reference.

The cold no-reuse transfer upper bound is about 114.85 GiB/request, so this
capacity proof makes no latency claim. Reuse, transfer coalescing, and overlap
belong after semantic correctness and cap enforcement are live-proven.

## Remaining non-claims

This run does not establish:

- 240K runtime;
- multi-request concurrency scaling;
- MTP follower correctness;
- causal resident selection without future-turn knowledge;
- exact hidden-state/token parity against full-history prefill;
- dynamic sticky replacement across multiple turns;
- selection-driven CPU reload correctness;
- <=64-row QSA sub-batch attention equivalence;
- production latency/TTFT;
- an engine-start arena allocation reduction (vLLM still preallocates its
  shared KV arena according to gpu_memory_utilization).

The claim is scheduler/block-pool ownership and per-request occupancy, which is
the prerequisite for useful long-session concurrency.

## Stop rule

Run exactly one 160K frozen semantic case. Stop after evidence collection.
Do not continue to 240K, MTP, multi-request, or dynamic replacement.

## Q2D CPU-authoritative streaming successor

The progressive frozen-mask policy above is closed as a semantic NO_GO, but
the bounded-ownership mechanism was successfully reused by Q2D. Final live
code SHA `c3f3d45ad8670dfef1d57401a80b46fafa0ab21f` keeps every completed QSA
page CPU-authoritative, divides the fixed GPU pool into 128 scheduler WRITE
pages plus 4,032 worker READ pages, and restores original selected history in
at most 64-query-row sub-batches.

Two fresh 160K runs both returned `violet-harbor-31, granite-comet-72`, covered
all 1,872 prefill layer/chunk records, reached exactly 4,160 combined real
pages, round-tripped at least 9,977 pages/layer bit-exactly, and ended with Xid
delta zero and an exact installed-QSA restore. End-to-end wall time remained
227--230 seconds and H2D traffic was about 68.7 GB/request, so Q2D is a
correctness/capacity qualification rather than a performance acceptance.

## Q2E transfer profile and direct-slot experiment

Q2E first added phase timing, an append-only CRC-protected access trace, and a
memory census without changing Q2D data flow. The initial staged profile at
exact SHA `24733aaffc18bf3225918a3864d38f13ef218b99` was a semantic GO and
closed all trace counts exactly: 33,120 records, 56,510,749 selected-history
pages, 2,102,171 assigned misses, and 12 layers with 2,760 records each.

That profile showed that transfer DMA alone was not the whole exposed cost.
During prefill, H2D worker wall time was 22.47 seconds, while total history
staging was 53.30 seconds; selection planning was 17.62 seconds, dynamic table
construction 18.27 seconds, and deliberately synchronous trace emission 19.35
seconds. This justified testing the already-installed vLLM worker's arbitrary
destination support before implementing reference-repo-style packing.

The direct candidate binds CPU backing to the dedicated QSA cache itself,
publishes from scheduler WRITE IDs, and loads each demand-miss set directly to
its assigned READ IDs in one job. It leaves the existing 128-page staging
tensor allocated but unused so boot memory and the QSA patch remain matched to
the staged control.

The first direct run at
`a7cb062c28a6b0fa143a0d0c4732c31e51ccfd8c` exposed a real ordering bug: all
prefill coverage, capacity, publication round-trip, and trace-integrity gates
passed, but the model emitted only `<|im_end|>`. A bounded byte oracle then
verified at least 1,851 early direct-load pages per layer against the CPU
backing and restored the correct semantic answer. Replacing that oracle with
an explicit post-load consumer barrier also restored semantics without any
readback. The barrier's measured prefill cost was only 0.25--0.26 seconds.

The final exact head is:

```text
5298916b68d15a4b73ca822585746fb5d2e884e4
```

Its fresh 160K direct run classified
`Q2D_CPU_AUTHORITATIVE_STREAMING_SEMANTIC_GO` and recorded:

- exact target answer and `finish_reason=stop`;
- 1,872/1,872 prefill records across 12 layers;
- WRITE peak 128, READ peak 4,032, combined peak 4,160;
- at least 9,977 published and bit-exact round-tripped pages/layer;
- 20,441 H2D jobs, versus 28,219 in the matched staged run;
- zero staging-to-READ D2D copies;
- 0.259 seconds total prefill consumer-barrier time;
- scheduler policy digest
  `52441ad0632294d31cf6ceabdbfb09d18062c477c495819bae16d381a4f0b514`,
  exactly matching the staged baseline;
- Xid delta zero, exact QSA restore, 14 MiB post-run GPU use, and no residual
  GPU/vLLM process.

The matched staged and direct profile wall times were 249.37 and 248.16
seconds. This is not evidence of a material end-to-end speedup. Direct I/O is
accepted as a simpler, correct transfer path with fewer jobs and no D2D bounce;
performance work should next target selection planning, table construction,
and avoidable cache misses before adding packing or overlap.

Cross-process access-trace SHA is retained as a diagnostic, not a hard gate.
The matched staged rerun was fully correct with the same plan and exact same
scheduler-policy digest, yet its access SHA differed from the earlier staged
profile. Hard trace gates therefore cover structural completeness and exact
agreement with the current run's worker counters; policy preservation is
gated by the canonical plan and normalized scheduler trace.

Q2E does not yet prove an event-fenced asynchronous consumer path, removal of
the allocated staging tensor, a broad prompt suite, 240K, MTP, multi-request
concurrency, recurrent-state restoration, or NVMe backing. Any replacement of
the synchronous consumer barrier requires a new semantic qualification.

## Q2E-2 persistent dynamic READ table

The first Q2E-2 candidate at exact SHA
`adbe46e59c120394f7951ea88faac2d434c5e385` tested small page-planning
changes. It remained a semantic GO but regressed the request to 250.57 seconds:
prefill selection rose to 18.26 seconds and table construction to 20.91
seconds. Split timing showed that table allocation itself cost only 0.73
seconds; about 19.68 seconds was spent repeatedly rebuilding Python mappings
and destination tensors. That candidate was rejected.

The successor keeps one persistent logical-to-physical table per QSA layer.
Publication, miss assignment, and eviction update only their bounded mapping
deltas; the scheduler WRITE view is refreshed once per forward. The original
selection, 64-row split, LRU residency policy, READ/WRITE partition, transfer
path, and 4,160-page cap are unchanged. Extra valid entries for other resident
pages are semantically inert because QSA indexes only the original selected
logical tokens.

An initially failed paired-policy gate exposed a gate-definition issue rather
than a scheduler change. The baseline generated 104 completion tokens while
the candidate generated 143, so a digest over the whole request included two
additional decode page cycles. At the frozen 159,533-token prompt boundary,
both runs contained the same 312 canonical scheduler events and the exact same
digest:

```text
6cfb508dbd49e52b3f8fa4a55f9a8a92dd44cf881acce7dbfcc106fa0c4c2c76
```

The gate now hashes only the frozen prefill policy, rejects any event that
crosses that boundary, and retains the full-request digest as a diagnostic.
This prevents output-length variation from weakening or spuriously failing the
ownership-policy comparison.

Exact head `ddcf37feb86021e8c05288f099d86fe8f48177cc` added a diagnostic GPU
table oracle. With the oracle required, a complete 160K run checked at least
2,634 sub-batches and 3,964,525 actual device-table entries per layer against
the exact CPU READ/WRITE mapping. It returned the target answer and passed all
semantic, coverage, capacity, CPU-authority, publication, scheduler, trace,
restore, and machine-health gates. Its 253.82-second wall time is diagnostic
only because the synchronous oracle deliberately reconstructs and reads back
the reference mapping.

The final oracle-off performance qualification used the same exact head and
classified:

```text
Q2D_CPU_AUTHORITATIVE_STREAMING_SEMANTIC_GO
```

It recorded:

- exact target answer and `finish_reason=stop`;
- 1,872/1,872 prefill records across 12 layers;
- WRITE peak 128, READ peak 4,032, combined peak 4,160;
- at least 9,979 published and bit-exact round-tripped pages/layer;
- exact frozen-prefill scheduler digest and a clean boundary gate;
- zero D2D staging copies;
- Xid delta zero, exact installed-QSA restore, 14 MiB post-run GPU use, and no
  residual GPU/vLLM process;
- request wall time 235.08 seconds;
- prefill table-management time 0.254 seconds.

Against the matched direct baseline at `5298916b`, request wall time improved
from 248.16 to 235.08 seconds (13.08 seconds, about 5.3%), while prefill table
management fell from 18.60 to 0.254 seconds. Prefill exposed worker time fell
from 124.05 to 106.86 seconds. The candidate produced 141 completion tokens
versus the baseline's 104, so the end-to-end improvement was not obtained by
shortening generation.

The next optimization target is now history-stage bookkeeping and LRU victim
selection: about 52.91 seconds remains in `stage_history`, versus 21.62 seconds
of measured H2D worker wall time. Selection planning remains about 17.62
seconds and qualification trace emission about 19.13 seconds. A semantics-
preserving LRU data-structure change should be trace-replayed against the old
victim sequence before another live run. Fine-grained producer/consumer event
ordering remains later work; the proven synchronous barrier stays frozen until
that separate candidate passes its own qualification.
