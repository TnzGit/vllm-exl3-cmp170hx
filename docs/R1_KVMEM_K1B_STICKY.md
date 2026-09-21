# R1 K1B: persistent-topk diagnosis + sticky-residency replay

## Status

Research-only follow-up to K1A.

K1A showed:
- QSA retrieval quality remains excellent;
- fresh desired-set churn is only ~18-22% of a 64K resident budget;
- but exact same-query replay Jaccard is only ~0.67-0.69.

That classification was `HIGH_CHURN_OR_UNSTABLE`, but the evidence points to
**fresh-selector instability**, not excessive transfer volume.

K1B determines whether that instability is consistent with the SM80
`persistent_topk` tie path, and whether a stateful sticky resident planner can
absorb it without losing target recall.

No model engine is launched in K1B.

## Why persistent-topk is the leading mechanism

On NVIDIA QSA, cooperative top-k requires newer GPU capability. CMP170HX is
SM80, so the QSA top-k path uses `persistent_topk`.

For long rows the persistent kernel uses multiple CTAs and collects values equal
to the radix-selected cutoff pivot using atomic increments into the shared
output counter.

When many candidates have the same cutoff score, the exact subset of equal-score
indices admitted at the top-k boundary can depend on CTA scheduling.

The K1A synthetic archive history intentionally repeats filler text over most of
the context, creating a high-tie-density workload. That is an excellent stress
test for this mechanism.

This explanation is also consistent with K1A's otherwise surprising combination:

- exact query/history replay -> different coarse resident set;
- direct requested facts -> still 100% hit;
- target block recall -> still 100%.

In other words, semantically important high-score blocks are stable while much
of the interchangeable tail can move.

K1B does not assume this explanation is true. It measures it directly.

## GPU microdiagnostic

`tools/kvmem_persistent_topk_diagnose.py`

runs the installed vLLM `torch.ops._C.persistent_topk` directly on the
CMP170HX without loading the model.

Default:
- rows: 8
- width: 60,000
- top-k: 512
- repeats: 30

Two inputs:

### Unique-score case

Every column has a unique float score.

Expected if the kernel's general selection is stable:
- one distinct selected set per row;
- pairwise Jaccard = 1.

### Tied-cutoff case

- top 256 values: score 2
- next 2048 values: score 1
- rest: score 0
- k=512

The kernel must choose 256 of the 2048 equal-score cutoff candidates.

If repeated calls produce multiple selected sets while the unique-score case is
stable, that directly supports the tie-boundary mechanism on SM80.

A repeated `torch.topk` tied-case reference is recorded for context. It is not
used as a production backend recommendation.

## Sticky-residency replay

K1B reuses the exact 12 K1A shadow JSONLs.

It does not rerun 160K/240K inference.

The planner is stateful:

1. turn 0 takes the fresh QSA working set;
2. later turns begin from the previous resident set;
3. sink/recent mandatory blocks are always retained/admitted;
4. current QSA fresh-set candidates are ranked by current block vote count;
5. resident nonmandatory blocks are ranked from weakest to strongest by the
   same current vote count;
6. a fresh block may replace a resident block only when it has a strictly
   higher current score;
7. per-turn query-specific replacements are capped.

Ground-truth facts are **never used to select blocks**. They are used only after
the resident plan is built to evaluate recall.

## Replacement-cap sweep

For every:
- context: 160K / 240K
- block size: 128 / 256 / 512
- resident budget: 32K / 64K

K1B sweeps maximum query-specific replacement fractions:

- 5%
- 10%
- 15%
- 20%
- 25%

Primary:
- 256-token blocks
- 64K resident budget

For every policy report:
- stage-in blocks/tokens/GiB;
- stage-in fraction of resident budget;
- overlap with the fresh QSA desired set;
- target-fact recall;
- stateful turn0-vs-turn5 resident Jaccard.

Because the sticky planner is intentionally stateful, turn0 and turn5 do not
need to converge to the exact same resident set after four intervening semantic
queries. The important determinism property is that the transition rule itself
is deterministic given prior resident state + current votes.

## K1B decision signal

The primary output reports the **smallest replacement cap that preserves all
known target facts** across both 160K and 240K.

This is not a production-quality proof; the fact suite is small.

Interpretation:

### <=10% cap preserves 100% target recall

Strong signal that selection-diff runtime engineering is worthwhile.

At 64K:
- 5% cap ~= 3.2K tokens max stage-in;
- 10% cap ~= 6.4K tokens max stage-in.

Measured H2D bandwidth then converts this into a local copy-time floor.

### 15-25% needed

Still plausible, but the runtime should likely include score hysteresis,
prefetch overlap, and possibly larger blocks.

### Even 25% loses target recall

Do not implement real K1 yet. The sticky policy is too aggressive or the
fresh-selector signal is too noisy; remain in planner research.

## H2D bandwidth measurement

`tools/kvmem_h2d_bandwidth.py`

measures pinned CPU -> GPU memcpy using CUDA events at:
- 64 MiB
- 256 MiB
- 512 MiB

No model runs.

The 512 MiB effective GiB/s is used only to estimate raw copy floors for:
- 0.075 GiB
- 0.15 GiB
- 0.27 GiB
- 0.285 GiB
- 0.33 GiB

These figures exclude:
- planner time;
- gather/scatter;
- allocator work;
- synchronization;
- CPU backing lookup;
- recurrent/GDN state;
- attention.

They are transfer floors, not runtime latency predictions.

## Relationship to expected product impact

If bounded residency ultimately works:

### Concurrency

This is the strongest expected benefit.

Measured full-attention main KV:
- 24,576 B/token
- 160K: ~3.66 GiB/sequence
- 240K: ~5.49 GiB/sequence
- 64K resident: 1.50 GiB/sequence
- 32K resident: 0.75 GiB/sequence

For the main-KV component alone, 240K -> 64K reduces resident bytes by ~3.66x;
240K -> 32K by ~7.32x.

Actual end-to-end concurrency gain is lower because model weights, recurrent
state, QSA metadata, workspaces and allocator reserve do not scale away.

### Cold prefill

Do not expect an inherent speedup.

Initial ingestion still processes the full logical history and additionally
creates/backfills lower-tier state.

The first implementation may be slightly slower.

### Repeated-turn prefill / TTFT

Potentially a large benefit once CPU historical state and recurrent-state resume
are implemented, because unchanged long history can be reused instead of
recomputed.

That is a later K1/K2 feature, not measured here.

### Decode

Single-sequence TPOT should not be expected to improve dramatically.

Current production Amdahl already shows QSA as a small fraction of decode cost,
while MoE/dense work dominates.

The bigger decode-side value is aggregate throughput:
more long requests can coexist in GPU memory, enabling useful batching and
avoiding capacity/preemption pressure.

## Runner

K1B expects the prior K1A artifacts to remain available.

Default:

```bash
R0_REPO="$PWD" bash tools/r0_run_kvmem_k1b_sticky.sh
```

Override if needed:

```bash
K1A_DIR=/path/to/kvmem-k1a-churn \
R0_REPO="$PWD" \
bash tools/r0_run_kvmem_k1b_sticky.sh
```

No engine is started.

## Artifacts

- `k1b_sticky_summary.json`
- `k1b_sticky_summary.stdout.json`
- `persistent_topk_diagnosis.json`
- `persistent_topk_diagnosis.stdout.json`
- `h2d_bandwidth.json`
- `h2d_bandwidth.stdout.json`

## Non-goals

K1B does not:
- change installed vLLM source;
- launch the model;
- evict/offload KV;
- change QSA;
- change top-k backend;
- implement CPU KV backing;
- add MTP;
- add GDN replay;
- add NVMe;
- merge to production.
