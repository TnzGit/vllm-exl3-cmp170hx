# Post-COOP MTP requalification — CMP170HX

Branch `r0/mtp-post-coop-qualify`, cut from `origin/r0/coop-production`
(PR #6 head `d24ab6f3c5db336b70e539ce0ca80257820e2725`, verified ancestor).

## Headline: the previous "MTP is net negative" result was a harness bug

The earlier MTP qualification measured **stream chunks as output tokens**.
Under speculative decoding one chunk can carry several accepted tokens, so the
denominator was undercounted and MTP latency was inflated ~2-2.6x. Recomputing
the *original* k=1 raw data with the correct denominator:

| old cell | reported ms/tok | drafts+accepted | corrected ms/tok |
|---|---|---|---|
| k=1 4K sentinel | 37.60 | 50 + 46 = 96 | **19.98** |
| k=1 126K | 38.37 | 64 + 63 = 127 | **19.64** |

So k=1 was never slower than no-draft. The old conclusion was wrong in
direction, not merely in magnitude. The harness now uses
`stream_options.include_usage` for an authoritative token count and
cross-checks it against `drafts + accepted`, which agree exactly.

## Phase 1 — historical control (COOP=0)

| config | historical | remeasured | drift |
|---|---|---|---|
| no-draft | 30.10 | **30.156** ms/tok | +0.2% (in spec) |
| k=1 | 38.20 (buggy) | **20.094** ms/tok | harness artifact |

No-draft reproduces, so the runtime/config has not drifted; the k=1 difference
is entirely the measurement bug.

## Phase 2 — production COOP=1 matrix

Fresh engine per configuration, profiler OFF, PIECEWISE CUDA graph, C1, 4K
prompt (3,475 tokens), greedy, 128 output tokens, 3 measured repeats after a
discarded warm-up. Same checkpoint/pack, text-only, prefix cache off,
disk n-gram, `MADV=0`, `EXPERT_MATCH_CACHE=1`, `GC=0`, `COOP=1`.

| COOP | MTP k | ms/tok | tok/s | accepted/pass | emitted/pass | per-position acceptance |
|---|---:|---:|---:|---:|---:|---|
| 1 | no-draft | 19.054 | 52.482 | — | — | — |
| 1 | k=1 | **14.212** | **70.365** | 0.8143 | 1.8286 | [0.814] |
| 1 | k=2 | **11.081** | **90.241** | 1.8444 | 2.8444 | [0.956, 0.889] |
| 1 | **k=3** | **10.355** | **96.572** | 2.5278 | 3.5556 | [0.972, 0.861, 0.694] |

Relative to no-draft: k=1 **+34%** throughput (−25.4% latency), k=2 **+72%**
(−41.8%), k=3 **+84%** (−45.7%). Improvement is monotonic in k.

Spread was tight (k=3 cells: 10.838 / 10.047 / 10.355 ms/tok).

### Cleanliness per config

| config | PRESCAN | direct_plan | fallback | preemptions | idle after | Xid delta |
|---|---|---|---|---|---|---|
| coop1-k0 | 48 | 48 | 0 | 0 | running 0 / waiting 0 | 0 |
| coop1-k1 | 49 | 49 | 0 | 0 | 0 / 0 | 0 |
| coop1-k2 | 49 | 49 | 0 | 0 | 0 / 0 | 0 |
| coop1-k3 | 49 | 49 | 0 | 0 | 0 / 0 | 0 |

PRESCAN 49 for MTP configs = 48 main layers + 1 MTP layer. No fallback.

### Round latency

`unavailable`. No low-overhead per-verification-round timer exists, and adding
one would perturb the production numbers, which the task forbids.

## Phase 3 — dispatch / path proof

Done profiler-free from verified counters plus the plugin's own dispatch logic,
because this branch's launcher has no `--profiler-config` plumbing and the task
forbids introducing a latency-perturbing profiler.

Verify width from the spec counters (`draft_tokens / drafts`):

| config | drafts | draft_tokens | accepted | m = k+1 | dense family at m |
|---|---:|---:|---:|---:|---|
| k=1 | 70 | 70 | 57 | 2 | `exl3_gemv_int8_sq` (non-cooperative GEMV) |
| k=2 | 46 | 92 | 83 | 3 | cooperative trellis GEMM (3..16) |
| k=3 | 36 | 108 | 91 | 4 | cooperative trellis GEMM (3..16) |

- **Verify width is real**: `draft_tokens/drafts` = 1.0 / 2.0 / 3.0 exactly, so
  m = 2 / 3 / 4. This is measured, not inferred from the launcher.
- **The m=2 -> m=3 dense crossover DOES occur at k=2** (`_dense_forward` selects
  by row count; rows<=2 GEMV, 3..16 cooperative GEMM, >=17 reconstruct).
- **Coop MoE stays eligible at every k**: `m * topk` = 20 / 30 / 40, all <= 256,
  with hidden 2560 % 128 == 0 and intermediate 640 % 128 == 0.
- Acceptance counters are non-zero for every MTP config, which proves the
  speculative path is genuinely executing rather than silently disabled.

Kernel names cannot be captured on this branch without adding profiler
plumbing, so the dense-family column is derived from the plugin's row-count
dispatch rule rather than from a trace; that is a source-level fact, not a
guess about this run.

## Decision

**Case D** — a k is stably better than no-draft. Specifically k=3 at
10.355 ms/tok vs 19.054 no-draft, with clean acceptance, zero fallback,
zero preemption and zero Xid delta. There is no m=3 cliff; k=2 crossing the
GEMV -> cooperative-GEMM boundary makes it *faster*, not slower.

## Answers

1. **Does the old "MTP net negative" hold post-COOP?** No. It does not hold
   post-COOP, and it never held pre-COOP either — the old result was a
   chunk-vs-token counting bug in the harness.
2. **Has k=1 crossed break-even?** Yes, decisively: 14.212 vs 19.054 ms/tok
   (+34% throughput).
3. **Is there a discrete m=3 cliff at k=2?** No. k=2 (m=3) is 22% faster than
   k=1 (m=2) despite crossing the GEMV -> cooperative GEMM transition.
4. **Next step:** **resume MTP production qualification**, with k=3 as the
   leading candidate. Before adopting it, extend to long context (the earlier
   ~160K acceptance-cliff question is still unverified on this checkpoint) and
   confirm prefix-caching behaviour separately.

## What was not done

No production optimization was implemented. No inference math changed, no
64K head prototype, no dense dispatch change, no custom kernel. PR #7 was not
reopened and no PR was merged.
