# Post-MTP k=3 production Amdahl — CMP170HX

Branch `r0/mtp-k3-amdahl`, from `r0/mtp-k3-production-qualify@4042c4bef9c6dc8e61de78db88b253ccd611b3e0`.
No PR merged; no optimization implemented.

## 0. Ledger number clarification (resolved, not a typo)

Both numbers are real, measured at different output lengths:

| value | output tokens | meaning |
|---|---:|---|
| **19.054 ms/tok / 52.482 tok/s** | 128 | canonical no-draft production baseline |
| **19.554 ms/tok / 51.141 tok/s** | 256 | no-draft reference captured for parity |

The k=3 qualification used 256 tokens, so its like-for-like comparison is
**10.376 vs 19.554 (-46.9% latency, 96.377 vs 51.141 = +88.5% throughput)**.
Against the canonical 128-token baseline: **~10.3 vs 19.054, about -45.5%
latency and +83.6% throughput**. Both baselines are now length-tagged so a
future ledger cannot silently mix them.

## A. Provenance

| item | value |
|---|---|
| branch | `r0/mtp-k3-amdahl` |
| base / start SHA | `4042c4bef9c6dc8e61de78db88b253ccd611b3e0` |
| PR #6 COOP ancestor | **yes** (`d24ab6f3…`) |
| denominator-fix ancestor | **yes** (`cdfd0926…`) |
| EngineCore `/proc/<pid>/cwd` | `/home/base-node/vllm-exl3-k3a` |
| `/proc/<pid>/environ` | `VLLM_EXL3_COOP=1`, `TORCH_PROFILER_DIR=…/mtp-k3-amdahl/traces` (diagnostic) |
| cmdline | `num_speculative_tokens: 3` |
| graph mode | PIECEWISE (100% of kernels carry `graph node id`) |
| plugin source | `site-packages/vllm_exl3/exl3.py`, installed from this branch |
| GPU / port / Xid before launch | 14 MiB idle / 8002 free / 0 |

Regression coverage: 11 tests passing (6 denominator contract + 5 correlation
chain). The correlation test explicitly forbids the earlier one-hop mistake
(kernel.correlation looked up as a cpu_op External id).

## B. Method notes

- Window defined by **verification passes**, never stream chunks: 14 passes per
  context, with `draft_tokens/drafts = 3.0` confirming **m = 4**.
- Per-token normalisation uses `emitted = accepted + passes` (the API usage
  count was not available inside the truncated window, so the equivalent
  authoritative counter identity was used).
- `passes=14`, emitted: **51 (4K)** and **43 (160K)**.
- Profiler-ON wall time is never treated as a production number; production
  figures come from the profiler-OFF qualification (10.3-11.3 ms/tok).

## C. Production kernel Amdahl (4K, 14 passes, 51 emitted)

| component | ms/win | ms/pass | ms/token | share | calls/pass |
|---|---:|---:|---:|---:|---:|
| coop_moe_a | 101.98 | 7.284 | 1.9996 | **19.9%** | 51 |
| dense_gemm (m=4 coop GEMM) | 73.40 | 5.243 | 1.4393 | 14.3% | 145 |
| dense_gemv | 67.83 | 4.845 | 1.3300 | 13.2% | 270 |
| coop_moe_b | 65.22 | 4.658 | 1.2788 | 12.7% | 51 |
| OTHER | 61.42 | 4.387 | 1.2043 | 12.0% | 539 |
| dtype_copy | 37.82 | 2.701 | 0.7416 | 7.4% | 833 |
| bf16_gemm | 30.75 | 2.197 | 0.6030 | 6.0% | 403 |
| **lm_head** | **18.52** | **1.323** | **0.3632** | **3.6%** | **3** |
| elementwise | 16.14 | 1.153 | 0.3164 | 3.2% | 500 |
| hyper_conn | 10.48 | 0.748 | 0.2054 | 2.0% | 314 |
| qsa | 9.21 | 0.658 | 0.1807 | 1.8% | 82 |
| fill_zero | 8.51 | 0.608 | 0.1669 | 1.7% | 339 |
| index_scatter | 4.22 | 0.302 | 0.0828 | 0.8% | 86 |
| topk_sort | 4.03 | 0.288 | 0.0790 | 0.8% | 16 |
| gdn | 2.64 | 0.188 | 0.0517 | 0.5% | 36 |
| ple_ngram | 0.05 | 0.004 | 0.0010 | 0.0% | 1 |

**MoE a+b = 167.20 ms/win = 32.6%** vs **lm_head = 18.52 ms = 3.6%** → **MoE is
9.0× larger than lm_head**.

### 160K (14 passes, 43 emitted)

Same ordering and near-identical per-pass costs:

| component | ms/pass | share |
|---|---:|---:|
| coop_moe_a | 7.516 | 20.1% |
| dense_gemm | 5.246 | 14.0% |
| dense_gemv | 4.842 | 13.0% |
| coop_moe_b | 4.769 | 12.8% |
| lm_head | 1.323 | 3.5% |

**MoE a+b = 32.9%**, **lm_head = 3.5%** (ratio 9.3×). The mix is stable across
context; only `qsa` grows materially (1.8% → 2.4%).

## D. Draft vs verify, and the 248K draft lm_head question

Only **14% of GPU time** has a correlated CPU op (86% is graph-replay and cannot
be attributed by caller), so this is a partial but decisive slice.

Within the attributed slice:

| side | ms (attributed) | calls/pass | key kernels |
|---|---:|---:|---|
| **DRAFT m=1** | 34.49 | 181 | `exl3_gemv_int8_sq_kernel<5>` (lm_head), `<3>`, `exl3_gemv_kernel<3>` |
| **VERIFY m=4** | 32.00 | 147 | `exl3_gemm_kernel<5…>`, `gdn_decode_post_conv_mtp`, `persistent_topk` |

**lm_head is confirmed DRAFT-side and m=1.** The caller attribution shows
`exl3_gemv_int8_sq_kernel<5, 1, true, false>` reached through
`vllm::exl3_linear_forward` with input dims **`[[1, 2560], []]`** — a single row,
42 calls over 14 passes = **3 calls/pass**, i.e. **one per draft step**. No m=4
lm_head call appears.

By contrast the verify path shows `exl3_gemm_kernel<5…>` with dims
**`[[4, 2560], []]`** — m=4 rows, the cooperative GEMM branch.

### Answer to the headline question

**No — 64K draft lm_head pruning is not the first optimization target.**

- lm_head total: **3.6% of GPU kernel time**, 0.363 ms/output-token
- even **100% elimination** of draft lm_head is worth only **~0.37 ms/token**
  (~3.6% e2e) — below the 5% gate on its own
- MoE is **9× larger** and is the dominant component

Confidence: lm_head share and call count **HIGH** (direct kernel + caller +
dims). The draft/verify split is **HIGH** for the lm_head rows (observed
`[[1,2560]]`) and **INFERRED** for the rest where only call-count arithmetic
applies.

## E. CPU / runtime gap

| signal | count in window | note |
|---|---:|---|
| `cudaStreamSynchronize` | 84 | 6/pass |
| `cudaEventSynchronize` | 14 | 1/pass |
| `cudaDeviceSynchronize` | 1 | startup/teardown |
| `cudaMemcpyAsync` | 490 | 4.09 ms total |
| `aten::item` / `_local_scalar_dense` | 1036 | 74/pass — host-sync risk |

There is a **recurring fixed per-pass synchronization pattern** (~6 stream syncs
and ~74 scalar extraction ops per verification pass). Absolute wall impact is
**not reliably quantifiable** from this trace because profiler overhead
dominates host-side timing; this is flagged as a gap signature worth a dedicated
low-overhead probe, not as a measured cost.

## F. CUDA graph coverage

- **100% of 51,366 kernels carry `graph node id`** → the entire k=3 pipeline
  (draft steps, m=4 verify, sampler/metadata) executes inside PIECEWISE graph
  replay.
- No eager island was observed; no shape-miss was identified in this run.
- Consequence: caller attribution is structurally limited (86% unmatched), which
  is why several draft/verify assignments are marked INFERRED.

## G. Context delta decomposition (4K → 160K)

`ms/output-token = ms/pass / emitted_per_pass`, so the delta splits cleanly:

| | 4K | 160K |
|---|---:|---:|
| verification passes | 14 | 14 |
| accepted | 37 | 29 |
| emitted | 51 | 43 |
| **emitted/pass** | **3.643** | **3.071** |
| acceptance rate | **88.1%** | **69.0%** |
| ms/pass (GPU) | 36.588 | 37.353 |
| GPU ms/output-token | 10.044 | 12.162 |

Decomposition (closes exactly):

- **acceptance effect: +1.869 ms/token (+18.6%)**
- **per-pass kernel cost effect: +0.249 ms/token (+2.1%)**
- total: **+2.118 ms/token (+21.1%)**

**Interpretation:** the 4K → 160K slowdown is almost entirely an **acceptance
effect** (fewer tokens emitted per verification pass), not a kernel slowdown.
Per-pass compute grows only ~2%. Measured wall went 10.38 → 11.33 ms/token
(+9.2%), smaller than the GPU-window +21.1% because the wall clock includes the
full production path and the window is a profiler-perturbed sample — so the
decomposition is a *relative* split, not an absolute latency prediction.

## H. Ceilings vs production 10.3 ms/output-token (4K)

| component | share | full removal | e2e if 100% removed | 50% fix | e2e at 50% |
|---|---:|---:|---:|---:|---:|
| **MoE a+b** | **32.6%** | **3.36 ms/tok** | **32.6%** | **1.68** | **16.3%** |
| dense_gemm | 14.3% | 1.48 | 14.3% | 0.74 | 7.2% |
| dense_gemv | 13.2% | 1.36 | 13.2% | 0.68 | 6.6% |
| OTHER | 12.0% | 1.24 | 12.0% | 0.62 | 6.0% |
| dtype_copy | 7.4% | 0.76 | 7.4% | 0.38 | 3.7% |
| bf16_gemm | 6.0% | 0.62 | 6.0% | 0.31 | 3.0% |
| **lm_head** | **3.6%** | **0.37** | **3.6%** | **0.19** | **1.8%** |
| elementwise | 3.2% | 0.32 | 3.2% | 0.16 | 1.6% |

A component share is an upper bound, not an achievable gain.

## I. Optimization candidates (max 3, evidence-backed)

### 1. EXL3 routed MoE (coop a+b) — P0 candidate

- current contribution: **32.6%** (3.28 ms/output-token at 4K, 4.00 at 160K)
- suspected removable fraction: unknown; coop already replaced the stock path,
  so realistic headroom is modest — perhaps 20-40% with further work
- projected e2e: **6.5-13%** at 20-40%; **not** the full 32.6%
- complexity/risk: high — kernel work, needs parity + stability
- next minimal experiment: extend the existing coop A/B to k=3 m=4 shapes and
  measure whether any remaining knob (slots, split-k, kernel selection) moves
  end-to-end at all before writing code

### 2. Dense EXL3 verify path (m=4 cooperative GEMM + m=1 draft GEMV)

- current contribution: **dense_gemm 14.3% + dense_gemv 13.2% = 27.5%**
- suspected removable fraction: unclear; the earlier dispatch sweep on m=1/m=2
  found no win, and m=3/m=4 already use cooperative GEMM
- projected e2e: up to ~14% if half-removable, but **prior evidence says the
  existing dispatch space is exhausted**, so realistic is low
- complexity/risk: medium-high (kernel), and prior negative result applies
- next minimal experiment: exact-shape m=1/2/3/4 microbench only if candidate 1
  proves unpromising

### 3. Framework/copy/elementwise + speculative metadata

- current contribution: **dtype_copy 7.4% + elementwise 3.2% + fill_zero 1.7% +
  index_scatter 0.8% ≈ 13.1%**, and much of it is plausibly
  speculative-specific (hidden feedback, acceptance bookkeeping, per-draft copies)
- suspected removable fraction: moderate — fusion/buffer reuse often removes a
  large fraction of pure copy traffic
- projected e2e: **~5-7%** if half of the copy/elementwise traffic is removed
- complexity/risk: **lowest of the three** — no kernel math, no numerical change
- next minimal experiment: eager caller attribution restricted to copy/fill/
  index ops to identify the speculative-specific subset

### Chosen P0 next experiment

**Candidate 1 first, but as a measurement, not an implementation:**
run one k=3 m=4 coop A/B to establish whether any MoE headroom exists at the new
shape. If it shows nothing, pivot to **candidate 3** (framework/copy), which has
the best risk-adjusted profile and a plausible ~5-7% e2e.

Explicitly **not** chosen: 64K draft lm_head pruning — measured at 3.6%, below
the 5% gate even if fully eliminated.

## What was not done

No optimization, no 64K head pruning, no lm_head vocab slicing, no host-sync
removal, no custom kernel, no dense dispatch change, no MoE kernel change, no
framework copy elimination, no graph-mode change, no prefix cache, no MTP k
adjustment, no production launcher adoption or merge.

## J. P0 update after upstream/static audit — #55054 outranks COOP re-A/B

The original section I selected a k=3 COOP on/off measurement as the next
experiment. That remains a valid historical comparison, but it is **superseded
as P0** by stronger evidence found after this Amdahl was written.

The same 14-pass trace records **84 `cudaStreamSynchronize` calls = 6/pass**.
With k=3, that is exactly **2 stream synchronizations per draft step**.

vLLM upstream PR #55054 ("Optimize PLE MTP metadata transfers") removes exactly
two synchronous request-index CPU->GPU transfers per Qwen4Exp MTP step by using
`async_tensor_h2d`. The v0.29.0 source used by this project contains the same
synchronous anchors and already imports `async_tensor_h2d`, so the delta is a
small, clean selective-backport candidate.

Upstream measured the mechanism on GB300/FP8, not CMP170HX/EXL3, so its reported
+9.877% C1 throughput is **not** used as a local performance claim. The reason
for P0 promotion is the independent cardinality match in our own trace.

Repository-side backport preparation and the hardware A/B contract live in
`docs/R0_MTP_ASYNC_METADATA.md`.

Therefore the next order is:

1. A/B #55054-equivalent async metadata transfers on the qualified k=3 path.
2. If <3% e2e, close that direction.
3. Then choose between COOP-kernel research and dense dtype-boundary work using
   the measured residual map.

The measured Amdahl itself is unchanged.
