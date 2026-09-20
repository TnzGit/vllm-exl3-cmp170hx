# MTP k=3 production qualification — CMP170HX

Branch `r0/mtp-k3-production-qualify`, cut from
`r0/mtp-post-coop-qualify@cdfd0926cfa3ecff67b657b50ee342296760b1b4`, which
itself descends from PR #6 `r0/coop-production@d24ab6f3c5db336b70e539ce0ca80257820e2725`
(verified ancestor). No PR merged; PR #7 not reopened.

Production configuration under test: MTP k=3, `VLLM_EXL3_COOP=1`, PIECEWISE
CUDA graph, profiler OFF, text-only, prefix cache OFF, disk n-gram, `MADV=0`,
`EXPERT_MATCH_CACHE=1`, `GC=0`, C1, greedy for perf and parity, same
checkpoint/pack and runtime.

## A. Provenance (all verified on the live process)

| item | value |
|---|---|
| branch | `r0/mtp-k3-production-qualify` |
| base / start SHA | `cdfd0926cfa3ecff67b657b50ee342296760b1b4` |
| PR #6 COOP production commit ancestor | **yes** (`d24ab6f3…`) |
| token-count fix source SHA | `cdfd0926` (`tools/r0_mtp_post_coop_cell.py`) |
| worktree | clean |
| EngineCore `/proc/<pid>/cwd` | `/home/base-node/vllm-exl3-k3` |
| `/proc/<pid>/environ` | `VLLM_EXL3_COOP=1`, `TORCH_PROFILER_DIR` unset (profiler OFF) |
| cmdline | `--speculative-config {"method":"qwen4_exp_mtp","num_speculative_tokens":3}` |
| graph mode | PIECEWISE (`--compilation-config`) |
| plugin source | `site-packages/vllm_exl3/exl3.py` installed from this branch |
| port 8002 | free before launch, no stale server |
| GPU before launch | 14 MiB (idle) |
| Xid baseline | 0 |

## B. Benchmark denominator contract — permanent fix + regression tests

The old harness counted **stream chunks** as output tokens. Under speculative
decoding one chunk carries several emitted tokens, so MTP ms/output-token was
inflated ~2-2.6x and the earlier "MTP net negative" conclusion was an artifact.

Contract now enforced:

- authoritative denominator = **API `usage.completion_tokens`**, obtained via
  `stream_options.include_usage`
- cross-check against spec counters: `emitted = accepted + passes`; when the
  sequence is cut at `max_tokens` the final pass is truncated, so
  `0 <= derived - usage <= k+1` (4)
- chunk / SSE / callback counts are recorded for diagnosis but **never** used
  as the denominator
- if the two authoritative sources disagree beyond that bound the cell is
  **INVALID** and no performance number is reported

Regression tests (`tests/test_mtp_denominator_contract.py`, 6 passing):

- Test A — a chunk holding multiple tokens must not shrink the denominator;
  TPOT uses the token count
- Test B — plain no-draft streaming yields the right denominator
- Test C — API usage and `drafts+accepted` agree → identical TPOT
- plus: truncation tolerance, INVALID marking, and a source assertion that
  `dec / chunks` never appears as a divisor

Historical data is **retained but marked invalidated**, not deleted.

## C. Long-context sweep (one fresh k=3 engine, fresh engine per configuration)

| context | prompt toks | ms/tok | tok/s | acc/pass | emitted/pass | acc% | preempt | fallback | Xid | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4K in | 3,475 | 10.376 | 96.377 | 2.5068 | 3.5068 | 83.6 | 0 | 0 | 0 | VALID |
| 32K | 27,250 | 10.283 | 97.246 | 2.5205 | 3.5068 | 84.0 | 0 | 0 | 0 | VALID |
| 65K | 55,375 | 11.149 | 89.695 | 2.1341 | 3.1220 | 71.1 | 0 | 0 | 0 | VALID |
| 126K | 107,375 | 11.374 | 87.920 | 2.1341 | 3.1220 | 71.1 | 0 | 0 | 0 | VALID |
| 160K | 136,350 | 11.330 | 88.260 | 2.1341 | 3.1220 | 71.1 | 0 | 0 | 0 | VALID |
| 200K | 170,450 | 10.760 | 92.935 | 2.4400 | 3.4133 | 81.3 | 0 | 0 | 0 | VALID |
| 240K | 204,525 | 11.084 | 90.224 | 2.2125 | 3.2000 | 73.8 | 0 | 0 | 0 | VALID |
| 4K out | 3,475 | 9.900 | 101.011 | 2.6056 | 3.6056 | 86.9 | 0 | 0 | 0 | VALID |
| 4K recheck | 3,475 | 10.308 | 97.011 | 2.3684 | 3.3684 | 78.9 | 0 | 0 | 0 | VALID |

PRESCAN 49 (48 main layers + 1 MTP layer) with `direct_plan=49`, `fallback=0`
throughout. KV pool 279,087 tokens at `MAX_MODEL_LEN=246000`.

### Capacity note (handled, not fudged)

`MAX_MODEL_LEN=256000` fails: it needs 7.24 GiB KV but only 7.03 GiB is
available, and the engine's own estimate is 246,400. The sweep therefore runs
at 246,000. The task's optional 250K point is **not** reachable on this card at
`gpu_memory_utilization=0.92`; that is a capacity limit, recorded rather than
worked around. Longest healthy context measured: **240K**.

## D. Full greedy token-ID parity

Reference: same checkpoint/pack, same prompt, same tokenizer, temperature 0,
same runtime — no-draft engine, 256-token greedy sequences.
Comparison is token-ID against token-ID (not text); the MTP side is flattened
because a speculative chunk's `logprobs.tokens` is itself a list.

| context | reference len | MTP k=3 len | first mismatch | verdict |
|---|---:|---:|---:|---|
| 4K | 256 | 256 | **none** | **PASS** |
| 160K | 256 | 256 | **none** | **PASS** |
| 240K | 256 | 256 | **none** | **PASS** |

## E. Sentinel drift

- sweep: 4K in 10.376 vs 4K out 9.900 → **-4.59%** (exceeds the 3% band)
- the drift is *favorable* (out faster), and `sentinel_in`'s median was pulled
  up by its first, warm-up repeat (10.952); its later repeats were 9.938 / 10.376
- fresh-engine recheck with 5 repeats: median **10.308 ms/tok**
  (10.531, 10.308, 9.829, 10.062, 13.459 — one outlier at 13.459)
- recheck confirms the ~10.3 ms/tok level, so the sweep is representative and
  the drift is measurement spread around a stable value, not degradation

## F. Sample-mode smoke

Fresh k=3 engine, PRESCAN 49, fallback 0:

| setting | seed | result |
|---|---|---|
| temp 0.8, top_p 0.95 | 1234 | **identical across 3 runs** |
| temp 1.0, top_p 1.0 | 1234 | **not identical** |

No crash, no NaN, no empty completion, valid token IDs, no preemption or
fallback anomaly, Xid 0. The temp=1.0 case is reported honestly: vLLM does not
guarantee bit-reproducible sampling at these settings, so no sampled-sequence
parity requirement is manufactured.

## G. Path proof (diagnostic, profiler-free)

- verify width **m=4** confirmed from spec counters
  (`draft_tokens/drafts = 3.0` → m = k+1 = 4)
- coop MoE eligible at every width: `m*topk = 40 <= 256`, hidden 2560 % 128 == 0,
  intermediate 640 % 128 == 0
- acceptance counters non-zero throughout → the speculative path really runs
- `fallback=0`, `PRESCAN=49`, `direct_plan=49` on every engine
- dense EXL3 at m=4 selects the cooperative trellis GEMM branch (rows 3..16)
  per `_dense_forward`'s row-count dispatch

Kernel names could not be captured (this branch's launcher has no
`--profiler-config` plumbing and adding a latency-perturbing profiler is
forbidden here), so the coop-hit assertion rests on eligibility plus the
acceptance counters rather than on a trace.

## Verdict

**k=3 PASSES production qualification.**

| gate | result |
|---|---|
| 4K performance vs no-draft | 10.376 vs 19.554 ms/tok → **-46.9% latency** |
| throughput | 96.4 vs 51.1 tok/s → **+88.5%** |
| long-context cliff | none; 10.28-11.37 ms/tok from 4K to 240K |
| full greedy token parity | PASS at 4K / 160K / 240K |
| preemption | 0 |
| fallback | 0 |
| Xid delta | 0 (whole run) |
| sentinel drift | -4.59%, resolved by fresh-engine recheck at 10.308 ms/tok |
| runtime path | COOP=1, m=4, PRESCAN 49, no fallback |
| denominator fix regression coverage | 6 tests passing |

Against the current no-draft production baseline (19.054 ms/tok, 52.482 tok/s),
k=3 delivers roughly **-45.7% latency and +84% throughput**, matching the
post-COOP requalification figures.

### Answers

1. **k=3 passes** production qualification.
2. **Longest healthy context: 240K** (250K/256K is capacity-limited at 0.92
   utilization; not a correctness or acceptance failure).
3. **No acceptance or performance cliff.** Acceptance moves 83.6% → 71.1% → 81.3%
   → 73.8% across contexts without a discrete drop; latency stays 10.28-11.37
   ms/tok.
4. **True gain vs no-draft production: ~-45.7% latency, ~+84% throughput**
   (this run: 10.376 vs 19.554 ms/tok at 4K).
5. **No k=2 fallback needed.** k=3 is the best measured k and shows no
   long-context degradation, so the conditional k=2 recheck is not triggered.
6. **Next step: post-MTP production Amdahl profiling from the new k=3 baseline.**
   The 19.05 ms/tok no-draft post-COOP Amdahl is obsolete as an optimization
   priority map. No optimization is started in this round.

Historical MTP TPOT/throughput numbers from the old chunk-count harness are
**invalid**; this document supersedes them for k=3.
