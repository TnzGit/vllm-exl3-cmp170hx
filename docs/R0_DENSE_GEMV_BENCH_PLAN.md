# Dense EXL3 C1 GEMV microbenchmark gate — CMP170HX

Post-COOP profiling moved the R0 production baseline to about **19.05 ms/output-token** and re-established the decode Amdahl on the cooperative-MoE path.

Measured dense EXL3 cost:

| family | ms/decode-step | share of 19.05 ms/token baseline |
|---|---:|---:|
| K=4 int8-sq GEMV | 3.659 | 19.2% |
| K=5 int8-sq GEMV | 1.718 | 9.0% |
| other K=4 EXL3 GEMV | 1.592 | 8.4% |
| K=5 EXL3 GEMM | 0.307 | 1.6% |
| all dense EXL3 | 7.275 | 38.2% of wall baseline / 32.22% of traced GPU kernels |

The post-COOP trace remains the source of truth for attribution. This microbenchmark only asks whether ExLlamaV3's already-shipped SM80 dispatch choices leave an exact-shape win on this CMP170HX.

## Why existing dispatch first

ExLlamaV3 1.5.0 already exposes Ampere-relevant knobs:

- EXL3_INT8_GEMV=2 — current plain int8-activation GEMV
- EXL3_INT8_GEMV=0 — disable it and fall through to fp16 paths
- EXL3_INT8_GEMV_MAX_K — current non-Hopper default is 5
- EXL3_GEMV=1/2 — heuristic / forced QTIP-style fp16 small-m GEMV
- EXL3_GEMV_SMEM=-1/0/1 — automatic / shuffle / shared-memory extraction

Upstream ExLlamaV3 comments report Ampere/RTX3090 int8 wins around 9% at K=4 and 6% at K=5 versus fp16 on their measured shapes. That is useful prior evidence, not a result for this checkpoint/CMP170HX.

Do **not** write a new CUDA kernel before testing the exact Qwen pack shapes.

## Tool

`tools/r0_bench_dense_exl3.py`

The parent process reads only safetensors metadata and launches a fresh Python child for each dispatch variant. Fresh children are mandatory because the ExLlamaV3 C++ code caches several environment variables on first use.

Each child:

1. loads the real .trellis/.suh/.svh tensors for one dense checkpoint key;
2. preserves the actual mcg / mul1 codebook flags;
3. allocates one row of fp16 input and fp32 output, matching the production dense C1 path;
4. calls exllamav3_ext.exl3_gemm directly;
5. warms up;
6. times repeated calls with CUDA events;
7. captures a one-call CUDA-profiler kernel-name diagnostic.

It does not load the full model, run vLLM, or include HTTP/scheduler cost.

## First-pass matrix

For the three K=4 GDN projection families:

- current int8 (INT8=2, max K=5)
- fp16 heuristic
- fp16 forced QTIP automatic extraction
- fp16 forced QTIP shuffle extraction
- fp16 forced QTIP shared-memory extraction

For K=5 families:

- current int8
- INT8_GEMV_MAX_K=4 (force K=5 crossover)
- int8 disabled / regular fp16 fallback

Forced QTIP K=5 variants are intentionally skipped: the ExLlamaV3 QTIP GEMV hard gate supports K <= 4.

## Qualification gate

The production baseline is ~19.05 ms/output-token. A candidate must project to at least **5% end-to-end** improvement before an engine A/B is allowed:

`19.05 * 0.05 = 0.953 ms/token`

Consequences:

- K=4 int8-sq is 3.659 ms/step, so a K=4-only candidate needs roughly **26% family-level reduction** if it affects the whole K=4 int8-sq family.
- K=5 int8-sq is 1.718 ms/step, so a K=5-only candidate needs roughly **55% reduction**.
- If a candidate affects multiple families, project the savings using their measured post-COOP ms/step contributions. Do not sum percentages from the pre-COOP trace.

These are screening thresholds, not promises of end-to-end gain.

## Stop rules

Reject without a fresh-engine production A/B when any of these is true:

- exact-pack median kernel-family gain projects to <5% e2e;
- gain exists only on a minor shape and does not cover enough production calls;
- run-to-run spread is too large to distinguish the candidate;
- the expected dispatch kernel is not actually observed;
- candidate requires a new quantized weight format or changes inference semantics merely to recover a small microbenchmark win.

If an existing dispatch variant clears the >=5% projected gate, then and only then run one fresh-engine profiler-OFF C1 A/B on the full model, with greedy parity and Xid checks.

## Accuracy

The microbenchmark output hash is a catastrophic-error guard only. fp16 and int8 activation paths are not required to produce identical floating-point outputs in isolation.

Any candidate promoted to production must pass the existing deterministic greedy token parity gate on the full model. If a faster path materially changes the numerical mode (for example disabling current int8 activation GEMV), also record a longer greedy/sample smoke before adopting it.
