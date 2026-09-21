# Post-COOP eager attribution plan — CMP170HX

## Why this diagnostic exists

Post-COOP CUDA-graph profiling established the production baseline at about 19.05 ms/output-token, but only 6.02% of GPU kernel time could be correlated back to CPU launch ops because 100% of decode kernels were replayed from CUDA graphs.

The largest unresolved residual families are:

- framework elementwise/copy: 4.221 ms/decode-step (18.69% of traced GPU time)
- BF16 GEMM/GEMV unattributed: 3.099 ms/decode-step (13.72%)

Together they are about 32.4% of traced post-COOP GPU time.

Dense EXL3 existing-dispatch tuning has already been closed by exact-pack microbenchmark evidence: all fp16/QTIP/K5 crossover alternatives were slower than the current Ampere int8-sq path and none cleared the >=5% projected end-to-end gate.

## Diagnostic mode

Use vLLM 0.29 `--enforce-eager`, which its own debug documentation defines as disabling both torch.compile and CUDA graphs.

This mode is **attribution only**.

Do not compare eager/profiler-on latency with the production 19.05 ms/token CUDA-graph baseline.

Keep all other relevant runtime identity fixed:

- same prepared EXL3 pack
- vLLM 0.29.0
- ExLlamaV3 1.5.0
- text-only
- no draft
- prefix cache off
- disk n-gram
- C1
- VLLM_EXL3_COOP=1
- qualified low-risk load policy

## Tools

- `tools/r0_serve_coop_eager_profiler.sh`: sets the profiling-only eager contract.
- `tools/r0_profile_decode_window.py`: starts profiling only after decode has begun.
- `tools/r0_capture_eager_attribution.sh`: captures 16 decode chunks by default, identifies the new trace, hashes it and runs the analyzer.
- `tools/r0_analyze_eager_attribution.py`: follows CPU external-id -> CUDA runtime correlation -> kernel correlation, then attempts to find the closest containing Python/function event.

The analyzer focuses on:

- gemv2T / gemvx
- cutlass_80_wmma
- direct/bfloat16 copy
- memcpy32_post
- FillFunctor
- compare_scalar
- index_elementwise / scatter-gather
- persistent top-k / bitonic sort

It also infers decode steps from `exl3_moe_coop_a_kernel` count / 48 layers when possible.

## Success criterion

The diagnostic succeeds when target-kernel correlation coverage is high enough to assign most of the 7.32 ms/step BF16+framework residual to concrete CPU ops / model code paths.

Prefer:

- target correlation coverage >=70%, or
- lower overall coverage but one dominant target family with a repeated, unambiguous CPU/Python caller.

If correlation remains poor even under `--enforce-eager`, do not guess. Use a shorter profiler trace with stack/shape data or add narrowly scoped model annotations before optimizing.

## Optimization gate after attribution

Do not optimize a kernel family merely because it is visually prominent.

For any newly attributed candidate, require:

1. exact production caller / shape / frequency;
2. measured post-COOP ms/decode-step contribution;
3. projected end-to-end ceiling on the 19.05 ms/token production baseline;
4. plausible >=5% end-to-end opportunity before new kernel work;
5. if an existing low-risk implementation/flag exists, test that before writing a new kernel.

QSA (~2.5%), GDN recurrent core (~2.5%), HyperConnection native kernels (~3%), PLE (~0.02%) remain below the current optimization threshold unless attribution shows their surrounding BF16/framework work was previously classified elsewhere.

## Permanent warning

`--enforce-eager` changes the execution architecture. Its timings are not production timings. The only valid use here is restoring launch attribution that CUDA-graph replay structurally hides.
