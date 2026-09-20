# Eager caller attribution for the post-COOP residual — CMP170HX

Branch `r0/eager-attribution` (PR #8, Draft), based on
`r0/coop-amdahl@a84d5d9f`.

Goal: turn the 32.4% of traced production GPU time whose callers were invisible
under CUDA graphs into concrete model modules. **Eager latency is not a
performance number** and is never compared to the 19.053 ms/token production
baseline; only caller identity is taken from this trace.

## Runtime identity (verified on the live process)

`/proc/<pid>/cwd` = `/home/base-node/vllm-exl3-eager` for both API server and
EngineCore. `--enforce-eager` present in the real cmdline.
`ENFORCE_EAGER=1`, `VLLM_EXL3_COOP=1`, `VLLM_EXL3_MADV_AFTER_H2D=0`,
`VLLM_EXL3_EXPERT_MATCH_CACHE=1`, `VLLM_EXL3_GC_AFTER_MOE_LAYER=0`,
`TORCH_PROFILER_RECORD_SHAPES=1`. Server log: `Cudagraph is disabled under eager
mode`, `execution mode EAGER (profiling only)`. PRESCAN 48, direct_plan 48,
fallback 0.

Trace: `…1789919781982908139.pt.trace.json.gz`, 39,188,220 B,
sha256 `7a8314fba6385255…e6beac41e`. 16-chunk 4K window, 1.6M events.

## Method: the two-hop correlation chain

torch profiler links launches as:

```text
cpu_op."External id" == cuda_runtime."External id"
cuda_runtime."correlation" == kernel."correlation"
```

Matching `kernel.correlation` directly to `cpu_op."External id"` skips the
middle hop. That mistake gave a **uniform 31.2% coverage across every family**
and paired kernels with unrelated `aten::empty`/`aten::view` ops — a pattern
that is obviously wrong. With the correct two-hop chain:

```text
target_kernel_ms=268.9  matched_ms=253.0  target_correlation_coverage=94.1%
```

Per-family coverage: dense EXL3 100%, coop MoE 100%, direct_copy 100%,
gemv2T 100%, gemvx 100%, **cutlass_wmma 0%** (still unattributed).

Kernel mix matches production exactly, so the attribution transfers:
gemv2T 97/step, cutlass 97/step, dense EXL3 385/step; `coop_a = 720 = 48 x 15`.

## Caller remap table

Production cost is taken from the COOP-on graph trace; only the caller comes
from eager.

| production kernel family | production ms/step | eager CPU op | eager Python/module parent | count/step | confidence |
|---|---:|---|---|---:|---|
| `gemv2T_kernel` (BF16) | 1.292 | `aten::mm` | HyperConnection `input_mix_weight_down_block_inject` | 96 | **HIGH** |
| `gemvx::kernel` (BF16) | 0.654 | `aten::mm` | attention/HC glue, two shapes | 84 | MEDIUM |
| `cutlass_80_wmma` (BF16) | 1.055 | `<unmatched>` | — | 97 | **unattributed** |
| `direct_copy_kernel` | 1.342 | `aten::copy_` | `Tensor.to` / `Tensor.half` (dtype convert) | 365 | HIGH |
| `bfloat16_copy_kernel` | 0.539 | `aten::copy_` | `Tensor.to` | 256 | HIGH |
| `memcpy32_post` | (in framework bucket) | — | — | — | LOW |
| `FillFunctor` | 0.578 | `aten::fill_` | `torch.zeros` (allocation materialization) | 349 | HIGH |
| `compare_scalar_kernel` | 0.305 | `aten::gt` / `lt` / `ge` | `vllm_exl3/exl3.py:1397 map_topk_to_local` | 150 | HIGH |
| `index_elementwise_kernel` | 0.043 | — | — | 10 | LOW |
| `_scatter_gather_elementwise` | 0.164 | `aten::scatter_add_` | MoE routing accumulation | 48 | MEDIUM |
| `persistent_topk` / `bitonicSort` | 0.217 | `_C::persistent_topk` | `persistent_topk` (pybind) | 12 | HIGH |

### The HyperConnection hypothesis: confirmed

The open question was whether the BF16 GEMV/GEMM excess came from
HyperConnection projections. It does, and the arithmetic is exact:

`GatedResidual.input_mix_weight_down_block_inject` is a
`MergedColumnParallelLinear(hyper_hidden_size, [lora_rank, hc_count, pad_size])`
with `hyper_hidden_size = hidden_size * hc_count = 2560 * 4 = 10240`,
`lora_rank = hc_lowrank = 320`, `hc_count = 4`, and
`pad_size = (-(320 + 4)) % 16 = 12`.

That gives output `320 + 4 + 12 = 336`, i.e. exactly the observed eager shape
`aten::mm([1, 10240], [10240, 336])`, at 96 calls/step (2 per layer x 48).

Confidence HIGH: the shape is not merely plausible, it is uniquely determined by
the module's own padding formula.

## Corrected component Amdahl

The earlier `HyperConnection = 2.96%` counted only the native `_hc_*` elementwise
kernels. Its BF16 projections were sitting in the `BF16 GEMM/GEMV` bucket. With
the remap:

| component | before (ms/step) | corrected (ms/step) | share of 22.58 |
|---|---:|---:|---:|
| HyperConnection (native `_hc_*` + `gemv2T` projections) | 0.669 | **1.961** | **8.7%** |
| BF16 GEMM/GEMV remaining | 3.099 | 2.134 (cutlass 1.055 + gemvx 0.654 + residual) | 9.5% |
| framework elem/copy | 4.221 | 4.221 | 18.7% |
| dense EXL3 | 7.275 | 7.275 | 32.2% |
| coop MoE residual | 4.509 | 4.509 | 20.0% |

**HyperConnection more than doubled** as a share (2.96% → ~8.7%) once its BF16
projections are included. QSA and GDN were checked the same way: no significant
`direct_copy`/BF16/fill traffic mapped back to them, so their ~2.5% each remains
a floor rather than a corrected total — but neither grew materially.

## What is still unattributed

`cutlass_80_wmma_tensorop_bf16` at **1.055 ms/step** with 0% correlation match,
despite the same count/step (97) as `gemv2T`. Same call frequency but a
different kernel suggests a second GEMM on the same codepath whose launch is not
correlated in this trace. It is reported as unknown rather than folded into the
HC figure.

## Gate check for a next candidate

Production baseline 19.053 ms/token; 5% = 0.953 ms/token.

| candidate family | production contribution | can it clear 5% alone? |
|---|---|---|
| HyperConnection total | 1.961 ms/step | **marginally yes in principle** |
| framework elem/copy | 4.221 ms/step | yes |
| cutlass BF16 (unknown) | 1.055 ms/step | only if nearly eliminated |

The largest single newly-attributed item is HyperConnection at ~1.96 ms/step.
The biggest actionable-looking block remains framework elem/copy at 4.221
ms/step, dominated by dtype conversions (`Tensor.to`/`half`: 1.88 ms/step) and
allocation materialization (`torch.zeros`: 0.578 ms/step).

No optimization was implemented in this round, and no performance A/B was run:
this engine is eager + profiler-on, which is attribution-only by construction.
