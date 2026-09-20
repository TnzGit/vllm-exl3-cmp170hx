# Post-COOP decode Amdahl — CMP170HX Qwen3.8-Flash-Next EXL3

Branch `r0/coop-amdahl`, base commit `d5047ddf858984e290fe794c0d6ca3234279a017`.

The pre-COOP percentages (MoE 42%, dense 21.7%) were measured against
30.188 ms/output-token and **no longer describe the production candidate**.
This stage re-measures the **19.053 ms/output-token** COOP=1 path and picks the
next target from that.

## Runtime under profile

```text
vLLM 0.29.0, ExLlamaV3 1.5.0, same prepared pack
text-only, no draft, prefix cache OFF, disk n-gram, PIECEWISE, C1, 4K
VLLM_EXL3_MADV_AFTER_H2D=0
VLLM_EXL3_EXPERT_MATCH_CACHE=1
VLLM_EXL3_GC_AFTER_MOE_LAYER=0
VLLM_EXL3_COOP=1
```

`VLLM_EXL3_COOP=1` was confirmed in the live process, not the launcher:
`/proc/<pid>/environ` of the API server shows `PWD=…/vllm-exl3-coopamd`,
`VLLM_EXL3_COOP=1`, `TORCH_PROFILER_DIR=…/coop-amdahl/traces`. PRESCAN 48,
direct_plan 48, staging fallback 0.

## Traces

| | main profile | shapes profile |
|---|---|---|
| purpose | Amdahl attribution | BF16 caller resolution |
| file | `…1789914010834793363.pt.trace.json.gz` | `…1789914468809617277.pt.trace.json.gz` |
| size | 24,119,915 B | 10,438,802 B |
| sha256 | `3381230764baedea…e48d501c` | `93ccf2355362cc05…e1b6f18e` |
| actual prompt tokens | **3,475** | 3,475 |
| profile window | 20.01 s, 32 chunks | 14 chunks |
| decode steps | **31** | ~13 |

Decode steps are derived from the coop MoE call count: 1,488 calls / 48 layers
= 31. Traces are not committed; only path, size and hash are recorded.

Profiler-on latency was **not** used as a performance number anywhere.

## Cooperative MoE is genuinely active

The trace contains `exl3_moe_coop_a_kernel<3,2,true>` and
`exl3_moe_coop_b_kernel<3,2,true>` at **1,488 calls each** = 48 layers x 31
steps. The pre-COOP `exl3_moe_kernel` does not appear at all, and the server log
has no fallback. So the residual MoE cost below is the real coop kernel, not a
silent fallback.

## Post-COOP Amdahl (4K decode, 31 steps)

| component | calls | ms/window | % | ms/decode-step |
|---|---|---|---|---|
| **dense EXL3** | 11,935 | 225.54 | **32.22%** | **7.275** |
| **coop MoE (residual)** | 2,976 | 139.77 | **19.97%** | **4.509** |
| framework elem/copy | 49,042 | 130.86 | 18.69% | 4.221 |
| **BF16 GEMM/GEMV (unattributed)** | 8,618 | 96.06 | **13.72%** | 3.099 |
| OTHER | 14,229 | 51.93 | 7.42% | 1.675 |
| HyperConnection | 8,990 | 20.74 | 2.96% | 0.669 |
| GDN/recurrent | 2,232 | 17.56 | 2.51% | 0.566 |
| QSA/indexer | 2,294 | 17.48 | 2.50% | 0.564 |
| PLE/n-gram | 31 | 0.12 | 0.02% | 0.004 |
| total GPU kernels | | 700.06 | 100% | 22.58 |

### Dense EXL3 breakdown (the new #1)

| kernel | calls | ms | ms/step | calls/step |
|---|---|---|---|---|
| `exl3_gemv_int8_sq_kernel<4,1,true,false>` | 6,696 | 113.42 | 3.659 | 216 |
| `exl3_gemv_int8_sq_kernel<5,1,true,false>` | 1,891 | 53.27 | 1.718 | 61 |
| `exl3_gemv_kernel<4,true,2,0,0,false>` | 2,914 | 49.34 | 1.592 | 94 |
| `exl3_gemm_kernel<5,true,2,16,32,128,4,3>` | 434 | 9.51 | 0.307 | 14 |

`calls/step` identifies the families without guessing:

- **216 K=4 calls/step** = 108 K=4 linears x 2. The K=4 dense set is exactly the
  GDN projections `linear_attn.in_proj_qkv` (36), `in_proj_z` (36) and
  `out_proj` (36). By weight bytes these are **21.2% + 12.7% + 12.7% = 46.6%**
  of all dense EXL3 bytes (28.52 + 17.11 + 17.11 MiB of 134.7 MiB total).
- **61 K=5 calls/step** = `lm_head` (1, and the single largest K=5 tensor at
  17.6% of dense bytes), `self_attn.q_proj` (12) + `o_proj` (12), indexer
  (12), `k_proj`/`v_proj` (24).

So the dominant dense kernel is K=4 and its dominant owner is the **GDN
linear-attention projections**, with `lm_head` the largest single K=5 linear.

## Correlation coverage: why BF16 stays unattributed

`correlated_kernel_coverage = 6.02%`. The shapes trace explains why: **100% of
decode kernels carry a `graph node id`**, i.e. every one is a CUDA-graph replay.
Graph-replayed kernels have no per-launch CPU op by construction, so
CPU-op -> kernel attribution is structurally unavailable for this workload, not
merely under-sampled. Per the agreed rule, the BF16 GEMM/GEMV family is therefore
reported as unattributed with names and counts rather than assigned to a module.

| BF16 kernel | calls | ms | ms/step |
|---|---|---|---|
| `gemv2T_kernel_val<...bf16...>` (cuBLAS) | 3,007 | 39.04 | 1.259 |
| `cutlass_80_wmma_tensorop_bf16_s161616gemm` | 3,007 | 31.81 | 1.026 |
| `internal::gemvx::kernel<...bf16...>` | 2,604 | 25.21 | 0.813 |

A call-count coincidence worth noting but **not** treated as proof:
`gemv2T` + cutlass are 3,007 calls = 97/step, which equals the `_hc_gate_mix`
count/step. HyperConnection projections are the leading hypothesis, but with
correlation unavailable this remains a hypothesis to confirm, not a finding.

## `VLLM_EXL3_COOP_GEMM` is not a candidate

`_dense_forward()` only branches on row count: rows 1-2 always take the plain
non-cooperative GEMV, and `VLLM_EXL3_COOP_GEMM` only changes the 3..144 row
cooperative-GEMM dispatch. C1 decode is rows=1, so this switch cannot affect the
19 ms decode path. It was not A/B'd.

## Residual MoE verdict

Residual coop MoE is **19.97%**, just under the 20% re-check line, and the
dispatch is confirmed per-layer with zero fallback. Combined with the earlier
finding that the 30.188 -> 19.053 ms/token win already captured ~88% of the
pre-COOP MoE cost, **the MoE direction is finished**. No further MoE work.

## Next target: dense EXL3

Dense EXL3 is now the single largest component at **32.22% (7.275 ms/step)**,
ahead of residual MoE. Within it:

- K=4 GEMV (`gemv_int8_sq<4>`) is the largest single kernel at 3.659 ms/step,
  owned by the GDN `linear_attn` projections
- `lm_head` is the largest single K=5 linear

Because no existing low-risk switch targets the dense C1 GEMV path, this stage
does **not** implement an A/B. The next step is an isolated SM80 shape
microbenchmark on the K=4 and K=5 GEMV shapes, gated on a projected end-to-end
gain of >=5% before any production experiment.

Amdahl ceiling: if dense EXL3 (32.22%) could be removed entirely, the ceiling is
1/(1-0.3222) = **1.475x**, i.e. 19.053 -> 12.92 ms/output-token. That bounds the
whole dense direction.

Upstream SM90/SM100 items (#54560 Hopper LL-GEMM, #54524 FlashInfer CuTeDSL,
#54687 HC combine-norm) are out of scope: wrong architecture, and #54687 targets
MTP input which R0 production does not use.
