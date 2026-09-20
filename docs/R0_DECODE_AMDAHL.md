# no-draft decode Amdahl profile on CMP170HX

Branch `r0/nodraft-amdahl`, base commit `501ce8160b1b304a5464f2b56d629bfe81af4ac0`.

Question: **of the ~30.1 ms/output-token, which component is the largest one
that can actually be reduced?**

## Method

Server-side profiling only. The engine was launched with `TORCH_PROFILER_DIR`
so vLLM 0.29 registers `/start_profile` and `/stop_profile`; worker traces carry
CPU + GPU events. A client-side `torch.profiler` was never used for GPU
attribution.

`tools/r0_profile_decode_window.py` streams a request, waits for prefill plus a
few emitted chunks, then opens the profiling window so the trace is decode-only.

| | 4K | 160K |
|---|---|---|
| actual prompt tokens | 3,475 | 136,350 |
| profile window | 32 chunks | 24 chunks |
| decode steps in window | ~30 (1,457 MoE calls / 48 layers) | ~23 (1,081 / 48) |
| trace file | `…1789909367111552337.pt.trace.json.gz` | `…1789909589407358061.pt.trace.json.gz` |
| trace sha256 | `fa52f133…78cb76d9` | `13542e93…88d281c` |

Traces are not committed; only paths, sizes and hashes are recorded.

## Amdahl table (GPU kernel time, decode window)

| component | 4K ms | 4K % | 160K ms | 160K % |
|---|---|---|---|---|
| **EXL3 routed MoE** | **437.907** | **42.04%** | **324.583** | **41.79%** |
| **dense EXL3** | **225.783** | **21.67%** | **167.173** | **21.52%** |
| OTHER (unclassified family) | 319.186 | 30.64% | 239.180 | 30.79% |
| GEMM_OTHER | 34.304 | 3.29% | 25.371 | 3.27% |
| QSA / indexer | 10.711 | 1.03% | 10.147 | 1.31% |
| GDN / recurrent | 13.153 | 1.26% | 9.738 | 1.25% |
| PLE / n-gram | 0.698 | 0.07% | 0.521 | 0.07% |
| total GPU kernels | 1041.743 | 100% | 776.713 | 100% |

EXL3 (MoE + dense) is **63.7% / 63.3%** of decode GPU time.

### Largest and second-largest

1. **EXL3 routed MoE — 42%.** Single kernel
   `exl3_moe_kernel<3,128,2,16>`: 428.972 ms over 1,457 calls at 4K
   (294 µs/call), 317.965 ms over 1,081 calls at 160K.
2. **dense EXL3 — 21.7%.** Dominated by `exl3_gemv_int8_sq_kernel<4,1,true,false>`
   (112.042 ms / 6,696 calls), `<5,1,true,false>` (53.615 ms / 1,891) and
   `exl3_gemv_kernel<4,true,2,0,0,false>` (50.362 ms / 2,914).

### Unclassified GPU time (OTHER, 30.6%)

Reported as kernel names with counts and durations, not attributed to a
component I cannot prove:

| kernel | 4K ms | count |
|---|---|---|
| `gemv2T_kernel_val<...bf16...>` (cuBLAS) | 39.473 | 3,007 |
| `at::native::direct_copy_kernel` | 34.436 | 7,843 |
| `cutlass_80_wmma_tensorop_bf16` gemm | 32.254 | 3,007 |
| `internal::gemvx::kernel<...bf16...>` | 25.012 | 2,604 |
| `at::native::bfloat16_copy_kernel` | 17.423 | 7,843 |
| `memcpy32_post` | 13.286 | 7,440 |
| `FillFunctor<long>` elementwise | 12.376 | 7,626 |
| `compare_scalar_kernel<long>` | 11.708 | 4,619 |
| `_hc_combine_norm_kernel` | 9.779 | 2,945 |
| `_scatter_gather_elementwise` ReduceAdd | 8.160 | 2,976 |
| `bitonicSortKVInPlace` | 8.054 | 1,488 |
| `persistent_topk_kernel<512,4>` | 6.924 | 372 |
| `index_elementwise_kernel` (int64) | 6.628 | 1,612 |
| `_hc_gate_mix_kernel` | 6.227 | 3,007 |
| `splitKreduce_kernel` | 7.404 | 3,007 |
| `_hc_silu_kernel` | 4.945 | 3,007 |

The `_hc_*` family (hyper-connection mixer, `hc_count=4`) totals ~21 ms (2.0%)
and is a real architectural component that the bucket classifier does not know.
Several entries are int64 index/fill/compare elementwise ops, which is a
candidate family for a later look but is not attributed here.

### Instrumentation caveat

CPU annotation rows are **not** component attribution inside a CUDA-graph
region: `vllm::qwen_gdn_attention_core_fused_norm_packed` reports 616.96 ms CPU
total but only 20.45 ms of its own GPU time across 1,116 calls — it is a fused
graph-scope annotation, not 39% of decode work. The GPU kernel rows above are
the evidence; the heuristic buckets were only a grouping aid.

## Why decode is flat in context

| | 4K | 160K |
|---|---|---|
| prompt tokens | 3,475 | 136,350 |
| GPU kernel time per decode step | **34.32 ms** | **34.49 ms** |
| EXL3 MoE share | 42.04% | 41.79% |
| dense EXL3 share | 21.67% | 21.52% |
| QSA + GDN share | 2.29% | 2.56% |

Normalising by decode steps (MoE calls / 48 layers) gives **34.32 vs 34.49 ms
per step** — a 0.5% difference across a 39x prompt-length increase, and the
component mix is unchanged. Decode is dominated by weight-bound EXL3 GEMM/GEMV;
QSA sparse attention is ~1% and GDN ~1.3% at both lengths. The sparse/recurrent
architecture genuinely decouples decode cost from KV length, which is why the
measured no-draft curve is flat at ~30.1 ms/token from 4K to 250K.

These per-step figures are profiler-inflated and are used only for the
*relative* comparison, per the rule that profiler-on latency is not a
performance number.

## Candidate A/B: `VLLM_EXL3_COOP=1`

Coop eligibility was read from the actual model geometry, not assumed:

```text
topk = 10                 (num_experts_per_tok)
tokens = 1                (C1 decode)
tokens * topk = 10  <= 256              OK
hidden = 2560,  2560 % 128 == 0         OK
intermediate = 640, 640 % 128 == 0      OK
codebook = mul1 -> flags uniform        OK
exllamav3_ext.exl3_moe_coop present     OK
fat route: tokens 1 <= 256 -> impossible OK
```

Fresh engine per config, profiler **OFF** (profiler-on latency is not a valid
performance number), C1, 3 repeats, median:

| config | 4K ms/output-tok | 4K output tok/s | 160K ms/output-tok | 160K output tok/s |
|---|---|---|---|---|
| stock `COOP=0` | 30.188 | 33.126 | 30.144 | 33.174 |
| **candidate `COOP=1`** | **19.053** | **52.484** | **19.037** | **52.529** |
| change | **−36.9%** | **+58.4%** | **−36.9%** | **+58.4%** |

- run-to-run spread 0.040 ms (stock) and 0.025 ms (coop) at 4K
- preemptions 0 in every cell
- greedy parity: identical token IDs on the deterministic prompt, both configs
- Xid delta 0 for both configs
- 4K and 160K gains are identical, so the win is context-independent

### Relationship to the Amdahl table

A 30.188 → 19.053 ms/token reduction removes 11.135 ms, i.e. **36.9% of total
decode**. MoE accounted for 42.0% of GPU time, so coop eliminated roughly 88%
of the MoE kernel cost. Against an absolute ceiling of 1/(1−0.42) = 1.72x
(17.5 ms/token) the measured 1.58x is about 92% of what killing MoE entirely
could ever give.

**Verdict: keep `VLLM_EXL3_COOP=1`.** It is a large end-to-end win with parity
and no Xid, not a standalone-kernel win.

## Recommendation

1. **Adopt `VLLM_EXL3_COOP=1`** as the no-draft production setting. It is
   already implemented and gated; no new code is required.
2. Because the MoE direction is ~92% exhausted, the next target is **dense EXL3
   (21.7%)**, followed by the **OTHER long tail (30.6%)** — in particular the
   cuBLAS `gemv2T`/`gemvx` bf16 GEMVs and the int64 index/fill/compare
   elementwise family, which may be reducible without kernel work.
3. QSA (1.0-1.3%) and GDN (1.3%) are **not** worth targeting at R0; the
   upstream QSA candidates (#54513, #54873) should not be prioritised on this
   evidence. Disk n-gram is 0.07% and should stay out of scope.

## Production adoption

The CMP170HX service launcher now defaults `VLLM_EXL3_COOP=1` while keeping it
environment-overridable. The plugin/library default remains unchanged; this is a
hardware-specific service-profile decision backed by the fresh-engine A/B above.

Because this optimization changes the decode bottleneck distribution materially,
the pre-coop Amdahl percentages must not be reused to choose the next target.
Re-profile the coop-on production profile before changing dense EXL3 or OTHER kernels.
