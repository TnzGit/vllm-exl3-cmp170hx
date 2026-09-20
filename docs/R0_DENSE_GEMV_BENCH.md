# Dense EXL3 SM80 GEMV dispatch bench — CMP170HX

Branch `r0/dense-gemv-bench` (PR #7, Draft), based on
`r0/coop-amdahl@a84d5d9fb0dbaf8217bd6c66b7a714935d35eb97`.

Question: **does ExLlamaV3 1.5.0's existing SM80 dense-GEMV dispatch have
>=5% end-to-end headroom left on this exact Qwen pack?**

No CUDA kernel was written and no full vLLM engine was started. Every variant
runs in its own child process because ExLlamaV3 caches these switches in C++ on
first use:

```text
EXL3_INT8_GEMV, EXL3_INT8_GEMV_MAX_K, EXL3_GEMV, EXL3_GEMV_SMEM
```

Production control group:

```text
EXL3_INT8_GEMV=2  EXL3_INT8_GEMV_MAX_K=5  EXL3_GEMV=1  EXL3_GEMV_SMEM=-1
```

## Pack identity

477 dense EXL3 entries; **0 routed `.experts.`, 0 MTP, 0 n-gram** — the catalog
excludes them as required. K distribution 242x K=4 and 235x K=5. All entries
`mul1=True, mcg=False`.

The GDN families are **34 layers at K=4 plus 2 layers at K=5** (layers 0 and 1):

| family | K=4 layers | K=5 layers | in | out |
|---|---|---|---|---|
| `linear_attn.in_proj_qkv` | 34 | 2 (layer 0,1) | 2560 | 10240 |
| `linear_attn.in_proj_z` | 34 | 2 (layer 0,1) | 2560 | 6144 |
| `linear_attn.out_proj` | 34 | 2 (layer 0,1) | 6144 | 2560 |

`lm_head` K=5 in=2560 out=248320; `self_attn.q_proj` K=5 in=2560 out=12288.

Note: a bare `--selector linear_attn.in_proj_qkv` resolves to the **first**
match, which is layer 0 (K=5). The K=4 matrix below therefore selects
`layers.10.*` explicitly so the measurement lands on the 34-layer K=4 majority
rather than the 2-layer K=5 outlier.

## K=4 matrix (3 GDN families, median of 5, graph-replay timed)

`exl3_gemv_int8_sq_kernel<4,1,true,false>` is the production kernel and the
baseline; every candidate was confirmed to switch to a different kernel.

| family | current | fp16_default | fp16_force_auto | fp16_force_shuffle | fp16_force_smem |
|---|---|---|---|---|---|
| `in_proj_qkv` | **21.580 us** | 36.108 | 36.320 | 36.148 | 37.981 |
| `in_proj_z` | **17.093 us** | 28.764 | 22.574 | 22.573 | 28.713 |
| `out_proj` | **18.495 us** | 32.495 | 32.477 | 32.483 | 34.373 |

Kernel names observed (dispatch genuinely changed):

```text
current            -> exl3_gemv_int8_sq_kernel<4, 1, true, false>
fp16_default       -> exl3_gemv_kernel<4, true, 2, 0, 1, false>   (or <4,...>)
fp16_force_auto    -> exl3_gemv_kernel<4, true, 2, 0, 0, false>
fp16_force_shuffle -> exl3_gemv_kernel<4, true, 2, 0, 0, false>
fp16_force_smem    -> exl3_gemm_kernel<4, true, 2, 16, 16|32, 512|256, 4, 3>
```

## K=4 projected end-to-end gate

Calibration fixed to this round's measured values:
`K4 int8-sq production cost = 3.659 ms/step`, `production latency = 19.053 ms/token`.
Full-engine gate requires projected e2e **>= 5%**, i.e. >= 0.953 ms/token.

| variant | K4 ratio | save ms/token | e2e % | gate |
|---|---|---|---|---|
| `fp16_default` | 1.7032 | **-2.5730** | **-13.50%** | **FAIL** |
| `fp16_force_auto` | 1.5983 | -2.1892 | -11.49% | FAIL |
| `fp16_force_shuffle` | 1.5954 | -2.1784 | -11.43% | FAIL |
| `fp16_force_smem` | 1.7679 | -2.8097 | -14.75% | FAIL |

Every candidate is a **regression of 11-15%**, not an improvement. The K=4-only
direction needed >=26% family-level reduction; the best candidate is 1.60x
*slower*.

## K=5 crossover check

K=5 needed >=55% family reduction to clear 5% e2e on its own.

| family | current | int8_k4_cap | fp16_default |
|---|---|---|---|
| `lm_head` | **580.636 us** | 805.942 (1.388x) | 812.564 (1.399x) |
| `self_attn.q_proj` | **32.820 us** | 46.905 (1.429x) | 46.954 (1.431x) |

Both candidates switch the kernel to `exl3_gemm_kernel<5,...>` and are
**1.39-1.43x slower**. No expansion to `o_proj`/`k_proj`/`v_proj`/`indexer` was
justified.

## Verdict

**dense existing-dispatch direction = CLOSED.**

ExLlamaV3 1.5.0's existing SM80 dense-GEMV dispatch space does not contribute
>=5% R0 end-to-end on this pack; the production `int8_sq` path is already the
fastest of every variant available. This matches ExLlamaV3's own upstream note
that K=4 int8 is ~9% faster than fp16 on Ampere — here the margin is much
larger, so there is nothing left to recover by env tuning.

Per the agreed plan, this closes the low-risk dense route. **No custom CUDA
kernel follows from this result.** The next targets are the two components that
together exceed any remaining single one:

```text
framework elem/copy   4.221 ms/step   18.69%
BF16 GEMM/GEMV        3.099 ms/step   13.72%
```

`EXL3_INT8_GEMV=1` (error-feedback / higher precision) was deliberately not
tested: it is a quality mode that upstream documents as slightly slower, and
this round is a throughput search.

No full-engine A/B was run, because no candidate passed the gate.
