# R0 MTP k=3 dense EXL3 dtype-boundary probe

## Status

This is a **diagnostic measurement branch**, not a production optimization.

It starts from the accepted post-MTP Amdahl baseline:

- base: `f2c6a719a1709ba40d08b97a4cc8d64b3c2256d8`
- MTP k=3
- `VLLM_EXL3_COOP=1`
- PIECEWISE graphs
- prefix cache OFF
- 4K production baseline ~10.3 ms/output-token

The immediately previous upstream #55054 async-metadata experiment was hardware
qualified and closed negative:

- BASE 10.320 ms/tok
- ASYNC 10.349 ms/tok
- latency gain -0.281%
- parity PASS
- Xid/preemption/fallback clean

PR #9 is closed and must not be merged.

## Why the dtype-copy bucket is no longer "framework miscellany"

The post-MTP Amdahl measured:

- non-lm-head `dense_gemv`: **270 calls/pass**
- non-lm-head `dense_gemm`: **145 calls/pass**
- draft `lm_head`: **3 calls/pass**
- `dtype_copy`: **833 calls/pass**

The plugin's `Exl3LinearMethod._apply_impl` has two explicit dtype boundaries for
ordinary BF16-input dense EXL3 calls:

1. input:
   `x_2d.to(torch.float16).contiguous()`
2. output:
   `y.to(dtype=x.dtype)`

For the 415 non-lm-head dense calls:

`(270 + 145) * 2 = 830`

The three draft lm_head calls contribute the remaining input-side conversions:

`830 + 3 = 833`

This exactly matches the observed dtype-copy call count. The 7.4% bucket can
therefore be treated as an EXL3 dtype-boundary cost with high confidence rather
than an unattributed framework bucket.

## ExLlamaV3 v1.5.0 source audit

Exact dependency commit:

`0740edc2da569fb99174023c1d2988b1e98cb41e`

Relevant facts:

1. The plugin constructs `LinearEXL3` with default `out_dtype=torch.float16`.
2. The plugin then overrides every dense EXL3 call with
   `out_dtype=torch.float32`.
3. ExLlamaV3 `BC_LinearEXL3::run_alloc` currently exposes only a boolean
   `output_fp32` distinction:
   - true -> FP32 output
   - false -> FP16 output
4. Dense EXL3 CUDA inputs are typed as `const half*`.
5. Existing epilogues similarly specialize FP32 or FP16 output. There is no
   native BF16 dense I/O path.

Therefore simply asking for BF16 output is **not available** in the current
native ABI.

## Why the first experiment uses FP16 output

A true production-quality boundary removal would ideally:

- read BF16 input directly and convert on-load exactly as needed by the existing
  FP16 arithmetic, eliminating BF16->FP16 materialization;
- retain the historical FP32 epilogue arithmetic;
- store the final result directly as BF16, eliminating FP32->BF16
  materialization.

That requires native ExLlamaV3 kernel/ABI work.

Before paying that implementation cost, this branch measures the output-side
ceiling using functionality the dependency already has.

With:

`VLLM_EXL3_DENSE_FP16_OUT_PROBE=1`

only linears satisfying all of the following use the existing FP16 EXL3
epilogue:

- input dtype is BF16;
- no mixed BF16 shards;
- not an lm_head.

The final returned tensor is still cast to the original BF16 dtype.

This isolates the hot ordinary dense path and avoids lm_head/mixed-shard
confounds.

## Important correctness warning

The FP16 epilogue changes internal rounding relative to the historical FP32
epilogue.

Therefore this probe is **not** a production candidate even if it is faster.

Full greedy token parity is measured only to characterize sensitivity. A
performance win from this probe is evidence to justify implementing a
BF16-native output path that preserves FP32 arithmetic, not evidence to enable
the probe itself in production.

## Benchmark counter fix carried on this branch

The speculative cross-check is also corrected.

For a measured interval:

`derived = verification_passes + accepted_draft_tokens`

The API usage count can be one larger because the initial target token can
precede the first speculative pass. At the opposite end, max-token truncation
can make the final pass account for up to `k` additional derived tokens.

The valid boundary is therefore:

`-1 <= derived - usage <= k`

where `k` is measured from draft tokens/pass in the same interval.

The API `usage.completion_tokens` remains the authoritative performance
denominator.

This fixes the false INVALID observed in the #55054 A/B
(`derived=255, usage=256`) without weakening the cross-check into an arbitrary
absolute tolerance.

## One-shot executor

Run:

```bash
R0_REPO="$PWD" bash tools/r0_run_dense_output_probe.sh
```

The runner performs:

1. CPU gates.
2. BASE: historical FP32 epilogue, 4K, 5 repeats, profiler OFF.
3. PROBE: FP16 epilogue, 4K, 5 repeats, profiler OFF.
4. Flattened greedy-token parity.
5. 160K PROBE only if 4K gain >=3% and parity PASS.
6. A 14-pass PROBE profiler trace regardless of e2e result.
7. Amdahl comparison against the frozen f2c6a71 4K trace:
   - dense_gemm
   - dense_gemv
   - dtype_copy
   - lm_head
   - combined dense + dtype boundary.
8. Xid and cleanup.

Profiler-on wall latency is never a production number.

## Decision rules

### Probe e2e <2% and per-pass dense+boundary reduction small

Close the output-side direction. Do not implement a BF16-native epilogue.

Move on to another measured target (likely MoE or input-side native BF16
boundary work only if separately justified).

### Probe e2e 2-5% or clear >=1 ms/pass dense+boundary reduction

The output side is material but the FP16 probe itself is still not
production-safe.

Next step: implement a native BF16 output path that preserves the FP32
epilogue arithmetic and compare exact hidden/output parity.

### Probe >=5%

Strong evidence that native BF16 output support is worth implementing.
Still do **not** productionize `VLLM_EXL3_DENSE_FP16_OUT_PROBE=1`.

### Greedy parity mismatch

Record it. A mismatch rejects the FP16 probe as a production candidate but does
not invalidate its use as a performance-ceiling measurement.

## Explicit non-goals

Do not in this experiment:

- modify MoE kernels;
- change MTP k;
- change COOP;
- enable prefix cache;
- alter lm_head;
- use the 6bpw hybrid n-gram table;
- change graph mode;
- merge a production PR.
