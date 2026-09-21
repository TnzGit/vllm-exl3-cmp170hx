# R0 MTP async-metadata backport experiment

## Why this is now P0

Post-MTP k=3 profiling at `f2c6a719a1709ba40d08b97a4cc8d64b3c2256d8`
found a recurring host/runtime signature of:

- 84 `cudaStreamSynchronize` calls over 14 verification passes
- exactly **6 stream synchronizations/pass**
- k=3 means 3 MTP draft steps/pass, i.e. **2 syncs/draft step**
- 1036 `aten::item` / `_local_scalar_dense` events over the same window

That 2-per-draft-step signature matches upstream vLLM PR #55054, **Optimize
PLE MTP metadata transfers**, which replaced synchronous CPU->GPU request-index
`.to(device)` copies with `async_tensor_h2d`.

Upstream reported, on a different platform/configuration (GB300, FP8, TP1,
MTP3, C1):

- `cudaStreamSynchronize`: 2/step -> 0
- GPU activity span: 11.185 -> 9.272 ms/step (-17.1%)
- summed GPU kernel time unchanged
- C1 output throughput: +9.877%

Those numbers do **not** transfer directly to CMP170HX. The important point is
that our trace independently shows the same synchronization cardinality.

## Why this outranks another COOP on/off A/B

The current k=3 production candidate already proves:

- `VLLM_EXL3_COOP=1` is active
- stock MoE fallback = 0
- MoE remains the largest GPU-kernel bucket
- m=4 verify is healthy

Turning COOP off can re-prove that the existing optimization is valuable, but it
does not measure *remaining* headroom inside the already-selected cooperative
kernel.

By contrast #55054 targets an observed, recurring MTP-only host synchronization
signature and changes no model math.

## Repository-side preparation

This branch adds:

- `tools/patch_vllm_qwen4_exp/patch_vllm_mtp_async_metadata.py`
  - exact v0.29-compatible backport of the merged upstream #55054 delta
  - fail-fast on layout drift
  - idempotent
  - compile-checked
  - first-application `.orig` backup
- `tests/test_mtp_async_metadata_patch.py`
  - exact v0.29 anchor application
  - idempotence
  - fail-closed layout drift
  - required helper import guard

No production default is changed.

Candidate patch command:

```bash
python tools/apply_qwen4_exp_patches.py <site-packages/vllm> \
  --profile text-mtp \
  --mtp-async-metadata
```

Baseline uses the identical command without `--mtp-async-metadata`.

For the qualified local R0 layout, the prepared one-shot executor is:

```bash
R0_REPO=/path/to/r0/mtp-async-metadata-backport \
  bash tools/r0_run_mtp_async_metadata_ab.sh
```

It runs the CPU gates, BASE 4K/160K capture, ASYNC 4K A/B, exact flattened-token
parity recheck, and only if the 4K candidate clears 3% it runs ASYNC 160K plus
a short profiler diagnostic and `r0_trace_sync_summary.py`. The script refuses
to start if the installed vLLM is already async-patched, to prevent a
contaminated BASE.

## CMP170HX hardware A/B contract

Use the same qualified k=3 production configuration for both cells:

- MTP k=3
- `VLLM_EXL3_COOP=1`
- PIECEWISE CUDA graph
- prefix cache OFF
- disk-backed n-gram
- C1
- same prepared checkpoint
- profiler OFF for formal performance numbers
- fresh engine per patch configuration

Cells:

1. **BASE** — normal vLLM 0.29 + existing Qwen EXL3 patches
2. **ASYNC** — BASE + `patch_vllm_mtp_async_metadata.py`

At 4K, run at least 3 measured repeats per cell and compare median:

- ms/output-token
- output tok/s
- accepted/pass
- emitted/pass
- preemption
- fallback
- Xid delta
- full greedy token-ID parity against BASE

Use `stream_options.include_usage` completion tokens as the authoritative
denominator. Never use stream chunk count.

If ASYNC gains >=3-5% e2e with parity/stability clean, repeat the decisive cell
at 160K. If the gain is <3%, keep the backport as a negative/low-value result
and do not productionize it.

## Diagnostic proof after the profiler-OFF A/B

Only after formal performance numbers, take a short server-side diagnostic
trace and confirm:

- `cudaStreamSynchronize/pass` goes from ~6 to ~0 (or explain every residual)
- summed GPU kernel work remains materially unchanged
- no graph-shape miss/eager island is introduced
- the two removed syncs correspond to the request-index H2D path

Profiler-on wall latency is diagnostic only.

## Next decision

- PASS >=5%: qualify/productionize the backport before further kernel work.
- PASS 3-5%: keep if complexity remains this small; then rebuild the production
  baseline.
- <3%: close this direction and move to the next measured target.
