# R0 MTP k=3 cooperative-MoE prelude probe

## Status

Measurement/qualification branch only. Production default remains unchanged.

Base:
`f2c6a719a1709ba40d08b97a4cc8d64b3c2256d8`

Qualified runtime:
- MTP k=3
- `VLLM_EXL3_COOP=1`
- PIECEWISE CUDA graph
- C1
- prefix cache OFF
- ~10.3 ms/output-token at 4K

Closed negative directions before this branch:
- PR #9 async MTP metadata: -0.281% e2e
- PR #10 dense FP16-output probe: -0.538% e2e
- PR #11 coop wide/narrow geometry: narrow was -12.78%; both A and B prefer wide

## Why this direction

The cooperative kernels remain the largest GPU bucket, but tile geometry is now
measured-optimal. The next low-risk question is whether framework work executed
*before* the already-selected coop path is redundant.

Current `apply_exl3_fused_moe` computes, before checking the coop branch:

1. `flat_token = arange(...).repeat_interleave(topk)`
2. FP16 routing weights
3. `expert_count = zeros(...)`
4. `ones(...)` + `expert_count.scatter_add_(...)`
5. `out = zeros(...)`
6. BF16/other input -> FP16
7. `counts = expert_count[:n_exp]`
8. `fat = counts > FAT_EXPERT_THRESHOLD`
9. `fat_route = zeros_like(...)`
10. only then checks `exl3_moe_coop`

For the qualified decode path:

- top-k = 10
- draft m=1 -> 10 slots
- verify m=4 -> 40 slots
- coop contract -> slots <= 256
- production `FAT_EXPERT_THRESHOLD=256`

Therefore, when:
`slots <= min(256, FAT_EXPERT_THRESHOLD)`

no expert count can exceed the fat threshold even in the worst case where every
slot routes to the same expert. The standard/fat bookkeeping is mathematically
unnecessary before the cooperative call.

## Probe 1: EARLY

`VLLM_EXL3_COOP_EARLY_PRELUDE=1`

The branch attempts the exact same `exl3_moe_coop` call immediately after
`map_topk_to_local`.

It only does so when:
- COOP is enabled;
- the extension exposes `exl3_moe_coop`;
- 1 <= slots <= 256;
- slots <= FAT_EXPERT_THRESHOLD;
- codebook flags are uniform;
- hidden/intermediate dimensions satisfy the existing coop contract.

Otherwise it returns to the historical path unchanged.

EARLY preserves:
- the same input FP16 conversion;
- the same routing-weight FP16 conversion;
- the same coop scratch shapes;
- the same counter zero-init;
- the same `exl3_moe_coop` arguments;
- the same FP32 output zero-init.

It removes only fallback bookkeeping that cannot be used for this shape.

## Probe 2: EARLY+EMPTY

Additionally set:
`VLLM_EXL3_COOP_OUT_EMPTY=1`

This changes only the coop output allocation:
- historical / EARLY: `torch.zeros(tokens, hidden, fp32)`
- EARLY+EMPTY: `torch.empty(tokens, hidden, fp32)`

Why this is a valid measurement candidate:

The audited ExLlamaV3 1.5.0 coop B kernel writes each output chunk after the
last active-slot arrival. For a token with no active local route,
`write_empty_row_chunk` explicitly writes zeros (or the shared-expert term;
this plugin passes no shared expert). Thus the plugin-side pre-zero is not
needed for the cooperative path.

Parity and health still gate any adoption.

## Expected trace signature

The accepted 4K Amdahl has:
- fill_zero: 339 calls/pass, 0.608 ms/pass
- index_scatter: 86 calls/pass, 0.302 ms/pass
- elementwise: 500 calls/pass, 1.153 ms/pass
- coop calls: 51/pass

Per cooperative call, the historical prelude has approximately:
- expert_count zeros
- ones tensor
- fat_route zeros
- output zeros
- counter zeros
- one expert_count scatter
- fat comparison / ancillary elementwise work

EARLY retains output zeros + counter zeros, so relative to BASE it should remove
about:
- **153 fill/zero calls/pass** (3 * 51)
- **51 scatter calls/pass**
- associated compare/arange/repeat work

EARLY+EMPTY should remove one additional output zero per layer:
- **204 fill/zero calls/pass** total reduction vs BASE (4 * 51)

These call-count deltas are mechanism checks, not performance promises.

## Runner installation hygiene

PR #10 exposed that `vllm_exl3` is installed as a non-editable pip copy.

This runner therefore:

1. proves this branch has **no csrc delta** vs the frozen base;
2. materializes the exact base `exl3.py` from Git and requires the installed
   plugin to match it byte-for-byte;
3. backs up the installed plugin;
4. copies this branch's pure-Python `exl3.py` into site-packages;
5. runs BASE/EARLY/EARLY+EMPTY using flags off/on against the same code;
6. records the compiled `.so` hash;
7. EXIT trap restores the exact original installed plugin;
8. verifies the compiled extension did not change.

No manual site-packages editing is required.

## Speculative denominator

Carries the corrected cross-check:
`-1 <= drafts + accepted - usage.completion_tokens <= k`

API `usage.completion_tokens` remains the authoritative performance
denominator.

## Executor

```bash
R0_REPO="$PWD" bash tools/r0_run_coop_prelude_ab.sh
```

It runs:
- CPU gates including `bash -n` contract tests;
- BASE 4K x5, profiler OFF;
- EARLY 4K x5, profiler OFF;
- EARLY+EMPTY 4K x5, profiler OFF;
- token parity vs BASE;
- 14-pass profiler traces for all three;
- component comparison for:
  - coop_moe_a/b
  - fill_zero
  - index_scatter
  - elementwise
  - dtype_copy;
- 160K only for a >=3% parity-clean winner;
- Xid and cleanup.

Profiler wall time is diagnostic only.

## Decision

### <1% e2e and expected framework kernels do not materially disappear
Close the direction as ineffective / probe failure.

### <1% e2e but expected call-count mechanism is confirmed
Close as a real but too-small optimization unless it is nearly free to retain.
Do not spend CUDA/kernel engineering on it.

### 1-3% e2e, parity/stability clean, mechanism confirmed
This is a plausible low-risk stackable framework optimization. Qualify at 160K
before deciding whether to carry it into the production baseline.

### >=3% e2e
Strong pass. Runner automatically takes a 160K sentinel.

### EARLY wins but EARLY+EMPTY regresses
Keep the early-bookkeeping change only; do not remove output zero-init.

### EARLY+EMPTY adds measurable gain over EARLY
The full-write assumption is supported by parity + trace and can be qualified
further.

## Non-goals

Do not in this experiment:
- change MoE tile geometry;
- change K-split;
- rebuild ExLlamaV3;
- modify CUDA kernels;
- change MTP k;
- change lm_head;
- use the 6bpw hybrid n-gram;
- enable prefix cache.
