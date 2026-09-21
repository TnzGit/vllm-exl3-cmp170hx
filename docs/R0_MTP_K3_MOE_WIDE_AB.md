# R0 MTP k=3 cooperative MoE wide-vs-narrow A/B

## Status

Measurement branch only. No production default changes.

Base:

`f2c6a719a1709ba40d08b97a4cc8d64b3c2256d8`

Qualified runtime:

- MTP k=3
- `VLLM_EXL3_COOP=1`
- PIECEWISE CUDA graph
- C1
- prefix cache OFF
- current 4K production baseline ~10.3 ms/output-token

Closed negative experiments before this branch:

1. vLLM #55054 async MTP metadata:
   - 10.320 -> 10.349 ms/token
   - -0.281% gain
   - PR #9 closed, not merged
2. dense EXL3 FP16-output probe:
   - 10.222 -> 10.277 ms/token
   - -0.538% gain
   - dense + dtype-boundary cost increased by 0.670 ms/pass
   - PR #10 closed, not merged

## Why return to MoE

The accepted post-MTP Amdahl is still dominated by cooperative routed MoE:

- `coop_moe_a`: 7.284 ms/pass, 19.9%
- `coop_moe_b`: 4.658 ms/pass, 12.7%
- combined: **11.942 ms/pass / 32.6% of GPU time**

This is the largest remaining measured component.

The old suggestion "run COOP on/off" is not useful enough: COOP is already the
qualified path and an off/on comparison only re-proves an optimization we have
already adopted.

The next question is instead:

> Is the current cooperative kernel geometry itself wrong for the actual k=3
> draft/verify shapes on CMP170HX?

## ExLlamaV3 1.5.0 source finding

Exact audited dependency commit:

`0740edc2da569fb99174023c1d2988b1e98cb41e`

In `exl3_moe_coop.cu`:

```cpp
static bool moe_coop_pick_wide(int kslices, int slots, int device)
{
    const int mode = moe_coop_wide_mode();
    if (mode >= 0) return mode != 0;
    if (DevCtx::instance().get_cc(device) < CC_BLACKWELL) return true;
    return kslices >= 256 || (kslices >= 128 && slots >= 32);
}
```

Thus on SM80 / Ampere the automatic policy is simply:

**wide for both stage A and stage B, for every decode shape.**

The source comment directly above this logic says the intended policy is wide
for long-k and narrow otherwise, but the pre-Blackwell branch bypasses that
shape heuristic.

For this checkpoint's actual geometry:

- hidden / gate-up input = **2560** -> 160 k-slices
- routed intermediate / down input = **640** -> 40 k-slices
- top-k = 10
- target verification m=4 -> 40 slots
- draft m=1 -> 10 slots

If the Blackwell shape heuristic were applied literally:

### target verify m=4

- A: 160 slices, 40 slots -> **wide**
- B: 40 slices, 40 slots -> **narrow**

### draft m=1

- A: 160 slices, 10 slots -> **narrow**
- B: 40 slices, 10 slots -> **narrow**

This does **not** prove that the Blackwell heuristic is right for Ampere. It
does show that the current unconditional-wide Ampere rule deserves a direct
measurement at our production shape.

## Why the first test does not patch/rebuild CUDA

ExLlamaV3 already exposes:

`EXL3_MOE_COOP_WIDE=0/1`

The existing flag forces the geometry for **both** A and B.

This branch therefore first compares:

1. AUTO:
   - environment unset
   - on CMP170HX/SM80 this means wide A + wide B
2. NARROW:
   - `EXL3_MOE_COOP_WIDE=0`
   - narrow A + narrow B

Two same-session-style 14-pass traces are also captured.

The traces keep `coop_moe_a` and `coop_moe_b` separate.

Therefore even if global narrow loses end-to-end, the experiment can still
answer whether one stage improves while the other regresses. Only then is a
stage-specific CUDA/source patch justified.

## Source-attestation guard

The previous PR #10 execution exposed an important operational risk: the
installed `vllm_exl3` was a non-editable pip copy from an older worktree.

The runner on this branch refuses all GPU work unless:

`installed vllm_exl3/exl3.py == this branch src/vllm_exl3/exl3.py`

byte-for-byte.

It also verifies the installed ExLlamaV3 source contains the exact
`EXL3_MOE_COOP_WIDE` policy markers used by this experiment.

The runner never patches or rebuilds ExLlamaV3.

## Speculative denominator contract

This branch carries the corrected cross-check:

`-1 <= (drafts + accepted - usage.completion_tokens) <= k`

with measured `k = draft_tokens / passes`.

API `usage.completion_tokens` remains the authoritative TPOT denominator.

## Executor

```bash
R0_REPO="$PWD" bash tools/r0_run_moe_wide_ab.sh
```

It performs:

1. plugin/source attestation before GPU work;
2. CPU gates;
3. AUTO wide/wide 4K, 5 repeats, profiler OFF;
4. NARROW narrow/narrow 4K, 5 repeats, profiler OFF;
5. flattened greedy token parity;
6. 14-pass AUTO diagnostic trace;
7. 14-pass NARROW diagnostic trace;
8. stage-separated comparison:
   - coop_moe_a
   - coop_moe_b
   - combined
   - theoretical best mixed geometry from the two measured extremes;
9. Xid and cleanup.

Profiler wall latency is diagnostic only.

## Decision rules

### 1. Global NARROW wins >=3% e2e, parity/stability clean

The existing env knob is a real candidate.

Do not productionize in this round. Next step is long-context qualification and
full parity.

### 2. Global NARROW does not win, but one stage clearly does

Use `moe_wide_compare.json`.

If:

- one stage prefers narrow,
- the other prefers wide,
- and the theoretical stage-specific combination saves at least about
  **1.0 ms/pass**,

then prepare a tiny stage-specific ExLlamaV3 patch next.

If the estimated saving approaches **1.8 ms/pass**, it reaches the project's
~5% end-to-end optimization gate at ~3.5 emitted tokens/pass and should be high
priority.

Do not rebuild CUDA in this round.

### 3. Both A and B are flat or worse under NARROW

Close the coop tile-geometry direction.

The existing wide policy is adequate for CMP170HX.

### 4. Parity mismatch

Record it. Geometry tuning should normally preserve arithmetic ordering within
the selected kernel specialization, but a mismatch blocks direct adoption.

The diagnostic stage timings remain useful.

## Explicitly not tested here

- `EXL3_MOE_COOP_KSPLIT`: upstream source explicitly labels split-k as
  "Measured ineffective"; do not reopen without new evidence.
- native p2b MoE: this checkpoint uses the ExLlamaV3 coop path and prior p2b
  evidence is from different shapes/platforms.
- dense output BF16 work: closed by PR #10.
- 64K lm_head pruning: measured ceiling only 3.6%.
- 6bpw hybrid n-gram: deferred accuracy experiment.
