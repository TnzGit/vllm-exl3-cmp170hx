# R1 K1-Q1: real-Qwen resident-visibility replay

## Goal

K1-T proved the generic CPU transfer plane is correct and close to the raw
pinned-H2D floor. K1-Q1 now isolates the next risk: model semantics.

The question is narrow:

Does the real Qwen4Exp QSA path still answer correctly when the historical
tokens visible to QSA are restricted to the frozen 64K / 5% sticky resident
plan?

K1-Q1 deliberately does NOT yet shrink physical GPU KV allocation. The full
main KV cache remains available as the safe reference source.

## Why selected-token visibility is patched instead of the whole block table

vLLM chunked prefill can place history-tail rows and query rows in the same
forward call. A block table is request-scoped, not row-scoped. Replacing the
whole block table at that point would also sparsify some pre-query history
rows and confound the experiment.

QSA's selected logical indices are row-scoped.

K1-Q1 therefore changes only selected indices for rows whose logical position
is at or after apply_min_pos. This gives an exact query/replay boundary even
when a prefill chunk straddles that boundary.

For affected rows:

- historical selected token is kept if its 256-token planner region is in
  the frozen sticky resident set;
- historical selected token outside the resident set becomes -1;
- current query/decode selected tokens at or after active_from_pos are always
  kept;
- QSA scoring and persistent_topk are not changed.

The QSA kernel already treats negative selected indices as invalid.

## Frozen policy source

K1-Q1 does not invent a new resident policy.

The plan is generated from the prior K1B stateful policy:

- context: 160K
- turn: ask_d_e
- planner region: 256 tokens
- historical resident budget: 64K tokens
- sticky replacement fraction: 5%
- resident regions: exactly the K1B ask_d_e resident set after the preceding
  fixed-history turns

The ask_d_e query targets two facts:

- KVMEM_FACT_D_426853 -> violet-harbor-31
- KVMEM_FACT_E_537961 -> granite-comet-72

This is stronger than the single-fact cases because both historical facts
must survive the real QSA visibility restriction.

## Active suffix

The historical resident budget is 64K.

Current query/decode positions are an active append region and are not charged
against that historical budget in K1-Q1. They remain visible even when their
planner region is not one of the 256 historical resident regions.

This matches the intended bounded-KV runtime model: historical residency is
bounded, while the live suffix must remain writable/visible.

## Default-off patch

tools/patch_vllm_qwen4_exp/patch_vllm_qsa_visibility.py

adds a research-only marker:

# KVMEM_QSA_RESIDENT_VISIBILITY_V1

If VLLM_QWEN_KVMEM_RESIDENT_PLAN is unset, the helper immediately returns and
QSA behavior is unchanged.

When enabled, K1-Q1 requires eager execution. CUDA graph capture fails closed.

The helper writes per-call evidence to
VLLM_QWEN_KVMEM_VISIBILITY_STATS_PATH.

## Visibility evidence

Per layer/call the stats include:

- rows_applied
- first/last applied logical position
- selected_valid_before
- historical_selected
- historical_resident_kept
- historical_selected_dropped
- active_selected_kept

The summary reports aggregate historical resident hit rate and overall selected
visible rate.

A GO is not allowed unless historical_selected_dropped > 0. This prevents a
false positive where the mask was technically enabled but never excluded any
QSA selection.

## Baseline and masked runs

Both runs use fresh eager engines with:

- no MTP
- COOP=1
- prefix cache off through the standard launcher
- one sequence
- exact same prompt token IDs
- temperature 0
- seed 0
- max_tokens 512

The installed QSA source is patched once before both engines.

Baseline engine:

- resident plan env unset
- visibility stats env unset
- patched helper therefore default-off

Masked engine:

- exact K1B resident plan enabled
- visibility stats enabled

After both runs, installed qsa.py must be restored byte-identically.

## Acceptance

Hard semantic gate:

- baseline must contain both expected recovery codes in order;
- masked output must contain both expected recovery codes in order;
- mask stats must be non-empty;
- at least one historical selected token must actually be dropped;
- Xid delta must be zero;
- QSA source must restore byte-identically.

Classification:

BASELINE_INVALID
- full-QSA baseline does not reach the expected target answer inside the
  measurement window;
- no conclusion about resident visibility may be drawn.

INVALID_MASK_EVIDENCE
- mask stats are missing/incomplete, layer coverage is incomplete, accounting
  does not close, or no historical selection was actually dropped.

EXACT_PARITY_GO
- baseline measurement is valid;
- hard semantic gate passes;
- baseline and masked completion token strings are exactly equal.

SEMANTIC_GO_NONEXACT
- baseline measurement is valid;
- hard semantic gate passes;
- exact completion tokens differ.

MASK_SEMANTIC_NO_GO
- baseline reaches the target answer;
- masked run does not.

Exact token parity is useful evidence but not the hard gate. SM80
persistent_topk tie-boundary behavior is already known to be set-nondeterministic,
and bounded historical visibility is intentionally approximate.

## What K1-Q1 proves

If GO, K1-Q1 proves:

- the frozen K1B sticky resident policy survives the real Qwen4Exp forward
  path;
- real QSA attention can tolerate dropping nonresident selected-history tail
  candidates for this long-context multi-fact query;
- active query/decode visibility can remain separate from historical residency;
- the next integration step can move from visibility semantics to physical
  resident-cache remapping.

## What K1-Q1 does not prove

K1-Q1 does not claim:

- GPU KV memory reduction;
- higher concurrency;
- real CPU-backed model inference;
- repeated-turn TTFT improvement;
- GDN/recurrent replay correctness;
- MTP correctness;
- production performance.

The full physical main KV cache remains allocated in both baseline and masked
runs.

## Next phase after GO

K1-Q2 will connect the already-proven K1-T CPU backing/transfer plane to a real
Qwen main-KV resident cache and physical QSA page mapping.

That phase will require explicit publication of historical main-KV pages and
resident physical-page remapping, while keeping target-only / no-MTP.

GDN checkpoint/replay remains a later K2 dependency before claiming cheap
multi-turn replay.

## First hardware attempt and measurement correction

The first K1-Q1 execution used max_tokens=32. Both the default-off baseline and
the masked arm exhausted that window inside the model's <think> preamble before
either target code was emitted.

That run is measurement-invalid rather than evidence against bounded
visibility.

Its mechanism evidence remains valid:

- 12/12 QSA layers covered;
- 912 replay rows observed;
- 1,834,100 historical selected tokens;
- 1,722,276 resident historical selections retained;
- 111,824 historical selections genuinely dropped;
- historical resident selected-token hit rate 93.903%;
- accounting invariants exact;
- Xid delta zero;
- QSA restore byte-identical.

The rerun widens only the generation window to 512. Exact prompt token IDs,
resident plan, 64K budget, 5% replacement policy, region size, eager/no-MTP
engine contract and QSA visibility patch are unchanged.
