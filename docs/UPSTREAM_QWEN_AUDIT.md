# vLLM upstream audit for Qwen3.8-Flash-Next EXL3

Audit date: 2026-09-20.

## Runtime lanes

### R0 — known-good compatibility baseline

- vLLM `0.29.0`
- vllm-exl3 plugin fork, exact SHA
- EXL3 row-wise n-gram through the plugin's `Exl3EmbeddingMethod`
- `VLLM_EXL3_NGRAM_TABLE=disk` for the first CMP170HX capacity profile
- text-only first (`--language-model-only`), no draft for first boot

The upstream Qwen EXL3 recipe still reports vLLM 0.29.0 as its measured vLLM
baseline. Keep R0 reproducible even if later runtimes win.

### R1 — upstream-Qwen candidate

Do **not** define R1 as simply "latest vLLM". Pin an exact vLLM commit and
re-qualify the plugin/source contract. R1 exists to test specific upstream
Qwen improvements after R0 is healthy.

## What vLLM 0.29.0 does not contain

Direct inspection of tag `v0.29.0` (`98dff2a81d747d1dba01a47f939f48c3526d4206`)
shows that it predates these later Qwen4Exp mainline changes:

| upstream work | status in v0.29.0 | EXL3 action |
|---|---|---|
| #54513 separate QSA prefill/decode indexer paths | absent | high-priority selective candidate after R0 |
| #54517 fused Qwen4Exp PLE kernels | absent | do not cherry-pick blindly; changes the PLE embedding contract |
| #55272 remove torch.compile from NVIDIA Qwen4Exp | absent | selective candidate after R0 |
| #54371 Engram/UVA PLE offload | absent | idea/reference only for packed EXL3 disk mode unless an adapter is designed |

These merged upstream changes are not evidence that the known-good 0.29.0
EXL3 recipe is wrong; they are potential second-stage improvements.

## Why current vLLM main is not a drop-in EXL3 upgrade

At audited vLLM main commit `92b40f5a144763c8b2ef6ebe848793de634ada05`:

- fused PLE kernels are present;
- separate QSA decode/prefill indexer ops are present;
- the NVIDIA Qwen4Exp model no longer carries the old `support_torch_compile` decorator;
- `EngramConfig` / pinned-host PLE offload is present;
- main and MTP `ParallelLMHead` constructors still do not pass EXL3 quant config;
- vision split-q/k/v filtering is still not the plugin's EXL3 policy;
- PLE construction has been refactored from the 0.29 `PLEVocabParallelEmbedding`
  call site into `Qwen4ExpNGramEmbedding`.

The new `Qwen4ExpPLEEmbeddingMethod.from_quant_config()` supports native
unquantized / FP8 / ModelOpt cases and raises `NotImplementedError` for an
`Exl3Config`. More importantly, the new PLE embedding abstraction assumes
upstream `weight` / `dequantize` / optional pinned-host semantics, while
`Exl3EmbeddingMethod` stores packed trellis tensors and owns its own
resident/disk decode path.

Therefore the 0.29 PLE patch must **not** be force-applied to current main.
A current-main EXL3 PLE adapter is a separate design task and requires CPU
contract tests plus GPU qualification.

## Selective-upstream order after R0

1. **#54513 QSA prefill/decode split** — profile first; candidate because it is
   outside the packed EXL3 PLE storage path.
2. **#55272 no-torch.compile Qwen NVIDIA path** — evaluate boot/compile time and
   steady performance; keep only if the exact plugin/runtime remains stable.
3. **#54517 PLE fusion** — blocked on an explicit EXL3 adapter to the new PLE
   abstraction. Do not cherry-pick as a normal performance patch.
4. **#54371 UVA/Engram offload** — not a substitute for EXL3 disk n-gram.
   Upstream offload addresses native embedding weights; EXL3 disk mode keeps
   packed mmap rows and dequantizes selected rows on the GPU.

Every selective backport gets a clean R0/R1 same-checkpoint A/B. Do not stack
multiple Qwen upstream changes before attribution.

## Open upstream items that are not R0 blockers

- vLLM #56964: quantization-only MoE backends should fall back for unquantized
  layers. Relevant only if a service explicitly selects such a backend and
  hits an unquantized draft/MTP layer; the first EXL3 profile should not add
  that confound.
- vLLM #55334: ModelOpt NVFP4 + FP8 PLE loading. Different quantization path;
  not an EXL3 blocker.

## Gate for changing the vLLM baseline

Do not replace R0 merely because a newer commit exists. Promote an R1 runtime
only after the same target checkpoint passes:

- pack/load correctness;
- greedy output sanity;
- no Xid delta;
- no silent PLE/lm_head fallback;
- actual EXL3 dispatch;
- 4K and long-context C1;
- MTP qualification if enabled;
- memory ledger;
- measurable end-to-end benefit or a necessary correctness fix.
