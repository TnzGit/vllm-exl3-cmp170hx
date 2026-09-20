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

## Fresh correctness/performance watchlist

These items were reviewed after the initial R0/R1 split. They are **not** a
request to bulk-backport current vLLM main.

### Before enabling MTP on R0

- **#56742 (open): Qwen4Exp MTP buffer placement/config/warmup.** Direct
  inspection of vLLM 0.29.0 confirms `_mtp_hidden_buffer = torch.empty(...)`
  has no explicit `device=` argument. The old recipe nevertheless served MTP,
  so do not assume failure from source alone. On the first MTP boot, record
  the actual `_mtp_hidden_buffer.device`. If it is wrong, backport only the
  configured-device allocation with a discriminating CPU/source test before
  changing other MTP code.
- **#55054 (merged after v0.29.0): async PLE MTP metadata transfers.** Reported
  removal of two per-step stream synchronizations and sizable C1 gains on
  newer Qwen code. Treat as a high-value R1/selective-backport candidate only
  after the R0 MTP baseline is measured.

### Prefix-cache / multi-request / PP gates, not first-boot blockers

- **#57616 (open): plain MTP can zero prefix-cache insertions.** Keep prefix
  caching out of the first MTP throughput gate. When prefix caching is later
  enabled, repeated identical prompts must produce measured cache hits before
  any TTFT/cache claim is accepted.
- **#55390 / #56026 (open): draft KV-group annotation on hybrid Qwen layouts.**
  Relevant to prefix-cache/offload policies. Do not backport for C1 no-offload
  baseline; audit if cache/offload is enabled.
- **#55506 (open): Mamba spec-decode block tables indexed by batch row instead
  of request slot.** The reported reproducer requires PP>=2 + MTP + prefix
  caching + at least three concurrent slots. It is especially notable because
  the PR reports silent corruption on SM80 and includes a 4x CMP170HX A/B.
  Our first target is PP1/C1, so this is not an R0 blocker. If the project
  later adds PP or that concurrency shape, this becomes a P0 correctness gate
  before performance work.
- **#57253 (open): cache-registered Mamba state retirement.** Relevant to
  long-prefix reuse under cache pressure; not a C1 first-boot blocker.

### QSA / PLE candidates after R0

- **#54873 (merged after v0.29.0): sparse-GQA valid-count optimization.**
  Strong microbench gains for short prefill/decode but modest e2e gains on
  GB300. Add to R1 after the QSA prefill/decode split, and remeasure on SM80.
- **#54890 (merged): FP8 QSA indexer cache.** Potential long-context bandwidth
  win, but published e2e short-context results were mostly noise and were on
  GB300. Do not change cache precision before the baseline memory/quality
  ledger exists.
- **#55557 (merged): FP8 main QSA KV cache.** Can materially increase KV
  capacity; this is a later capacity experiment, not a first-boot default.
  Re-qualify quality and SM80 kernel support before adopting.
- **#55375 (merged): fused-PLE state-index stride correctness.** It fixes the
  fused PLE implementation introduced after v0.29.0; R0 does not contain that
  fused PLE path, so do not backport it to R0. It is mandatory context if an
  R1 EXL3 adapter ever adopts upstream fused PLE.
- **#55309 (merged): PLE residual / QSA output-gate fusion.** Current-main
  optimization only; depends on the newer Qwen kernel/dataflow and is not an
  R0 patch.

### Open items to watch rather than preemptively patch

- #54912 QSA raw-key ring widening for certain MTP depths.
- #56500 bounded QSA prefill-logits workspace.
- #57105 QSA indexer workspace fragmentation.
- #55122 persistent-topk determinism.
- #56577 FP8 Qwen4Exp MTP proposal head.

Only promote an open upstream change into this fork after either (a) our exact
R0 configuration reproduces its defect, or (b) the change becomes a required
dependency of an explicitly chosen R1 experiment. Preserve a minimal A/B for
every such backport.

### MTP async-metadata optimization #55054 — not an R0 rescue target

Upstream #55054 is mechanically close to vLLM 0.29.0: the 0.29 short-conv metadata builder still performs the same synchronous request-index `.to(device)` calls and already imports `async_tensor_h2d`.

However, upstream's same-node GB300 C1 measurement for MTP3 improved output throughput by **9.877%** (GPU activity span -17.1%). The CMP170HX R0 measurements show:

- k=1: ~21-25% slower than no-draft
- k=2: ~33% slower
- k=3: ~45% slower

Therefore #55054 is not expected to overturn the R0 production decision. Even applying the upstream ~10% throughput gain in full leaves k=1 materially behind no-draft; the more optimistic 17% span reduction is still insufficient to establish a win.

Do not create a dedicated R0 backport solely to rescue MTP. If a later R1 runtime already contains #55054, remeasure MTP opportunistically as part of that runtime qualification.
