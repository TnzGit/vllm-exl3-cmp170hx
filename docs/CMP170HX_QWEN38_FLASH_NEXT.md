# CMP170HX Qwen3.8-Flash-Next EXL3 bring-up

This document defines the CMP170HX research contract for serving
Qwen3.8-Flash-Next EXL3 through `vllm-exl3`.

It is intentionally separate from the Apache-licensed
`TnzGit/vLLM-CMP170HX-Qwen3.8` Qwen3.8-27B project. This repository is
AGPL-3.0-only and remains an out-of-tree vLLM serving plugin.

## Target

Initial model target:

- `Lygodactylus/Qwen3.8-Flash-Next-Uncensored-exl3-3bpw`
- Qwen4Exp / Flash-Next
- EXL3 3.05 bpw language model
- EXL3 3 bpw MTP
- EXL3 3 bpw row-wise n-gram / PLE table
- MTP retained
- single CMP170HX / SM80 is the first hardware target

The model card reports roughly 68 GB on disk and a 3 bpw n-gram table. Do not
infer actual tensor geometry or codebook from the repository name or average
bpw: the pack scan is authoritative.

## Branches and provenance

Two long-lived references exist in the fork:

- `baseline/qwen38-flash-next-94c29ba`
  - exact commit `94c29ba25a66a7583bfa49546c783d3e70859cc6`
  - Qwen known-good anchor after upstream PR #23
  - contains native Qwen EXL3, Qwen4Exp plumbing, ExLlamaV3 1.5.0 MoE ABI
    compatibility, unsharded n-gram support, and disk-backed n-gram mode
- `research/qwen38-flash-next-cmp170hx`
  - starts from fork `main` at
    `08ed1bfc57cfeaf87204559f2a3a9cec916361df`
  - all CMP170HX-specific research belongs here or under child research branches

Do not rewrite the baseline branch. It is a reproducibility reference, not a
development branch.

Initial runtime contract to qualify:

- vLLM: `0.29.0`
- ExLlamaV3: `1.5.0`, exact tag commit `0740edc2da569fb99174023c1d2988b1e98cb41e`
- vllm-exl3: exact branch SHA, never just a package version
- model: exact Hugging Face revision once downloaded
- driver / CUDA / PyTorch: record exact values before the first benchmark

If current `main` fails while the baseline branch works, compare only the
20-commit interval `94c29ba..08ed1bf` before changing vLLM or the model.

See also `docs/UPSTREAM_QWEN_AUDIT.md` for the R0/R1 vLLM runtime lanes and the
reason newer Qwen PLE changes are not a drop-in EXL3 upgrade.

## Why disk-backed n-gram is the first baseline on CMP170HX

`VLLM_EXL3_NGRAM_TABLE=disk` leadves the packed n-gram table in the checkpoint
mmap instead of copying it into a device-resident int16 tensor. Each lookup:

1. forms unique row ids,
2. synchronizes ids to the host,
3. gathers packed rows from the mmap/page cache,
4. copies those packed rows to the GPU,
5. decodes them on the GPU,
6. expands them back to the requested positions.

This is a capacity feature first, not a free performance win.

Disk mode requires PIECEWISE CUDA graphs with
`vllm::exl3_ngram_lookup_out` as a splitting op. FULL graph results from the
27B project must never be compared as if graph semantics were unchanged.

The current implementation deliberately uses blocking uploads because an
earlier non-blocking copy from short-lived pinned temporaries could outlive the
temporary. Any future async staging optimization must prove lifetime ownership
with persistent buffers/events rather than simply restoring
`non_blocking=True`.

## Bring-up sequence

### F0  freeze identity

Before touching GPU performance, record:

- plugin SHA
- vLLM version and source artifact
- ExLlamaV3 revision and extension ABI
- PyTorch / CUDA / driver
- model revision
- model `config.json` hash
- safetensors index hash
- pack scan output
- effective environment variables

No benchmark is valid without the runtime identity.

### F1 - pack contract, CPU only

Run the native pack tools before modifying the pack:

- `qwen_pack_scan.py`
- `qwen_pack_config.py --dry-run`
- index validation
- n-gram layout and bit width
- dense EXL3 map
- routed expert K / codebook
- MTP tensors
- lm_head bits
- vision leftovers / fused qkv contract

The current target model has a Hugging Face config parsing warning. Treat
safetensors metadata and the pack scanner as authoritative; do not patch code
around malformed or ambiguous pack metadata.

### F2 - minimum service boot

First boot target:

- one CMP170HX
- C1
- 4K runtime length or another deliberately small first-boot context
- no speculation
- `VLLM_EXL3_NGRAM_TABLE=disk`
- PIECEWISE graphs with the n-gram out-variant split
- exact worker-side runtime diagnostics captured

The first goal is correct service, not 262K.

Gates:

- server reaches healthy state
- coherent greedy output
- no Xid delta
- no silent fallback to unquantized n-gram/lm_head
- actual EXL3 dispatch recorded
- memory ledger recorded



### Text-only first profile

vLLM 0.29 exposes `--language-model-only` for multimodal hybrid models.
Qwen4Exp has an explicit text-only branch: the visual module is replaced by a
missing-layer placeholder and `visual.*` checkpoint weights are dropped during
load. QSA also selects text-only-specific RoPE/fusion behavior.

Use this mode for the first CMP170HX profile unless multimodal serving is an
explicit requirement. It reduces the resident-weight envelope and removes the
vision tower from the first performance question.

This flag is part of the runtime identity. Do not compare a text-only run with
a multimodal-loaded run as if only weights changed. If multimodal support is
needed later, qualify it as a separate service profile.

### F3
 - MTP qualification

Do not import the 27B project's k values.

Measure at least:

- no draft
- k=1
- k=2
- k=3

Only add k=4 if the exact runtime survives compile/capture and there is a
specific reason. Existing Qwen recipe history has seen k=4 wedge and an MTP
acceptance cliff at ultra-long context; both must be re-qualified on this
checkpoint and SM80.

Report:

- output tok/s
- ms/output-token
- actual draft iterations
- accepted tokens/pass
- per-position acceptance
- draft time/pass
- target time/pass
- TTFT
- Xid delta

### F4 - context envelope

First-pass C1 points:

- 4K
- 32K
- 65K
- 126K
- around 160K
- 200K
- 250K

The point near 160K exists to test the previously observed Qwen Flash-Next MTP
acceptance cliff. Do not interpolate across it without data.

Only after C1 is understood should C2/C4 be attempted. Reuse the 27B
concurrency rules: admission, residency, prompt work and steady decode are
different measurements.

### F5 - current-head Amdahl profile

Profile the current real service before writing new kernels. Keep at least:

- EXL3 routed MoE
- dense EXL3 linears
- row-wise n-gram lookup:
  - id transfer / synchronization
  - host mmap gather
  - H2D packed-row copy
  - n-gram decode
- QSA indexer
- QSA sparse attention
- GDN / recurrent state
- MTP
- graph split overhead
- scheduler / other

Client-side `torch.profiler` is not server GPU attribution. Use the server
process or Nsight.

## First optimization candidates

### 1. Disk n-gram pipeline

Highest-priority structural candidate on a discrete PCIe GPU.

Audit opportunities:

- persistent pinned staging instead of per-lookup temporaries
- explicit event-owned buffer lifetime
- fixed-address graph-friendly output/staging buffers
- reduce D2H synchronization of row ids
- batch/merge row gathers
- page-cache warming policy
- hot-row cache
- overlap only when correctness proves the source buffer lifetime

A kernel-only n-gram decoder speedup is not enough if host synchronization
dominates.

### 2. EXL3 MoE cooperative decode path

Upstream PR #27 measured the cooperative MoE path directly on CMP170HX/SM80,
but on a different model shape. Qwen must be re-tested with its actual:

- hidden size
- top-k
- K / codebook
- routed rows/call
- C1/C2/C4 decode shapes

Never claim the DeepSeek result transfers to Qwen without an end-to-end A/B.

### 3. MTP service profiles

If acceptance collapses beyond a context threshold, prefer a simple static
service profile such as:

- normal context: MTP enabled with measured best k
- ultra-long: MTP disabled

over adding a complex adaptive policy before the underlying geometry and graph
constraints are understood.

## Measurement contracts inherited from the 27B project

These rules carry forward unchanged:

- fresh engine for formal context-to-context comparisons
- exact-token prompts
- unique prompt content where shared-prefix reuse would distort the test
- record actual runtime state, not launcher intent
- Xid delta for every GPU experiment
- TTFT, prefill, output latency and target-pass latency are distinct
- `ms/output-token` is never called `ms/step`
- speculative iteration counters must cover the same time interval as their
  denominator
- for every ratio, numerator and denominator must span the same interval or the
  result is `null`
- concurrent steady window:
  - start = latest first-output timestamp
  - end = earliest finish timestamp
  - all intended requests must be resident
  - prompt work during that window must be reported
- `running`, `waiting`, preemption and KV usage are required for concurrency
- fresh CUDA graph addresses and replay correctness are separate gates
- standalone kernel wins are not production wins
- measured and modeled quantities are labeled separately
- negative results are preserved

## 27B migration matrix

| 27B result / technique | Flash-Next action |
|---|---|
| exact-token benchmark harness discipline | reuse |
| server-side profiling / Xid / sanitizer workflow | reuse |
| concurrency steady-window and admission telemetry | reuse |
| block-id int64 before large stride multiplication | preserve as an audit invariant |
| GDN mixed spec `a/b` alignment | already in vLLM 0.29; verify, do not re-patch |
| draft Gumbel stream isolation | already in vLLM 0.29; verify, do not re-patch |
| M7 NSEG35 mixed-FP8 verifier | do not port |
| M7 E4M3 LUT / page carry / half-warp verifier | do not port blindly |
| INT8-G64 verifier experiments | do not port |
| dense Marlin gate/up tuning | not a Flash-Next MoE starting point |
| 27B k=3/5/7 winners | do not port values; only reuse sweep methodology |
| FULL CUDA graph assumptions | invalid for disk n-gram; re-qualify PIECEWISE |
| `max_num_batched_tokens` conclusions | re-test on the new scheduler/model |
| MAX_SEQS service-policy A/B | reuse methodology |
| client-side profiler attribution | prohibited |
| large-number subtraction for decode cost | prohibited |

## Address-width correctness invariant

The 27B project found a real IMA caused by multiplying a valid int32 physical
block id by a large byte stride before pointer arithmetic. Flash-Next uses
different attention kernels, so no code is copied. The invariant is retained:

> Any physical block/page/slot index that can participate in address arithmetic
> above INT32 range must be widened before multiplication.

Audit QSA, paged KV, recurrent-state block tables and any plugin-native pointer
tables with this rule. Do not wait for a >2 GiB pool crash to rediscover it.

## Patch-stack rules

- Prefer semantic, version-aware patch scripts over blind source replacement.
- Every patch must fail closed if its expected source contract is absent.
- Keep vLLM patches minimal and separately revertible.
- Before benchmarking, verify the installed vLLM source after patching.
- A patch script existing in the repo does not prove the installed runtime is
  patched.
- Do not backport fixes already present in vLLM 0.29.

The CMP170HX research branch adds a vLLM 0.29-aware PLE patch contract so EXL3
can coexist with Qwen's model-specific FP8 PLE method.

## License boundary

This fork remains AGPL-3.0-only. The old 27B repository is Apache-2.0.

Move ideas, measurements, independently written harness logic and correctness
contracts as appropriate. Do not copy AGPL implementation code into the
Apache-licensed 27B repository.

## Stop conditions

Do not start a new kernel merely because one subsystem is visible in a trace.
A candidate should have a plausible whole-request impact.

Kill or defer an experiment when:

- its theoretical whole-step ceiling is below roughly 1%
- it adds graph instability for a sub-percent gain
- it requires unsupported Qwen geometry
- it repeats a 27B negative result without a materially different dataflow
- it improves a microkernel but loses end-to-end after host/graph overhead
- it consumes VRAM needed for the target context envelope

The goal is a simple, reproducible CMP170HX Flash-Next service profile, not a
large patch count.


## Baseline-to-current regression boundary

The immutable Qwen anchor `94c29ba` is 20 commits behind the fork's initial
research base `08ed1bf`. If current-head Qwen bring-up fails while the baseline
works, keep vLLM, ExLlamaV3 and the model revision fixed and bisect only this
plugin interval first. Do not respond by upgrading every runtime component at
once.

The main implementation delta is concentrated in `src/vllm_exl3/exl3.py`;
the research base also contains later MoE work such as the cooperative decode
path. This small, explicit interval is the first regression boundary.



### Classification of the 20 post-baseline commits

For first-boot triage, do not treat all 20 commits as equally likely Qwen
regressions.

Directly relevant candidates:

- PR #27 / `exl3_moe_coop`: optional decode-shaped MoE path, explicitly
  interesting on CMP170HX/SM80 but disabled unless requested.
- `61613b1` / PR #29: removes an extra CUDA synchronization after a blocking
  trellis copy; generic load-path behavior and worth keeping unless evidence
  says otherwise.

Mostly feature/platform-scoped changes:

- UVA routed-expert arena placement
- UMA `MADV_DONTNEED` source-page reclaim and compatibility wrappers
- parity/EP placement diagnostics
- DeepSeek-V4.1 tensor-metadata and mixed-K planning

These should be inactive on the initial single-CMP170HX Qwen configuration
unless explicitly enabled. If a regression appears, verify the guards in the
effective process before reverting them wholesale.

## Cooperative MoE qualification gate

The existing `exl3_moe_coop` path is relevant to CMP170HX because upstream
already measured it on SM80, but that result was obtained with a different
model geometry and does not transfer automatically to Qwen.

Current plugin eligibility requires all of the following:

- `VLLM_EXL3_COOP=1`
- `tokens * topk <= 256`
- `exllamav3_ext.exl3_moe_coop` is present
- no fat expert route for the call
- gate/up/down use a uniform codebook-flag pair
- hidden width divisible by 128
- expert intermediate width divisible by 128

The Qwen pack scan must establish the actual top-k, physical K/codebook and
expert widths before enabling this path. If the model uses top-k 10, the
current slot cap means at most 25 token rows per eligible call; treat this as a
derived eligibility bound, not as a measured performance claim.

Qualification order:

1. stock path parity baseline
2. coop kernel parity on actual Qwen shapes
3. C1 decode A/B
4. C2/C4 only when calls remain eligible and resident
5. end-to-end output-token latency and throughput

A kernel-level win is rejected if graph/scratch/dispatch overhead erases it
end-to-end.

### Engine reuse for same-configuration sweeps

Fresh-engine isolation is a correctness tool, not a requirement to reload the model for every prompt length. Within an unchanged runtime configuration, context-only sweeps may reuse one engine if a short-context sentinel is measured before and after the sweep and scheduler/GPU state is clean between requests.

Always restart after a runtime/configuration change, crash/OOM/Xid, or when confirming a capacity/acceptance/performance cliff.

## R0 MTP qualification result

Hardware qualification on one CMP170HX found no production win from Qwen4Exp MTP.

- no-draft: ~30.1 ms/output-token / ~33.2 tok/s
- k=1: ~37.5-38.4 ms/output-token / ~26.1-26.7 tok/s
- k=2: ~45.2 ms/output-token / ~22.1 tok/s
- k=3: ~53.9-54.4 ms/output-token / ~18.4-18.6 tok/s

Acceptance is high rather than poor: k=1 rises from ~0.90 accepted/pass at 4K to ~0.98-1.00 at long context. The previously reported ~163,840-token acceptance cliff did not reproduce through 204,525 prompt tokens.

Conclusion: **the R0 production profile is no-draft across the measured context envelope.** Do not spend further optimization time on MTP unless a later runtime materially changes draft cost.

Full evidence: `docs/R0_MTP_QUALIFY.md`.
