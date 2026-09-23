# R0 C2/C4 MTP k=2 vs k=3 matched-load plan

**Protocol only; no C2/C4 results exist in this plan.** The local prior
artifacts establish C1 behavior, not concurrent-serving performance. The
historical C1 k=2/k=3 figures in [`R0_MTP_POST_COOP.md`](R0_MTP_POST_COOP.md)
must not be projected onto these loads.

## Frozen comparison

Use the same Qwen3.8-Flash-Next Qwen4Exp EXL3 3bpw model pack and runtime
identity for every cell; record pack/revision hashes, vLLM 0.29.0 and
ExLlamaV3 revision, installed EXL3 source SHA, driver/CUDA/PyTorch, and full
R0 source SHA. The only A/B variable is `num_speculative_tokens` (2 or 3).
Keep the revised non-KVMEM benchmark profile fixed: COOP=1, PIECEWISE graphs
with explicit capture sizes `[1,2,4,8,16,24]`,
text-only, disk n-gram, profiler off, prefix caching off, `GPU_MEM_UTIL=0.92`,
`MAX_MODEL_LEN=240000`, `MAX_NUM_SEQS=4`, and
`MAX_NUM_BATCHED_TOKENS=auto` (log the effective value and require it to match
across all cells). Port 8002. Do not tune the envelope per k or per workload.

**Revised 240K capacity profile after failed live1:** The original 246000
maximum-length profile failed before `/health` or any request: default
PIECEWISE graph capture used 0.56 GiB, leaving 6.74 GiB KV against 6.83 GiB
required. That failure remains a separate 246K boot result. The user accepted
`MAX_MODEL_LEN=240000` as the final configuration label for this study. A
subsequent default-graph k=3 attempt still failed boot by about 0.04 GiB KV
memory; the user approved a second prospective amendment: explicitly pin the
smaller PIECEWISE capture list `[1,2,4,8,16,24]` uniformly for **both** k
values and all load points. Retain `max_num_seqs=4`, `.92` memory fraction,
MTP settings, and exact token-ID inputs. This is a new fixed service envelope,
not evidence that 246K or default-graph k=3 works. Record effective graph sizes and KV headroom
from startup logs, and preserve all original correctness and machine-hygiene
gates. If 240K still cannot boot, retain the failure artifacts and stop for
diagnosis; do not silently raise memory utilization, disable graph-memory
accounting, or change any other capacity parameter.

| Cell family | Simultaneous requests | Per-request input | Output per request |
|---|---:|---:|---:|
| C2 | 2 | 16K: 15,533 exact tokens; 80K: 79,533 exact tokens | 256 tokens |
| C4 | 4 | 16K: 15,533 exact tokens; 32K: **BLOCKED** | 256 tokens |

The runnable matrix is exactly c2_16k, c2_80k, and c4_16k, each at k=2 and
k=3 (six fresh-engine cells). The archived historical turn JSON files are now
available under `evidence/c2c4-source-prompts/ctx16000/` and
`ctx80000/`. They were copied from the remote archive
`/home/base-node/.codex_tasks/qwen38-flashnext-r0/results/kvmem-k1q2e-benchmark-61a8eb2-live3/turns/`;
the committed files and generated manifest retain SHA-256 provenance. The
additional `prompt_15533.json` and `prompt_79533.json` came from
`r0-prefill-scan-9da3ace-live2/prompt_cases/` and are historical references,
not substitutes for concurrent request slots. Manifest generation derives
each request from a distinct archived
turn in slot order A, B, A-paraphrase, C (C2 uses A and B). It preserves all
archived token IDs except token 0, which is replaced by a distinct ordinary
single-token lead; neutral single-token filler IDs are inserted immediately
before the archived query span to reach exactly 15,533 or 79,533 tokens. For
every request the manifest records the parent relative path, parent file
SHA-256, parent token-ID SHA-256, query spans, lead replacement, filler
ID/count, decoded query, recovery marker, expected single recovery code, and
derived token-ID SHA-256. Generation verifies the decoded query against the
archived `query_text`, checks that exactly one recovery fact is declared and
its marker/code survive decoding, and confirms exact length. The same frozen
manifest supplies identical derived IDs to k=2 and k=3; simultaneous requests
must have no common prefix.

These are **derived inputs, not historical byte-identical inputs**: token 0 is
changed and neutral IDs are inserted. Results must be described accordingly;
they must not be represented as measurements on byte-identical historical
prompts. The historical 27,250-token C4 input is not recoverable, so c4_32k is
explicitly **BLOCKED** and is excluded from generation and runnable scheduling.
Dry-run output reports the blocked marker, and cell execution fails closed if
c4_32k is requested. It must not be synthesized or run.

Use greedy requests (`temperature=0`, seed 0, `ignore_eos=true`) and require
API `usage.completion_tokens=256`. This is a fixed-length decode-throughput
diagnostic, not normal post-EOS user behavior. Preserve raw text and output
hashes, and separately check each prompt's known target answer. Do not claim
to have located the EOS boundary unless the API exposes it.
The APC short-context probe already showed that forced post-EOS continuation
can produce different hashes on identical requests; whole-output or inferred
token-ID parity is therefore not a sound standalone semantic gate. If the
actual API exposes exact generated IDs, record them; never reconstruct them
from text and label them exact. A meaningful-answer mismatch requires separate
correctness investigation. Disable retries. Wrong token counts or an
unresolved answer mismatch invalidate performance comparison.

## Arrival, repetition, and measurement

Run one unmeasured, barrier-released warm-up wave per fresh engine, drain it,
and verify idle before measurement. Then run 10 measured waves per cell. Each
wave is a closed-loop batch: prepare all C requests, release their workers on
one monotonic-clock barrier, wait for all completions, then wait for idle
before the next wave. Require client dispatch skew <=50 ms; otherwise discard
and repeat that wave. This tests synchronized bursts at fixed concurrency, not
an open-loop arrival-rate curve. Counterbalance k order within each paired load
point; start a fresh engine for every k/load cell and never overlap engines.

Capture per request: actual prompt/completion token counts, dispatch/start,
first-token and finish timestamps, TTFT, end-to-end latency, and decode TPOT
`(finish - first-token)/(completion_tokens - 1)`. Report raw samples and
median/p95 TTFT, TPOT, and end-to-end latency per cell (p95 is descriptive at
this sample size). Per wave report aggregate output tok/s as summed API output
tokens divided by barrier-release-to-last-finish time. Also report the common
decode-overlap window using `start = latest first-token`, `end = earliest
finish`; use only waves with positive overlap and state its duration. Keep
prefill/TTFT distinct from decode TPOT.

Bracket each wave with `/metrics` snapshots; preserve counter deltas for MTP
drafts, draft tokens, accepted tokens, and `vllm:num_preemptions_total`.
Report emitted/draft width (`draft_tokens / drafts`), accepted/pass, and
acceptance by position where exposed. Poll `running`, `waiting`, and KV-cache
usage throughout each wave; report peak waiting/KV usage and the time all C
requests are simultaneously resident/running. Use API usage as the throughput
denominator, not stream chunks. These counters are aggregate for the cell,
not per-request evidence.

## Provenance, capacity, and stop gates

- Before launch, require exact `EXPECTED_SHA` == full `git HEAD` in an isolated
  `R0_REPO`, clean status, and recorded installed-plugin SHA matching the
  approved source fingerprint. Verify model/config hashes and token-ID hashes.
  Refuse any mismatch; no in-place checkout or source edits.
- Require port 8002 free, GPU idle/no compute process, baseline Xid count,
  expected launcher arguments/config, then `/health` and `/v1/models` healthy.
  Preserve metadata, startup/serve/cell logs, prompt hashes, request JSON/IDs,
  timestamped metrics, GPU/process/port snapshots, Xid before/after, and a
  per-cell result/status manifest.
- Before requests, record the actual KV pool and require capacity for the
  largest live set (2 x (79,533 + 256) tokens, plus at least 20% headroom) under
  the fixed 0.92/240000 envelope. The earlier k3 C1 pool of 279,087 tokens is
  context only, not a capacity guarantee for k2/k3 with `MAX_NUM_SEQS=4`.
  If startup/admission cannot meet this gate, mark that cell NO-GO; do not
  raise utilization, shorten context, or otherwise alter the comparison.
- Stop the current cell on unhealthy service, wrong runtime/config, failed
  capacity/admission, nonzero preemption, OOM/crash, Xid delta, MTP counters
  inconsistent with requested k, wrong token count, unresolved meaningful-answer
  mismatch, or loss of the
  synchronized/resident C-request decode window. Preserve partial evidence;
  no automatic retry of failed cells. Classify prompt/admission queueing
  separately from decode overlap and do not call it steady C2/C4 service.
- Launch only in an owned `setsid` process group. On every exit, terminate
  only that group; verify its processes are gone, port 8002 is closed, GPU is
  idle/no compute process, and Xid delta is recorded. If cleanup/hygiene fails,
  stop the matrix and retain the failure artifacts for review.

## Decision rule (pre-register; not a result)

Report each of the three runnable load points independently and retain c4_32k
as BLOCKED. A k is a candidate for a
point only with verified prompt-token identity and meaningful-answer correctness,
valid identical-work waves, zero preemptions
and Xid delta, and a repeatable >=5% median aggregate-throughput gain without
more than 3% regression in p95 TTFT or TPOT. Otherwise report no clear winner
or NO-GO; do not infer a universal k from one context/concurrency point.

Protocol references: [`CMP170HX_QWEN38_FLASH_NEXT.md`](CMP170HX_QWEN38_FLASH_NEXT.md)
for exact-input/runtime and concurrent-residency contracts;
[`R0_MTP_POST_COOP.md`](R0_MTP_POST_COOP.md) for corrected C1 MTP counters and
128-token C1 methodology; [`R0_MTP_K3_PRODUCTION.md`](R0_MTP_K3_PRODUCTION.md)
for the existing 27,250-token “32K” case and fixed 246000/.92 envelope; and
[`R0_PREFILL_SCAN_LIVE2.md`](R0_PREFILL_SCAN_LIVE2.md) for exact prompt counts,
source fingerprinting, auto batching, and cleanup precedent. None is a C2/C4
measurement.
