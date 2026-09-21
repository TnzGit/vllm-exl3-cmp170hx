# Deferred hybrid PLE / n-gram accuracy qualification

This branch is documentation-only and intentionally isolated from the active
performance lane.

## Candidate

Keep the current main model / MTP path from:

- `Lygodactylus/Qwen3.8-Flash-Next-Uncensored-exl3-3bpw`

but substitute only the PLE / n-gram table from:

- `Lygodactylus/Qwen3.8-Flash-Next-Uncensored-exl3-4bpw`

Do not perform this swap until the current k=3 performance baseline is stable.

## What the model cards actually say

The 3bpw repository uses:

- language model: 3.05 bpw
- lm_head: 5 bpw
- MTP: 3 bpw
- n-gram / PLE: **3 bpw**

The 4bpw repository uses:

- language model: 4.05 bpw
- lm_head: 6 bpw
- MTP: 8 bpw
- n-gram / PLE: **6 bpw** (`-ngb 6`)

So this hybrid is not a simple "3-bit -> 4-bit PLE" change. It is a
**3 bpw -> 6 bpw PLE table** precision experiment.

Sources:

- https://huggingface.co/Lygodactylus/Qwen3.8-Flash-Next-Uncensored-exl3-3bpw
- https://huggingface.co/Lygodactylus/Qwen3.8-Flash-Next-Uncensored-exl3-4bpw

A Hugging Face discussion contains one community report that the 4bpw-repo
n-gram table was mixed into the 3bpw model successfully and that speed appeared
similar. Treat this as compatibility evidence only, not as our qualification:

- https://huggingface.co/Lygodactylus/Qwen3.8-27B-Uncensored-exl3-4bpw/discussions/1

## Performance expectation

Current CMP170HX R0 uses disk-backed n-gram. A higher-bpw table therefore does
not consume the main model's GPU weight residency, but **do not assume exactly
zero performance cost**.

A 6bpw packed row is larger than a 3bpw packed row, so the hybrid can increase:

- mmap/page-cache footprint,
- cold-page NVMe traffic,
- host gather bytes,
- packed-row H2D bytes,
- PLE decode work if bit-width changes the decode path.

The current profiling evidence says PLE/n-gram is a very small steady-decode
component, so the expected impact on decode throughput is small. The larger
risk is cold/page-cache/TTFT behavior, not the already-profiled GPU decode
kernel share.

## Qualification contract

### 1. Compatibility / provenance

Before any quality claim, record:

- exact 3bpw and 4bpw model revisions,
- exact n-gram tensor names and shapes,
- physical PLE bit width / K / codebook metadata,
- index entries / shard locations,
- prepared-pack rewrite or mount method,
- runtime proof that only the PLE/n-gram tensors came from the 4bpw source.

Do not silently copy unrelated 4bpw lm_head, MTP, vision, or LM tensors.

### 2. Performance sentinel

Keep every other runtime setting fixed. Compare:

- baseline: 3bpw LM + 3bpw PLE
- hybrid: 3bpw LM + 6bpw PLE

Use profiler OFF and at least:

- C1 4K
- C1 160K
- 3 repeats median

Record:

- TTFT,
- ms/output-token,
- output tok/s,
- MTP accepted/pass and emitted/pass,
- preemption,
- fallback,
- Xid delta,
- cold first request vs warm repeated request where practical.

A steady-decode difference below the normal noise band should be treated as
effectively neutral, but cold/page-cache effects must be reported separately.

### 3. Accuracy / quality A/B

Only after the performance sentinel is clean:

- fixed prompt/eval set,
- exact same decoding settings,
- deterministic seeds where supported,
- baseline 3bpw PLE vs hybrid 6bpw PLE.

Include tasks where PLE precision could plausibly matter:

- coding,
- reasoning,
- factual/knowledge prompts,
- long-context retrieval,
- multilingual prompts if relevant.

Prefer task-level exact/graded outcomes over subjective spot checks.

## Decision rule

Keep the hybrid as a production-quality candidate only if it provides a
repeatable quality gain without a meaningful latency/TTFT regression or new
stability/fallback issue.

No production launcher/default change is made on this branch.
