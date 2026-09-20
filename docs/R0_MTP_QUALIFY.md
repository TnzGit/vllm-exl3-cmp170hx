# MTP qualification on one CMP170HX — Qwen3.8-Flash-Next EXL3

Branch `r0/mtp-qualify`, cut from `origin/r0/load-low-risk-cuts`.
Base SHA `57358f5923cc567cfc609e90c132f4c39ad37cfb`.

Question this stage answers: **when should this model run MTP, at what k, and
when should MTP be off?**

## Runtime and policy

Unchanged R0 runtime plus the qualified low-risk load policy:

```text
vLLM 0.29.0, ExLlamaV3 1.5.0, same prepared pack
text-only, no draft for the baseline, prefix caching OFF
disk n-gram, PIECEWISE graphs, C1, 4K first
VLLM_EXL3_MADV_AFTER_H2D=0
VLLM_EXL3_EXPERT_MATCH_CACHE=1
VLLM_EXL3_GC_AFTER_MOE_LAYER=0
```

Verified per fresh engine: `PRESCAN ready = 49` (48 main MoE layers + 1 MTP
layer), `mode=direct_plan = 49`, staging fallback 0, load 93-97 s.

## Hard gate: `_mtp_hidden_buffer.device` (upstream #56742)

vLLM 0.29.0 allocates the buffer without an explicit `device=`. Measured on the
live engine via a diagnostic inserted into the installed vLLM
(`tools/r0_probe_mtp_buffer_device.py`, never changes the allocation):

```text
EXL3_MTP_HIDDEN_BUFFER_DEVICE device=cuda:0 dtype=torch.bfloat16
  shape=(2048, 10240) default_device=cuda:0 cuda_current=0
```

The buffer lands on the correct CUDA device in every MTP boot (k=1, k=2, k=3).
**No backport of #56742 is needed for this configuration**, and none was made.

## k sweep: 4K and 126K, fresh engine per k

| config | context | prompt tok | ms/output-tok | output tok/s | accepted/pass | accepted/output |
|---|---|---|---|---|---|---|
| no-draft | 4K | 3,475 | **30.10** | **33.23** | — | — |
| no-draft | 126K-class | 107,375-class | ~30.1 | ~33.2 | — | — |
| k=1 | 4K | 3,475 | 38.204 | 26.176 | 0.8955 | 0.8824 |
| k=1 | 126K | 107,375 | 38.367 | 26.064 | 0.9844 | 0.9692 |
| k=2 | 4K | 3,475 | 45.209 | 22.120 | 1.8043 | 1.7660 |
| k=2 | 126K | 107,375 | 45.338 | 22.056 | 1.8043 | 1.7660 |
| k=3 | 4K | 3,475 | 54.381 | 18.389 | 2.4865 | 2.4211 |
| k=3 | 126K | 107,375 | 53.875 | 18.561 | 2.2821 | 2.2250 |

k=3 per-position acceptance: 4K `[0.946, 0.838, 0.703]`,
126K `[1.000, 0.718, 0.564]`.

### Result: every k loses, and higher k loses more

| k | 4K tok/s | vs no-draft |
|---|---|---|
| no-draft | 33.23 | — |
| k=1 | 26.18 | −21% |
| k=2 | 22.12 | −33% |
| k=3 | 18.39 | −45% |

**The winner is no-draft.** Best MTP k is k=1, and it still loses by ~21%.

### Acceptance is not the problem

Acceptance is high and rises with context: k=1 goes 0.896 at 4K to 0.984 at
126K, and k=2 sustains ~0.90 per position. So MTP's loss is **not** a draft
quality problem — the per-pass draft forward cost exceeds the tokens it saves.
That is why raising k (more accepted tokens per pass) still makes throughput
worse: the draft pass grows faster than the acceptance gain.

## Long-context acceptance cliff

Best MTP variant (k=1), one engine, 4K sentinels at both ends:

| context | prompt tok | ms/output-tok | output tok/s | accepted/pass |
|---|---|---|---|---|
| 4K sentinel in | 3,475 | 37.599 | 26.596 | 0.9200 |
| 150K | 127,825 | 38.049 | 26.282 | 0.9792 |
| 160K | 136,350 | 37.618 | 26.583 | **1.0000** |
| 164K | 139,750 | 37.473 | 26.686 | **1.0000** |
| 200K | 170,450 | 37.911 | 26.378 | 0.9792 |
| 240K | 204,525 | 37.490 | 26.674 | **1.0000** |
| 4K sentinel out | 3,475 | 37.606 | 26.591 | 0.9200 |

4K sentinel drift **+0.02%** — the engine stayed healthy across the whole
ladder. Xid delta 0, preemptions 0, no OOM.

### The ~163,840-token cliff does not reproduce

Acceptance stays between 0.920 and 1.000 from 4K all the way to 204,525 prompt
tokens, and per-token decode is flat at 37.5-38.0 ms/output-token. There is no
collapse at ~160K, at 164K, or anywhere beyond. The external recipe's cliff is
**not** a property of this checkpoint on CMP170HX.

Because there is also no region where MTP wins, there is no cliff-driven profile
switch to design: MTP is a net loss at every context measured.

## Long-context comparison, no-draft vs best MTP

| context | no-draft ms/tok | k=1 ms/tok | k=1 penalty |
|---|---|---|---|
| 4K | 30.10 | 37.60 | +25% |
| 160K | 30.11 | 37.62 | +25% |
| 240/250K | 30.01 | 37.49 | +25% |

The ~25% MTP penalty is flat across the entire context envelope.

## Data-quality notes

- k=3 4K sentinel drift was −6.8%, above the 2-3% health band. It is recorded
  rather than hidden, and no fresh-engine recheck was spent on it because k=3 is
  last by a wide margin at every cell and no conclusion depends on its exact
  value. The k=1 cliff ladder, which all conclusions rest on, drifted +0.02%.
- k=3 at `max_model_len=250000` fails to start on capacity: 7.12 GiB KV needed
  vs 7.10 GiB available (server-estimated maximum 249,600). This is marked as a
  capacity limit, not worked around. k=3 was run at 240,000.
- Both k=3 and the earlier no-draft run show `_mtp_hidden_buffer` allocation
  fine; no sampler/acceptance/state anomaly appeared.

## Answers

1. **Should MTP be enabled?** No. It loses at every k and every context.
2. **Best k?** k=1 is the least-bad MTP setting (26.2 tok/s vs 33.2 no-draft).
   It is not competitive, so it is not a production setting.
3. **Cliff?** None. Acceptance never collapses; the old ~163,840-token cliff is
   not reproduced.
4. **Production profile:** a single static no-draft profile covering the whole
   context envelope. There is no normal/ultra-long MTP split to make.
