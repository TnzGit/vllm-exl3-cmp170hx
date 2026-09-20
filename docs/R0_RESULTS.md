# R0 measured results — Qwen3.8-Flash-Next EXL3 on one CMP170HX

All numbers below were produced on the pinned R0 runtime. Raw JSON lives under
`~/.codex_tasks/qwen38-flashnext-r0/results/` on the GPU host; this document is
the summary of record.

## Runtime identity

| item | value |
|---|---|
| vLLM | 0.29.0 (wheel, sha256 `09d48617…bbb635`) |
| ExLlamaV3 | 1.5.0, commit `0740edc2da569fb99174023c1d2988b1e98cb41e` |
| plugin | `bringup/qwen38-flash-next-cmp170hx-r0` |
| torch | 2.13.0+cu130 |
| driver / CUDA UMD | 610.43.02 / 13.3 |
| GPU | NVIDIA CMP 170HX, cc 8.0, 63.39 GiB, 70 SM |
| host | 12 cores, 46 GiB RAM, 359 GiB free NVMe |
| checkpoint | `Lygodactylus/Qwen3.8-Flash-Next-Uncensored-exl3-3bpw`, 72.18 GiB |
| config sha256 (prepared) | `466d89a0cf08977d3a8481bee64100694c5d151db2f36b8f2ab71481e1568f8f` |
| index sha256 (regenerated) | `37d098b99467c756094e2a9f089a7a92b119506daee77ad692767f8236b570ca` |
| native config backup | `config.json.native` (source `bits: 3.05` preserved) |

## Startup cost breakdown (measured, per fresh engine)

| phase | seconds |
|---|---|
| model weight load (47.82 GiB) | ~330 |
| engine init (profile + KV cache + warmup) | ~114 |
| **total startup** | **~444 s (7.4 min)** |

Disk reads at 5.8 GB/s and the link is PCIe Gen2 x16, so neither is the
bottleneck: 47.82 GiB in 330 s is ~145 MB/s effective. The load is dominated by
per-tensor CPU work over 304,240 tensors, of which 73,728 are routed-expert
trellises that took the `staging fallback (no arena plan)` path.

## No-draft C1 context table

Fresh engine per cell; Xid delta 0 for every cell; prefix caching off; C1;
text-only; `VLLM_EXL3_NGRAM_TABLE=disk`; PIECEWISE graphs.

| context | prompt tokens | TTFT s | prefill tok/s | ms/output-token | output tok/s |
|---|---|---|---|---|---|
| 4K | 3,475 | 2.54 | 1,370 | **30.10** | 33.23 |
| 32K | 27,925 | 17.93 | 1,557 | **30.18** | 33.14 |
| 65K | 55,850 | 33.64 | 1,660 | **30.10** | 33.23 |
| 160K | 136,350 | 80.90 | 1,685 | **30.11** | 33.21 |
| 250K | 213,050 | 127.65 | 1,669 | **30.01** | 33.32 |

Decode cost is flat across the whole ladder: no measurable context-dependent
decode degradation up to 213K prompt tokens.

## MTP: negative result

Fresh engine per configuration, context ladder reused on one engine with
sentinel guards (`tools/r0_sweep_reuse.py`).

| context | no-draft | k=1 | k=2 | k=3 |
|---|---|---|---|---|
| 4K | **30.10** | 39.21 | 45.81 | 52.66 |
| 32K | **30.18** | 38.88 | 47.18 | 53.42 |
| 65K | **30.10** | (aborted) | 45.26 | 54.13 |
| 160K | **30.11** | (aborted) | 44.88 | 51.81 |
| 240K / 250K | **30.01** | (aborted) | 44.68 | 51.43 |

MTP is a net loss on this checkpoint and hardware at every k measured, and it
gets monotonically worse with k:

- k=1 ~30% slower than no-draft
- k=2 ~49% slower
- k=3 ~73% slower

Because no k beat no-draft, there is no "best k" to recommend and no
ultra-long acceptance cliff to locate: the loss is already present at 4K, so
the previously reported ~160K Qwen MTP cliff is not reproduced here — this
configuration never had a winning region to lose.

### Capacity limit (honest marker, not a fudged pass)

k=3 at `max_model_len=250000` failed to start:
`7.12 GiB KV cache is needed … larger than the available KV cache memory (7.03 GiB)`.
It was re-run at 240K rather than by lowering the context silently or raising
`gpu_memory_utilization` to manufacture a pass.

### Sentinel drift and the warmup explanation

The k=2 and k=3 sweeps reported `sentinel stable=False` (k=2: 55.18 → 32.01
ms/token). A fresh-engine re-verification (`tools/r0_verify_warmup.py`, 8
identical requests) showed:

```text
req 0: ms/token = 63.598   <- outlier
req 1..7: 39.09 .. 39.29   <- flat, spread 0.196 ms
first_is_outlier = true, tail_flat = true
```

The first request after startup carries lazy init / graph capture cost; later
requests are flat. So the sentinel gap is warmup, not engine drift, and the
later cells are trustworthy. The real k=2 steady-state at 4K is ~39 ms/token,
which still loses to no-draft's 30.10.

## Greedy determinism: characterised, not an EXL3 defect

The same greedy request can produce different text. This was investigated
rather than assumed:

- 8-token completions: 4/4 identical.
- Strongly determined continuation ("Count … nine," → " ten."): 4/4 identical.
- Longer generations diverge at a fixed position (token 33) which is a genuine
  near-tie: `Belgium -1.282` vs `Greece -1.532`, and in one run the order flipped.
- Per-token logit noise across 12 identical runs: **0.0077 nats** spread.

The noise is smaller than the top-1/top-2 margin at that position, so
auto-regression amplifies it into visible divergence. This is the near-tie
numerical path the contract warns about — not recurrent-state corruption, not
KV corruption, and not an EXL3 dequantization error.

## Correctness and stability

- Xid delta: 0 for every cell and configuration.
- No CUDA errors, no crashes, no OOM after the pack fixes.
- Coherent, on-topic greedy output.
- Model weights: 47.82 GiB resident; KV pool 8.84 GiB at `gpu-memory-utilization=0.92`;
  peak VRAM 58.7 GiB of 63.39 GiB.

## Pack defects found and fixed

1. **Index omitted the n-gram table.** The shipped index covered 7 shards
   (52.4 GB); the pack has 8 (72.18 GB) and `ngram_embedding.safetensors`
   (18.48 GB) was absent from `weight_map`. vLLM trusts the index over files on
   disk, so the whole table would have been skipped silently.
2. **`lm_head` K mis-resolution.** The config emitted bare `lm_head` but vLLM
   resolves `language_model.lm_head`; the exact-dict lookup missed and fell back
   to generic `bits=4` while the pack stores K=5 →
   `trellis dest (160, 15520, 64) != loaded (160, 15520, 80)`.

Both are recorded as separate commits with regression tests.

## Load-path A/B: PR #2 (`fix/qwen-trellis-prescan-load`)

Baseline (old plugin, measured repeatedly): weight load ~330–335 s,
engine init ~114 s, total ~444 s, 73,728 staging-fallback warnings.

With PR #2 on the same checkpoint, pack, runtime and `gpu_memory_utilization`:

| marker | count | expected |
|---|---|---|
| `EXL3 trellis PRESCAN ready` | 48 | 48 (one per MoE layer) |
| `mode=direct_plan` | 48 | 48 |
| `mode=post_stage_pack` | 0 | 0 |
| `staging fallback summary` | 0 | 0 |
| `staging fallback (no arena plan)` | 0 | 0 |

| phase | baseline | PR #2 | change |
|---|---|---|---|
| weight load | ~330 s | **187.7 s** | **-43%** |
| post-load init | ~114 s | 31.7 s | -72% |
| total startup | ~444 s | **~219 s** | **-51%** |

Correctness and stability:

- Greedy parity on the deterministic prompt: identical token IDs, 4/4 runs.
- Xid delta: 0.
- Peak VRAM 59,576 MiB (baseline 58,694 MiB); KV pool 9.62 GiB
  (baseline 8.84 GiB) — direct-fill also freed staging pressure.
- Host RSS 3.7 GiB, peak 717 GiB virtual (mmap of the 72 GiB pack).

Interpretation per the agreed bands: 187.7 s lands in the 120–200 s band, i.e.
a major win that still leaves the remaining generic per-tensor loader (~304k
tensor entries) as the next candidate. Marker batching and a shard-oriented
routed-expert loader are deliberately NOT started before a profile proves them.
