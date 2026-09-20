# PR #4 low-risk load policy — hardware qualification

Branch `r0/load-low-risk-qualify`, based on `r0/load-profile` plus the PR #4
low-risk toggles and the existing section instrumentation.

- base `a9925cb` (`r0/loadfix-qualify`) -> `r0/load-profile`
- qualification head at run time: `fa65d7eb039553d298b597b2474eb7a5e72bb0a7`
- PR #2 and PR #4 were not merged; `r0/load-profile` history was not rewritten

## Policy under test

```text
VLLM_EXL3_LOAD_PROFILE=1
VLLM_EXL3_MADV_AFTER_H2D=0        (discrete GPU: let Linux reclaim page cache)
VLLM_EXL3_EXPERT_MATCH_CACHE=1    (memoize successful first-match)
VLLM_EXL3_GC_AFTER_MOE_LAYER=0    (skip the UMA-era full-heap collect)
```

Everything else held at R0: same checkpoint and prepared pack, vLLM 0.29.0,
ExLlamaV3 1.5.0, text-only, no draft, prefix caching off, disk n-gram,
PIECEWISE graphs, C1, 4K, same `gpu_memory_utilization`.

## CPU gate

```text
py_compile src/vllm_exl3/exl3.py tools/cmp170hx_qwen_preflight.py tools/r0_bench_trellis_h2d.py   OK
pytest test_load_low_risk_cuts test_direct_fill_regression test_trellis_prescan test_madv_view_range
       33 passed, 3 skipped
pytest scaffold + arena + trellis + mixed-K prescan
       29 passed
```

Imported plugin verified before boot (`_match_expert_mapping`,
`_exl3_gc_after_moe_layer_enabled`, `load_profile_summary` all present and
loaded from the qualification worktree).

## Section counters (the comparable metric)

| section | baseline | run 1 | run 2 |
|---|---|---|---|
| **expert.name_match** | 22.24 s | **5.448 s** | **5.150 s** |
| **scale.madv** | 13.07 s | **0.270 s** | **0.259 s** |
| **trellis.madv** | 10.06 s | **0.154 s** | **0.148 s** |
| **per_layer.gc_collect** | 20.81 s | **0 calls** | **0 calls** |
| trellis.direct_H2D | 72.12 s | 62.28 s | 60.76 s |
| scale.copy_H2D | 4.92 s | 4.775 s | 4.732 s |
| marker.fill | 1.64 s | 1.545 s | 1.483 s |
| prescan | 2.67 s | 2.615 s | 2.538 s |
| instrumented total | 127.11 s | 77.10 s | 75.08 s |

Raw loads:

| metric | baseline | run 1 | run 2 |
|---|---|---|---|
| `Loading weights took` | 159.40 s | 108.20 s | 105.14 s |
| `Model loading took` | 195.12 s | 124.14 s | 124.00 s |

### Attributable gain (the three low-risk items only)

| item | baseline | run 1 | run 2 | saved |
|---|---|---|---|---|
| name_match | 22.24 | 5.448 | 5.150 | 16.79 / 17.09 |
| madv (scale+trellis) | 23.13 | 0.424 | 0.407 | 22.71 / 22.72 |
| gc.collect | 20.81 | 0 | 0 | 20.81 / 20.81 |
| **total** | | | | **60.31 / 60.62 s** |

Both runs land at ~60.5 s of attributable recovery, comfortably past the 30 s
success bar and near the 45–60 s "very good" band.

`trellis.direct_H2D` measured 62.3 s and 60.8 s against a 72.1 s baseline. That
is consistent across both runs but is **not** counted above: the code path is
unchanged and page-cache state moves this section by ~25 s run to run. It is
plausibly a real side effect of leaving safetensors pages resident (MADV off),
but it is reported separately rather than claimed.

## Memory gate

| metric | before | during-load min | after |
|---|---|---|---|
| MemAvailable | 45.99 GB | **43.09 GB** | 42.97 GB |
| Cached | 30.90 GB | — | 26.86 GB |
| Swap used | 0.32 GB | — | 0.32 GB |
| GPU VRAM | 14 MiB | — | 14 MiB |

- No OOM, no swap growth, no runaway host growth.
- Engine `VmHWM` 8.91 GB peak, `VmRSS` 2.48 GB.
- Memory recovers after the server exits.

## Correctness / stability

```text
PRESCAN ready = 48        (main model has 48 MoE layers)
mode=direct_plan = 48
mode=post_stage_pack = 0
staging fallback summary = 0
staging fallback (no arena plan) = 0
greedy token IDs: identical to R0 baseline, 4/4 identical
Xid delta = 0
```

## Trellis H2D microbenchmark (lifetime-safe pipeline)

`tools/r0_bench_trellis_h2d.py --bytes 655360 --calls 4096 --repeat 3 --slots 4 8 16 32`
(2.500 GiB total, no model load):

| method | per call | throughput | runs |
|---|---|---|---|
| pageable blocking (current) | 128.34 µs | 4.756 GiB/s | 0.527 / 0.5256 / 0.5257 |
| pageable nonblocking | 98.44 µs | 6.200 GiB/s | 0.4032 ×3 |
| pinned ring slots=4 | 98.44 µs | 6.200 GiB/s | 0.4033 / 0.4032 / 0.4032 |
| pinned ring slots=8 | 98.44 µs | 6.200 GiB/s | 0.4032 ×3 |
| pinned ring slots=16 | 98.43 µs | 6.201 GiB/s | 0.4032 ×3 |
| pinned ring slots=32 | 98.66 µs | 6.186 GiB/s | 0.4042 / 0.4041 / 0.4041 |

### Gate verdict: do not build the pipeline

- Pinned ring is **1.30×** the current blocking path and **identical** to plain
  nonblocking (98.44 µs both). Ring size buys nothing.
- The agreed gate required >=3× or >=30 s projected full-model saving. 1.30×
  fails it outright.
- The primitive is already at ~6.2 GiB/s, which is the practical PCIe Gen2 x16
  ceiling. So the ~824–978 µs per call seen in the real load is **not** a copy-API
  problem and a pinned ring cannot address it.
- The real gap is ~7×: 73,728 calls × ~640 KB = ~45 GiB, which at 6.2 GiB/s is
  ~7.3 s of transfer but measures 61–72 s. Whatever the remaining ~55 s is, the
  microbenchmark shows it is outside the copy primitive — most plausibly
  first-touch page faults on the mmap-backed source views, which the ring does
  not change.

Writing a production trellis pipeline is therefore not justified. No
marker batching, scale batching, shard loader, QSA, MTP, context sweep or R1
work was done.
