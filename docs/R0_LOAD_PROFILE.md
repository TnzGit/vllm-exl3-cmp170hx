# R0 load-path profile — where the remaining 187.7 s goes

Branch: `r0/load-profile` (child of `r0/loadfix-qualify` @ `a9925cb`).
PR #2 itself is untouched.

## Method

`nsys` is not installed on this host. `py-spy` 0.4.2 and `perf` 7.0.14 were
used instead; both needed `sudo` (`perf_event_paranoid=4`, `yama.ptrace_scope=1`,
relaxed to 2/0 for the run and restored afterwards).

Two instruments:

1. `tools/r0_profile_load.sh` — one no-draft 4K startup with `py-spy record`
   and `perf record` attached to the live EngineCore, stopping as soon as the
   server reports healthy.
2. `VLLM_EXL3_LOAD_PROFILE=1` — diagnostic-only counters in the plugin
   (default OFF), aggregating wall time per load section. These separate
   *blocking CUDA time* from *Python dispatch time*, which sampling alone
   cannot do for synchronous copies.

## Perf (native, self time)

| DSO | share |
|---|---|
| python3.12 | 52.7% |
| kernel | 15.5% |
| **libcuda** (real H2D) | **11.0%** |
| libtorch_cpu | 5.3% |
| libgomp (+ OpenMP barriers) | 3.9% |
| libtorch_python | 2.6% |
| libc | 2.4% |
| safetensors | 1.5% |

Top leaf symbols: `PyUnicode_Contains` 12.0%, `_PyEval_EvalFrameDefault` 9.1%,
`gomp_barrier_wait_end` 2.6%, `kernel_init_pages` 2.0%, `mangle_path` 0.8%,
`show_vma_header_prefix` 0.5%. The kernel/string symbols are `/proc/self/maps`
parsing; the rest is Python dispatch over ~295k routed tensor entries.

## Counter attribution (one 4K no-draft startup)

`Loading weights took 159.40 s`, `Model loading took 195.12 s`.

| section | calls | wall | avg |
|---|---|---|---|
| **trellis.direct_H2D** | 73,728 | **72.12 s** | 978 µs |
| **expert.name_match** | 294,912 | **22.24 s** | 75.4 µs |
| scale.madv | 148,611 | 13.07 s | 88.0 µs |
| trellis.madv | 73,728 | 10.06 s | 136.5 µs |
| scale.copy_H2D | 148,611 | 4.92 s | 33.1 µs |
| prescan | 48 | 2.67 s | 55.5 ms |
| **marker.fill** | 73,728 | **1.64 s** | 22.2 µs |
| per_layer.empty_cache | 1 | 0.006 s | — |
| instrumented total | | **127.11 s** | |

Plus, outside that window: `per_layer.gc_collect` = 48 calls, **20.81 s total**
(0.43 s per MoE layer).

## What this disproves

**Marker batching is not the answer.** The 73,728 `dest.fill_(marker)` ops the
hypothesis predicted do exist, but they cost **1.64 s total** (22 µs each).
Even reducing them to 192 bulk copies saves under 1.6 s of a 195 s load.

**Scale copies are also not dominant.** 148,611 `suh/svh` H2D copies cost
**4.92 s**. Batching them to ~192 copies is bounded by that.

So neither of the two candidates the theory nominated is where the time is.

## Where the time actually is

1. **`trellis.direct_H2D` — 72.1 s (37%).** 73,728 blocking
   `arena[idx].copy_(src, non_blocking=False)` calls at 978 µs each for ~640 KB
   per tensor. The payload totals ~45 GB, which at PCIe Gen2 x16 is ~7.5 s of
   pure transfer; the other ~65 s is per-call synchronous-copy latency. This is
   the single largest lever and it is *latency*-bound, not bandwidth-bound.
2. **`expert.name_match` — 22.2 s (11%).** Pure Python substring scan
   (`weight_name not in qual_name`) over 294,912 routed entries.
3. **`gc.collect()` per MoE layer — 20.8 s (11%).** 48 full-heap collections on
   a process holding ~300k tensors. It exists to release transients on UMA
   hosts; this is a discrete GPU.
4. **madv VMA lookup — 23.1 s (12%).** `trellis.madv` + `scale.madv`. Each of
   ~222k calls re-opened and line-parsed `/proc/self/maps` (1,078 lines here);
   measured at ~104 µs per call.

## Implemented fix: cached VMA table

`_find_containing_vma` now parses `/proc/self/maps` once into a sorted table and
refreshes only on a miss. Mapping set only grows during a load (shards open
lazily), so a stale hit at worst makes one `madvise` fail harmlessly and a
stale miss only skips one reclaim. The function's contract (return the
containing VMA with its pathname, heap or file) is unchanged and its existing
test still passes.

`VLLM_EXL3_LOAD_PROFILE=1` and the VMA cache stay diagnostic/profiling scoped:
the counters are default-OFF.

### A/B (same pack, runtime, flags, `gpu_memory_utilization`)

Two VMA-cache runs were taken because the first showed a larger drop than the
change can plausibly explain.

| metric | baseline | VMA run 1 | VMA run 2 |
|---|---|---|---|
| `Loading weights took` | 159.40 s | 110.87 s | 135.28 s |
| `Model loading took` | 195.12 s | 146.21 s | 170.95 s |
| **trellis.madv + scale.madv** | **23.13 s** | 7.59 s | **7.10 s** |
| trellis.direct_H2D | 72.12 s | 46.90 s | 66.16 s |
| scale.copy_H2D | 4.92 s | 3.61 s | 4.86 s |
| expert.name_match | 22.24 s | 22.02 s | 21.78 s |
| marker.fill | 1.64 s | 1.56 s | 1.61 s |
| prescan | 2.67 s | 2.54 s | 2.54 s |
| instrumented total | 127.11 s | 84.60 s | ~107 s |

**What is attributable.** The madv path drops from 23.13 s to 7.10–7.59 s with
identical call counts. That is a direct, isolated, reproducible effect of the
change: −16 s. Every other section is stable across the three runs
(`name_match` 22.2/22.0/21.8, `marker.fill` 1.64/1.56/1.61, `prescan`
2.67/2.54/2.54, `scale.copy_H2D` 4.92/3.61/4.86).

**What is not attributable.** `trellis.direct_H2D` measured 72.12 s, 46.90 s and
66.16 s across the three runs — a 25 s spread on an unchanged code path. That
variance is page-cache state, not the fix, so the run-1 drop must not be
credited to the VMA cache. An earlier draft of this document did credit it;
this is the corrected reading.

**Honest net.** The VMA cache reliably removes ~16 s of the ~195 s startup.
Total-startup improvement measured between 16 s and 49 s depending on
page-cache state, and the wide end is not reproducible.

Correctness on the VMA build: `PRESCAN ready 48`, `direct_plan 48`, staging
fallback 0, greedy token IDs identical to the R0 baseline (4/4), Xid delta 0.

## Recommended next cuts (evidence-ranked)

1. **Batch / pipeline the trellis H2D (up to ~65 s).** Gather a layer's expert
   trellis views into one CPU staging buffer, then one arena copy per layer
   (48 large copies instead of 73,728 synchronous small ones). Needs a staging
   memory budget check; must not reintroduce short-lived `non_blocking=True`
   temporaries without explicit event/lifetime ownership.
2. **Cache / fast-path `expert.name_match` (~22 s).** Requires care: the
   candidate list comes from vLLM's `get_expert_mapping()`, so the match order
   must be preserved exactly.
3. **Reduce `gc.collect()` frequency (~20 s).** It is a UMA-era guard on a
   discrete GPU; verify host RSS and page-cache behaviour before changing.
4. **Marker batching: do not do it.** Bounded at 1.6 s.
