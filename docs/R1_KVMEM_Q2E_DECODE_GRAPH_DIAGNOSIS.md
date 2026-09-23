# Q2E decode diagnosis: eager overhead, graph execution, and residual page cost

## Matched controls and graph sweep

The original Q2E 16K–240K sweep used eager execution and emitted 256 output
tokens per cell. A stock-QSA control at the identical 15,533-token prompt,
sampling settings, max model length, and 256-token denominator measured
96.013 ms/token eager versus 20.404 ms/token with piecewise CUDA graph.
Both controls returned the target facts. This establishes that eager execution
alone accounts for most of the original 16K decode latency; it does not make
the Q2E-specific cost disappear.

| Prompt tokens | Q2E eager ms/token | Q2E graph ms/token | Q2E graph tok/s | Graph READ+WRITE peak | Working-set peak |
|---:|---:|---:|---:|---:|---:|
| 15,533 | 110.395 | 37.920 | 26.37 | 1,114 | 971 |
| 79,533 | 115.966 | 44.722 | 22.36 | 4,160 | 3,157 |
| 159,533 | 121.972 | 54.380 | 18.39 | 4,160 | 3,840 |
| 239,533 | 126.228 | 54.632 | 18.30 | 4,160 | 4,096 |

The graph probe preserves the 16-token QSA pages, 128 WRITE + 4,032 READ
partition, 4,160-page GPU cap, 1,024-token scheduler chunks, direct transfers,
consumer synchronization, CPU-authoritative history, original selection and
semantic gates. MTP and prefix caching remain off. Each measured cell produced
256 tokens and both target codes in order; CPU round-trip checks and all 12 QSA
layer coverage passed. The 80K–240K cells ran under one model load after a 4K
warmup. The 16K graph probe was a separate single-cell run.

The graph path is **experimental**, not production-qualified. vLLM performs
synthetic all-padding QSA forwards while warming up and capturing graphs. The
probe skips these forwards so they cannot corrupt real WRITE mapping, CPU
backing or LRU state. A first implementation failed during capture because it
did not skip them. A later attempted shortcut that checked padding only in
`CUDAGraphMode.NONE` also failed during engine initialization: vLLM has a
pre-capture `PIECEWISE` warmup with all-padding rows. That failure occurred
before any real request; the installed QSA was restored. The working guard
checks the all-padding condition in both modes, so its GPU-to-host check may
itself contribute to the remaining latency.

## Residual cost, with limits on attribution

At 16K there were no decode reload misses, yet Q2E graph took 37.920 versus
stock graph's 20.404 ms/token: approximately 17.5 ms/token of Q2E-associated
overhead in this matched case. Worker `forward_exposed_wall_seconds` summed
over 12 layers and 255 timed decode tokens to 13.46 ms/token. Its components
included WRITE mapping 3.49, selection planning 2.56, table refresh 1.05 and
attention submission 1.47 ms/token. These are Python wall-clock diagnostics,
not additive GPU kernel measurements or guaranteed savings.

At 240K the same worker measure was 28.35 ms/token. Stage-history accounted
for 14.72 ms/token, of which slot assignment was 8.71, transfer calls 3.37,
table delta 1.06, residency lookup 0.69 and touch 0.71 ms/token. The 255 timed
decode tokens incurred 36,724 reload-miss pages across all layers. This
supports prioritizing bounded LRU victim selection and Python page bookkeeping
before redesigning PCIe transfers. It does not prove that all stage-history
wall time is on the end-to-end critical path.

Graph execution also left prefill throughput broadly unchanged: 16K 1,122.86,
80K 1,100.85, 160K 882.59 and 240K 701.94 prompt tok/s. These are
client-observed TTFT-derived rates, not kernel-only throughput. Graph capture
and model-load time are outside the request timings.

## Time-boxed LRU candidate: correct but not worth retaining

A stable partial-victim selector was tested at `3b458f87d`. Its CPU tests
passed, and a 16K/240K graph run passed semantic, CPU round-trip, capacity,
QSA restore, and Xid gates. At 16K decode was 38.119 ms/token versus the
earlier 37.920; at 240K it was 53.743 versus 54.632 ms/token. That 1.6% 240K
difference is not a reliable causal speedup: the two runs had 20,604 versus
36,724 decode reload misses, so they did not exercise the same access sequence.

To isolate algorithm behavior, both selectors were replayed over the existing
complete 160K access trace from `kvmem-k1q2e-array-lru-c54fff7-live1` (SHA-256
`36c5b462b3be166c4588430b71216d08be43a1ead223d07b4807e1feb58fb2dd`).
All 33,120 events and 2,097,938 assigned pages matched exactly, including
missing pages, victims, slots, and reuse generations. Assignment-call time was
11.697 s for the original full sort and 11.095 s for the partial selector,
only 0.602 s saved across the trace. A selection-only microbenchmark showed
larger speedups, but it omits candidate construction and the real page-event
mix. The partial selector was therefore **rejected and removed from the active
code**; the benchmark/replay experiment remains visible in Git history.

## Provenance and next gates

- Stock eager/graph control: `7eb0fa5e7babf74ec4779c29bc8035ca8d76e9fe`,
  `/home/base-node/.codex_tasks/qwen38-flashnext-r0/results/kvmem-q2e-decode-controls-7eb0fa5`.
- Q2E eager four-context baseline: `61a8eb2f8dde5e48f025024589494dbf5b9178cf`,
  documented in `R1_KVMEM_Q2E_CONTEXT_BENCHMARK.md`.
- Q2E graph 16K probe: `d1310ea33b4f2e14115edb1aae62b5a476a741fd`,
  `/home/base-node/.codex_tasks/qwen38-flashnext-r0/results/kvmem-q2e-graph-d1310ea`.
- Q2E graph 80K–240K sweep: `c8b03ce289d04ae9f9b480f415b5510f4896f4df`,
  `/home/base-node/.codex_tasks/qwen38-flashnext-r0/results/kvmem-q2e-graph-c8b03ce`.
- Failed pre-capture padding shortcut: `a590694e5755dc3fa539dee4d94bb71ab6f243bc`,
  `/home/base-node/.codex_tasks/qwen38-flashnext-r0/results/kvmem-q2e-graph-a590694`;
  boot failure, no runtime result.
- Rejected LRU candidate: `3b458f87df5af71a7de8300095d7a2cdadad867c`,
  `/home/base-node/.codex_tasks/qwen38-flashnext-r0/results/kvmem-q2e-graph-lru-3b458f8`.
- Exact trace-replay implementation: `c7fa37c413c020d6a5e470ae6bdd508df00c7285`.

The high-confidence improvement in this time box is enabling piecewise graph
execution: decode latency is roughly halved or better across 16K–240K while
the measured semantic and capacity gates remain intact. Further optimization
is **paused** under the user's one-hour decision rule because the next LRU
candidate did not demonstrate a reliable end-to-end benefit. Before treating
the graph probe as a production mode, it still needs repeated runs, broader
semantic/byte-oracle qualification, and a capture lifecycle that does not rely
on a per-layer padding host sync. Prefill, MTP, concurrency and persistent
sessions remain separate work.
