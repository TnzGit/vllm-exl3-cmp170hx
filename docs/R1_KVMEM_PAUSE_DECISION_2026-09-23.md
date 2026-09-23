# KVMEM research pause decision (2026-09-23)

## Status and scope

**PAUSED, not qualified for production and not a verdict that the KVMEM
architecture is impossible.** Stop new KVMEM implementation, tuning, and GPU
runs. Preserve the research branch, PR discussion, exact-SHA experiment
artifacts, and installed-source restore discipline. Do not merge the research
PR. Reopening requires the evidence gates below, not a calendar date.

The two possible performance objectives are distinct:

1. Beat a matched non-KVMEM baseline on single-request cold prefill/decode.
2. Keep single-request latency reasonably close to that baseline while
   materially improving *measured* long-context concurrent service throughput
   and latency, not merely the reported KV capacity.

The present evidence favors investigating objective 2 if the line resumes.
Neither objective has been demonstrated.

## Measured evidence and limits

| Observation | Result | Interpretation |
|---|---:|---|
| Matched 16K stock, no MTP, piecewise graph | 20.404 ms/output token; 49.01 tok/s | The relevant single-request control. |
| Matched 16K Q2E, no MTP, piecewise graph | 37.920 ms/output token; 26.37 tok/s | 17.516 ms/token slower, despite zero decode reload misses. |
| Matched 16K prefill TTFT | stock 10.966 s; Q2E 13.833 s | Q2E is about 26% slower. |
| Q2E graph, 80K / 160K / 240K | 44.722 / 54.380 / 54.632 ms/output token | One measured sample per length; no concurrent run. |
| Q2E 240K stage-history | 14.72 ms/token, including 8.71 ms slot assignment and 3.37 ms transfer calls | Python wall-time components are not guaranteed end-to-end savings. |
| Q2E 240K 255-token timed decode | 36,724 reload-miss pages, about 1.121 GiB H2D | PCIe matters, but copying alone is not the entire gap. |

All Q2E graph cells returned the target facts under the frozen semantic test,
kept the 4,160-page real-QSA cap and CPU round-trip checks, and used MTP off,
prefix caching off, one request, and an experimental graph-capture guard.
This is not a production qualification or a broad quality evaluation. See
`R1_KVMEM_Q2E_DECODE_GRAPH_DIAGNOSIS.md` and
`R1_KVMEM_Q2E_CONTEXT_BENCHMARK.md` for exact SHAs, run directories, and
measurement definitions. The separate non-KVMEM MTP-k3 production result
of 90.224 tok/s used a different prompt/configuration and is a service target,
**not** a matched Q2E A/B (`R0_MTP_K3_PRODUCTION.md`).

The 16K zero-miss gap makes fixed QSA integration overhead the first obstacle.
Even eliminating the measured 240K transfer-call time entirely would recover
only about 3.37/54.63 = 6.2% of that Q2E decode time, and transfer-call time
is not necessarily all exposed. A stable LRU partial-sort candidate saved
only 0.602 s in an exact 33,120-event 160K replay; its apparent live gain
was confounded by different miss counts and it was rejected. Do not repeat
minor LRU sorting work without new evidence.

## Capacity hypothesis, not a throughput result

The present single-request dedicated QSA pool is 4,160 pages/layer x 12
layers x 32 KiB/page, or about 1.523 GiB GPU. Independently replicating that
pool for each request would cost 1.523 GiB GPU per request; that isolation is
not implemented yet. Full unbounded QSA
would be roughly 5.49 GiB at 240K tokens (about 3.65 GiB at 160K), so the
potential 240K QSA GPU saving is about 3.97 GiB/request. This does not include
the non-QSA vLLM KV groups, graph capture, scratch, allocator fragmentation,
or any MTP follower. CPU backing at the configured 246K maximum is roughly
5.63 GiB/request before other host memory.

The stock no-MTP 240K boot log reports generic KV capacity for about 1.49
246K-token requests. The Q2E log's generic KV estimate of 14.31 such requests
**must not** be read as supported KVMEM concurrency: the dedicated QSA pool is
separate, `max_num_seqs=1`, and current virtual-ID/worker/CPU-backing state is
not isolated per request. A rough static budget suggests two independent 240K
QSA pools could fit on this GPU, and four might fit with much less margin, but
that is not a boot, semantic, or throughput result. Host RAM would become a
separate constraint at four maximum-length requests.

Smaller per-request pools are not an established shortcut. A diagnostic replay
of the existing complete 160K access trace, reconstructed from the *warm
prefill state*, found 5,559 baseline decode misses after the first ten tokens
versus 10,218 with a 2,048-page pool over the remaining 94 tokens. At 1,024
pages it found 25,492. This is one trace and an offline policy simulation,
not a live capacity/performance qualification. It shows why the small
instantaneous selection set cannot alone justify aggressive pool shrinking.

## Ranked directions if evidence warrants reopening

1. **Reduce fixed decode overhead.** Replace per-layer Python/host-sync page
   planning and READ-table bookkeeping with a batched native/GPU-aware plan;
   make graph capture lifecycle safe without an all-padding GPU-to-host check
   on every real forward. Preserve original selection, exact logical K/V
   mapping, generations, and the 4,160 cap. A zero-reload 16K comparison is the
   cheapest falsification test. Direct transfer/prefetch improvements follow
   only after timing shows exposed waits.
2. **Implement actual C2 isolation.** Give each request its own virtual ID
   namespace, CPU authoritative history, READ/WRITE mappings, slot generations,
   admission/preemption lifecycle, and independently bounded GPU pool. Test
   simultaneous 240K requests and cross-request non-aliasing before C4. A
   memory-capacity gain is not a throughput or tail-latency gain.
3. **Restore production features deliberately.** Re-enable and validate
   MTP-k3 target/draft residency and rollback only after C1/C2 correctness;
   account for its memory and acceptance-rate effects. Treat persistent
   sessions, recurrent/GDN state checkpointing, and later-turn CPU-KV reuse as
   a separate possible TTFT win, compared fairly against stock prefix caching.
   Do not reintroduce the progressive mask already shown to cause semantic
   failure or change the frozen semantic gate to obtain a speed result.

## Explicit reopening and acceptance gates

Before substantial implementation, run repeated, identical-prompt,
identical-output-length 240K stock-versus-Q2E comparisons with the same graph,
MTP, and prefix-cache settings; report cold prefill, decode, whole-request wall,
variance, and evidence artifacts separately. The existing 16K matched control
remains a fixed-overhead diagnostic. Check whether a bounded, correctness-safe
prototype can bring Q2E C1 within an agreed latency budget (provisionally
10-15%, **not** a passed criterion) before claiming objective 2.

Then qualify C2 on matched 240K concurrent requests against stock's feasible
concurrency: aggregate output throughput, per-request TPOT/TTFT and p95,
semantic/byte-oracle coverage, GPU and host peaks, transfer counts, Xid,
process/port cleanup, and byte-exact installed-QSA restore. Report any stock
preemption/admission behavior. Only advance to C4 or MTP with a demonstrated C2
benefit and sufficient memory margin. Cold-request and repeated-turn benefits
must be reported as separate claims. Capacity logs or linearly multiplied
C1 tok/s are not substitutes for these measurements.

## Repository disposition

Keep the KVMEM branch and all hardware evidence for reproducibility. Close the
Draft research PR as **paused, unmerged**, with a pointer to this decision;
it may be reopened when the above gates and resource priorities justify it.
No hardware or installed-vLLM change is required to make this decision.
