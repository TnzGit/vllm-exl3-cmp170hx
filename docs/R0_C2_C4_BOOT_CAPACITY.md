# C2/C4 boot-capacity finding and selected 240K revision

## Classification

The copied `c2_16k`, k=2 startup is a **boot-capacity NO-GO under the frozen
envelope; runtime performance is unmeasured**. PIECEWISE graph capture completed,
but EngineCore then rejected KV-cache sizing: **6.83 GiB was needed and 6.74
GiB was available**, a 0.09 GiB shortfall. The log estimates a 241,600-token
maximum against the required 246,000. The server exited before `/health`; the
startup proof remained incomplete. **No prompt or other benchmark request was
sent.** This is a startup-capacity result, not a C2 performance or correctness
cell.

The graph-capture log reports **0.56 GiB** captured. It also reports that CUDA
graph memory profiling is enabled and explains the utilization equivalence
when profiling is disabled. That capture-memory observation does not establish
that a reduced capture set will free enough memory or make the engine boot.

Cleanup and host state were clean: Xid count was 0 before and after (delta 0);
GPU use was 14 MiB before launch and after shutdown, with no compute process;
port 8002 had zero listeners and was closed both before and after. The owned
server process exited with status 1, the cell wrapper with status 2, and its
owned process group was stopped.

Evidence: [`evidence/r0-c2c4-fd0d6dc-c2_16k-k2-live1/`](../evidence/r0-c2c4-fd0d6dc-c2_16k-k2-live1/).
The recorded source and repository HEAD are `fd0d6dca871ca5a512393b48e334fe26a5086e73`;
the frozen manifest SHA-256 is
`0522e679e0d7daac5a735500e1109e331a6c877f32ea5fd8ae93ee42ea6c96ca`.

## Comparison with prior C1 evidence

The successful C1 production record reports a 279,087-token KV pool at
`MAX_MODEL_LEN=246000`, `gpu_memory_utilization=0.92`, and successful C1 MTP
measurements with zero Xid delta. The earlier prefill scan also documents a
separate 4096 boot-capacity NO-GO under its frozen envelope, followed by clean
Xid/GPU/port checks. These are useful historical controls for interpreting
startup admission and cleanup. The C1 pool is **not** a capacity guarantee for
this C2/k=2 boot: this run used `MAX_NUM_SEQS=4`, and its own current startup
estimate is the applicable evidence. Neither prior C1 success nor the C1
279,087-token figure turns this failed boot into a runnable C2/C4 cell.

References: [`R0_MTP_K3_PRODUCTION.md`](R0_MTP_K3_PRODUCTION.md),
[`R0_MTP_POST_COOP.md`](R0_MTP_POST_COOP.md), and
[`R0_PREFILL_SCAN_LIVE2.md`](R0_PREFILL_SCAN_LIVE2.md).

## Selected prospective protocol revision

The user has selected **`max_model_len=240000` with the default PIECEWISE
capture sizes** as the prospective final benchmark configuration. Preserve
`gpu_memory_utilization=0.92` and `max_num_seqs=4`; retain the runtime's default
effective capture list rather than changing it. This selection updates the
prospective protocol only. It has **not been booted or benchmarked**, and does
not change the factual 246000 boot-capacity NO-GO above.

The explicit PIECEWISE capture-size set `[1,2,4,8,16,24]` is deferred and
unrun; it is not part of the selected configuration. If reconsidered later,
record the launch configuration and source/model identity, effective capture
list, graph-capture memory, available KV GiB and token pool, estimated maximum
model length, health and exit status, GPU/process snapshots, Xid counts, and
port state. Keep graph-memory profiling/accounting enabled: setting
`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` changes the accounting basis and
is not an accepted way to claim real graph-memory reduction.

For the selected 240000 configuration, preserve the same startup and cleanup
evidence. A failed boot remains NO-GO with no requests; any request cell must
pass the applicable capacity/admission gate. No performance result is
available from the evidence summarized here.
