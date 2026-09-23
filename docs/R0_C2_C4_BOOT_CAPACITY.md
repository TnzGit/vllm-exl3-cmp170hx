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

## First 240K protocol revision (historical)

The user initially selected **`max_model_len=240000` with the default
PIECEWISE capture sizes** for the benchmark. Preserve
`gpu_memory_utilization=0.92` and `max_num_seqs=4`; retain the runtime's default
effective capture list rather than changing it. This selection updates the
protocol only and does not change the factual 246000 boot-capacity NO-GO above.

The explicit PIECEWISE capture-size set `[1,2,4,8,16,24]` was deferred at
that point. For any later revision, record the launch configuration and
source/model identity, effective capture list, graph-capture memory, available
KV GiB and token pool, estimated maximum model length, health and exit status,
GPU/process snapshots, Xid counts, and port state. Keep graph-memory
profiling/accounting enabled: setting
`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` changes the accounting basis and
is not an accepted way to claim real graph-memory reduction.

## First 240K startup attempt at `67bc16f`

With the default nine PIECEWISE graph sizes, `max_model_len=240000`, `.92`
memory utilization and `max_num_seqs=4`, EngineCore reported **6.78 GiB**
available KV memory and a **242,944-token** GPU KV pool. The API reached
`/health` and `/v1/models`; the original 246K sizing failure did not recur.
This establishes startup admission for the 240K service envelope, not
concurrent-request correctness or throughput. The available pool clears the
largest scheduled C2 live-set plus the preregistered 20% headroom criterion;
the per-cell runtime proof still needs to validate that gate.

The runner then rejected the live API process before building its startup
proof or sending a request. Its identity check expected `vllm` as argv[0],
whereas the installed entrypoint runs as `python /venv/bin/vllm serve`.
Classification: **runner-gate false refusal; runtime cell unmeasured**. The
owned server group was stopped, Xid stayed 0, GPU returned to 14 MiB with
no compute process, and port 8002 closed. A subsequent exact-SHA revision
must correct only this process-identity gate and repeat the full cell.

Evidence: [`evidence/r0-c2c4-67bc16f-c2_16k-k2-live1/`](../evidence/r0-c2c4-67bc16f-c2_16k-k2-live1/).
Frozen manifest SHA-256:
`43d983a7a074293d16065bde8a572aa4839205ae9d8f8150996292f9541c6ef5`.
No performance result is available from either attempt summarized here.

## Matched default-graph attempts at `c611ea2`

The corrected runner used the same fixed 240K manifest
(`27487834307b69f25c57c8421f1545100a451184c293fbd717d8a19027e6e555`)
for both k values, with `.92` memory utilization, `max_num_seqs=4`, and the
default PIECEWISE capture list. The k=2 C2/16K cell passed startup proof,
ten valid measured waves, semantic checks, and cleanup: median aggregate
output **20.45 tok/s**, median TTFT **14.55 s**, median TPOT **36.73 ms**,
0 preemptions, 0 discarded waves, 0 Xid delta, GPU 14 MiB afterward, and
port 8002 closed. Its startup log and proof agree on a **272,392-token** KV
pool. This is aggregate request throughput including prefill, not pure decode
speed. Evidence: [`evidence/r0-c2c4-c611ea2-c2_16k-k2-live1/`](../evidence/r0-c2c4-c611ea2-c2_16k-k2-live1/).

The matched k=3 C2/16K cell failed **before `/health` and before requests**:
EngineCore required 6.83 GiB KV for 240,000 tokens, had 6.79 GiB available,
and estimated a 238,400-token maximum. Its earlier log line rounded
available KV to 6.83 GiB; the exception's 6.79 GiB is the final admission
value. This is a k=3 **boot-capacity NO-GO under the default graph profile**,
not a runtime performance or correctness measurement. Xid remained 0, GPU
returned to 14 MiB with no compute process, and port 8002 closed. Evidence:
[`evidence/r0-c2c4-c611ea2-c2_16k-k3-live1/`](../evidence/r0-c2c4-c611ea2-c2_16k-k3-live1/).

The two startup KV pools differ materially; a pool measured in a different
engine cannot be substituted for k=3's failed admission. There is no valid
matched k=2/k=3 speed comparison yet. A smaller fixed graph-capture list
would constitute a further protocol revision and require both k values to
be rerun under the same exact configuration; these default-graph results
must not be pooled with such a revision.

## Approved reduced-graph 240K revision (prospective)

The user approved the uniform `[1,2,4,8,16,24]` PIECEWISE capture list after
the default-graph k=3 failure. The runner now pins and checks that list in
its environment and actual vLLM argv, while the runtime proof must read the
effective list from EngineCore's normalized compilation config. This changes
neither the 240K maximum length nor `.92` memory utilization, `max_num_seqs=4`,
MTP k values, prompts, resident/semantic gates, or graph-memory accounting.
Every cell under this revision needs a fresh engine and a new exact source SHA;
none of the earlier default-graph results counts as a matched speed result.
If k=3 still fails admission, retain the boot NO-GO and stop rather than
altering another memory parameter.
