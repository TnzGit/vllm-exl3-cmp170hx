# Non-KVMEM prefix-cache compatibility probe

This is a separate, diagnostic run, not a throughput qualification. It keeps
the installed EXL3 model, MTP k=3, 246000 max length, auto batching, and
0.92 memory fraction from the frozen non-KVMEM configuration; only
`PREFIX_CACHING=1` changes the launcher. The default remains off.

Run `tools/r0_run_prefix_cache_probe.sh` from a clean, exact-SHA isolated
checkout with an empty artifact directory. It refuses an occupied port or
GPU compute process, starts one server in an owned process group, sends the
same 15533-token prompt twice, and requires both a query-token counter
increase and a cached-token hit on the repeated request. It retains raw
metrics, logs, output hashes, token-ID provenance, source fingerprints, and
Xid/GPU/port cleanup evidence. An engine boot failure is a compatibility
failure for this exact configuration, not a semantic result. A cache hit
demonstrates only that the cache was exercised; it does not establish
correctness under branching, eviction, long sessions, or C2/C4 workloads.

No changes to the installed vLLM source, resident policy, or GPU memory
fraction are authorized by this probe.

The first hardware attempt at the production 246000-token envelope (exact
head `e57cdff437e241d1d592a5da749bc43e45fab2d8`) stopped before `/health`:
the hybrid prefix-cache layout required 7.16 GiB KV but only 7.03 GiB was
available. The engine estimated a 240000-token maximum. There was no request,
so cache-hit and semantic behavior remain untested. Xid was zero, the port
closed, and the GPU returned idle. The earlier `d3b25d9` attempt failed even
earlier due to a runner shell error and is not a model result.

`PREFIX_DIAGNOSTIC_SHORT_CONTEXT=1` selects a separately labeled 32768-token
mechanism-only cell using the same 15533-token prompt. Its result cannot
qualify the 246000-token production envelope; it only separates boot sizing
from hybrid/MTP prefix-cache behavior. The default remains the original
246000-token envelope.

The short-context cell at exact head
`8f4ce1559245ade4f34833a02626acc04a152076` booted and completed two
identical 15533-token requests. Raw Prometheus totals rose by 31066 queried
tokens and 12800 cached-token hits. The probe's final output-hash equality
gate failed, so APC semantics are **not qualified**. The first probe version
retained only the error, not each generated answer; it is not enough to
attribute the mismatch to APC versus MTP/runtime nondeterminism or generation
after EOS. Xid remained zero; port and GPU processes were clean after owned
shutdown. A later controlled comparison must preserve both outputs and run
the same short prompt with APC off before assigning root cause. This is
lower priority than the independently scheduled refusal-edit baseline.
