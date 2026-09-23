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
