# CMP170HX non-KVMEM prefill configuration scan

## Purpose

Measure whether vLLM's prefill token budget improves single-request TTFT and
prefill throughput on the already qualified MTP k=3 / cooperative MoE service.
Then test a chosen setting with an overlapping decode request before considering
it a service default. This experiment does not change model weights, sparse
attention policy, quantization, or the production installation.

## Frozen baseline

- Source parent: `f2c6a719a1709ba40d08b97a4cc8d64b3c2256d8`, the post-MTP
  k=3 production Amdahl head.
- Installed EXL3 plugin SHA-256 on base-node before this scan:
  `dc7d3dd16a39aed839f04f68b1ee30ce2cf186b5e206c847273964f809347aba`.
  This matched the parent source on inspection. Recheck before every live run.
- Installed vLLM: `0.29.1rc1.dev5+ge52be1a62`.
- The installed `SchedulerConfig.DEFAULT_MAX_NUM_BATCHED_TOKENS` is 2048.
  The launcher omits `--max-num-batched-tokens` when unset; verify the effective
  value from the new engine before treating `auto` as an exact 2048 control.
- CMP170HX, text-only, MTP k=3, `VLLM_EXL3_COOP=1`, PIECEWISE graph,
  prefix caching off, `max_model_len=246000`, `max_num_seqs=1`,
  `gpu_memory_utilization=0.92`, same model pack and vLLM patches.

The previous MTP qualification measured 10.28–11.37 ms/output token from
4K to 240K with greedy parity at 4K/160K/240K. Those figures establish a
decode reference, not a prefill result. See `R0_MTP_K3_PRODUCTION.md` and
`R0_MTP_K3_AMDAHL.md`.

## A/B contract

Start with `auto` (verify effective value), 1024, and 4096 tokens per scheduler
step. Do not reload the model between contexts within the same configuration;
do restart for a different token budget. The 8192 candidate is conditional on
the first scan showing useful headroom and a healthy boot. A failed 4096/8192
boot is a capacity result; do not lower `max_model_len` or change GPU memory
utilization to make it pass.

For each configuration, issue the same deterministic input at 16K, 80K and
160K targets. Record the API's actual prompt token count, require equality
across configurations, and use 256 requested output tokens. Run a short
unmeasured warmup first, then multiple measured repetitions and a short
sentinel at the end. The results must distinguish cold engine load, request
TTFT, prefill tokens/second, decode TPOT and whole-request wall. API
`usage.completion_tokens` is authoritative for output count; streamed chunks
are not tokens. Decode TPOT uses `completion_tokens - 1` because the first
token is included in TTFT.

Save the exact source SHA, startup command/config, plugin/vLLM identities,
raw per-request JSON and engine logs. Require no preemption, no Xid increase,
correct output count, stable process/port cleanup, and unchanged installed
source hashes. A single fast cell is a lead, not a qualification. A candidate
worth retaining should improve repeated 80K/160K TTFT by at least roughly 5%
without a reproducible decode or short-context regression above 3%. These
thresholds are investment gates, not predicted speedups.

After C1, a separate C2 overlap probe should submit a new prefill while an
existing request is decoding, and report both the new request's TTFT and the
existing request's p95 token interval. Do not infer a concurrency win from C1.

## Stop conditions

Stop on boot failure, source mismatch, invalid denominator, unexpected
preemption, semantic output anomaly, Xid, or contaminated process state.
Preserve raw artifacts and inspect machine health before changing code or
rerunning. Prefix caching and MTP k=4 are separate experiments.
