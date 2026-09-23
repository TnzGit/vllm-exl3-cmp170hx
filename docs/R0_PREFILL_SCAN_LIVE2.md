# Non-KVMEM prefill scheduling scan: live2 partial qualification

Date (UTC): 2026-09-23. Host: single CMP 170HX on `base-node`.
Exact experiment source: `9da3ace393e56f1f0375e13e2e3f4cd52e4ed5b9`.
Remote raw artifacts: `/home/base-node/.codex_tasks/qwen38-flashnext-r0/results/r0-prefill-scan-9da3ace-live2`.
The earlier `11c4dd1` live1 artifacts remain separately preserved; that run
stopped after a strict output-text-hash gate found one differing 16K repeat.

## Frozen workload and limits

- Installed EXL3 plugin and experiment source both SHA-256
  `dc7d3dd16a39aed839f04f68b1ee30ce2cf186b5e206c847273964f809347aba`.
- Existing non-KVMEM EXL3 model, MTP k=3, cooperative path enabled, piecewise
  graphs, prefix caching off, `max_model_len=246000`, `max_num_seqs=1`,
  `gpu_memory_utilization=0.92`.
- Same exact token-ID prompts across candidates: 15,533 / 79,533 / 159,533
  input tokens; one warmup and three measured requests per context, then a 16K
  sentinel. API `usage` confirmed 256 generated tokens per measured request,
  with `temperature=0`, seed 0 and `ignore_eos=true`.
- `auto` resolved to a 2,048 scheduled-token budget in the installed vLLM log.
  The explicit 1,024 and 4,096 values were verified in launcher arguments.

## Measured C1 prefill results

Numbers are medians of the three measured requests; TTFT includes the first
output token and is not a kernel-only prefill measurement.

| Input tokens | Auto/2048 TTFT | 1024 TTFT | 1024 change | Auto effective prompt tok/s | 1024 effective prompt tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 15,533 | 9.198 s | 11.298 s | +22.8% | 1,688.8 | 1,374.8 |
| 79,533 | 46.764 s | 57.463 s | +22.9% | 1,700.7 | 1,384.1 |
| 159,533 | 95.952 s | 117.029 s | +22.0% | 1,662.6 | 1,363.2 |

The three TTFT samples within each completed cell were tightly clustered:
auto 160K 95.908–95.976 s; 1024 160K 117.010–117.067 s. The 16K sentinels
were 9.233 s and 11.288 s, respectively. These observations reject 1024 as
a candidate for this frozen C1 workload; they do not establish a universal
best budget or C2/C4 behavior.

## 4096 capacity boundary

The 4096 engine died during KV-cache sizing, before `/health` and before any
request. vLLM reported that serving max length 246000 requires 7.0 GiB KV
cache but only 6.63 GiB was available, estimating a 232000-token maximum.
Classification: **4096 boot capacity NO_GO under the frozen envelope; runtime
performance unmeasured**. We did not raise GPU utilization, lower model length,
or otherwise alter the comparison envelope to make it boot.

The runner retained the raw startup log and exited nonzero. Post-failure
check: Xid count 0 before/after, GPU 14 MiB with no compute process, port
8002 closed, isolated worktree clean, installed/source plugin hashes equal.

## Output caveat and decision

All 18 measured auto/1024 responses began their first visible final answer
with the correct two recovery codes in order. The two sentinels also did.
However, 80K/160K output-text hashes varied because `ignore_eos=true` forced
generation after the answer, sometimes into synthetic later turns. These
256-token continuations are **prefill-only evidence**. Their decode TPOT and
whole-request wall are recorded in raw JSON but are not qualified as normal
answer-length decode performance or semantic parity.

Decision for this narrow config scan: retain auto/2048; reject 1024 on measured
TTFT and 4096 on fixed-envelope boot capacity. Do not continue this particular
batched-token sweep to 8192. The next high-value non-KVMEM probes remain native
prefix-cache compatibility and C2/C4 service behavior, each as a separate
controlled experiment. No installed source or model files were changed.
