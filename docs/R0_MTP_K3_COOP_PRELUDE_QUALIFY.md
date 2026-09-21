# MTP k=3 cooperative-MoE EARLY prelude qualification

## Final result — NOT QUALIFIED

Hardware qualification completed on exact branch head
`b162c36639ccf69e7445f99f304bae68eabe54bd`.

The mechanism remains real, but it did **not** survive the production-context
qualification gate.

| context | BASE ms/output-token | EARLY ms/output-token | latency gain |
|---|---:|---:|---:|
| 4K | 10.085 | 9.981 | +1.031% |
| 160K | 11.524 | 11.545 | -0.182% |
| 240K | 11.031 | 11.044 | -0.118% |

4K sentinels:

| sentinel | BASE | EARLY | gain |
|---|---:|---:|---:|
| in | 10.256 | 10.291 | -0.341% |
| out | 10.142 | 9.917 | +2.219% |

Gate summary:

- mean 4K sentinel gain: **+0.939%** — PASS
- median context gain: **-0.118%** — FAIL (required >= +0.5%)
- worst context gain: **-0.182%** — PASS (required >= -0.5%)
- greedy token parity: PASS at 4K / 160K / 240K
- denominator: VALID for every checked cell
- preemption / fallback / Xid: 0
- restore: frozen production source and compiled extension byte-identical after run
- final decision: **`qualified=false`**

The 4K signal is not stable enough to justify a separate production mode:
sentinel-in was negative while sentinel-out was +2.219%, and both long-context
cells were slightly negative. Do not create a 4K-only EARLY production profile.

This line is closed as a preserved negative result. Do not revisit OUT_EMPTY or
lower the qualification threshold.

## Candidate

Qualify only:

`VLLM_EXL3_COOP_EARLY_PRELUDE=1`

The rejected output-empty probe is deliberately absent from this branch.

Base production source:

`f2c6a719a1709ba40d08b97a4cc8d64b3c2256d8`

Research evidence from PR #12:

- BASE 4K: 10.149 ms/output-token
- EARLY: 10.027 ms/output-token
- latency gain: +1.202%
- throughput gain: +1.210%
- full greedy token parity: PASS
- denominator/preemption/fallback/Xid: clean
- mechanism:
  - fill_zero: 339 -> 186 calls/pass (-153)
  - index_scatter: 86 -> 35 calls/pass (-51)
  - elementwise: 500 -> 398 calls/pass
  - framework prelude: -0.670 ms/pass

EARLY+EMPTY was hardware-negative (-0.552% e2e) and is not part of
qualification.

## Production-vs-candidate comparison

PR #12 used one candidate source with the flag OFF/ON, which was correct for
mechanism attribution. This qualification is stricter.

BASE runs the exact installed `f2c6a71` production plugin.

Only after all BASE cells complete does the runner install the EARLY-only
candidate `exl3.py` and start a fresh candidate engine.

This avoids counting the qualification helper's Python call overhead in the
production baseline.

The branch is also required to satisfy:

- no `csrc` delta vs `f2c6a71`;
- the only runtime-package source delta under `src/vllm_exl3` is
  `src/vllm_exl3/exl3.py`;
- no `VLLM_EXL3_COOP_OUT_EMPTY` code remains.

## Runtime contract

Both configurations use:

- one CMP170HX / SM80
- Qwen3.8-Flash-Next Uncensored EXL3 3bpw
- MTP k=3
- `VLLM_EXL3_COOP=1`
- PIECEWISE CUDA graphs
- prefix cache OFF
- disk n-gram / PLE
- text-only
- C1
- profiler OFF for all formal performance cells

Only the candidate sets:

`VLLM_EXL3_COOP_EARLY_PRELUDE=1`

## Contexts / repeats

Per fresh configuration engine:

1. 4K sentinel-in: 5 repeats
2. 4K main cell: 7 repeats
3. 160K: 3 repeats
4. 240K: 3 repeats
5. 4K sentinel-out: 5 repeats

BASE runs first on exact frozen production code.

Candidate runs second on a fresh engine after the pure-Python source swap.

## Correctness

Candidate cells at:

- 4K
- 160K
- 240K

compare against same-run BASE token sequences.

Inline parity now flattens speculative multi-token chunks before comparison,
matching `r0_k3_parity_recheck.py`.

Required:

- full greedy token-ID parity PASS at all three contexts;
- authoritative denominator is API `usage.completion_tokens`;
- speculative counter cross-check uses:
  `-1 <= derived - usage <= k`.

## Qualification threshold

This is a low-risk framework optimization with already-proven mechanism, so it
is not required to hit the project's 5% single-target discovery gate.

It qualifies if all of the following hold:

1. parity PASS at 4K / 160K / 240K;
2. every formal cell denominator VALID;
3. preemption/fallback/Xid clean;
4. mean of 4K sentinel-in/out latency gain >= **0.5%**;
5. median latency gain across 4K / 160K / 240K >= **0.5%**;
6. worst single context latency gain >= **-0.5%**.

Rationale:

The mechanism was already directly measured (-153 fill and -51 scatter
calls/pass, -0.670 ms/pass framework prelude). Qualification now asks whether
that real mechanism translates into a stable production improvement without a
context-specific regression.

## Installation / restore guard

The runner:

1. derives torch/NVIDIA library paths into `LD_LIBRARY_PATH`;
2. requires installed `vllm_exl3/exl3.py` to equal frozen `f2c6a71`;
3. stores an exact backup;
4. runs all BASE cells before any candidate source install;
5. installs only candidate `exl3.py`;
6. records compiled `vllm_exl3_c.so` hash;
7. runs EARLY cells;
8. restores the original plugin in EXIT trap;
9. verifies the compiled extension hash never changed.

## Executor

```bash
R0_REPO="$PWD" bash tools/r0_run_coop_prelude_qualify.sh
```

Primary artifacts:

- `ref_4096.json`
- `ref_160000.json`
- `ref_240000.json`
- `parity_k3_4096.json`
- `parity_k3_160000.json`
- `parity_k3_240000.json`
- BASE/EARLY 4K sentinel-in/out cells
- `parity.txt`
- `qualification_summary.json`
- `health_summary.json`
- BASE/EARLY serve logs

## Post-qualification

If qualified:

- do not rerun discovery/profiler work;
- normalize the production launcher/default to EARLY;
- update the canonical k=3 production baseline;
- retain the old path as automatic fallback whenever EARLY eligibility is not
  provable.

If not qualified:

- close EARLY as a measured framework optimization that did not survive
  production qualification;
- do not tune OUT_EMPTY again.
