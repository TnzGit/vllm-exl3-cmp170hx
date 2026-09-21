# R1 K0: QSA-native shadow retrieval

## Status

Research-only diagnostic lane.

This experiment does **not** change:
- attention indices;
- KV residency;
- KV eviction;
- MTP;
- production defaults.

It only exports Qwen4Exp QSA's already-computed logical token selections for
the final query and asks whether those selections can support a KVMem-style
coarse historical working set.

Base production remains the frozen MTP-k3/COOP baseline at:

`f2c6a719a1709ba40d08b97a4cc8d64b3c2256d8`

PR #13 EARLY prelude qualification failed and is closed. K0 begins only after
that exact-context optimization lane is frozen.

## Why start from QSA instead of porting KVMem mean-K

The audited `kvmem/kvmem-llama.cpp` design uses a compact mean-K index and
query-conditioned historical-block scoring.

Qwen4Exp on vLLM 0.29 already has a model-native retrieval pipeline:

1. `QSAIndexer` projects normalized Q/K index features.
2. complete groups are compressed into `QSACompressedKeyCache`;
3. `qsa_select_paged_tokens()` scores all visible compressed groups;
4. top groups are expanded back to logical token indices;
5. those exact indices are consumed by `qsa_sparse_paged_attention()`.

Therefore the first question should be:

> Are the token indices QSA already selects good enough to rank coarse
> historical blocks for a bounded KV working set?

If yes, a KVMem-style vLLM implementation can reuse a signal the model already
computes instead of adding another mean-K projection/index in the hot path.

If no, KVMem mean-K remains the next reference selector.

## Important QSA cache distinction

Exact vLLM 0.29 source:

- `QSAKeyStateCache` is only a small circular raw-key ring for the open
  compression group plus speculative rows.
- `QSACompressedKeyCache` stores one normalized BF16 key per complete
  compression group using an MLA-like paged cache.

The raw-key ring is not a persistent historical index.

The compressed-key cache and the resulting `selected` logical indices are
the relevant K0 substrate.

## Shadow patch

`tools/patch_vllm_qwen4_exp/patch_vllm_qsa_shadow.py`

patches only:

`vllm/models/qwen4_exp/nvidia/qsa.py`

Immediately after:

```python
selected = self.indexer(...)
```

and its existing shape check, it calls a diagnostic helper that appends the
already-produced logical indices to JSONL.

The helper never mutates `selected`, `topk_indices_buffer`, attention
metadata, KV cache, or output.

When `VLLM_QWEN_KVMEM_SHADOW_PATH` is unset it returns immediately.

## Eager-only requirement

The export performs small GPU-to-CPU diagnostic copies.

It refuses CUDA graph capture and must only run with:

`--enforce-eager`

No wall time from K0 is a production performance number.

This run is about retrieval quality / index suitability only.

## Bounded diagnostic export

Long prefills are chunked. Exporting tail rows from every chunk would create a
large diagnostic file.

The runner starts one eager engine for each context and sets:

- 160K run: `VLLM_QWEN_KVMEM_SHADOW_MIN_POS=158976`
- 240K run: `VLLM_QWEN_KVMEM_SHADOW_MIN_POS=238976`
- `VLLM_QWEN_KVMEM_SHADOW_ROWS=128`

The helper first checks the final logical position of the current chunk. Chunks
that have not reached the final 1024 logical positions are ignored.

This keeps only the region containing the final user query while leaving QSA
model behavior unchanged.

## Exact-token needle suite

`tools/kvmem_qsa_make_needles.py` uses the real local tokenizer and creates
prompt **token IDs**, not approximate text-length prompts.

Contexts:
- 160000 tokens
- 240000 tokens

Per context:

1. single early needle (~10%)
2. single middle needle (~50%)
3. single late needle (~84%)
4. multi-needle case (~12%, ~52%, ~82%)

Each needle contains:
- a unique marker;
- a unique recovery code;
- an exact token span.

The final query explicitly names the relevant marker(s).

The prompt is sent as token IDs to avoid decode/re-encode span drift.

K0 does not grade the generated answer. It requests only one output token to
complete the transaction; retrieval is scored from QSA shadow records.

## Query-row filtering

The shadow analyzer reads only rows whose logical position falls inside the
recorded final-query span.

It also:
- ignores any accidental `skip_topk=True` follower records;
- discards selected indices at or after the query start;
- therefore votes only on historical tokens.

This ensures prefill filler tokens and any later decode row do not become part
of the retrieval score.

## Coarse block simulation

QSA selects logical tokens/groups, while KVMem-style residency would operate at
coarser blocks.

The analyzer aggregates votes into:

- 128-token blocks
- 256-token blocks
- 512-token blocks

It simulates finite active historical budgets:

- 32K tokens
- 64K tokens

Within the same finite budget it reserves:
- first 512 tokens as sink blocks;
- latest 4096 historical tokens as mandatory recent blocks.

Remaining capacity is filled by QSA block vote count, descending.

This is not yet an eviction implementation. It is an offline simulation over
the shadow selections.

## Metrics

For every needle:

### Direct QSA token hit

Did any final-query QSA selection directly include a token inside the exact
needle span?

This is the strongest evidence that the model-native selector recognizes the
fact itself.

### Coarse block recall

For each block size / budget:

- which blocks overlap the needle span;
- which of those blocks enter the simulated working set;
- fraction recalled;
- whether all required blocks are present.

Suite summary also reports:
- direct hit rate;
- 128/256/512 sensitivity;
- 32K vs 64K budget recall;
- number of QSA layers/query rows observed.

## Initial GO signal

Primary policy for K0:

- 256-token coarse blocks
- 64K active historical budget

GO signal requires:

- >=99% fully recalled needle rate at 64K;
- >=95% fully recalled needle rate at 32K.

The 32K threshold is intentionally labelled desirable rather than a production
quality guarantee. With this small deterministic suite, report raw counts as
well as percentages.

A GO does **not** mean sparse KV is production-ready. It means QSA-native
selection is strong enough to justify K1 engineering.

## Interpretation

### Strong GO

If:
- all or almost all needles are directly hit;
- 256/64K fully recalls every needle;
- 32K remains high;
- results are consistent at 160K and 240K;

then K1 should reuse QSA-derived relevance first.

Next step:
- design CPU-tier historical KV residency;
- selection-diff planner;
- retained-resident block reuse;
- no NVMe yet;
- no MTP follower until target path works.

### Block GO but weak direct hits

If coarse blocks recall needles mostly because neighboring selected tokens land
in the same block, QSA may still be useful for coarse residency, but block size
sensitivity becomes important.

Prefer 128/256-token K1 experiments before 512.

### NO-GO

If 64K/256 recall is unstable or misses clear marker-addressed needles, do not
build a KV eviction layer around QSA selection.

Next research step is an independent KVMem-style mean-K reference selector,
still in shadow mode.

## Why no MTP in K0

K0 asks whether the **target model's retrieval signal** is suitable.

MTP would add:
- step-0/follower reuse semantics;
- duplicate shadow records;
- acceptance as a confound.

The K0 server therefore uses:

`NUM_SPEC_TOKENS=0`

If K1 target sparse residency later succeeds, MTP must follow the target
selection in lockstep, matching the KVMem design constraint.

## Runner

```bash
R0_REPO="$PWD" bash tools/r0_run_kvmem_qsa_shadow.sh
```

The runner:

1. resolves installed vLLM;
2. refuses an already shadow-patched or stale-backup QSA source;
3. copies exact installed `qsa.py`;
4. runs patcher `--check-only` and proves it made no write;
5. runs CPU tests;
6. generates exact 160K/240K token-ID cases;
7. applies only the research QSA logging patch;
8. starts eager no-draft engines, one per context;
9. executes all eight cases;
10. stores one JSONL shadow file per case;
11. produces per-case and suite retrieval summaries;
12. records Xid/log health;
13. restores installed `qsa.py` byte-identically on normal exit or failure.

No production plugin, EXL3 CUDA extension, cache policy, or launcher default is
modified.

## Artifacts

Primary:

- `cases/manifest.json`
- eight exact-token case JSONs
- `shadows/<case>.jsonl`
- `summaries/<case>_response.json`
- `summaries/<case>_retrieval.json`
- `k0_suite_summary.json`
- `logs/serve_ctx160000.log`
- `logs/serve_ctx240000.log`
- `qsa.base.py`

## Explicit non-goals

K0 must not:
- evict KV;
- offload KV;
- modify block tables;
- alter attention top-k;
- change QSA indexer math;
- enable MTP;
- benchmark eager wall time as production performance;
- implement NVMe paging;
- copy llama.cpp adapter code;
- merge into production.

## After K0

If QSA-native retrieval passes, the next code design is K1:
vLLM-native bounded historical KV residency with CPU as the first lower tier.

If it fails, remain in shadow mode and implement a small independently-written
mean-K selector for comparison before touching KV residency.
