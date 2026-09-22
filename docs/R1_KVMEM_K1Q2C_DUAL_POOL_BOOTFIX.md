# K1-Q2C follow-up: private QSA pool / dual backing arena

## Why PR #32 could not boot

The first live Q2C attempt at
`9fc014ab0da18a8c36aeb30cb8c2e517fecb64c0` passed every CPU gate but
failed before EngineCore became healthy:

```text
Available KV cache memory: 10.07 GiB
161000-token request requires: 12.86 GiB
```

No request was issued, so that run is **BOOT_INVALID / runtime unproven**, not
a Q2C ownership or semantic NO_GO.

The failure exposed two vLLM 0.29 assumptions that the CPU preflight did not
exercise.

### 1. Hybrid page-size alignment runs before KV grouping

`Platform._align_hybrid_block_size()` sees Qwen4Exp as QSA + Mamba and, under
the stock single-pool assumption, expands the attention manager block to 1568
tokens so one attention page is at least as large as a Mamba state page. Mamba
is then padded to the same page.

That happens before the later mixed-page BLHNC grouping path.

### 2. One global BlockPool means mixed pages still share the widest stride

Even if the 1568-token alignment is skipped, stock vLLM 0.29 creates one
scheduler `BlockPool` and one worker backing allocation. Every group therefore
shares one physical block-ID domain, and `num_blocks` is derived from the
widest group stride.

Q2C needs 4,160 independently addressable 16-token QSA pages. Charging those
IDs at a Mamba-sized stride defeats the memory reduction.

Therefore "mixed-page grouping works" is necessary but not sufficient. QSA
must have its own physical block-ID domain.

## Follow-up architecture

This branch keeps regular/Mamba state on the stock vLLM pool and gives QSA a
private pool plus a dedicated worker backing allocation.

### Scheduler QSA pool

```text
private QSA BlockPool total blocks = 4161
  block 0 = permanent shared null block
  blocks 1..4160 = usable QSA pages

QSA page = 16 tokens
per-layer page = 32 KiB
usable logical/physical cap = 4160
```

The QSA manager checks this private pool itself. Once private capacity is
confirmed, its public stock-coordinator allocation count is zero, so QSA IDs
do not consume regular/Mamba BlockPool capacity.

Private IDs are never returned to the stock pool and never enter the stock
worker zeroing/copy ID lists.

### Worker QSA arena

There are 12 QSA layers:

```text
one private scheduler block:
12 * 32 KiB = 393,216 bytes

private backing:
4161 * 393,216
= 1,636,171,776 bytes
~= 1.524 GiB
```

All 12 QSA layer views bind to this private backing. Regular/Mamba views bind
to a separate stock backing allocation.

Block IDs may numerically overlap between the two domains because each KV
cache group carries its own block table and indexes its own tensor.

### Boot memory accounting

Boot-time memory checks are also split:

```text
QSA request memory
  = 4160 * QSA-group block bytes

regular request memory
  = regular blocks/request * regular widest group stride

null overhead
  = one QSA private block + one regular stock block
```

The QSA private allocation is reserved first from the profiled available KV
memory. The remainder sizes the stock regular/Mamba pool.

This is not a higher `gpu_memory_utilization` workaround and does not shorten
`max_model_len`.

## Narrow platform exception

Only when all of the following are true:

- `VLLM_QWEN_KVMEM_Q2C_PLAN` is active;
- model type is `qwen4_exp`;
- backend is `QWEN4_EXP_QSA_TRITON`;
- prefix caching is off;
- Mamba cache mode is `none`;
- attention block size before hybrid alignment is 16;
- layout is explicitly `BLHNC`;

the old QSA/Mamba shared-page alignment is skipped.

Every other runtime follows stock vLLM behavior.

## Scheduler granularity

Without the old 1568 alignment, the scheduler-level LCM can be much larger
than 16 because Mamba none-mode carries its own state block size.

For this proof that value is not used as the QSA physical allocation
granularity:

- prefix caching is off;
- KV connectors are off;
- Mamba align mode is off;
- request chunking is explicitly limited to 1024 tokens;
- QSA allocates from its private 16-token pool.

The runtime evidence still hard-gates the actual private QSA peak at 4160.

## New boot hard gate

Before sending the 160K request the runner requires
`Q2C_DUAL_POOL_BOOT_GO`.

Evidence must show:

- the platform skipped legacy shared-page alignment;
- the old 1568 alignment log is absent;
- core config reports exactly 4161 private QSA blocks;
- private QSA bytes are exactly 1,636,171,776;
- worker reports a distinct private allocation with the same byte count;
- stock regular pool remains nonempty;
- reported capacity is at least one 161K request.

A boot failure still means runtime ownership/semantic gates remain unproven.

## Request-time conservation proof

Scheduler events report both real QSA pages and private pool free blocks.

At the frozen boundary:

```text
real QSA pages + private free pages = 4160 usable pages
private total pool blocks = 4161
```

Worker evidence additionally requires every QSA layer view to report:

```text
shape[0] = 4161
underlying storage bytes = 1,636,171,776
scheduler block stride = 393,216 bytes
```

Only after these dual-pool gates pass do the existing frozen-plan CPU-authority,
visibility and semantic gates decide the final Q2C result.

## Scope remains frozen-plan only

A successful run still does not prove:

- causal resident selection without future-query knowledge;
- exact full-history token/hidden-state parity;
- 240K;
- MTP;
- multi-request concurrency scaling;
- dynamic multi-turn replacement;
- production latency.

The purpose is to prove the physical allocator substrate that PR #32 could not
reach.
