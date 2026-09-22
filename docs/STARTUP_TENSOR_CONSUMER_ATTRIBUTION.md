# Lazy safetensors tensor-consumer attribution

## Motivation

Two transport-side experiments are now negative:

1. Reusable pinned staging:
   - pinned H2D itself reached ~5.04 GiB/s;
   - pageable->pinned staging remained ~0.68 GiB/s;
   - end-to-end wall/GiB became ~6.9% worse.

2. Same-shard background prefetch:
   - major-fault pressure fell;
   - storage traffic amplified heavily;
   - total wall/GiB became ~28% worse.

The next question is therefore above the transport layer:

**Which tensor classes consume the remaining model-load wall and CPU time once
safetensors has yielded a tensor to the model loader?**

This lane is attribution only.

## Generator boundary

vLLM's default lazy iterator does:

```python
param = f.get_tensor(name)
yield name, param
```

A Python generator resumes only when the downstream consumer asks for the next
item.

Therefore:

- time around `get_tensor()` measures creation of the lazy safetensors
  tensor/view;
- time from `yield` to generator resume covers downstream mapper,
  AutoWeightsLoader recursion/dispatch, EXL3/native weight-loader work, and any
  source-page faults triggered by that work.

The latter is the useful per-tensor consumer denominator.

## Instrumentation

The temporary vLLM patch is enabled only by:

`VLLM_EXL3_TENSOR_ATTR_PATH=/path/to/stats.json`

The patch requires the default lazy safetensors strategy. It does not enable
eager loading, prefetch, pinning, async copy, multithread loading or ordering
changes.

Per tensor it measures:

- tensor bytes;
- get_tensor wall;
- get_tensor process CPU;
- yield->resume consumer wall;
- yield->resume process CPU.

To avoid diagnostic I/O perturbing the loader, records are aggregated in
memory and written once after the iterator is exhausted.

## Aggregation

Each shard records totals plus breakdowns by:

- suffix: trellis, suh, svh, mcg, mul1, weight, bias, other;
- scope: routed expert, shared expert, ngram, layer non-expert, other;
- size bin;
- scope + suffix.

Each shard keeps only its slowest 24 individual consumer tensors. The summary
retains the global top 40.

## Copy reconciliation

The repo EXL3 diagnostic already measures:

- direct routed-trellis copy wall;
- direct-trellis prep wall;
- generic EXL3 blocking copy wall.

These timers are nested inside iterator consumer wall and must be subtracted,
never added.

The closest denominators are:

- `routed_expert|trellis` consumer wall vs direct-trellis copy wall;
- all other EXL3 suffix consumer wall vs generic copy wall.

This shows how much downstream wall remains after the already-measured copy
calls.

## Interpretation

Examples:

### get_tensor dominates

If `get_tensor_wall_s` is a large fraction of main-weight wall, optimize the
safetensors access/materialization path before AutoWeightsLoader.

### routed trellis consumer ~= direct copy

If routed-trellis consumer wall is almost entirely explained by direct-fill
copy wall, Python dispatch around trellis weights is not the next target.

### many-small-tensor consumer dominates

If small size bins or suh/svh/mcg/mul1 categories consume substantial wall/CPU
despite little byte volume, focus on call count, mapper recursion and loader
dispatch/coalescing.

### weight/other consumer dominates

If native/dense `weight` or `other` classes dominate, inspect those concrete
weight_loader paths instead of routed trellis.

## Scope

- one target main-model boundary boot;
- no prompt;
- no health wait;
- no page-cache drop;
- no loader-policy change;
- no tensor reordering;
- no prefetch;
- no pinned staging;
- no async H2D;
- installed vLLM weight_utils restored byte-identically afterward;
- no K1 changes;
- no merge without explicit instruction.
