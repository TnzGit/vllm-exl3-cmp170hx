# Bounded safetensors same-shard prefetch A/B

## Motivation

Pinned CPU staging was negative:

- control direct-trellis path: 0.638 GiB/s
- pageable->pinned stage: 0.676 GiB/s
- pinned->GPU H2D: 5.041 GiB/s
- pinned end-to-end: 0.596 GiB/s

The H2D leg is healthy once data is pinned. The unresolved denominator is
source-page availability / storage / CPU-side materialization.

The same run showed about 40.9 GiB of kernel-accounted reads and 109k major
faults.

## Why not force vLLM full prefetch

vLLM 0.29 already has a safetensors prefetch mode. It reads checkpoint files
into the OS page cache in background.

For this model the checkpoint is about 67.26 GiB while available RAM is only
about 41 GiB. Full-checkpoint prefetch therefore exceeds the normal vLLM
fits-in-RAM threshold and risks page-cache churn.

This lane does not enable global `--safetensors-load-strategy=prefetch`.

## Single-boot A/B

The temporary diagnostic patch modifies only the default lazy safetensors
iterator when both opt-in variables are set.

Natural-order checkpoint shards are split by index:

- even index: control, unchanged lazy mmap path;
- odd index: prefetch arm.

For a prefetch-arm shard, immediately before `safe_open`/consumption, one
background thread calls vLLM's existing `_prefetch_checkpoint()` on that same
file.

The thread and mmap/tensor consumer run concurrently.

The thread is joined before leaving that shard. No prefetch I/O is allowed to
leak into the next shard's measurement.

This is intentionally a **same-shard background reader A/B**, not yet a
one-shard-ahead production design.

## Per-shard evidence

Every shard records:

- natural index / filename / file bytes;
- arm;
- tensor-consumption wall;
- prefetch-thread wall;
- join wait at shard end;
- total shard wall;
- process read_bytes;
- major/minor faults;
- user/system CPU.

Summary comparison is normalized by total checkpoint-file GiB in each arm:

- total seconds/GiB;
- consume seconds/GiB;
- major faults/GiB;
- storage read GiB / file GiB.

The A/B is valid only if both arms have at least two shards and the prefetch arm
contains 40-60% of total shard bytes.

## Interpretation

Positive evidence requires lower **total shard wall/GiB**, not merely lower
mmap-consumption wall.

Useful positive pattern:

- total wall/GiB decreases;
- major faults/GiB decreases;
- join wait remains bounded.

That would justify a follow-up true one-shard-ahead design that overlaps next
shard prefetch with useful work on the current shard.

Negative pattern:

- total wall/GiB unchanged or worse;
- read amplification increases;
- fault rate does not improve materially.

Then extra page-cache reader threads are not the right direction; investigate
tensor access order, safetensors materialization and CPU dispatch instead.

## Safety

The installed vLLM `weight_utils.py` is temporarily patched only for the
diagnostic boot and restored byte-identically afterward. Marker and backup must
both be absent at completion.

No prompt, no full server health wait, no page-cache drop, no pinned staging,
no async H2D, and no K1 changes.
