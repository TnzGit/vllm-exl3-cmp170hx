# R1 K1-Q2C: scheduler-visible KV shrink preflight

## Purpose

Q2A proved byte-exact logical->physical mapping into independent 16-token QSA
resident pages. Q2B proved the same 64K historical resident set can survive a
real vLLM generic CPU-offload GPU->CPU->GPU round trip while preserving target
semantics. Both still retained the scheduler-owned full QSA GPU KV as a
source/reference.

Q2C is the ownership transition: the scheduler/cache manager must stop treating
full historical QSA KV as permanently GPU-resident while the logical request
history remains complete.

This PR is deliberately a CPU/read-only preflight. It does not boot a model and
does not patch installed vLLM.

## Frozen geometry

- resident page: 16 tokens
- QSA KV: 2 KV heads x 256 dim x K+V x BF16 = 2048 B/token
- page: 32 KiB
- historical resident budget: 65,536 tokens = 4096 pages
- active reserve: 1,024 tokens = 64 pages
- bounded physical cap: 4160 pages = 130 MiB/layer
- 12 QSA layers: 1.5234375 GiB bounded resident storage
- 161K logical table: ceil(161000/16) = 10,063 entries
- 240K logical table: 15,000 entries

The critical contract is therefore:

```text
logical table width grows with complete history
physical GPU pages stay bounded at 4160/layer
```

The prior Q2A/Q2B hardware source page was 1568 tokens, so the physical remap
granularity remains 1568/16 = 98. The resident page must not be changed to 1568.

## vLLM 0.29 route being tested

vLLM 0.29 already exposes an out-of-tree KVCacheSpec registry and custom
SingleTypeKVCacheManager selection. It also has a block-outermost packed-group
path that runs before page-size unification.

The preflight defines a non-production QSAResidentContractSpec:

- AttentionSpec-compatible tensor geometry, BF16, 2x256 K/V;
- block_size = 16;
- page_size_bytes = 32768;
- max_num_blocks_per_req = full logical history width;
- max_memory_usage_bytes = 4160 resident pages only;
- prefix caching disabled for this auxiliary resident ownership group.

A probe manager is registered only to validate the vLLM registry contract. It
is not the eventual residency implementation.

## Hard preflight gates

The installed environment must prove all of the following:

1. vLLM is 0.29.x.
2. Current Qwen4Exp QSA main KV is still returned as FullAttentionSpec.
3. The custom spec maps to the custom manager through KVCacheSpecRegistry.
4. The registry treats the QSA spec as its own uniform type.
5. 161K logical table width exceeds the 4160 physical cap.
6. Q2B geometry is exact: 16-token/32KiB pages, 4096+64 physical pages.
7. The prior 1568-token source page remains exactly 98 resident pages.
8. Packed grouping is attempted before generic page-size unification.
9. At least one block-outermost compact layout exists.
10. The base manager can represent null holes in request block tables.
11. Under BLHNC packed grouping, the custom QSA spec remains 16-token/32KiB
    rather than being inflated to the stock hybrid page size.

GO classification:

`Q2C_SCHEDULER_SHRINK_PREFLIGHT_GO`

## What GO does not prove

Even a full GO does not prove:

- a real request holds <=4160 physical QSA pages;
- full QSA GPU source can be deleted;
- Q2B CPU history is authoritative during a live engine;
- arbitrary sticky READ residency and active WRITE slot mapping coexist;
- hardware memory/concurrency improvement;
- MTP follower correctness.

Those belong to the next Q2C runtime branch.

## Planned runtime ownership after GO

```text
QSA get_kv_cache_spec
  -> custom 16-token resident spec
  -> scheduler manager keeps full logical row
     [resident block IDs | null host-backed holes]
  -> coordinator owns sticky residency transitions
  -> Q2B generic CPU backing owns historical bytes
  -> active WRITE slots remain resident and separate from READ remap
```

The current upstream HiSparse design uses the same high-level ownership split
(normal KV manager for resident GPU leases/tables, separate coordinator for host
identity/residency, worker for host bytes/transfers). It is architecture
reference only; this project remains pinned to the installed vLLM 0.29 API.

## Stop rule

The local executor runs only the CPU/read-only preflight and returns the JSON.
No engine boot, no installed patch, no merge, no automatic runtime experiment.
