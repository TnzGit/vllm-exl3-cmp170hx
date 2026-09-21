# R1 K1-Q2B: real-model generic CPU backing

## Goal

K1-Q2A proved that the frozen 64K / 5% sticky resident set can be remapped
from the hybrid scheduler's 1568-token full-cache blocks into independent
16-token resident pages with byte-exact selected K/V mapping and preserved
target semantics.

K1-Q2B replaces Q2A's direct full-GPU D2D historical bootstrap with the real
vLLM 0.29 generic CPU offload manager/worker path already qualified by K1-T.

The scheduler-owned full QSA KV cache remains allocated only as a same-forward
reference/source for this diagnostic. Q2B does not yet claim GPU-memory
reduction or long-session concurrency improvement.

## Frozen policy

- context: 160K
- turn: ask_d_e
- historical resident budget: 65,536 tokens
- planner region: 256 tokens
- sticky replacement fraction: 5%
- resident page size: 16 tokens
- historical resident pages: 4,096
- active reserve: 1,024 tokens = 64 resident pages
- resident physical pages: 4,160

## GPU staging geometry

Publishing all 4,096 historical pages does not allocate another 4,096-page
GPU staging cache.

Q2B uses a 128-page reusable publication window:

- resident+active physical pages: 4,160
- publication staging pages: 128
- transfer tensor pages: 4,288
- QSA page size: 32 KiB at 2 KV heads × 16 tokens × 512 BF16 values
- resident view: 130 MiB/layer
- staging overhead: 4 MiB/layer
- total transfer tensor: 134 MiB/layer

Across 12 QSA layers the extra staging overhead relative to Q2A is only
48 MiB of GPU memory.

## Real generic offload path

Each QSA layer owns one `VllmCPUPageBacking` built through
`single_tensor_cpu_backing()`.

The adapter still uses the same installed vLLM primitives qualified by K1-T:

- CPUOffloadingManager
- CPUOffloadingWorker
- prepare_store / submit_store / wait / get_finished / complete_store
- lookup
- prepare_load / submit_load / wait / get_finished / complete_load

The Qwen patch itself does not directly construct either generic manager or
worker.

## Publication

At the first affected query row for each layer:

1. the normal scheduler/full KV cache is updated;
2. 128 frozen resident logical pages at a time are gathered from 1568-token
   full blocks into the reusable 16-token GPU staging window;
3. that chunk is published D2H through `VllmCPUPageBacking.publish()`;
4. the same 128 staging physical page IDs are reused for the next chunk;
5. after 32 jobs all 4,096 historical logical pages must be CPU-backing HITs.

Expected D2H bytes per layer:

`4096 × 32768 = 134,217,728 bytes = 128 MiB`

Expected publication jobs per layer:

`ceil(4096 / 128) = 32`

## H2D stage-in hard proof

Before H2D, Q2B zeroes resident history pages 0..4095 and synchronizes.

It then stages all 4,096 CPU-backed logical pages into resident physical pages
0..4095 through `VllmCPUPageBacking.stage_in()`.

Expected H2D bytes per layer are also 128 MiB.

After stage-in, every one of the 4,096 resident historical pages is compared
byte-for-byte against the corresponding full-cache source page. Any mismatch
is a hard `Q2B_CPU_TRANSFER_NO_GO`.

This full-page bootstrap verification is stronger than the later selected-token
check: it proves the complete 64K historical resident payload survived the
GPU -> CPU -> GPU round trip.

## Same-forward mapping and semantic gates

After CPU stage-in, Q2B reuses the Q2A gates:

- active suffix K/V is written into resident physical pages 4096..4159;
- historical visibility is restricted by the frozen sticky plan;
- selected K/V payloads from full and resident caches must remain byte-exact;
- full-cache and resident attention are evaluated on the same selected indices;
- attention numerical drift across PAGE_SIZE=1568 vs PAGE_SIZE=16 is measured
  but no atol/rtol acceptance is introduced;
- the final answer must preserve both expected recovery codes.

## Classifications

`Q2B_CPU_TRANSFER_NO_GO`

- CPU publication/stage-in byte counts are wrong, backing keys are missing,
  full-page H2D verification fails, publication job count is wrong, or staging
  tensor geometry does not match the plan.

`Q2B_INPUT_MAPPING_NO_GO`

- CPU round-trip passes but selected K/V logical-address mapping is not byte-exact.

`Q2B_SEMANTIC_NO_GO`

- CPU transfer and selected K/V mapping pass but the target answer is lost.

`Q2B_CPU_BACKED_EXACT_GO`

- all hard gates pass and full-vs-resident attention is bit-exact.

`Q2B_CPU_BACKED_MAPPING_EXACT_SEMANTIC_GO_NONEXACT`

- CPU round-trip and selected K/V mapping are byte-exact;
- target semantics are correct;
- PAGE_SIZE-specialized attention is numerically non-exact.

The last classification is a GO, but is intentionally not called exact GO.

## Timing interpretation

Q2B records D2H and H2D worker event/wall time for economics only.

This diagnostic publishes and stages the complete 64K resident history on the
first affected call. That is not the production turn-transition design.

The product target remains the K1B sticky transition economics:

- 5% replacement
- 12 regions
- 192 × 16-token pages across the all-layer logical transfer
- 72 MiB stage-in per semantic transition

Q2B must therefore not use its full 1.5 GiB all-layer bootstrap transfer time
as a production TTFT estimate.

## What GO proves

If Q2B passes, it proves that the real Qwen target path can:

- publish frozen QSA history into vLLM generic CPU backing;
- retrieve the same history through real H2D offload primitives;
- retain byte-exact physical resident K/V;
- preserve selected-token logical mapping;
- preserve target semantics.

## What remains

Q2B still retains the scheduler-owned full QSA GPU cache.

After Q2B GO, the next engineering phase should stop treating full QSA KV as
the authoritative resident store and introduce scheduler/cache-manager
ownership for bounded GPU QSA pages plus CPU logical history.

That later phase must also explicitly handle active suffix lifecycle, resident
transitions, host misses and reclaim. GDN/recurrent replay and MTP follower
state remain separate dependencies before claiming cheap general multi-turn
long-context service.
