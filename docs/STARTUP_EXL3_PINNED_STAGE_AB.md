# EXL3 direct-trellis pinned staging A/B

## Why this experiment

The completed loader attribution measured:

- main model weights: 100.09 s
- direct trellis blocking copies: 57.95 s
- total instrumented EXL3 copy wall: 62.60 s
- direct-fill effective throughput: ~0.715 GiB/s
- pinned raw H2D reference: 6.3494 GiB/s
- kernel storage reads: 36.94 GiB
- major faults: ~26k

The next question is narrower:

**Does decoupling mmap/pageable source access from H2D with a reusable pinned
CPU staging buffer reduce per-byte direct-trellis wall time?**

This PR is an A/B probe, not a production optimization.

## Single-boot design

Two separate boots would be strongly confounded by Linux page-cache state.
Instead, one model load is split deterministically inside each equal-shape
trellis arena group:

- even arena slot index: existing control path
  `mmap/pageable -> GPU, blocking`
- odd arena slot index: pinned path
  `mmap/pageable -> reusable pinned CPU buffer -> GPU, blocking`

Because the selector is the slot index *within one exact-shape group*, mixed-K
or expert shape distribution cannot systematically put large tensors in one arm.

The pinned path remains synchronous. There is no async copy and no overlap yet.

## Measurements

Control:

- calls
- bytes
- blocking copy wall
- effective GiB/s
- wall seconds per GiB

Pinned arm:

- calls
- bytes
- one-time/growth pinned allocation wall
- pageable-to-pinned CPU stage wall
- pinned-to-GPU H2D wall
- total stage+H2D wall excluding allocation
- CPU stage GiB/s
- H2D GiB/s
- end-to-end GiB/s
- max reusable pinned buffer footprint

The A/B is valid only if:

- both arms execute;
- both have nonzero bytes;
- pinned byte share is 45-55%;
- control+pinned bytes exactly partition the direct-trellis byte total;
- the trace attests `pinned_stage_ab=true`.

## Projection versus qualification

The summary can project the observed per-GiB costs to:

- all-control direct-trellis wall;
- all-pinned direct-trellis wall;
- all-control main-weight load;
- all-pinned main-weight load.

These are **within-boot projections only**. They are not production
qualification.

If pinned staging has a material per-GiB advantage, the next step is a true
full-pinned boundary boot. Only after that would an async double-buffer /
I/O-H2D overlap design be justified.

If pinned staging does not materially improve total per-GiB wall, the next
target is storage readahead / page-fault behavior instead.

## Corrected storage window

The proc watcher now accepts both:

- `Starting to load model`
- `Loading model from scratch`

The A/B runner hard-requires the ideal
`model_start_to_main_weights_done` window and will reject fallback to the
coarser EngineCore-start window.

## Scope

- one boundary boot;
- 4K loader geometry;
- target model only;
- no prompt;
- no page-cache drop;
- no compile-cache changes;
- no installed-package modification;
- no async copy;
- no K1 changes;
- no merge without explicit instruction.
