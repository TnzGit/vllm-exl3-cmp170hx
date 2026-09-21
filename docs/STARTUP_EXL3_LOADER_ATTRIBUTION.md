# Startup EXL3 loader attribution

## Question

After 161K AOT prewarm, recurring startup is dominated by about 100 seconds of
main-model weight loading. This lane attributes that window without changing
loader policy.

It does **not** optimize anything.

## Boundary

The target loader boundary is deliberately narrower than server startup:

1. vLLM GPU runner logs `Starting to load model ...`.
2. `DefaultModelLoader.load_weights()` executes `model.load_weights(...)`.
3. vLLM logs the first `Loading weights took ...` immediately after that call.
4. vLLM later enters `process_weights_after_loading(...)`.

The EXL3 diagnostic writes its final cumulative loader snapshot on the first
EXL3 post-load hook. Therefore all checkpoint weight-loader copies are already
complete, while later draft-model loading, compile, KV warmup and graph capture
are outside the intended measurement window.

The runner stops as soon as both the first vLLM weight timing and the EXL3
post-load boundary are visible.

## Evidence planes

### Kernel/process evidence

A read-only watcher records `/proc/<EngineCore>/io` and
`/proc/<EngineCore>/stat` around model load:

- `read_bytes`
- `rchar`
- read syscalls
- major/minor page faults
- user/system CPU time
- RSS

`read_bytes` is kernel-accounted storage I/O charged to the process. It is not
treated as a pure safetensors counter.

### EXL3 copy evidence

Diagnostic timing is enabled only with `VLLM_EXL3_LOAD_TRACE_PATH`.

It accumulates:

- generic blocking destination-copy calls, bytes and wall time;
- direct trellis preparation calls, bytes and wall time;
- direct trellis blocking-copy calls, bytes and wall time;
- existing direct-fill/fallback counters.

The production copy mode remains `non_blocking=False`; no asynchronous copy,
pinning, prefetch or loader-policy experiment is introduced.

The copy wall can include source mmap page faults and H2D time. Consequently
storage I/O and copy wall are overlapping evidence and must not be added
together as independent phases.

### H2D reference

The summary compares the instrumented-copy byte volume with the previously
qualified pinned-copy reference of 6.3494 GiB/s.

That is a lower-bound economics reference, not an assumption that the actual
loader source is pinned.

## Interpretation

The result determines the next lane:

- Large kernel `read_bytes` / major-fault volume plus copy wall far above the
  raw H2D floor: source-page availability/storage faults materially contribute.
- Low storage I/O but copy wall far above the floor: pageable/materialization or
  layout/copy overhead is more important.
- Copy wall close to the H2D floor while the 100-second loader remains: most
  time lies outside the instrumented EXL3 copies and the weight iterator /
  safetensors materialization path should be instrumented next.
- Copy wall itself accounts for most of main-weight time: investigate a pinned
  chunked/overlapped transfer pipeline only after this attribution is proven.

No conclusion is made before hardware evidence.

## Safety / scope

- 4K max model length is sufficient because only target main-weight loading is
  measured; long-context engine initialization is intentionally excluded.
- MTP k=3 remains configured, but the runner stops before draft loading is
  required.
- No prompt.
- No Linux page-cache drop.
- No installed-package modification.
- No compile-cache deletion.
- No K1 runtime changes.
