# CMP170HX EXL3 model-load fast path

This note documents the load-time diagnosis for the Qwen3.8-Flash-Next EXL3 checkpoint on one CMP170HX.

## Observed symptom

A ~47.82 GiB model load took ~330 s, or roughly 145 MiB/s effective end-to-end load rate, despite multi-GB/s NVMe and PCIe Gen2 x16.

This is not evidence that PCIe itself is limited to ~145 MiB/s. The path is dominated by very small per-tensor operations and synchronization.

The target pack contains about 304k tensors. Routed experts account for almost all of them:

- trellis: 48 layers * 512 experts * 3 projections = 73,728 tensors
- suh/svh: 48 * 512 * 3 * 2 = 147,456 tensors
- one codebook marker per projection: 48 * 512 * 3 = 73,728 tensors

Total: 294,912 routed-expert tensor entries before other model tensors.

## Root causes found

### 1. Arena prescan was never enabled by the launcher

_try_prescan_trellis_shapes() requires either VLLM_EXL3_MODEL_DIR or VLLM_ENGRAM_MODEL_DIR.
The CMP170HX first-boot launcher did not set either variable. Therefore prescan returned None before reading the checkpoint index and every routed trellis fell back to host staging.

The observed 73,728 EXL3 trellis staging fallback messages exactly match one fallback per routed expert projection.

### 2. Fallback causes two-stage trellis handling

Without a prescan plan each trellis is loaded, staged on host, retained until process_weights_after_loading(), and then copied again into the final GPU arena.
With a valid plan the final arenas are allocated before the first trellis load and each tensor is copied directly to its final slot.

### 3. Existing prescan was itself too fine-grained

The old fallback metadata scanner called safe_open() once per trellis tensor. For this model that can mean up to 73,728 safetensors opens/header traversals.

The fast-path patch now:

- parses the large safetensors index once per process;
- keeps only routed trellis key/shard metadata;
- accepts both w1/w3/w2 and gate_proj/up_proj/down_proj source names;
- separates main-model and MTP key namespaces;
- groups requested keys by shard;
- opens each shard header once per layer rather than once per tensor.

### 4. Routed suh/svh loading performed a stream-wide sync per tensor

The routed scale path copied each small suh/svh tensor and then executed torch.cuda.current_stream().synchronize().
That can occur 147,456 times for the 48x512-expert model.

This is the same redundant pattern already removed from direct trellis fill in upstream plugin PR #29: CPU->CUDA copy_ with non_blocking=False already preserves the CPU source lifetime. A second stream-wide synchronization only serializes the loader.

The patch routes these copies through _copy_weight_blocking() and removes the extra stream-wide synchronization. The same fix is applied to the dense EXL3 load path.

### 5. Warning volume was itself non-trivial

The fallback path emitted one warning per trellis tensor. The patch keeps the first three examples, emits one suppression notice, then reports one summary per affected layer. Counters/bytes remain available through direct_fill_stats() / runtime_diagnostics().

## What this patch does not solve

Even with the fast path, this checkpoint still has around 300k tensor entries. The generic vLLM/safetensors loader still performs Python dispatch for a very large number of small tensors, and marker tensors are still loaded individually.

Therefore a successful patch should materially reduce startup time, but it is not yet justified to expect raw NVMe/PCIe streaming speed.

If the load remains far above ~1-2 minutes after this patch, profile the remaining loader by tensor family before redesigning it. A possible next step would be batched marker staging or a shard-oriented routed-expert loader, but neither should be implemented before the direct-fill/sync A/B is measured.

## Qualification gate

Compare old vs fast-path with the exact same prepared checkpoint, vLLM 0.29.0, ExLlamaV3 1.5.0, GPU memory utilization, context/no-draft service profile, and disk n-gram setting.

Measure:

- wall time from server launch to first healthy;
- weight-load interval;
- engine init/warmup interval;
- direct-fill calls/bytes;
- fallback calls/bytes;
- number of PRESCAN ready layers;
- number of fallback summaries;
- peak host RAM;
- peak VRAM;
- Xid delta;
- one deterministic greedy output for parity.

Acceptance:

- correctness and Xid clean;
- DIRECT_FILL_FALLBACK_CALLS == 0 for the main routed expert bank;
- direct-fill calls match the expected main-model trellis count;
- startup time materially improves.

Do not merge based only on fewer log lines.
