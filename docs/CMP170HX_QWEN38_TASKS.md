# CMP170HX Qwen3.8-Flash-Next task board

Primary plan: `docs/CMP170HX_QWEN38_FLASH_NEXT.md`.

Machine-side work should use branch `bringup/qwen38-flash-next-cmp170hx-r0`, cut from the reviewed scaffold. Keep `research/qwen38-flash-next-cmp170hx` as the scaffold lane.

Before changing the vLLM runtime, review `docs/UPSTREAM_QWEN_AUDIT.md`.

## CPU gate before touching GPU

- [ ] `python -m py_compile tools/apply_qwen4_exp_patches.py tools/cmp170hx_qwen_preflight.py tools/cmp170hx_qwen_pack_manifest.py`
- [ ] `python -m pytest -q tests/test_qwen4_exp_patch_script.py tests/test_qwen4_exp_patch_stack.py tests/test_cmp170hx_qwen_preflight.py tests/test_pack_tools_ngram_fixture.py`
- [ ] Run the full CPU-capable repository test suite if dependencies are available.
- [ ] If any test fails, fix the scaffold first; do not compensate in the serve command.

GitHub Actions has not produced a run for this fork/PR yet, so local CPU validation is a hard
bring-up gate rather than an assumed CI result.

## Runtime identity

- [ ] Record exact vLLM artifact/version.
- [ ] Use ExLlamaV3 v1.5.0 tag commit `0740edc2da569fb99174023c1d2988b1e98cb41e` for R0 and record the extension ABI.
- [ ] Record exact vllm-exl3 SHA.
- [ ] Record PyTorch, CUDA and driver.
- [ ] Record exact model revision and config/index hashes.
- [ ] Save pack scan output.
- [ ] Save worker-side runtime diagnostics.

## F1 - pack contract

- [ ] Run `qwen_pack_scan.py`.
- [ ] Run `qwen_pack_config.py --dry-run`.
- [ ] Verify n-gram layout/bits from tensor metadata.
- [ ] Verify dense EXL3 map.
- [ ] Verify routed expert K/codebook.
- [ ] Verify lm_head and MTP tensors.
- [ ] Verify vision split/fused-qkv contract.
- [ ] Do not patch around ambiguous pack metadata.

## F2 - minimum service boot

- [ ] One CMP170HX, C1, deliberately small first context.
- [ ] No speculation.
- [ ] `VLLM_EXL3_NGRAM_TABLE=disk`.
- [ ] PIECEWISE graphs with `vllm::exl3_ngram_lookup_out` split.
- [ ] Run `tools/cmp170hx_qwen_preflight.py`.
- [ ] Confirm installed vLLM source is actually patched.
- [ ] Record actual EXL3 dispatch.
- [ ] Coherent greedy output.
- [ ] Zero Xid delta.
- [ ] Memory ledger.

## F3 - MTP

Start the first MTP comparison with prefix caching disabled and C1 so current upstream hybrid-cache annotation issues are not an attribution confound. Prefix caching is a separate later qualification gate.

- [ ] Before MTP boot, apply `--profile text-mtp` and run the matching preflight.
- [ ] Record the actual `_mtp_hidden_buffer.device`; vLLM 0.29 source lacks explicit `device=` and upstream #56742 is still open.
- [ ] no draft
- [ ] k=1
- [ ] k=2
- [ ] k=3
- [ ] k=4 only if there is a specific reason and compile/capture is clean
- [ ] report per-position acceptance
- [ ] report accepted tokens/pass and actual draft iteration count
- [ ] report target/draft time/pass
- [ ] report output tok/s and ms/output-token

### Prefix-cache qualification after the C1 MTP baseline

- [ ] Enable prefix caching only after the no-prefix MTP baseline is stable.
- [ ] Repeated byte-identical prompts must produce measured cache hits; do not infer cache activity from lower TTFT alone.
- [ ] If hits remain zero, review upstream #57616 / #55390 / #56026 before tuning performance.

## F4 - context envelope

C1 first:

- [ ] 4K
- [ ] 32K
- [ ] 65K
- [ ] 126K
- [ ] around 160K
- [ ] 200K
- [ ] 250K

Then, only if useful:

- [ ] C2 admission/residency qualification
- [ ] C4 admission/residency qualification
- [ ] running/waiting/preemption/KV telemetry
- [ ] valid steady-decode windows only

## F5 - profile

Before any new kernel work:

- [ ] EXL3 routed MoE share
- [ ] dense EXL3 share
- [ ] n-gram id D2H synchronization
- [ ] host mmap gather
- [ ] H2D packed rows
- [ ] n-gram decode
- [ ] QSA indexer
- [ ] QSA sparse attention
- [ ] GDN/recurrent state
- [ ] MTP
- [ ] graph split overhead
- [ ] scheduler/other

## Optimization gates

### Disk n-gram

- [ ] Prove host/sync cost is material before changing staging.
- [ ] If justified, prototype persistent pinned buffers.
- [ ] Give every async source buffer explicit event/lifetime ownership.
- [ ] Never restore short-lived `non_blocking=True` temporaries.

### EXL3 MoE

- [ ] Measure actual Qwen routed shape.
- [ ] A/B stock vs `exl3_moe_coop` on CMP170HX.
- [ ] Validate parity first.
- [ ] Keep only an end-to-end win.

### Service policy

- [ ] Locate/disprove the ultra-long MTP acceptance cliff.
- [ ] Prefer simple static profiles before adaptive runtime policy.

## Permanent inherited guards

- [ ] widen address indices before large-stride multiplication when needed
- [ ] exact-token formal prompts
- [ ] fresh engine per runtime/configuration change or after failure; same-config context sweeps may reuse one healthy engine with start/end sentinel checks
- [ ] Xid delta on every GPU experiment
- [ ] numerator and denominator share the same interval
- [ ] never label ms/output-token as ms/step
- [ ] client profiler is not server GPU attribution
- [ ] preserve negative results

## Do not port from the 27B project

- M7 NSEG35 mixed-FP8 verifier
- M7 LUT/page-carry/half-warp code
- INT8-G64 verifier work
- dense Marlin tuning
- 27B k=3/5/7 values
- FULL-graph assumptions for disk n-gram

## Engine reuse rule

Model startup is expensive for this EXL3 pack. Do not restart mechanically for every context cell.

A fresh engine is required when changing runtime/env/patches, MTP k, graph mode, prefix-cache policy, cache dtype/geometry, memory configuration, or after crash/OOM/Xid.

For a same-configuration C1 context sweep, reuse one healthy engine:

- run a short-context sentinel first;
- run the requested context cells;
- verify running=0 / waiting=0 between requests;
- require no Xid/preemption/abnormal VRAM growth;
- repeat the sentinel at the end.

If the sentinel moves by more than roughly 2-3%, or an anomalous/cliff point appears, rerun the decisive cell with a fresh engine. Final boundary/anomaly claims still need fresh-engine confirmation.

## MTP qualification — complete

- [x] hidden buffer landed on `cuda:0`; no #56742 backport required
- [x] k=1 / k=2 / k=3 measured at short and mid context
- [x] best MTP setting is k=1, but it is still ~21-25% slower than no-draft
- [x] long-context acceptance cliff did not reproduce through 204,525 prompt tokens
- [x] production decision: **MTP OFF / no-draft only**

Next gate: profile the no-draft path before choosing MoE, disk n-gram, QSA or GDN optimization work.
