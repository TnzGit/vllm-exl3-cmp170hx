# CMP170HX Qwen3.8-Flash-Next task board

Primary plan: `docs/CMP170HX_QWEN38_FLASH_NEXT.md`.

Before changing the vLLM runtime, review `docs/UPSTREAM_QWEN_AUDIT.md`.

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

- [ ] no draft
- [ ] k=1
- [ ] k=2
- [ ] k=3
- [ ] k=4 only if there is a specific reason and compile/capture is clean
- [ ] report per-position acceptance
- [ ] report accepted tokens/pass and actual draft iteration count
- [ ] report target/draft time/pass
- [ ] report output tok/s and ms/output-token

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
- [ ] fresh engine across formal context cells
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
