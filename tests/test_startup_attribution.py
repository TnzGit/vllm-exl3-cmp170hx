import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "r0_startup_attribution.py"


def _load():
    spec = importlib.util.spec_from_file_location("startup_attr", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def test_nested_phase_attribution_separates_main_and_draft_weights():
    mod = _load()
    log = """
INFO Filesystem type for checkpoints: EXT4. Checkpoint size: 68.00 GiB. Available RAM: 120.00 GiB.
INFO EXL3 trellis PRESCAN ready layer=0 experts=64 tensors=192 shards=4 provider=no elapsed_ms=20.0
INFO EXL3 trellis PRESCAN ready layer=1 experts=64 tensors=192 shards=4 provider=no elapsed_ms=30.0
INFO Loading weights took 40.00 seconds
INFO Loading weights took 8.00 seconds
INFO Model loading took 52.00 GiB memory and 58.000000 seconds
INFO Available KV cache memory: 9.75 GiB
INFO GPU KV cache size: 240,128 tokens, Maximum concurrency for 161,000 tokens per request: 1.49x
INFO Setting attention block size to 1568 tokens to ensure that attention page size is >= mamba page size.
INFO Using cache directory: /tmp/vllm/cache123/rank_0_0/backbone for vLLM's torch.compile
INFO Dynamo bytecode transform time: 10.85 s
INFO Compiling graph(1, 2048) took 19.13 s
INFO torch.compile took 33.25 s in total
INFO Initial profiling/warmup run took 60.81 s
INFO saved AOT compiled function to /tmp/torch_aot_compile/bcd650c067/rank_0_0/model
INFO (eagle_head) Dynamo bytecode transform time: 0.39 s
INFO (eagle_head) Compiling a graph for compile range (1, 2048) takes 2.09 s
INFO (eagle_head) torch.compile took 2.67 s in total
INFO (eagle_head) Initial profiling/warmup run took 0.02 s
INFO Graph capturing finished in 5 secs, took 1.00 GiB
INFO init engine (profile, create kv cache, warmup model) took 120.70 s
"""
    out = mod.parse_log(
        log,
        {"start_monotonic_s": 100.0, "health_monotonic_s": 280.0},
    )
    assert out["timings"]["weights_s"] == 40.0
    assert out["timings"]["main_weights_s"] == 40.0
    assert out["timings"]["draft_weights_s"] == 8.0
    assert out["timings"]["total_weights_s"] == 48.0
    assert out["timings"]["weight_loads_s"] == [40.0, 8.0]
    assert out["timings"]["model_construct_postload_s"] == 10.0
    assert out["timings"]["engine_non_graph_s"] == 115.70
    assert out["timings"]["graph_capture_s"] == 5.0
    assert out["timings"]["torch_compile_total_s"] == 33.25
    assert out["timings"]["initial_profiling_warmup_s"] == 60.81
    assert out["timings"]["dynamo_bytecode_s"] == 10.85
    assert out["timings"]["compile_graph_s"] == [19.13, 2.09]
    assert out["timings"]["backbone_compile"]["torch_compile_s"] == [33.25]
    assert out["timings"]["backbone_compile"]["initial_profiling_warmup_s"] == [60.81]
    assert out["timings"]["eagle_head_compile"]["torch_compile_s"] == [2.67]
    assert out["timings"]["eagle_head_compile"]["initial_profiling_warmup_s"] == [0.02]
    assert out["compile_cache"]["aot_saved_ids"] == ["bcd650c067"]
    assert out["runtime_geometry"]["available_kv_cache_gib"] == 9.75
    assert out["runtime_geometry"]["gpu_kv_cache_size_tokens"] == 240128
    assert out["runtime_geometry"]["capacity_request_tokens"] == 161000
    assert out["runtime_geometry"]["kv_max_concurrency"] == 1.49
    assert out["runtime_geometry"]["effective_attention_block_tokens"] == 1568
    assert out["compile_cache"]["cache_dirs"] == [
        "/tmp/vllm/cache123/rank_0_0/backbone"
    ]
    assert out["exl3_prescan"]["sum_ms"] == 50.0
    assert out["phase_ranking"][0]["phase"] == (
        "engine_profile_cache_warmup_ex_graph"
    )


def test_weights_throughput_is_labeled_non_storage_pure():
    mod = _load()
    out = mod.parse_log(
        """
Filesystem type for checkpoints: EXT4. Checkpoint size: 60.00 GiB. Available RAM: 100.00 GiB.
Loading weights took 30.00 seconds
Model loading took 50.00 GiB memory and 40.000000 seconds
init engine (profile, create kv cache, warmup model) took 10.00 s
"""
    )
    assert out["checkpoint"]["implied_checkpoint_gib_per_weights_second"] == 2.0
    assert "not pure storage throughput" in out["checkpoint"]["warning"]


def test_missing_graph_capture_is_supported_for_eager_logs():
    mod = _load()
    out = mod.parse_log(
        """
Loading weights took 20.00 seconds
Model loading took 40.00 GiB memory and 25.000000 seconds
init engine (profile, create kv cache, warmup model) took 7.00 s
"""
    )
    assert out["timings"]["graph_capture_s"] is None
    assert out["timings"]["engine_non_graph_s"] is None
    assert out["phase_ranking"][0]["phase"] == "main_model_weights_path"



def test_runtime_geometry_is_optional_when_logs_do_not_emit_it():
    mod = _load()
    out = mod.parse_log(
        """
Loading weights took 20.00 seconds
Model loading took 40.00 GiB memory and 25.000000 seconds
init engine (profile, create kv cache, warmup model) took 7.00 s
"""
    )
    assert out["runtime_geometry"]["available_kv_cache_gib"] is None
    assert out["runtime_geometry"]["gpu_kv_cache_size_tokens"] is None
    assert out["runtime_geometry"]["capacity_request_tokens"] is None
    assert out["runtime_geometry"]["kv_max_concurrency"] is None
    assert out["runtime_geometry"]["effective_attention_block_tokens"] is None



def test_compile_cache_hit_evidence_is_reported():
    mod = _load()
    out = mod.parse_log(
        """
Loading weights took 20.00 seconds
Model loading took 40.00 GiB memory and 25.000000 seconds
reconstructed serializable fn from standalone compile artifacts. num_artifacts=52 num_submods=1
Directly load AOT compilation from path /tmp/torch_aot_compile/bcd650c067/rank_0_0/model
Directly load the compiled graph(s) for compile range Range(start=1, end=2048) from the cache, took 0.050 s
torch.compile took 0.62 s in total
Initial profiling/warmup run took 1.28 s
INFO (eagle_head) torch.compile took 0.05 s in total
INFO (eagle_head) Initial profiling/warmup run took 0.02 s
init engine (profile, create kv cache, warmup model) took 3.00 s
"""
    )
    assert out["compile_cache"]["cache_hit_evidence"] is True
    assert out["compile_cache"]["aot_direct_load"] is True
    assert out["compile_cache"]["compiled_graph_cache_load"] is True
    assert out["compile_cache"]["standalone_artifact_reconstruction"] is True
    assert out["compile_cache"]["aot_loaded_ids"] == ["bcd650c067"]
    assert out["timings"]["torch_compile_total_s"] == 0.62
    assert out["timings"]["initial_profiling_warmup_s"] == 1.28
    assert out["timings"]["eagle_head_compile"]["torch_compile_s"] == [0.05]


def test_multiple_auxiliary_weight_loads_are_not_misnamed_as_one_draft():
    mod = _load()
    out = mod.parse_log(
        """
Loading weights took 100.0 seconds
Loading weights took 20.0 seconds
Loading weights took 3.0 seconds
Model loading took 50.00 GiB memory and 140.0 seconds
init engine (profile, create kv cache, warmup model) took 5.0 s
"""
    )
    assert out["timings"]["main_weights_s"] == 100.0
    assert out["timings"]["draft_weights_s"] is None
    assert out["timings"]["auxiliary_weights_s"] == 23.0
    assert out["timings"]["total_weights_s"] == 123.0
    assert out["timings"]["model_construct_postload_s"] == 17.0



def test_upstream_compile_range_timing_wording_is_supported():
    mod = _load()
    out = mod.parse_log(
        """
Loading weights took 10.0 seconds
Model loading took 20.00 GiB memory and 12.0 seconds
Compiling a graph for compile range Range(start=1, end=2048) takes 4.25 s
init engine (profile, create kv cache, warmup model) took 8.0 s
"""
    )
    assert out["timings"]["compile_graph_s"] == [4.25]



def test_backbone_scalars_do_not_take_last_eagle_head_match():
    mod = _load()
    out = mod.parse_log(
        """
Loading weights took 100.0 seconds
Loading weights took 20.0 seconds
Model loading took 50.0 GiB memory and 140.0 seconds
Dynamo bytecode transform time: 10.85 s
torch.compile took 33.25 s in total
Initial profiling/warmup run took 60.81 s
(eagle_head) Dynamo bytecode transform time: 0.39 s
(eagle_head) torch.compile took 2.67 s in total
(eagle_head) Initial profiling/warmup run took 0.02 s
init engine (profile, create kv cache, warmup model) took 120.7 s
"""
    )
    assert out["timings"]["torch_compile_total_s"] == 33.25
    assert out["timings"]["initial_profiling_warmup_s"] == 60.81
    assert out["timings"]["dynamo_bytecode_s"] == 10.85
    assert out["timings"]["eagle_head_compile"]["torch_compile_s"] == [2.67]


def test_aot_saved_and_loaded_ids_share_identity_across_log_forms():
    mod = _load()
    saved = mod.parse_log(
        """
Loading weights took 1.0 seconds
Model loading took 1.0 GiB memory and 2.0 seconds
saved AOT compiled function to /x/torch_aot_compile/abc123/rank_0_0/model
init engine (profile, create kv cache, warmup model) took 1.0 s
"""
    )
    loaded = mod.parse_log(
        """
Loading weights took 1.0 seconds
Model loading took 1.0 GiB memory and 2.0 seconds
Directly load AOT compilation from path /x/torch_aot_compile/abc123/rank_0_0/model
init engine (profile, create kv cache, warmup model) took 1.0 s
"""
    )
    assert saved["compile_cache"]["aot_saved_ids"] == ["abc123"]
    assert loaded["compile_cache"]["aot_loaded_ids"] == ["abc123"]
    assert set(saved["compile_cache"]["aot_saved_ids"]) & set(
        loaded["compile_cache"]["aot_loaded_ids"]
    )
