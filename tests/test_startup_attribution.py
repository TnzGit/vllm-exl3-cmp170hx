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


def test_nested_phase_attribution():
    mod = _load()
    log = """
INFO Filesystem type for checkpoints: EXT4. Checkpoint size: 68.00 GiB. Available RAM: 120.00 GiB.
INFO EXL3 trellis PRESCAN ready layer=0 experts=64 tensors=192 shards=4 provider=no elapsed_ms=20.0
INFO EXL3 trellis PRESCAN ready layer=1 experts=64 tensors=192 shards=4 provider=no elapsed_ms=30.0
INFO Loading weights took 40.00 seconds
INFO Model loading took 52.00 GiB memory and 50.000000 seconds
INFO Graph capturing finished in 12 secs, took 1.00 GiB
INFO init engine (profile, create kv cache, warmup model) took 30.00 s
"""
    out = mod.parse_log(
        log,
        {"start_monotonic_s": 100.0, "health_monotonic_s": 190.0},
    )
    assert out["timings"]["weights_s"] == 40.0
    assert out["timings"]["model_construct_postload_s"] == 10.0
    assert out["timings"]["engine_non_graph_s"] == 18.0
    assert out["timings"]["graph_capture_s"] == 12.0
    assert out["timings"]["frontend_spawn_preflight_other_s"] == 10.0
    assert out["exl3_prescan"]["sum_ms"] == 50.0
    assert out["phase_ranking"][0]["phase"] == "weights_path"


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
    assert out["phase_ranking"][0]["phase"] == "weights_path"
