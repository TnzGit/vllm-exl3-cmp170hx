from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "r0_run_startup_161k_cache_reuse.sh"


def test_161k_cache_reuse_runner_shell_syntax():
    run = subprocess.run(
        ["bash", "-n", str(RUNNER)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr


def test_161k_cache_reuse_runner_is_one_health_only_repeat():
    src = RUNNER.read_text()
    assert "serve_161k_warm.log" in src
    assert "baseline_161k_reparsed.json" in src
    assert "serve_161k_repeat.log" in src
    assert "repeat_161k.json" in src
    assert "161k_cache_reuse_comparison.json" in src
    assert "/health" in src
    assert "/v1/models" in src
    assert "/v1/completions" not in src


def test_161k_cache_reuse_runner_preserves_production_style_settings():
    src = RUNNER.read_text()
    assert "STARTUP_LONG_MAX_MODEL_LEN:-161000" in src
    assert "NUM_SPEC_TOKENS:-3" in src
    assert "VLLM_EXL3_COOP=1" in src
    assert 'GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"' in src
    assert "MAX_NUM_SEQS=1" in src
    assert "serve_cmp170hx_qwen_firstboot.sh" in src


def test_161k_cache_reuse_runner_does_not_clear_or_override_compile_cache():
    src = RUNNER.read_text()
    assert "drop_caches" not in src
    assert "sync; echo 3" not in src
    assert "rm -rf" not in src
    assert "VLLM_CACHE_ROOT=" not in src
    assert "VLLM_DISABLE_COMPILE_CACHE" not in src


def test_161k_cache_reuse_runner_compares_compile_and_warmup_phases():
    src = RUNNER.read_text()
    assert "torch_compile_total_s" in src
    assert "initial_profiling_warmup_s" in src
    assert "dynamo_bytecode_s" in src
    assert "compile_graph_sum_s" in src
    assert "engine_init_saved_s" in src
    assert "total_to_health_saved_s" in src
    assert "repeat_cache_hit_evidence" in src
    assert "same_cache_dirs" in src


def test_161k_cache_reuse_runner_keeps_health_evidence():
    src = RUNNER.read_text()
    assert "xid_before=" in src
    assert "xid_after=" in src
    assert "xid_delta=" in src
    assert "gpu_processes=" in src
    assert "vllm_processes=" in src
    assert "port_$PORT=" in src
