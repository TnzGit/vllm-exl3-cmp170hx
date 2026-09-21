from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "r0_run_startup_attribution.sh"


def test_startup_runner_shell_syntax():
    run = subprocess.run(
        ["bash", "-n", str(RUNNER)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr


def test_startup_runner_does_4k_current_4k_warm_and_161k_warm_health_boots():
    src = RUNNER.read_text()
    assert 'run_boot 1 "4k_current" "$SHORT_MAXLEN"' in src
    assert 'run_boot 2 "4k_warm" "$SHORT_MAXLEN"' in src
    assert 'run_boot 3 "161k_warm" "$LONG_MAXLEN"' in src
    assert "/health" in src
    assert "/v1/models" in src
    assert "/v1/completions" not in src
    assert "drop_caches" not in src
    assert "sync; echo 3" not in src


def test_startup_runner_preserves_production_weight_path_defaults():
    src = RUNNER.read_text()
    assert 'NUM_SPEC_TOKENS:-3' in src
    assert "VLLM_EXL3_COOP=1" in src
    assert "serve_cmp170hx_qwen_firstboot.sh" in src
    assert 'STARTUP_SHORT_MAX_MODEL_LEN:-4096' in src
    assert 'STARTUP_LONG_MAX_MODEL_LEN:-161000' in src


def test_startup_runner_outputs_cache_and_long_context_comparisons():
    src = RUNNER.read_text()
    assert "startup_4k_current.json" in src
    assert "startup_4k_warm.json" in src
    assert "startup_161k_warm.json" in src
    assert "startup_comparison.json" in src
    assert "cache_effect_4k" in src
    assert "long_context_effect_warm" in src
    assert "large_4k_weight_time_drop_when_warm" in src
    assert "long_context_added_engine_init_s" in src


def test_startup_runner_keeps_health_and_safety_evidence():
    src = RUNNER.read_text()
    assert "xid_before=" in src
    assert "xid_after=" in src
    assert "xid_delta=" in src
    assert "gpu_processes=" in src
    assert "vllm_processes=" in src
    assert "port_$PORT=" in src
