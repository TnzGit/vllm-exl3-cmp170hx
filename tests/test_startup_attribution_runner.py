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


def test_startup_runner_does_two_identical_health_only_boots():
    src = RUNNER.read_text()
    assert "for RUN in 1 2" in src
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
    assert 'MAX_MODEL_LEN:-4096' in src


def test_startup_runner_outputs_nested_phase_comparison():
    src = RUNNER.read_text()
    assert "startup_run1.json" in src
    assert "startup_run2.json" in src
    assert "startup_comparison.json" in src
    assert "large_weight_time_drop_on_run2" in src
