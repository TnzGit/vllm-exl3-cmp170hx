from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "r0_run_exl3_pinned_stage_ab.sh"


def test_pinned_ab_runner_shell_syntax():
    run = subprocess.run(
        ["bash", "-n", str(RUNNER)], capture_output=True, text=True, check=False
    )
    assert run.returncode == 0, run.stderr


def test_pinned_ab_runner_uses_exactly_one_existing_loader_boot():
    src = RUNNER.read_text()
    assert src.count("r0_run_exl3_loader_attribution.sh") == 1
    assert "VLLM_EXL3_PINNED_STAGE_AB=1" in src
    assert "/v1/completions" not in src
    assert "drop_caches" not in src


def test_pinned_ab_runner_requires_corrected_model_load_window():
    src = RUNNER.read_text()
    assert '"model_start_to_main_weights_done"' in src
    assert "ideal_model_load_window=true" in src


def test_pinned_ab_runner_does_not_qualify_projection():
    src = RUNNER.read_text()
    assert "projected full-pinned numbers are not qualification" in src
    assert "full-pinned" in src
    assert "VLLM_EXL3_PINNED_STAGE=1" not in src


def test_pinned_ab_runner_keeps_raw_h2d_reference():
    src = RUNNER.read_text()
    assert "LOADER_REFERENCE_H2D_GIB_S:-6.3494" in src
    assert "r0_exl3_pinned_stage_ab.py" in src
