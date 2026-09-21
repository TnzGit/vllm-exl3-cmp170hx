from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "r0_run_kvmem_k1_target_core.sh"


def test_k1_target_core_runner_shell_syntax():
    run = subprocess.run(
        ["bash", "-n", str(RUNNER)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr


def test_runner_is_cpu_read_only_and_never_starts_model():
    src = RUNNER.read_text()
    for forbidden in (
        "serve_cmp170hx_qwen_firstboot",
        "patch_vllm_qsa_shadow.py",
        "apply_qwen4_exp_patches.py",
        "nvidia-smi",
    ):
        assert forbidden not in src
    assert "kvmem_k1_runtime_preflight.py" in src
    assert "kvmem_runtime_replay_check.py" in src
    assert "no model engine or installed source modification" in src


def test_runner_requires_prior_k1a_k1b_evidence():
    src = RUNNER.read_text()
    assert "K1A_DIR" in src
    assert "K1B_DIR" in src
    assert "k1b_sticky_summary.json" in src
    assert "expected exactly 12 K1A shadow files" in src


def test_runner_emits_official_preflight_and_runtime_contract_artifacts():
    src = RUNNER.read_text()
    assert "k1_runtime_preflight.json" in src
    assert "k1_runtime_replay_check.json" in src
    assert "recommended_integration_route" in src
