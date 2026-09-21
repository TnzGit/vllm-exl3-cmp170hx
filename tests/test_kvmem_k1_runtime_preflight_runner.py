from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "r0_run_kvmem_k1_runtime_preflight.sh"


def test_k1_runtime_preflight_runner_shell_syntax():
    run = subprocess.run(
        ["bash", "-n", str(RUNNER)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr


def test_runner_never_starts_model_or_patches_installed_source():
    src = RUNNER.read_text()
    for forbidden in (
        "vllm serve",
        "serve_cmp170hx_qwen_firstboot",
        "patch_vllm_qsa_shadow.py",
        "apply_qwen4_exp_patches.py",
    ):
        assert forbidden not in src
    assert "no installed source was modified" in src


def test_runner_executes_preflight_and_fixed_topk_diagnostic():
    src = RUNNER.read_text()
    assert "kvmem_k1_runtime_preflight.py" in src
    assert "k1_runtime_preflight.json" in src
    assert "kvmem_persistent_topk_diagnose.py" in src
    assert "persistent_topk_diagnosis.json" in src
    assert "test_kvmem_persistent_topk_diagnose.py" in src
    assert "test_kvmem_k1_runtime_preflight.py" in src
