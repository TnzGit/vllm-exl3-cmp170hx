from pathlib import Path
import re
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

    # Health/provenance checks are allowed to *mention* a process pattern such
    # as "vllm serve". Reject executable launch forms instead of using a naive
    # substring blacklist that also catches pgrep/grep diagnostics.
    executable_lines = [
        line.strip()
        for line in src.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert not any(
        re.search(r"(^|[;&|])\s*(?:setsid\s+)?(?:[^#]*\s)?vllm\s+serve(?:\s|$)", line)
        and "pgrep" not in line
        and "grep" not in line
        for line in executable_lines
    )

    for forbidden in (
        "serve_cmp170hx_qwen_firstboot",
        "patch_vllm_qsa_shadow.py",
        "apply_qwen4_exp_patches.py",
    ):
        assert forbidden not in src

    # Read-only process accounting remains part of the final-state proof.
    assert "pgrep -af" in src
    assert "VLLM::EngineCore|vllm serve" in src
    assert "no installed source was modified" in src


def test_runner_executes_preflight_and_fixed_topk_diagnostic():
    src = RUNNER.read_text()
    assert "kvmem_k1_runtime_preflight.py" in src
    assert "k1_runtime_preflight.json" in src
    assert "kvmem_persistent_topk_diagnose.py" in src
    assert "persistent_topk_diagnosis.json" in src
    assert "test_kvmem_persistent_topk_diagnose.py" in src
    assert "test_kvmem_k1_runtime_preflight.py" in src
