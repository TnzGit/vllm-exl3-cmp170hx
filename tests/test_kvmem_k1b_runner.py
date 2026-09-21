from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "r0_run_kvmem_k1b_sticky.sh"


def test_k1b_runner_shell_syntax():
    run = subprocess.run(
        ["bash", "-n", str(RUNNER)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr


def test_k1b_runner_reuses_k1a_artifacts_without_engine():
    src = RUNNER.read_text()
    assert "K1A_DIR" in src
    assert "k1a_churn_summary.json" in src
    assert "turns/manifest.json" in src
    assert "shadows" in src
    for forbidden in (
        "vllm serve",
        "serve_cmp170hx_qwen_firstboot",
        "patch_vllm_qsa_shadow.py",
        "ENFORCE_EAGER",
    ):
        assert forbidden not in src


def test_k1b_runner_runs_tie_microbench_and_sticky_replay():
    src = RUNNER.read_text()
    assert "kvmem_persistent_topk_diagnose.py" in src
    assert "kvmem_qsa_sticky_replay.py" in src
    assert "persistent_topk_diagnosis.json" in src
    assert "h2d_bandwidth.json" in src
    assert "k1b_sticky_summary.json" in src
