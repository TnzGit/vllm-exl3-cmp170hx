from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "r0_run_kvmem_k1q2c_scheduler_preflight.sh"


def test_q2c_preflight_runner_shell_syntax():
    r = subprocess.run(["bash", "-n", str(RUNNER)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_q2c_preflight_runner_is_cpu_read_only_and_exact_sha_gated():
    src = RUNNER.read_text()
    assert 'EXPECTED_SHA="${EXPECTED_SHA:?set EXPECTED_SHA to the exact preflight head}"' in src
    assert "kvmem_q2c_scheduler_preflight.py" in src
    assert "--observed-full-block-tokens 1568" in src
    assert "Q2C_SCHEDULER_SHRINK_PREFLIGHT_GO" in src
    assert "NUM_SPEC_TOKENS" not in src
    assert "serve_cmp170hx" not in src
    assert "setsid bash" not in src
    assert "curl" not in src
    assert "patch_vllm" not in src
    assert "cp \"$QSA" not in src
    assert "STOP HERE" in src
