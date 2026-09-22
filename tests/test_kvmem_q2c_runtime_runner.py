from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "r0_run_kvmem_k1q2c_runtime_ownership.sh"
LAUNCHER = ROOT / "tools" / "serve_cmp170hx_qwen_firstboot.sh"


def test_q2c_runtime_runner_shell_syntax():
    r = subprocess.run(["bash", "-n", str(RUNNER)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_q2c_runtime_runner_contract():
    src = RUNNER.read_text()
    assert 'EXPECTED_SHA="${EXPECTED_SHA:?set EXPECTED_SHA to the exact Q2C runtime head}"' in src
    assert "patch_vllm_qsa_q2c_runtime.py" in src
    assert "VLLM_QWEN_KVMEM_Q2C_PLAN" in src
    assert "VLLM_QWEN_KVMEM_Q2C_SCHED_STATS_PATH" in src
    assert "VLLM_QWEN_KVMEM_Q2C_WORKER_STATS_PATH" in src
    assert "VLLM_KV_CACHE_LAYOUT=BLHNC" in src
    assert "ENFORCE_EAGER=1 NUM_SPEC_TOKENS=0" in src
    assert 'MAX_BATCHED="${K1Q2C_MAX_NUM_BATCHED_TOKENS:-1024}"' in src
    assert 'MAX_NUM_BATCHED_TOKENS="$MAX_BATCHED"' in src
    assert 'requires max-num-batched-tokens=1024' in src
    assert "q2c_runtime_summary.json" in src
    assert "restore_qsa" in src
    assert "qsa_sha256_after" in src
    assert "STOP HERE" in src
    assert "240K" in src


def test_firstboot_launcher_accepts_explicit_batch_token_cap():
    r = subprocess.run(["bash", "-n", str(LAUNCHER)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    src = LAUNCHER.read_text()
    assert 'MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"' in src
    assert 'ARGS+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")' in src
