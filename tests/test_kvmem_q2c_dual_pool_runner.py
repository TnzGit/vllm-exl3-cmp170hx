from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "r0_run_kvmem_k1q2c_dual_pool.sh"


def test_q2c_dual_pool_runner_shell_syntax():
    r = subprocess.run(["bash", "-n", str(RUNNER)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_q2c_dual_pool_runner_contract():
    src = RUNNER.read_text()
    assert 'EXPECTED_SHA="${EXPECTED_SHA:?set EXPECTED_SHA to the exact Q2C dual-pool head}"' in src
    assert 'MAX_BATCHED="${K1Q2C_MAX_NUM_BATCHED_TOKENS:-1024}"' in src
    assert 'MAXLEN="${MAX_MODEL_LEN:-161000}"' in src
    assert "patch_vllm_q2c_dual_pool.py" in src
    assert "patch_vllm_qsa_q2c_runtime.py" in src
    assert "test_kvmem_q2c_dual_pool_installed.py" in src
    assert "q2c_dual_pool_boot.json" in src
    assert "kvmem_q2c_dual_pool_boot_summarize.py" in src
    assert '--boot "$BOOT"' in src
    assert "BOOT_INVALID: Q2C runtime gates remain unproven" in src
    assert "VLLM_QWEN_KVMEM_Q2C_PLAN" in src
    assert "VLLM_KV_CACHE_LAYOUT=BLHNC" in src
    assert "ENFORCE_EAGER=1 NUM_SPEC_TOKENS=0" in src
    assert "restore_sources" in src
    assert "platform_sha256_after" in src
    assert "core_sha256_after" in src
    assert "worker_sha256_after" in src
    assert "STOP HERE" in src


def test_q2c_dual_pool_runner_restores_all_installed_sources():
    src = RUNNER.read_text()
    for name in (
        "qsa.base.py",
        "platform_interface.base.py",
        "kv_cache_utils.base.py",
        "worker_utils.base.py",
    ):
        assert name in src
    assert 'cp "$QSA_BACKUP" "$QSA"' in src
    assert 'cp "$PLATFORM_BACKUP" "$PLATFORM"' in src
    assert 'cp "$CORE_BACKUP" "$CORE"' in src
    assert 'cp "$WORKER_BACKUP" "$WORKER"' in src
