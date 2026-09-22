from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "r0_run_kvmem_k1q2d_reload_shadow.sh"


def test_reload_runner_freezes_identity_geometry_restore_and_health():
    src = RUNNER.read_text()
    for required in (
        'EXPECTED_SHA="${EXPECTED_SHA:?',
        'GPU_MEM_UTIL" != "0.92"',
        'MAXLEN" != "161000"',
        'MAX_BATCHED" != "1024"',
        'status --porcelain',
        'REFUSE: worktree is dirty',
        'VLLM_QWEN_KVMEM_Q2D_PLAN',
        'VLLM_QWEN_KVMEM_Q2D_STATS_PATH',
        ': > "$STATS"',
        'restore_qsa',
        'qsa_sha256_after',
        'xid_delta',
        'gpu_processes=',
        'vllm_processes=',
    ):
        assert required in src


def test_reload_runner_labels_shadow_as_diagnostic():
    src = RUNNER.read_text()
    assert "kvmem_q2d_reload_summarize.py" in src
    assert "patch_vllm_qsa_q2d_reload_shadow.py" in src
    assert "patch_vllm_qsa_q2c_runtime.py" not in src
