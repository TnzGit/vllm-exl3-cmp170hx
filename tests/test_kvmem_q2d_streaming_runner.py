from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "r0_run_kvmem_k1q2d_streaming_runtime.sh"


def test_streaming_runner_freezes_experiment_and_restores_qsa():
    src = RUNNER.read_text()
    for required in (
        'EXPECTED_SHA="${EXPECTED_SHA:?',
        'GPU_MEM_UTIL" != "0.92"',
        'MAXLEN" != "161000"',
        'MAX_BATCHED" != "1024"',
        'status --porcelain',
        'REFUSE: worktree is dirty',
        'VLLM_QWEN_KVMEM_Q2D_RUNTIME_PLAN',
        'VLLM_QWEN_KVMEM_Q2D_WORKER_STATS_PATH',
        'VLLM_QWEN_KVMEM_Q2D_SCHED_STATS_PATH',
        ': > "$WORKER_STATS"',
        ': > "$SCHED_STATS"',
        'restore_qsa',
        'qsa_sha256_after',
        'xid_delta',
    ):
        assert required in src
