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
        'K1Q2E_PROFILE',
        'K1Q2E_DIRECT_IO',
        'VLLM_QWEN_KVMEM_Q2E_DIRECT_IO',
        'direct_dedicated_slots',
        'K1Q2E_EXPECTED_PLAN_SHA256',
        'K1Q2E_EXPECTED_TRACE_SHA256',
        'K1Q2E_EXPECTED_SCHEDULER_DIGEST',
        '--expected-trace-sha256',
        '--expected-scheduler-digest',
        'VLLM_QWEN_KVMEM_Q2E_TRACE_PATH',
        'VLLM_QWEN_KVMEM_Q2E_TRACE_MAX_BYTES',
        'kvmem_q2e_trace_summarize.py',
        '--trace-summary',
        'q2e_boot_gpu_processes.csv',
        'q2e_request_wall_seconds.txt',
        ': > "$WORKER_STATS"',
        ': > "$SCHED_STATS"',
        'restore_qsa',
        'qsa_sha256_after',
        'xid_delta',
    ):
        assert required in src
