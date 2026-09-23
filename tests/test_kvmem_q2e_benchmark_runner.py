from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "r0_run_kvmem_k1q2e_benchmark.sh"


def test_benchmark_runner_freezes_performance_contract_and_restores_qsa():
    source = RUNNER.read_text()
    assert "CONTEXTS=(16000 80000 160000 240000)" in source
    assert "MAX_MODEL_LEN=246000" in source
    assert "MAX_BATCHED=1024" in source
    assert "MAX_TOKENS=256" in source
    assert "GPU_MEM_UTIL=0.92" in source
    assert 'ENFORCE_EAGER="$EAGER" NUM_SPEC_TOKENS=0' in source
    assert "VLLM_QWEN_KVMEM_Q2E_DIRECT_IO=1" in source
    assert "VLLM_QWEN_KVMEM_Q2E_DIRECT_CONSUMER_SYNC=1" in source
    assert "unset VLLM_QWEN_KVMEM_Q2E_TRACE_PATH" in source
    assert "--benchmark-case" in source
    assert "k1b_sticky_summary" not in source
    assert "--max-tokens 16 --allow-semantic-failure" in source
    assert "EXPECTED_SHA" in source
    assert "restore_qsa" in source
    assert "xid_now" in source
    assert "Q2E_BENCHMARK_VALID" in source
