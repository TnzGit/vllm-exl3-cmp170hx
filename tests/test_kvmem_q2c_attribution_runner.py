from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "r0_run_kvmem_k1q2c_attribution.sh"


def test_attribution_runner_shell_syntax():
    result = subprocess.run(
        ["bash", "-n", str(RUNNER)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_attribution_runner_freezes_three_matched_modes():
    src = RUNNER.read_text()
    assert 'EXPECTED_SHA="${EXPECTED_SHA:?set EXPECTED_SHA to the exact attribution head}"' in src
    assert 'MAX_BATCHED="${K1Q2C_MAX_NUM_BATCHED_TOKENS:-1024}"' in src
    assert "launch_full_control baseline" in src
    assert "launch_full_control progressive_mask" in src
    assert "launch_bounded" in src
    assert 'A_ONLY="${K1Q2C_ATTRIB_A_ONLY:-0}"' in src
    assert 'SPLIT_ONLY="${K1Q2C_ATTRIB_SPLIT_ONLY:-0}"' in src
    assert 'SPLIT_ROWS="${K1Q2C_SPLIT_ROWS:-64}"' in src
    assert "VLLM_QWEN_KVMEM_Q2C_WORKSET_ROW_BATCHES" in src
    assert "kvmem_q2c_working_set_summarize.py" in src
    assert "kvmem_q2c_split_summarize.py" in src
    assert "launch_full_control split_reference" in src
    assert "VLLM_QWEN_KVMEM_Q2C_ATTRIB_STATS_PATH" in src
    assert ': > "$stats"' in src
    assert "VLLM_QWEN_KVMEM_Q2C_SCHED_STATS_PATH" in src
    assert "ENFORCE_EAGER=1 NUM_SPEC_TOKENS=0" in src
    assert 'GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"' in src
    assert 'MAXLEN="${MAX_MODEL_LEN:-161000}"' in src
    assert "restore_qsa" in src
    assert "qsa_sha256_after" in src
    assert "xid_delta" in src
