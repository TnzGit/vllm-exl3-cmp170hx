from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
FIRSTBOOT = ROOT / "tools" / "serve_cmp170hx_qwen_firstboot.sh"
WRAPPER = ROOT / "tools" / "r0_serve_coop_eager_profiler.sh"
ANALYZER = ROOT / "tools" / "r0_analyze_eager_attribution.py"


def test_firstboot_exposes_explicit_eager_profile_mode():
    text = FIRSTBOOT.read_text(encoding="utf-8")
    assert 'if [[ "${ENFORCE_EAGER:-0}" == "1" ]]; then' in text
    assert "ARGS+=(--enforce-eager)" in text
    assert "EAGER (profiling only)" in text


def test_eager_wrapper_forces_coop_profiler_contract():
    text = WRAPPER.read_text(encoding="utf-8")
    assert "export ENFORCE_EAGER=1" in text
    assert 'VLLM_EXL3_COOP="${VLLM_EXL3_COOP:-1}"' in text
    assert 'TORCH_PROFILER_RECORD_SHAPES="${TORCH_PROFILER_RECORD_SHAPES:-1}"' in text
    assert "attribution-only" in text


def test_eager_analyzer_recovers_cpu_and_python_parent(tmp_path):
    trace = {
        "traceEvents": [
            {
                "ph": "X",
                "cat": "python_function",
                "name": "Qwen4Exp.forward",
                "pid": 1,
                "tid": 2,
                "ts": 0,
                "dur": 100,
                "args": {},
            },
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": "aten::mm",
                "pid": 1,
                "tid": 2,
                "ts": 10,
                "dur": 20,
                "args": {"External id": 42},
            },
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": "cudaLaunchKernel",
                "pid": 1,
                "tid": 2,
                "ts": 15,
                "dur": 2,
                "args": {"External id": 42, "correlation": 7},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "void gemv2T_kernel<float>()",
                "pid": 3,
                "tid": 4,
                "ts": 30,
                "dur": 50,
                "args": {"correlation": 7},
            },
        ]
    }
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(trace), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(ANALYZER), str(path), "--steps", "1"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "target_correlation_coverage=100.00%" in proc.stdout
    assert "aten::mm" in proc.stdout
    assert "Qwen4Exp.forward" in proc.stdout
    assert "gemv2T_kernel" in proc.stdout
