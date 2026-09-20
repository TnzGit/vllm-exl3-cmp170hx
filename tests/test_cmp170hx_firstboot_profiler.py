from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "serve_cmp170hx_qwen_firstboot.sh"


def test_firstboot_profiler_is_opt_in():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'if [[ -n "${TORCH_PROFILER_DIR:-}" ]]; then' in text
    assert 'ARGS+=(--profiler-config "$PROFILER_CONFIG")' in text
    assert '"profiler": "torch"' in text
    assert '"ignore_frontend": True' in text


def test_firstboot_keeps_profiler_disabled_without_directory():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'echo "  torch profiler               ${TORCH_PROFILER_DIR:-disabled}"' in text
