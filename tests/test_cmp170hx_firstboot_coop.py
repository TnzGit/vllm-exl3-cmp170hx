from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "serve_cmp170hx_qwen_firstboot.sh"


def test_cmp170hx_firstboot_defaults_to_qualified_coop_moe():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'VLLM_EXL3_COOP="${VLLM_EXL3_COOP:-1}"' in text
    assert '"VLLM_EXL3_COOP" "$VLLM_EXL3_COOP"' in text


def test_coop_default_remains_environment_overridable():
    text = SCRIPT.read_text(encoding="utf-8")

    assert '${VLLM_EXL3_COOP:-1}' in text
