from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "serve_cmp170hx_qwen_firstboot.sh"


def test_cmp170hx_firstboot_defaults_to_qualified_discrete_gpu_load_policy():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'VLLM_EXL3_MADV_AFTER_H2D="${VLLM_EXL3_MADV_AFTER_H2D:-0}"' in text
    assert 'VLLM_EXL3_EXPERT_MATCH_CACHE="${VLLM_EXL3_EXPERT_MATCH_CACHE:-1}"' in text
    assert 'VLLM_EXL3_GC_AFTER_MOE_LAYER="${VLLM_EXL3_GC_AFTER_MOE_LAYER:-0}"' in text


def test_library_policy_remains_overridable_from_environment():
    text = SCRIPT.read_text(encoding="utf-8")

    for name in (
        "VLLM_EXL3_MADV_AFTER_H2D",
        "VLLM_EXL3_EXPERT_MATCH_CACHE",
        "VLLM_EXL3_GC_AFTER_MOE_LAYER",
    ):
        assert f'${{{name}:-' in text
