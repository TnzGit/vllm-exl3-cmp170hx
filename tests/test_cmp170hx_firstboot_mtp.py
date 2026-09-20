from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "serve_cmp170hx_qwen_firstboot.sh"


def test_firstboot_selects_preflight_profile_from_speculation_setting():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-0}"' in text
    assert 'PREFLIGHT_PROFILE="text-no-draft"' in text
    assert 'PREFLIGHT_PROFILE="text-mtp"' in text
    assert 'python "$ROOT/tools/cmp170hx_qwen_preflight.py" --profile "$PREFLIGHT_PROFILE"' in text


def test_mtp_profile_is_selected_only_for_nonzero_spec_tokens():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'if [[ "$NUM_SPEC_TOKENS" == "0" ]]; then' in text
    assert 'else\n  PREFLIGHT_PROFILE="text-mtp"' in text
