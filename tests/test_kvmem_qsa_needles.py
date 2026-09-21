import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "kvmem_qsa_make_needles.py"


def _load():
    spec = importlib.util.spec_from_file_location("needle_gen", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(c) % 251 + 1 for c in text]


def test_build_case_hits_exact_token_target_and_records_spans():
    mod = _load()
    case = mod.build_case(
        FakeTokenizer(),
        target_tokens=12000,
        case_name="synthetic",
        needles=(
            ("MARK_A", "code-a", 0.10),
            ("MARK_B", "code-b", 0.75),
        ),
    )
    assert case["prompt_tokens"] == 12000
    assert len(case["prompt_token_ids"]) == 12000
    q0, q1 = case["query_span"]
    assert q1 == 12000
    assert q0 < q1
    assert len(case["needles"]) == 2
    for needle in case["needles"]:
        assert 0 <= needle["start"] < needle["end"] <= q0


def test_repeat_to_length_is_exact_and_deterministic():
    mod = _load()
    assert mod._repeat_to_length([1, 2, 3], 8) == [1, 2, 3, 1, 2, 3, 1, 2]
