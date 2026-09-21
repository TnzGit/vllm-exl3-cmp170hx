import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "kvmem_qsa_make_turns.py"


def _load():
    spec = importlib.util.spec_from_file_location("turn_gen", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(c) % 251 + 1 for c in text]


def test_turn_suite_reuses_identical_history_and_query_start():
    mod = _load()
    suite = mod.build_suite(FakeTokenizer(), context=20000, history_reserve=512)
    assert len(suite["turns"]) == 6
    starts = {tuple(t["history_span"]) for t in suite["turns"]}
    assert starts == {(0, 19488)}
    query_starts = {t["query_span"][0] for t in suite["turns"]}
    assert query_starts == {19488}
    histories = [
        t["prompt_token_ids"][:19488]
        for t in suite["turns"]
    ]
    assert all(h == histories[0] for h in histories[1:])


def test_first_and_last_turn_are_exact_query_replay():
    mod = _load()
    suite = mod.build_suite(FakeTokenizer(), context=20000, history_reserve=512)
    first = suite["turns"][0]
    last = suite["turns"][-1]
    assert first["query_text"] == last["query_text"]
    assert (
        first["prompt_token_ids"][first["query_span"][0]:]
        == last["prompt_token_ids"][last["query_span"][0]:]
    )


def test_facts_are_inside_fixed_history():
    mod = _load()
    suite = mod.build_suite(FakeTokenizer(), context=20000, history_reserve=512)
    for fact in suite["facts"]:
        assert 0 <= fact["start"] < fact["end"] <= suite["history_tokens"]
