import importlib.util
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "kvmem_qsa_sticky_replay.py"


def _load():
    spec = importlib.util.spec_from_file_location("sticky", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def test_sticky_update_respects_capacity_and_replacement_cap():
    mod = _load()
    previous = set(range(10))
    mandatory = {0, 1}
    fresh = {0, 1, 2, 3, 4, 10, 11, 12, 13, 14}
    scores = Counter({
        0: 100, 1: 100, 2: 50, 3: 40, 4: 30,
        10: 90, 11: 80, 12: 70, 13: 60, 14: 55,
    })
    resident, update = mod.sticky_update(
        previous,
        fresh,
        scores,
        mandatory,
        capacity_blocks=10,
        replacement_fraction=0.20,
    )
    assert len(resident) == 10
    assert mandatory <= resident
    assert len(update["stage_in"]) <= 2
    assert update["query_replacements"] <= 2
    assert len(update["stage_in"]) == len(update["stage_out"])


def test_sticky_hysteresis_refuses_lower_scoring_candidate():
    mod = _load()
    previous = {0, 1, 2, 3}
    mandatory = {0}
    fresh = {0, 1, 2, 4}
    scores = Counter({0: 100, 1: 90, 2: 80, 3: 70, 4: 60})
    resident, update = mod.sticky_update(
        previous,
        fresh,
        scores,
        mandatory,
        capacity_blocks=4,
        replacement_fraction=0.50,
    )
    assert resident == previous
    assert not update["stage_in"]


def test_target_scoring_does_not_feed_planner():
    src = SCRIPT.read_text()
    planner = src[src.index("def sticky_update"):src.index("def _target_rows")]
    assert "target_facts" not in planner
    assert "marker" not in planner
