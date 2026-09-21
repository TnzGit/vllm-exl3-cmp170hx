#!/usr/bin/env python3
"""Replay K1B primary shadows through the runtime resident coordinator.

This is the contract bridge from research planner evidence to runtime code.
It requires exact logical resident-set and stage-in-count agreement with the
previous K1B 5% / 256-token / 64K reference summary.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vllm_exl3.kvmem_resident import StickyResidentCoordinator

TOOLS = ROOT / "tools"
STICKY_SCRIPT = TOOLS / "kvmem_qsa_sticky_replay.py"

_spec = importlib.util.spec_from_file_location("k1b_sticky", STICKY_SCRIPT)
_k1b = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_k1b)

BLOCK_SIZE = 256
BUDGET = 65536
REPLACEMENT = 0.05
POLICY_KEY = "b256_budget65536"
CAP_KEY = "replace_5"


def replay_context(ctx_spec: dict, shadow_dir: Path, reference: dict) -> dict:
    turns, _layers = _k1b.load_context_turns(ctx_spec, shadow_dir)
    coordinator = StickyResidentCoordinator(
        capacity_blocks=BUDGET // BLOCK_SIZE,
        replacement_fraction=REPLACEMENT,
    )

    runtime_states = []
    runtime_transitions = []

    for index, tr in enumerate(turns):
        fresh, block_votes, mandatory = _k1b._fresh_set(
            tr["turn"],
            tr["votes"],
            block_size=BLOCK_SIZE,
            budget=BUDGET,
        )
        if index == 0:
            transition = coordinator.bootstrap(fresh)
        else:
            transition = coordinator.update(
                fresh,
                mandatory,
                block_votes,
            )
            runtime_transitions.append(
                {
                    "from": turns[index - 1]["turn"]["name"],
                    "to": tr["turn"]["name"],
                    "stage_in_blocks": len(transition.stage_in),
                    "stage_out_blocks": len(transition.stage_out),
                    "query_replacements": transition.query_replacements,
                    "mandatory_replacements": transition.mandatory_replacements,
                }
            )
        runtime_states.append(
            {
                "index": tr["turn"]["index"],
                "name": tr["turn"]["name"],
                "resident_blocks": list(coordinator.resident_blocks),
            }
        )

    ref_policy = reference["policies"][POLICY_KEY][CAP_KEY]
    ref_turns = ref_policy["turns"]
    ref_transitions = ref_policy["transitions"]

    turn_checks = []
    for got, expected in zip(runtime_states, ref_turns, strict=True):
        exact = got["resident_blocks"] == expected["resident_blocks"]
        turn_checks.append(
            {
                "name": got["name"],
                "resident_exact_match": exact,
                "runtime_count": len(got["resident_blocks"]),
                "reference_count": len(expected["resident_blocks"]),
            }
        )

    transition_checks = []
    for got, expected in zip(
        runtime_transitions,
        ref_transitions,
        strict=True,
    ):
        stage_in_match = (
            got["stage_in_blocks"] == expected["stage_in_blocks"]
        )
        stage_out_match = (
            got["stage_out_blocks"] == expected["stage_out_blocks"]
        )
        transition_checks.append(
            {
                "from": got["from"],
                "to": got["to"],
                "runtime_stage_in_blocks": got["stage_in_blocks"],
                "reference_stage_in_blocks": expected["stage_in_blocks"],
                "runtime_stage_out_blocks": got["stage_out_blocks"],
                "reference_stage_out_blocks": expected["stage_out_blocks"],
                "stage_in_exact_match": stage_in_match,
                "stage_out_exact_match": stage_out_match,
            }
        )

    return {
        "context": int(ctx_spec["context"]),
        "turn_checks": turn_checks,
        "transition_checks": transition_checks,
        "all_turn_sets_exact": all(
            row["resident_exact_match"] for row in turn_checks
        ),
        "all_transition_counts_exact": all(
            row["stage_in_exact_match"] and row["stage_out_exact_match"]
            for row in transition_checks
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k1a-dir", type=Path, required=True)
    ap.add_argument("--k1b-summary", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    manifest = json.loads(
        (args.k1a_dir / "turns" / "manifest.json").read_text()
    )
    reference = json.loads(args.k1b_summary.read_text())
    ref_by_context = {
        int(row["context"]): row for row in reference["contexts"]
    }

    contexts = []
    for ctx_spec in manifest["contexts"]:
        ctx = int(ctx_spec["context"])
        contexts.append(
            replay_context(
                ctx_spec,
                args.k1a_dir / "shadows",
                ref_by_context[ctx],
            )
        )

    all_exact = all(
        row["all_turn_sets_exact"] and row["all_transition_counts_exact"]
        for row in contexts
    )
    result = {
        "schema": 1,
        "policy": {
            "block_size": BLOCK_SIZE,
            "budget_tokens": BUDGET,
            "replacement_fraction": REPLACEMENT,
        },
        "contexts": contexts,
        "runtime_matches_k1b_reference": all_exact,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if all_exact else 3


if __name__ == "__main__":
    raise SystemExit(main())
