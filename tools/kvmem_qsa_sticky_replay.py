#!/usr/bin/env python3
"""Replay K1A QSA shadows through a stateful sticky-residency planner.

The planner is intentionally simple and implementable:
- turn 0 uses the fresh QSA working set;
- later turns retain the previous resident set;
- missing mandatory blocks are always admitted;
- nonresident blocks from the fresh QSA set compete against resident blocks
  using current-turn vote counts;
- only a bounded fraction of the resident budget may be replaced per turn.

No ground-truth target information is used to choose blocks. Ground truth is
used only after planning to score recall.
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import math
import statistics
from pathlib import Path


TOOLS = Path(__file__).resolve().parent
CHURN_SCRIPT = TOOLS / "kvmem_qsa_churn_analyze.py"

_spec = importlib.util.spec_from_file_location("k1a_churn", CHURN_SCRIPT)
_k1a = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_k1a)

BLOCK_SIZES = (128, 256, 512)
BUDGETS = (32768, 65536)
REPLACEMENT_CAPS = (0.05, 0.10, 0.15, 0.20, 0.25)


def _block_votes(
    votes: collections.Counter[int],
    block_size: int,
) -> collections.Counter[int]:
    out: collections.Counter[int] = collections.Counter()
    for token, count in votes.items():
        out[token // block_size] += count
    return out


def _fresh_set(
    turn: dict,
    votes: collections.Counter[int],
    *,
    block_size: int,
    budget: int,
) -> tuple[set[int], collections.Counter[int], set[int]]:
    chosen, block_votes, mandatory = _k1a.working_set(
        turn,
        votes,
        block_size=block_size,
        budget=budget,
    )
    return chosen, block_votes, mandatory


def _sorted_candidates(
    candidates: set[int],
    score: collections.Counter[int],
) -> list[int]:
    return sorted(candidates, key=lambda b: (-score[b], b))


def _sorted_victims(
    victims: set[int],
    score: collections.Counter[int],
    fresh: set[int],
) -> list[int]:
    # Prefer evicting blocks that are not in the current fresh desired set.
    # Within each class evict the lowest current vote count first.
    return sorted(
        victims,
        key=lambda b: (
            b in fresh,
            score[b],
            -b,
        ),
    )


def sticky_update(
    previous: set[int],
    fresh: set[int],
    block_votes: collections.Counter[int],
    mandatory: set[int],
    *,
    capacity_blocks: int,
    replacement_fraction: float,
) -> tuple[set[int], dict]:
    if len(previous) != capacity_blocks:
        raise ValueError(
            f"previous resident set is not full: {len(previous)} != "
            f"{capacity_blocks}"
        )
    if len(fresh) != capacity_blocks:
        raise ValueError(
            f"fresh desired set is not full: {len(fresh)} != "
            f"{capacity_blocks}"
        )
    if not mandatory <= fresh:
        raise ValueError("fresh desired set does not contain mandatory blocks")

    resident = set(previous)
    stage_in: set[int] = set()
    stage_out: set[int] = set()

    # Mandatory changes are not charged against the query-specific cap.
    missing_mandatory = mandatory - resident
    for block in _sorted_candidates(missing_mandatory, block_votes):
        victims = resident - mandatory - missing_mandatory
        if not victims:
            victims = resident - mandatory
        if not victims:
            raise ValueError("no evictable block available for mandatory admission")
        victim = _sorted_victims(victims, block_votes, fresh)[0]
        resident.remove(victim)
        stage_out.add(victim)
        resident.add(block)
        stage_in.add(block)

    replace_cap = max(
        0,
        math.floor(capacity_blocks * replacement_fraction),
    )
    replacements = 0

    candidates = _sorted_candidates(fresh - resident, block_votes)
    for candidate in candidates:
        if replacements >= replace_cap:
            break
        victims = resident - mandatory
        if not victims:
            break
        victim = _sorted_victims(victims, block_votes, fresh)[0]

        # Hysteresis: only move if the current query actually scores the new
        # block higher than the weakest evictable resident block.
        if block_votes[candidate] <= block_votes[victim]:
            break

        resident.remove(victim)
        stage_out.add(victim)
        resident.add(candidate)
        stage_in.add(candidate)
        replacements += 1

    if len(resident) != capacity_blocks:
        raise AssertionError("sticky planner changed resident capacity")
    if not mandatory <= resident:
        raise AssertionError("sticky planner lost a mandatory block")

    return resident, {
        "stage_in": stage_in,
        "stage_out": stage_out,
        "query_replacements": replacements,
        "query_replacement_cap_blocks": replace_cap,
    }


def _target_rows(
    turn: dict,
    resident: set[int],
    block_size: int,
) -> list[dict]:
    rows = []
    for fact in turn["target_facts"]:
        required = _k1a.fact_blocks(fact, block_size)
        hit = required & resident
        rows.append(
            {
                "marker": fact["marker"],
                "required_blocks": sorted(required),
                "selected_required_blocks": sorted(hit),
                "block_recall": (
                    len(hit) / len(required) if required else 1.0
                ),
                "all_required_blocks_selected": required <= resident,
            }
        )
    return rows


def analyze_policy(
    turns: list[dict],
    *,
    block_size: int,
    budget: int,
    replacement_fraction: float,
    bytes_per_token: int,
) -> dict:
    capacity = budget // block_size
    states = []
    previous: set[int] | None = None

    for tr in turns:
        fresh, block_votes, mandatory = _fresh_set(
            tr["turn"],
            tr["votes"],
            block_size=block_size,
            budget=budget,
        )
        if previous is None:
            resident = set(fresh)
            update = {
                "stage_in": set(fresh),
                "stage_out": set(),
                "query_replacements": len(fresh),
                "query_replacement_cap_blocks": capacity,
            }
        else:
            resident, update = sticky_update(
                previous,
                fresh,
                block_votes,
                mandatory,
                capacity_blocks=capacity,
                replacement_fraction=replacement_fraction,
            )

        fresh_intersection = resident & fresh
        fresh_union = resident | fresh
        targets = _target_rows(tr["turn"], resident, block_size)
        states.append(
            {
                "index": tr["turn"]["index"],
                "name": tr["turn"]["name"],
                "resident": resident,
                "fresh": fresh,
                "mandatory": mandatory,
                "targets": targets,
                "fresh_jaccard": (
                    len(fresh_intersection) / len(fresh_union)
                    if fresh_union
                    else 1.0
                ),
                "fresh_coverage": len(fresh_intersection) / len(fresh),
                "stage_in": update["stage_in"],
                "stage_out": update["stage_out"],
                "query_replacements": update["query_replacements"],
                "query_replacement_cap_blocks": update[
                    "query_replacement_cap_blocks"
                ],
            }
        )
        previous = resident

    transitions = []
    bytes_per_block = bytes_per_token * block_size
    for prev, cur in zip(states, states[1:]):
        a = prev["resident"]
        b = cur["resident"]
        retained = a & b
        union = a | b
        stage_in = cur["stage_in"]
        stage_out = cur["stage_out"]
        stage_in_bytes = len(stage_in) * bytes_per_block
        transitions.append(
            {
                "from": prev["name"],
                "to": cur["name"],
                "retained_blocks": len(retained),
                "stage_in_blocks": len(stage_in),
                "stage_out_blocks": len(stage_out),
                "stage_in_fraction_of_budget": len(stage_in) / capacity,
                "resident_jaccard": (
                    len(retained) / len(union) if union else 1.0
                ),
                "stage_in_tokens": len(stage_in) * block_size,
                "estimated_main_kv_stage_in_bytes": stage_in_bytes,
                "estimated_main_kv_stage_in_gib": stage_in_bytes / (1024**3),
                "query_replacements": cur["query_replacements"],
            }
        )

    target_rows = [
        target
        for state in states
        for target in state["targets"]
    ]
    stage_fracs = [
        x["stage_in_fraction_of_budget"] for x in transitions
    ]
    fresh_coverages = [
        s["fresh_coverage"] for s in states[1:]
    ]

    first = states[0]["resident"]
    replay = states[-1]["resident"]
    replay_union = first | replay
    replay_intersection = first & replay

    return {
        "block_size": block_size,
        "budget_tokens": budget,
        "budget_blocks": capacity,
        "replacement_fraction": replacement_fraction,
        "turns": [
            {
                "index": s["index"],
                "name": s["name"],
                "fresh_coverage": s["fresh_coverage"],
                "fresh_jaccard": s["fresh_jaccard"],
                "target_recall": (
                    sum(
                        int(x["all_required_blocks_selected"])
                        for x in s["targets"]
                    )
                    / len(s["targets"])
                    if s["targets"]
                    else 1.0
                ),
                "targets": s["targets"],
                "resident_blocks": sorted(s["resident"]),
            }
            for s in states
        ],
        "transitions": transitions,
        "summary": {
            "median_stage_in_fraction": (
                statistics.median(stage_fracs) if stage_fracs else 0.0
            ),
            "max_stage_in_fraction": (
                max(stage_fracs) if stage_fracs else 0.0
            ),
            "mean_fresh_coverage": (
                statistics.mean(fresh_coverages)
                if fresh_coverages
                else 1.0
            ),
            "min_fresh_coverage": (
                min(fresh_coverages) if fresh_coverages else 1.0
            ),
            "target_facts": len(target_rows),
            "target_facts_fully_recalled": sum(
                int(x["all_required_blocks_selected"])
                for x in target_rows
            ),
            "target_fact_recall_rate": (
                sum(
                    int(x["all_required_blocks_selected"])
                    for x in target_rows
                )
                / len(target_rows)
                if target_rows
                else 1.0
            ),
            "stateful_turn0_turn5_jaccard": (
                len(replay_intersection) / len(replay_union)
                if replay_union
                else 1.0
            ),
            "stateful_turn0_turn5_exact_set": first == replay,
        },
    }


def load_context_turns(
    ctx_spec: dict,
    shadow_dir: Path,
) -> tuple[list[dict], set[str]]:
    turns = []
    layers: set[str] = set()
    ctx = int(ctx_spec["context"])
    for turn_spec in ctx_spec["turns"]:
        turn = json.loads(Path(turn_spec["path"]).read_text())
        shadow = (
            shadow_dir
            / f"ctx{ctx}_turn_{turn['index']:02d}_{turn['name']}.jsonl"
        )
        records = _k1a.load_jsonl(shadow)
        ev = _k1a.extract_votes(turn, records)
        layers.update(ev["layers"])
        turns.append(
            {
                "turn": turn,
                "votes": ev["votes"],
                "targets": ev["targets"],
            }
        )
    return turns, layers


def choose_smallest_cap(contexts: list[dict]) -> dict:
    candidates = []
    for repl in REPLACEMENT_CAPS:
        per_ctx = [
            ctx["policies"]["b256_budget65536"][f"replace_{int(repl*100)}"]
            for ctx in contexts
        ]
        recall = min(
            p["summary"]["target_fact_recall_rate"] for p in per_ctx
        )
        median_stage = statistics.median(
            p["summary"]["median_stage_in_fraction"] for p in per_ctx
        )
        max_stage = max(
            p["summary"]["max_stage_in_fraction"] for p in per_ctx
        )
        mean_fresh = statistics.mean(
            p["summary"]["mean_fresh_coverage"] for p in per_ctx
        )
        row = {
            "replacement_fraction": repl,
            "min_target_recall": recall,
            "median_stage_in_fraction": median_stage,
            "max_stage_in_fraction": max_stage,
            "mean_fresh_coverage": mean_fresh,
            "target_recall_preserved": recall >= 0.99,
        }
        candidates.append(row)

    preserving = [
        x for x in candidates if x["target_recall_preserved"]
    ]
    selected = preserving[0] if preserving else None
    return {
        "candidates": candidates,
        "smallest_cap_preserving_target_recall": selected,
        "note": (
            "This is a shadow-planner signal, not a production quality gate. "
            "Ground-truth targets are used only to evaluate recall, never to "
            "select resident blocks."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--shadow-dir", type=Path, required=True)
    ap.add_argument("--model-config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    model_config = json.loads(args.model_config.read_text())
    results = []

    for ctx_spec in manifest["contexts"]:
        turns, layers = load_context_turns(ctx_spec, args.shadow_dir)
        geom = _k1a.kv_geometry(model_config, len(layers))
        policies = {}
        for block_size in BLOCK_SIZES:
            for budget in BUDGETS:
                key = f"b{block_size}_budget{budget}"
                policies[key] = {}
                for repl in REPLACEMENT_CAPS:
                    policies[key][f"replace_{int(repl*100)}"] = analyze_policy(
                        turns,
                        block_size=block_size,
                        budget=budget,
                        replacement_fraction=repl,
                        bytes_per_token=geom["main_kv_bytes_per_token"],
                    )
        results.append(
            {
                "context": int(ctx_spec["context"]),
                "history_tokens": int(ctx_spec["history_tokens"]),
                "layers": sorted(layers),
                "kv_geometry": geom,
                "policies": policies,
            }
        )

    result = {
        "schema": 1,
        "contexts": results,
        "primary": choose_smallest_cap(results),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
