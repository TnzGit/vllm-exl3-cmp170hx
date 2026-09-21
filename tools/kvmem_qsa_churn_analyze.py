#!/usr/bin/env python3
"""Analyze QSA shadow resident-set churn across fixed-history query turns."""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path


BLOCK_SIZES = (128, 256, 512)
BUDGETS = (32768, 65536)
SINK_TOKENS = 512
RECENT_TOKENS = 4096
BF16_BYTES = 2


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON") from exc
    if not rows:
        raise ValueError(f"empty shadow file: {path}")
    return rows


def mandatory_blocks(query_start: int, block_size: int) -> set[int]:
    out = set(
        range(
            0,
            (min(SINK_TOKENS, query_start) + block_size - 1) // block_size,
        )
    )
    recent_start = max(0, query_start - RECENT_TOKENS)
    if recent_start < query_start:
        out.update(
            range(
                recent_start // block_size,
                (query_start - 1) // block_size + 1,
            )
        )
    return out


def fact_blocks(fact: dict, block_size: int) -> set[int]:
    start = int(fact["start"])
    end = int(fact["end"])
    if end <= start:
        return set()
    return set(range(start // block_size, (end - 1) // block_size + 1))


def extract_votes(turn: dict, records: list[dict]) -> dict:
    q0, q1 = map(int, turn["query_span"])
    votes: collections.Counter[int] = collections.Counter()
    direct: collections.Counter[int] = collections.Counter()
    layers: set[str] = set()
    query_rows = 0
    records_used = 0

    for rec in records:
        if int(rec.get("schema", 0)) != 1 or rec.get("skip_topk"):
            continue
        used = False
        layer = str(rec.get("layer_name", ""))
        for row in rec.get("rows") or []:
            pos = int(row["pos"])
            if not (q0 <= pos < q1):
                continue
            query_rows += 1
            used = True
            layers.add(layer)
            for token in row.get("selected") or []:
                token = int(token)
                if 0 <= token < q0:
                    votes[token] += 1
                    direct[token] += 1
        if used:
            records_used += 1

    targets = []
    for fact in turn["target_facts"]:
        start, end = int(fact["start"]), int(fact["end"])
        hit_votes = sum(v for tok, v in direct.items() if start <= tok < end)
        targets.append(
            {
                "marker": fact["marker"],
                "start": start,
                "end": end,
                "direct_qsa_token_hit": hit_votes > 0,
                "direct_vote_count": int(hit_votes),
            }
        )

    return {
        "votes": votes,
        "layers": layers,
        "query_rows": query_rows,
        "records_used": records_used,
        "targets": targets,
    }


def working_set(
    turn: dict,
    votes: collections.Counter[int],
    *,
    block_size: int,
    budget: int,
) -> tuple[set[int], collections.Counter[int], set[int]]:
    q0 = int(turn["query_span"][0])
    block_votes: collections.Counter[int] = collections.Counter()
    for token, count in votes.items():
        block_votes[token // block_size] += count

    ranked = [
        block
        for block, _ in sorted(
            block_votes.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )
    ]
    cap = budget // block_size
    mandatory = mandatory_blocks(q0, block_size)
    if len(mandatory) > cap:
        raise ValueError(
            f"mandatory set exceeds budget: {len(mandatory)} > {cap}"
        )
    chosen = set(mandatory)
    for block in ranked:
        if len(chosen) >= cap:
            break
        chosen.add(block)
    return chosen, block_votes, mandatory


def kv_geometry(config: dict, qsa_layers: int) -> dict:
    kv_heads = int(config["num_key_value_heads"])
    head_dim = int(
        config.get("head_dim")
        or int(config["hidden_size"]) // int(config["num_attention_heads"])
    )
    bytes_per_token = (
        qsa_layers
        * 2  # K + V
        * kv_heads
        * head_dim
        * BF16_BYTES
    )
    return {
        "qsa_layers_observed": qsa_layers,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "dtype": "bfloat16",
        "bytes_per_element": BF16_BYTES,
        "main_kv_bytes_per_token": bytes_per_token,
        "formula": (
            "qsa_layers * 2(K+V) * num_key_value_heads * head_dim * 2(BF16)"
        ),
    }


def analyze_context(
    ctx_spec: dict,
    *,
    shadow_dir: Path,
    model_config: dict,
) -> dict:
    turn_results = []
    layer_union: set[str] = set()

    for turn_spec in ctx_spec["turns"]:
        turn = json.loads(Path(turn_spec["path"]).read_text())
        shadow = shadow_dir / f"ctx{ctx_spec['context']}_turn_{turn['index']:02d}_{turn['name']}.jsonl"
        records = load_jsonl(shadow)
        ev = extract_votes(turn, records)
        layer_union.update(ev["layers"])
        turn_results.append(
            {
                "turn": turn,
                "shadow_path": str(shadow),
                "records_total": len(records),
                "records_used": ev["records_used"],
                "query_rows": ev["query_rows"],
                "layers": sorted(ev["layers"]),
                "votes": ev["votes"],
                "targets": ev["targets"],
            }
        )

    geom = kv_geometry(model_config, len(layer_union))
    policies = {}

    for block_size in BLOCK_SIZES:
        for budget in BUDGETS:
            key = f"b{block_size}_budget{budget}"
            states = []
            for tr in turn_results:
                chosen, block_votes, mandatory = working_set(
                    tr["turn"],
                    tr["votes"],
                    block_size=block_size,
                    budget=budget,
                )
                target_rows = []
                for fact in tr["turn"]["target_facts"]:
                    required = fact_blocks(fact, block_size)
                    hit = required & chosen
                    target_rows.append(
                        {
                            "marker": fact["marker"],
                            "required_blocks": sorted(required),
                            "selected_required_blocks": sorted(hit),
                            "block_recall": (
                                len(hit) / len(required) if required else 1.0
                            ),
                            "all_required_blocks_selected": required <= chosen,
                        }
                    )
                states.append(
                    {
                        "index": tr["turn"]["index"],
                        "name": tr["turn"]["name"],
                        "selected_blocks": chosen,
                        "selected_block_count": len(chosen),
                        "mandatory_block_count": len(mandatory),
                        "voted_block_count": len(block_votes),
                        "targets": target_rows,
                    }
                )

            transitions = []
            for prev, cur in zip(states, states[1:]):
                a = prev["selected_blocks"]
                b = cur["selected_blocks"]
                retained = a & b
                stage_in = b - a
                stage_out = a - b
                union = a | b
                bytes_per_block = (
                    geom["main_kv_bytes_per_token"] * block_size
                )
                stage_in_bytes = len(stage_in) * bytes_per_block
                stage_out_bytes = len(stage_out) * bytes_per_block
                transitions.append(
                    {
                        "from": prev["name"],
                        "to": cur["name"],
                        "retained_blocks": len(retained),
                        "stage_in_blocks": len(stage_in),
                        "stage_out_blocks": len(stage_out),
                        "stage_in_fraction_of_budget": (
                            len(stage_in) / (budget // block_size)
                            if budget // block_size
                            else 0.0
                        ),
                        "stage_out_fraction_of_prev": (
                            len(stage_out) / len(a) if a else 0.0
                        ),
                        "jaccard": (
                            len(retained) / len(union) if union else 1.0
                        ),
                        "stage_in_tokens": len(stage_in) * block_size,
                        "stage_out_tokens": len(stage_out) * block_size,
                        "estimated_main_kv_stage_in_bytes": stage_in_bytes,
                        "estimated_main_kv_stage_out_bytes": stage_out_bytes,
                        "estimated_main_kv_stage_in_gib": (
                            stage_in_bytes / (1024**3)
                        ),
                        "estimated_main_kv_stage_out_gib": (
                            stage_out_bytes / (1024**3)
                        ),
                    }
                )

            # turn 0 and turn 5 are intentionally the exact same query text.
            first = states[0]["selected_blocks"]
            replay = states[-1]["selected_blocks"]
            replay_union = first | replay
            replay_intersection = first & replay

            stage_in_fracs = [
                float(x["stage_in_fraction_of_budget"]) for x in transitions
            ]
            jaccards = [float(x["jaccard"]) for x in transitions]
            all_targets = [
                target
                for state in states
                for target in state["targets"]
            ]
            policies[key] = {
                "block_size": block_size,
                "budget_tokens": budget,
                "budget_blocks": budget // block_size,
                "turns": [
                    {
                        **{
                            k: v
                            for k, v in state.items()
                            if k != "selected_blocks"
                        },
                        "selected_blocks": sorted(state["selected_blocks"]),
                    }
                    for state in states
                ],
                "transitions": transitions,
                "summary": {
                    "median_stage_in_fraction": (
                        statistics.median(stage_in_fracs)
                        if stage_in_fracs
                        else 0.0
                    ),
                    "max_stage_in_fraction": (
                        max(stage_in_fracs) if stage_in_fracs else 0.0
                    ),
                    "mean_transition_jaccard": (
                        statistics.mean(jaccards) if jaccards else 1.0
                    ),
                    "same_query_replay_jaccard": (
                        len(replay_intersection) / len(replay_union)
                        if replay_union
                        else 1.0
                    ),
                    "same_query_replay_exact_set": first == replay,
                    "target_facts": len(all_targets),
                    "target_facts_fully_recalled": sum(
                        int(x["all_required_blocks_selected"])
                        for x in all_targets
                    ),
                    "target_fact_recall_rate": (
                        sum(
                            int(x["all_required_blocks_selected"])
                            for x in all_targets
                        )
                        / len(all_targets)
                        if all_targets
                        else 1.0
                    ),
                },
            }

    direct_targets = [
        target
        for tr in turn_results
        for target in tr["targets"]
    ]
    return {
        "context": int(ctx_spec["context"]),
        "history_tokens": int(ctx_spec["history_tokens"]),
        "layers": sorted(layer_union),
        "kv_geometry": geom,
        "turns": [
            {
                "index": tr["turn"]["index"],
                "name": tr["turn"]["name"],
                "prompt_tokens": tr["turn"]["prompt_tokens"],
                "query_span": tr["turn"]["query_span"],
                "records_total": tr["records_total"],
                "records_used": tr["records_used"],
                "query_rows": tr["query_rows"],
                "layers_used": len(tr["layers"]),
                "unique_history_tokens_voted": len(tr["votes"]),
                "total_history_votes": int(sum(tr["votes"].values())),
                "targets": tr["targets"],
            }
            for tr in turn_results
        ],
        "direct_target_hits": sum(
            int(t["direct_qsa_token_hit"]) for t in direct_targets
        ),
        "direct_target_count": len(direct_targets),
        "direct_target_hit_rate": (
            sum(int(t["direct_qsa_token_hit"]) for t in direct_targets)
            / len(direct_targets)
            if direct_targets
            else 0.0
        ),
        "policies": policies,
    }


def classify(primary: list[dict]) -> dict:
    medians = [
        x["policies"]["b256_budget65536"]["summary"][
            "median_stage_in_fraction"
        ]
        for x in primary
    ]
    maxima = [
        x["policies"]["b256_budget65536"]["summary"]["max_stage_in_fraction"]
        for x in primary
    ]
    replays = [
        x["policies"]["b256_budget65536"]["summary"][
            "same_query_replay_jaccard"
        ]
        for x in primary
    ]
    recalls = [
        x["policies"]["b256_budget65536"]["summary"][
            "target_fact_recall_rate"
        ]
        for x in primary
    ]
    median_churn = statistics.median(medians) if medians else 1.0
    max_churn = max(maxima) if maxima else 1.0
    min_replay = min(replays) if replays else 0.0
    min_recall = min(recalls) if recalls else 0.0

    if (
        median_churn <= 0.25
        and max_churn <= 0.50
        and min_replay >= 0.99
        and min_recall >= 0.99
    ):
        label = "LOW_CHURN"
    elif (
        median_churn <= 0.50
        and max_churn <= 0.75
        and min_replay >= 0.95
        and min_recall >= 0.99
    ):
        label = "MODERATE_CHURN"
    else:
        label = "HIGH_CHURN_OR_UNSTABLE"

    return {
        "primary_policy": "b256_budget65536",
        "classification": label,
        "median_of_context_median_stage_in_fraction": median_churn,
        "max_stage_in_fraction_across_contexts": max_churn,
        "min_same_query_replay_jaccard": min_replay,
        "min_target_fact_recall_rate": min_recall,
        "thresholds": {
            "LOW_CHURN": {
                "median_stage_in_fraction_max": 0.25,
                "max_stage_in_fraction_max": 0.50,
                "same_query_replay_jaccard_min": 0.99,
                "target_fact_recall_rate_min": 0.99,
            },
            "MODERATE_CHURN": {
                "median_stage_in_fraction_max": 0.50,
                "max_stage_in_fraction_max": 0.75,
                "same_query_replay_jaccard_min": 0.95,
                "target_fact_recall_rate_min": 0.99,
            },
        },
        "note": (
            "Classification describes planner churn only; it is not a "
            "production-performance or quality qualification."
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
    contexts = [
        analyze_context(
            spec,
            shadow_dir=args.shadow_dir,
            model_config=model_config,
        )
        for spec in manifest["contexts"]
    ]
    result = {
        "schema": 1,
        "contexts": contexts,
        "planner_summary": classify(contexts),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
