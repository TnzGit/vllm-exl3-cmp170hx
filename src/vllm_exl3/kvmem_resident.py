"""Deterministic sticky resident-set core for QSA/KVMem residency.

The module has no vLLM imports. It owns only logical block residency policy and
stable logical-block -> physical-slot mapping. Non-resident logical pages
materialize as -1 for QSA sparse-attention block tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Collection, Mapping, Sequence


ScoreMap = Mapping[int, int | float]


@dataclass(frozen=True, slots=True)
class ResidentMove:
    logical_block: int
    physical_slot: int


@dataclass(frozen=True, slots=True)
class ResidentTransition:
    retained_blocks: tuple[int, ...]
    stage_in: tuple[ResidentMove, ...]
    stage_out: tuple[ResidentMove, ...]
    mandatory_replacements: int
    query_replacements: int
    query_replacement_cap: int
    logical_to_slot: tuple[tuple[int, int], ...]

    @property
    def stage_in_blocks(self) -> tuple[int, ...]:
        return tuple(move.logical_block for move in self.stage_in)

    @property
    def stage_out_blocks(self) -> tuple[int, ...]:
        return tuple(move.logical_block for move in self.stage_out)


def _score(scores: ScoreMap, block: int) -> float:
    return float(scores.get(block, 0))


def deterministic_fresh_set(
    scores: ScoreMap,
    mandatory_blocks: Collection[int],
    *,
    capacity_blocks: int,
) -> tuple[int, ...]:
    """Build a stable fresh desired set: score descending, block id ascending."""

    if capacity_blocks <= 0:
        raise ValueError("capacity_blocks must be positive")
    mandatory = {int(x) for x in mandatory_blocks}
    if any(x < 0 for x in mandatory):
        raise ValueError("logical block ids must be non-negative")
    if len(mandatory) > capacity_blocks:
        raise ValueError("mandatory set exceeds resident capacity")

    candidates = {int(x) for x in scores}
    candidates.update(mandatory)
    if any(x < 0 for x in candidates):
        raise ValueError("logical block ids must be non-negative")

    chosen = set(mandatory)
    for block in sorted(
        candidates - chosen,
        key=lambda b: (-_score(scores, b), b),
    ):
        if len(chosen) >= capacity_blocks:
            break
        chosen.add(block)
    if len(chosen) != capacity_blocks:
        raise ValueError("not enough candidate blocks to fill resident capacity")
    return tuple(sorted(chosen))


class StickyResidentCoordinator:
    """Stateful sticky residency with deterministic tie handling.

    Mandatory admissions are not charged against the semantic-query replacement
    cap. Equal-score fresh candidates never replace resident blocks, which
    suppresses the SM80 persistent-topk tie-boundary churn observed in K1A.
    """

    def __init__(
        self,
        *,
        capacity_blocks: int,
        replacement_fraction: float = 0.05,
    ) -> None:
        if capacity_blocks <= 0:
            raise ValueError("capacity_blocks must be positive")
        if not 0.0 <= replacement_fraction <= 1.0:
            raise ValueError("replacement_fraction must be in [0, 1]")
        self.capacity_blocks = int(capacity_blocks)
        self.replacement_fraction = float(replacement_fraction)
        self._logical_to_slot: dict[int, int] = {}
        self._slot_to_logical: list[int | None] = [None] * self.capacity_blocks

    @property
    def initialized(self) -> bool:
        return len(self._logical_to_slot) == self.capacity_blocks

    @property
    def resident_blocks(self) -> tuple[int, ...]:
        return tuple(sorted(self._logical_to_slot))

    @property
    def logical_to_slot(self) -> dict[int, int]:
        return dict(self._logical_to_slot)

    def reset(self) -> None:
        self._logical_to_slot.clear()
        self._slot_to_logical = [None] * self.capacity_blocks

    def bootstrap(self, desired_blocks: Collection[int]) -> ResidentTransition:
        desired = {int(x) for x in desired_blocks}
        if any(x < 0 for x in desired):
            raise ValueError("logical block ids must be non-negative")
        if len(desired) != self.capacity_blocks:
            raise ValueError("bootstrap set must exactly fill resident capacity")
        if self._logical_to_slot:
            raise RuntimeError("coordinator is already initialized")

        new_map = {
            logical: slot for slot, logical in enumerate(sorted(desired))
        }
        self._logical_to_slot = new_map
        self._slot_to_logical = [None] * self.capacity_blocks
        for logical, slot in new_map.items():
            self._slot_to_logical[slot] = logical

        stage_in = tuple(
            ResidentMove(logical, slot)
            for logical, slot in sorted(new_map.items(), key=lambda x: x[1])
        )
        return ResidentTransition(
            retained_blocks=(),
            stage_in=stage_in,
            stage_out=(),
            mandatory_replacements=0,
            query_replacements=self.capacity_blocks,
            query_replacement_cap=self.capacity_blocks,
            logical_to_slot=tuple(sorted(new_map.items())),
        )

    @staticmethod
    def _candidate_order(candidates: set[int], scores: ScoreMap) -> list[int]:
        return sorted(candidates, key=lambda b: (-_score(scores, b), b))

    @staticmethod
    def _victim_order(
        resident: set[int],
        *,
        mandatory: set[int],
        fresh: set[int],
        scores: ScoreMap,
    ) -> list[int]:
        return sorted(
            resident - mandatory,
            key=lambda b: (
                b in fresh,
                _score(scores, b),
                -b,
            ),
        )

    @staticmethod
    def _replace(
        logical_to_slot: dict[int, int],
        slot_to_logical: list[int | None],
        *,
        victim: int,
        candidate: int,
    ) -> tuple[ResidentMove, ResidentMove]:
        slot = logical_to_slot.pop(victim)
        logical_to_slot[candidate] = slot
        slot_to_logical[slot] = candidate
        return ResidentMove(victim, slot), ResidentMove(candidate, slot)

    def update(
        self,
        desired_blocks: Collection[int],
        mandatory_blocks: Collection[int],
        scores: ScoreMap,
    ) -> ResidentTransition:
        if not self.initialized:
            raise RuntimeError("coordinator must be bootstrapped first")

        fresh = {int(x) for x in desired_blocks}
        mandatory = {int(x) for x in mandatory_blocks}
        if any(x < 0 for x in fresh | mandatory):
            raise ValueError("logical block ids must be non-negative")
        if len(fresh) != self.capacity_blocks:
            raise ValueError("desired set must exactly fill resident capacity")
        if not mandatory <= fresh:
            raise ValueError("mandatory blocks must be contained in desired set")

        original = set(self._logical_to_slot)
        logical_to_slot = dict(self._logical_to_slot)
        slot_to_logical = list(self._slot_to_logical)
        stage_in: list[ResidentMove] = []
        stage_out: list[ResidentMove] = []
        mandatory_replacements = 0

        for candidate in self._candidate_order(
            mandatory - set(logical_to_slot), scores
        ):
            victims = self._victim_order(
                set(logical_to_slot),
                mandatory=mandatory,
                fresh=fresh,
                scores=scores,
            )
            if not victims:
                raise RuntimeError("no evictable block for mandatory admission")
            out_move, in_move = self._replace(
                logical_to_slot,
                slot_to_logical,
                victim=victims[0],
                candidate=candidate,
            )
            stage_out.append(out_move)
            stage_in.append(in_move)
            mandatory_replacements += 1

        replacement_cap = floor(
            self.capacity_blocks * self.replacement_fraction
        )
        query_replacements = 0

        for candidate in self._candidate_order(
            fresh - set(logical_to_slot), scores
        ):
            if query_replacements >= replacement_cap:
                break
            victims = self._victim_order(
                set(logical_to_slot),
                mandatory=mandatory,
                fresh=fresh,
                scores=scores,
            )
            if not victims:
                break
            victim = victims[0]
            if _score(scores, candidate) <= _score(scores, victim):
                break

            out_move, in_move = self._replace(
                logical_to_slot,
                slot_to_logical,
                victim=victim,
                candidate=candidate,
            )
            stage_out.append(out_move)
            stage_in.append(in_move)
            query_replacements += 1

        if len(logical_to_slot) != self.capacity_blocks:
            raise AssertionError("resident capacity changed")
        if not mandatory <= set(logical_to_slot):
            raise AssertionError("mandatory block lost")
        if len(set(logical_to_slot.values())) != self.capacity_blocks:
            raise AssertionError("physical slot alias detected")
        if any(x is None for x in slot_to_logical):
            raise AssertionError("resident slot unexpectedly empty")

        self._logical_to_slot = logical_to_slot
        self._slot_to_logical = slot_to_logical

        return ResidentTransition(
            retained_blocks=tuple(sorted(original & set(logical_to_slot))),
            stage_in=tuple(stage_in),
            stage_out=tuple(stage_out),
            mandatory_replacements=mandatory_replacements,
            query_replacements=query_replacements,
            query_replacement_cap=replacement_cap,
            logical_to_slot=tuple(sorted(logical_to_slot.items())),
        )

    def materialize_block_table(self, *, logical_page_count: int) -> list[int]:
        if logical_page_count < 0:
            raise ValueError("logical_page_count must be non-negative")
        table = [-1] * logical_page_count
        for logical, slot in self._logical_to_slot.items():
            if logical < logical_page_count:
                table[logical] = slot
        return table


def logical_blocks_from_selected_tokens(
    selected_tokens: Sequence[int],
    *,
    block_size: int,
    history_limit: int | None = None,
) -> tuple[int, ...]:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    blocks: set[int] = set()
    for token in selected_tokens:
        token = int(token)
        if token < 0:
            continue
        if history_limit is not None and token >= history_limit:
            continue
        blocks.add(token // block_size)
    return tuple(sorted(blocks))
