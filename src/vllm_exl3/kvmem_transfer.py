"""Pure transfer planning helpers for K1 target-only residency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .kvmem_resident import ResidentGeometry, ResidentMove, ResidentTransition


@dataclass(frozen=True, slots=True)
class PageMove:
    """One logical KV page loaded into one physical resident GPU page."""

    logical_page: int
    physical_page: int


@dataclass(frozen=True, slots=True)
class ExpandedTransfer:
    """Expanded page-level form of one resident transition."""

    stage_in_pages: tuple[PageMove, ...]
    stage_out_pages: tuple[PageMove, ...]
    pages_per_region: int

    def stage_in_bytes(self, bytes_per_page_all_layers: int) -> int:
        if bytes_per_page_all_layers < 0:
            raise ValueError("bytes_per_page_all_layers must be non-negative")
        return len(self.stage_in_pages) * bytes_per_page_all_layers


def _expand_moves(
    moves: Iterable[ResidentMove],
    geometry: ResidentGeometry,
) -> tuple[PageMove, ...]:
    pages: list[PageMove] = []
    ppr = geometry.pages_per_region
    for move in moves:
        logical0 = move.logical_block * ppr
        physical0 = move.physical_slot * ppr
        for offset in range(ppr):
            pages.append(
                PageMove(
                    logical_page=logical0 + offset,
                    physical_page=physical0 + offset,
                )
            )
    return tuple(pages)


def expand_resident_transition(
    transition: ResidentTransition,
    geometry: ResidentGeometry,
) -> ExpandedTransfer:
    """Expand region-level stage-in/out moves to physical KV-page moves."""

    stage_in = _expand_moves(transition.stage_in, geometry)
    stage_out = _expand_moves(transition.stage_out, geometry)

    in_phys = [move.physical_page for move in stage_in]
    out_phys = [move.physical_page for move in stage_out]
    if len(in_phys) != len(set(in_phys)):
        raise ValueError("stage-in physical page alias detected")
    if len(out_phys) != len(set(out_phys)):
        raise ValueError("stage-out physical page alias detected")

    # A replacement reuses the victim's physical page span.
    if sorted(in_phys) != sorted(out_phys):
        raise ValueError(
            "stage-in and stage-out physical page sets must match for replacements"
        )

    return ExpandedTransfer(
        stage_in_pages=stage_in,
        stage_out_pages=stage_out,
        pages_per_region=geometry.pages_per_region,
    )


def qsa_main_kv_bytes_per_token(
    *,
    qsa_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int,
) -> int:
    """Main K+V bytes per logical token across all QSA layers."""

    vals = (qsa_layers, num_kv_heads, head_dim, dtype_bytes)
    if any(v <= 0 for v in vals):
        raise ValueError("KV geometry values must be positive")
    return qsa_layers * 2 * num_kv_heads * head_dim * dtype_bytes


def page_bytes_per_layer(
    *,
    page_tokens: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int,
) -> int:
    if any(v <= 0 for v in (page_tokens, num_kv_heads, head_dim, dtype_bytes)):
        raise ValueError("KV page geometry values must be positive")
    return page_tokens * 2 * num_kv_heads * head_dim * dtype_bytes
