"""Shared selection-policy instrumentation for K1-Q2C attribution runs.

Both the full-KV control and the bounded Q2C implementation call this module.
Keeping the policy in one place makes the B/C comparison a test of storage and
addressing rather than two independently copied masking implementations.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

import torch


def _resident_tensor(
    resident_pages: Sequence[int], device: torch.device
) -> torch.Tensor:
    return torch.tensor(resident_pages, dtype=torch.int64, device=device)


def apply_progressive_visibility(
    plan: dict[str, Any],
    selected: torch.Tensor,
    positions: torch.Tensor,
    *,
    apply_mask: bool,
    resident_pages_tensor: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Apply, or only observe, the frozen Q2C progressive visibility rule.

    The returned integer moments are compact deterministic fingerprints of the
    selected logical token IDs after policy application. They are diagnostic
    evidence, not a cryptographic digest.
    """
    if selected.ndim != 2:
        raise RuntimeError("Q2C attribution requires a two-dimensional selection")
    pos = positions.to(device=selected.device, dtype=torch.int64).reshape(-1)
    if selected.shape[0] != pos.numel():
        raise RuntimeError("Q2C selection/position row count mismatch")

    page_tokens = int(plan["page_tokens"])
    logical = selected.clamp_min(0).to(torch.int64)
    valid = selected >= 0
    first_query_page = int(pos.min().item()) // page_tokens if pos.numel() else 0
    pages = torch.div(logical, page_tokens, rounding_mode="floor")
    resident = resident_pages_tensor
    if resident is None:
        resident = _resident_tensor(plan["resident_pages"], selected.device)
    elif resident.device != selected.device or resident.dtype != torch.int64:
        raise RuntimeError("Q2C resident tensor device/dtype mismatch")
    keep_resident = torch.isin(pages, resident)
    historical = logical < int(plan["active_from_pos"])
    processed_history = pages < first_query_page
    would_drop = valid & historical & processed_history & ~keep_resident

    valid_before = int(valid.sum().item())
    historical_total = int((valid & historical).sum().item())
    would_drop_count = int(would_drop.sum().item())
    valid_pages = pages[valid]
    historical_pages = pages[valid & historical]
    nonresident_historical_pages = pages[valid & historical & ~keep_resident]
    current_pages = torch.div(pos, page_tokens, rounding_mode="floor")

    def _unique_count(values: torch.Tensor) -> int:
        return int(torch.unique(values).numel()) if values.numel() else 0

    unique_pages_before = _unique_count(valid_pages)
    unique_historical_pages_before = _unique_count(historical_pages)
    unique_nonresident_historical_pages_before = _unique_count(
        nonresident_historical_pages
    )
    unique_with_writes_before = _unique_count(
        torch.cat((valid_pages, current_pages))
    )
    if apply_mask:
        selected.masked_fill_(would_drop, -1)

    post_valid = selected >= 0
    post = selected.clamp_min(0).to(torch.int64)
    post_values = post * post_valid.to(torch.int64)
    post_pages = torch.div(post, page_tokens, rounding_mode="floor")[post_valid]
    unique_pages_after = _unique_count(post_pages)
    unique_with_writes_after = _unique_count(
        torch.cat((post_pages, current_pages))
    )
    row_weights = torch.arange(
        1, selected.shape[0] + 1, dtype=torch.int64, device=selected.device
    ).reshape(-1, 1)
    col_weights = torch.arange(
        1, selected.shape[1] + 1, dtype=torch.int64, device=selected.device
    ).reshape(1, -1)

    return {
        "first_pos": int(pos.min().item()) if pos.numel() else -1,
        "last_pos": int(pos.max().item()) if pos.numel() else -1,
        "query_rows": int(pos.numel()),
        "selection_width": int(selected.shape[1]),
        "first_query_page": first_query_page,
        "apply_rows": int((pos >= int(plan["apply_min_pos"])).sum().item()),
        "valid_before": valid_before,
        "historical_total": historical_total,
        "would_drop": would_drop_count,
        "actual_dropped": would_drop_count if apply_mask else 0,
        "historical_kept_if_masked": historical_total - would_drop_count,
        "unique_pages_before": unique_pages_before,
        "unique_historical_pages_before": unique_historical_pages_before,
        "unique_nonresident_historical_pages_before": (
            unique_nonresident_historical_pages_before
        ),
        "unique_pages_with_current_writes_before": unique_with_writes_before,
        "unique_pages_after": unique_pages_after,
        "unique_pages_with_current_writes_after": unique_with_writes_after,
        "post_valid": int(post_valid.sum().item()),
        "post_sum": int(post_values.sum().item()),
        "post_square_sum": int((post_values * post_values).sum().item()),
        "post_row_weighted_sum": int((post_values * row_weights).sum().item()),
        "post_col_weighted_sum": int((post_values * col_weights).sum().item()),
    }


def write_attribution_event(payload: dict[str, Any]) -> None:
    path = os.environ.get("VLLM_QWEN_KVMEM_Q2C_ATTRIB_STATS_PATH")
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, separators=(",", ":")) + "\n")


def tensor_bit_fingerprint(tensor: torch.Tensor) -> dict[str, int]:
    """Return compact exact-bit moments for a BF16/FP16 output tensor."""
    if tensor.dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(
            f"Q2C attribution expected a 16-bit attention output, got {tensor.dtype}"
        )
    bits = tensor.detach().contiguous().view(torch.int16).to(torch.int64)
    metrics = torch.stack(
        (bits.sum(), (bits * bits).sum(), bits.abs().sum())
    ).cpu().tolist()
    return {
        "output_bit_count": int(bits.numel()),
        "output_bit_sum": int(metrics[0]),
        "output_bit_square_sum": int(metrics[1]),
        "output_bit_abs_sum": int(metrics[2]),
    }


FINGERPRINT_FIELDS = (
    "query_rows",
    "selection_width",
    "first_query_page",
    "valid_before",
    "historical_total",
    "would_drop",
    "historical_kept_if_masked",
    "unique_pages_before",
    "unique_historical_pages_before",
    "unique_nonresident_historical_pages_before",
    "unique_pages_with_current_writes_before",
    "unique_pages_after",
    "unique_pages_with_current_writes_after",
    "post_valid",
    "post_sum",
    "post_square_sum",
    "post_row_weighted_sum",
    "post_col_weighted_sum",
    "output_bit_count",
    "output_bit_sum",
    "output_bit_square_sum",
    "output_bit_abs_sum",
)
