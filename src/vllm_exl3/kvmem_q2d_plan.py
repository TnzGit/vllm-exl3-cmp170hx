"""Frozen plan validation for the Q2D CPU reload correctness oracle."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any


def validate_reload_plan(plan: dict[str, Any]) -> dict[str, Any]:
    if int(plan.get("schema", 0)) != 1:
        raise ValueError("Q2D reload plan schema mismatch")
    if plan.get("mode") != "qsa_cpu_reload_shadow":
        raise ValueError("Q2D reload plan mode mismatch")
    required = {
        "page_tokens": 16,
        "physical_page_count": 4160,
        "scheduler_chunk_tokens": 1024,
        "query_row_batch": 64,
        "staging_pages": 128,
    }
    for field, expected in required.items():
        if int(plan.get(field, 0)) != expected:
            raise ValueError(f"Q2D reload plan {field} must equal {expected}")
    if int(plan.get("cpu_page_count", 0)) < math.ceil(
        int(plan["max_model_len"]) / int(plan["page_tokens"])
    ):
        raise ValueError("Q2D CPU page count cannot cover max model length")
    if float(plan.get("allclose_atol", -1)) != 0.02:
        raise ValueError("Q2D allclose atol mismatch")
    if float(plan.get("allclose_rtol", -1)) != 0.01:
        raise ValueError("Q2D allclose rtol mismatch")
    return dict(plan)


def load_reload_plan(path: str | os.PathLike[str]) -> dict[str, Any]:
    return validate_reload_plan(json.loads(Path(path).read_text()))
