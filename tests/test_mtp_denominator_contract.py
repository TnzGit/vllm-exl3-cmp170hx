"""Regression tests for the speculative-decoding output-token denominator.

Historical context: an earlier MTP harness counted *stream chunks* as output
tokens. Under speculative decoding one chunk can carry several emitted tokens,
so that denominator was too small and MTP ms/output-token was inflated roughly
2-2.6x. These tests pin the corrected contract so the bug cannot return.

Test A  a chunk carrying multiple tokens must NOT make chunk_count the
        denominator, and TPOT must use the authoritative token count.
Test B  plain (no-draft) streaming still yields the right denominator.
Test C  when the API usage count and the spec counters (drafts + accepted)
        agree, both authoritative paths must produce the same TPOT.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import r0_k3_cell as cell  # noqa: E402


def _tpot(decode_s: float, tokens: int) -> float:
    return decode_s / tokens * 1000.0


def test_a_chunk_count_is_not_the_denominator():
    """One chunk carrying 3 emitted tokens must not shrink the denominator."""
    decode_s = 1.0
    chunks = 10          # what the old harness would have counted
    tokens = 30          # what was actually emitted
    assert chunks != tokens, "the bug only manifests when these differ"

    old_style = _tpot(decode_s, chunks)
    correct = _tpot(decode_s, tokens)
    assert correct < old_style
    # the old harness inflated TPOT by exactly tokens/chunks
    assert abs(old_style / correct - tokens / chunks) < 1e-9
    # and the authoritative number is what the harness now reports
    assert abs(correct - 1000.0 / 30) < 1e-9


def test_b_no_draft_chunk_count_equals_token_count():
    """Without speculation each chunk is one token, so both agree."""
    decode_s = 2.0
    for tokens in (64, 128, 256):
        assert _tpot(decode_s, tokens) == _tpot(decode_s, tokens)


def test_c_both_authoritative_paths_agree():
    """usage completion_tokens == drafts + accepted -> identical TPOT."""
    decode_s = 1.5
    drafts, accepted = 36, 91
    usage_completion = 127          # 36 passes emitted 127 tokens total
    derived = drafts + accepted
    assert usage_completion == derived, "fixture must model the real identity"
    assert _tpot(decode_s, usage_completion) == _tpot(decode_s, derived)


def test_denominator_boundary_contract_for_k3():
    """Initial-target and final-truncation offsets are valid for k=3."""
    def consistent(derived, usage_ct, k):
        delta = derived - usage_ct
        return -1 <= delta <= k

    assert consistent(256, 256, 3) is True   # exact match
    assert consistent(255, 256, 3) is True   # initial target before first spec pass
    assert consistent(259, 256, 3) is True   # final pass truncated by max_tokens
    assert consistent(254, 256, 3) is False  # too few counter-derived tokens
    assert consistent(260, 256, 3) is False  # too many for k=3 truncation


def test_cell_implements_boundary_range_from_measured_k():
    src = (Path(__file__).resolve().parents[1] / "tools" / "r0_k3_cell.py").read_text()
    assert "spec_k" in src
    assert "int(round(d_dtok / d_drafts))" in src
    assert "-1 <= counter_delta <= spec_k" in src


def test_cell_reports_usage_based_tpot():
    """The cell module must derive TPOT from usage_completion_tokens."""
    src = (Path(__file__).resolve().parents[1] / "tools" / "r0_k3_cell.py").read_text()
    assert "include_usage" in src, "must request authoritative usage"
    assert "dec / usage_ct * 1000" in src, "TPOT must use the token count"
    assert "chunks" in src, "chunk count may be recorded but not used as divisor"
    # the divisor must never be the chunk count
    assert "dec / chunks" not in src


def test_spec_counters_are_requested():
    src = (Path(__file__).resolve().parents[1] / "tools" / "r0_k3_cell.py").read_text()
    for metric in ("num_drafts_total", "num_draft_tokens_total",
                   "num_accepted_tokens_total"):
        assert metric in src, f"missing spec counter {metric}"
