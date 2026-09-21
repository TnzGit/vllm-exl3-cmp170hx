from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CELL = ROOT / "tools" / "r0_k3_cell.py"


def test_inline_parity_flattens_speculative_multi_token_chunks():
    src = CELL.read_text(encoding="utf-8")
    assert "def _flatten_token_pieces(seq):" in src
    assert "if isinstance(piece, list):" in src
    assert "flat.extend(piece)" in src
    assert "ref_tokens = _flatten_token_pieces(ref_tokens)" in src
    assert 'got = _flatten_token_pieces(cells[0]["token_pieces"])' in src


def test_inline_parity_remains_token_length_sensitive():
    src = CELL.read_text(encoding="utf-8")
    assert 'len(ref_tokens) == len(got)' in src
    assert '"first_mismatch_index": first' in src
