from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXL3 = ROOT / "src" / "vllm_exl3" / "exl3.py"


def test_pinned_stage_ab_is_opt_in_and_synchronous():
    src = EXL3.read_text()
    assert 'VLLM_EXL3_PINNED_STAGE_AB' in src
    assert '_PINNED_STAGE_AB = os.environ.get("VLLM_EXL3_PINNED_STAGE_AB", "0") == "1"' in src
    assert 'pin_memory=True' in src
    assert 'pinned.copy_(src, non_blocking=False)' in src
    assert 'arena[idx].copy_(pinned, non_blocking=False)' in src


def test_pinned_stage_ab_is_shape_stratified_by_arena_slot_parity():
    src = EXL3.read_text()
    assert 'use_pinned = _PINNED_STAGE_AB and (int(idx) & 1) == 1' in src
    assert 'DIRECT_TRELLIS_CONTROL_BYTES' in src
    assert 'DIRECT_TRELLIS_PINNED_BYTES' in src


def test_pinned_stage_buffer_is_reused_and_bounded_to_max_tensor():
    src = EXL3.read_text()
    assert '_PINNED_TRELLIS_STAGE' in src
    assert 'int(_PINNED_TRELLIS_STAGE.numel()) < numel' in src
    assert 'DIRECT_TRELLIS_PINNED_BUFFER_MAX_BYTES' in src


def test_pinned_stage_ab_keeps_source_reclaim_after_h2d():
    src = EXL3.read_text()
    direct = src[src.index("def _direct_fill_trellis_slot"):src.index("def _pack_trellis_arenas")]
    assert direct.index('arena[idx].copy_(pinned, non_blocking=False)') < direct.index('_madv_dontneed_cpu_tensor(src)')
    assert direct.index('arena[idx].copy_(src, non_blocking=False)') < direct.index('_madv_dontneed_cpu_tensor(src)')
