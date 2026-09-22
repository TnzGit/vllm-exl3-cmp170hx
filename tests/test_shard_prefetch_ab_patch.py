from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCHER = ROOT / "tools" / "patch_vllm_weight_utils_shard_prefetch_ab.py"
SUMMARY = ROOT / "tools" / "r0_shard_prefetch_ab_summary.py"


def test_shard_prefetch_patch_uses_vllm_primitive_and_same_shard_join():
    src=PATCHER.read_text()
    assert '_prefetch_checkpoint(path, block_size)' in src
    assert '_exl3_prefetch_thread.join()' in src
    assert 'VLLM_EXL3_SHARD_PREFETCH_AB' in src
    assert 'VLLM_EXL3_SHARD_PREFETCH_AB_STATS_PATH' in src
    assert 'full-checkpoint prefetch' in src


def test_shard_prefetch_patch_is_odd_even_single_boot_ab():
    src=PATCHER.read_text()
    assert '(_exl3_ab_idx & 1)' in src
    assert '"prefetch"' in src
    assert '"control"' in src
    assert '"file_bytes"' in src
    assert '"majflt"' in src
    assert 'startswith("model-")' in src
    assert '"excluded"' in src
    assert '"eligible"' in src
    assert '"/proc/self/io"' in src
    assert '"io": io' in src
    assert '"delta": _exl3_delta' in src

    summary = SUMMARY.read_text()
    assert '.get("io",{}).get("read_bytes",0)' in summary


def test_shard_prefetch_patch_has_restore_safe_backup_contract():
    src=PATCHER.read_text()
    assert '.exl3_shard_prefetch_ab.orig' in src
    assert 'stale backup exists' in src
    assert '--check-only' in src
