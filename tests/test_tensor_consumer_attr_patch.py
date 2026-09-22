from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCHER = ROOT / "tools" / "patch_vllm_weight_utils_tensor_attr.py"


def test_tensor_attr_patch_times_get_and_yield_resume_without_policy_change():
    src = PATCHER.read_text()
    assert "VLLM_EXL3_TENSOR_ATTR_PATH" in src
    assert "_exl3_get_t0 = time.perf_counter()" in src
    assert "_exl3_cons_t0 = time.perf_counter()" in src
    assert "yield name, param" in src
    assert "time.process_time()" in src
    assert "param.numel()" in src


def test_tensor_attr_patch_aggregates_in_memory_and_flushes_once():
    src = PATCHER.read_text()
    assert "\"by_suffix\"" in src
    assert "\"by_scope\"" in src
    assert "\"by_size_bin\"" in src
    assert "\"by_scope_suffix\"" in src
    assert "\"top_consumers\"" in src
    assert "_exl3_tensor_attr_flush(" in src
    assert "out.write_text(json.dumps(payload, indent=2)" in src


def test_tensor_attr_patch_is_lazy_only_and_does_not_optimize_loader():
    src = PATCHER.read_text()
    assert "requires default lazy safetensors strategy" in src
    assert "_prefetch_checkpoint(" not in src
    assert "pin_memory=True" not in src
    assert "non_blocking=True" not in src
    assert "ThreadPoolExecutor" not in src


def test_tensor_attr_patch_has_byte_exact_backup_contract():
    src = PATCHER.read_text()
    assert ".exl3_tensor_attr.orig" in src
    assert "shutil.copyfile(path, backup)" in src
    assert "--check-only" in src
