from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXL3 = ROOT / "src" / "vllm_exl3" / "exl3.py"


def test_loader_trace_is_diagnostic_only_and_records_exact_boundaries():
    src = EXL3.read_text()
    assert 'VLLM_EXL3_LOAD_TRACE_PATH' in src
    assert 'FIRST_EXL3_COPY_BEFORE' in src
    assert 'ALL_WEIGHTS_LOADED_BEFORE_POSTLOAD' in src
    assert 'GENERIC_COPY_WALL_S' in src
    assert 'DIRECT_TRELLIS_COPY_WALL_S' in src
    assert 'DIRECT_TRELLIS_PREP_WALL_S' in src
    assert 'proc_io' in src
    assert 'ru_majflt' in src


def test_loader_trace_times_blocking_copy_without_changing_copy_mode():
    src = EXL3.read_text()
    assert 'dest.copy_(src, non_blocking=False)' in src
    assert 'arena[idx].copy_(src, non_blocking=False)' in src
    assert 'time.perf_counter()' in src
    assert 'non_blocking=True' not in src[src.index("def _copy_weight_blocking"):src.index("def _direct_fill_trellis_slot")]


def test_loader_trace_final_boundary_is_called_from_postload_hook():
    src = EXL3.read_text()
    assert src.count("_load_trace_weights_complete(layer)") >= 2
    assert 'vLLM calls quant_method.process_weights_after_loading only after model' in src
