import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCHER = ROOT / "tools" / "patch_vllm_qwen4_exp" / "patch_vllm_qsa_q2c_runtime.py"


def _load():
    spec = importlib.util.spec_from_file_location("q2c_patch", PATCHER)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_q2c_patch_matches_installed_qsa_without_modifying_it():
    import vllm
    mod = _load()
    root = Path(vllm.__file__).resolve().parent
    target = root / mod.TARGET
    before = target.read_bytes()
    status = mod.patch(target, check_only=True)
    after = target.read_bytes()
    assert before == after
    assert "anchors/postconditions validated" in status


def test_q2c_patch_contains_scheduler_owned_contract():
    mod = _load()
    import vllm
    target = Path(vllm.__file__).resolve().parent / mod.TARGET
    src = target.read_text()
    out = src.replace(mod.IMPORT_ANCHOR, mod.IMPORT_BLOCK, 1)
    out = out.replace(mod.HELPER_ANCHOR, mod.HELPER, 1)
    out = out.replace(mod.INIT_ANCHOR, mod.INIT_BLOCK, 1)
    out = out.replace(mod.BIND_ANCHOR, mod.BIND_BLOCK, 1)
    out = out.replace(mod.SPEC_ANCHOR, mod.SPEC_BLOCK, 1)
    out = out.replace(mod.RUN_ANCHOR, mod.RUN_BLOCK, 1)
    assert mod.MARKER in out
    assert "make_qsa_runtime_spec" in out
    assert "_q2c_dedicated_kv_cache" in out
    assert "register_buffer(" in out
    assert "_q2c_placeholder_shape" in out
    assert "self.kv_cache = dedicated" in out
    assert "Q2C dedicated KV must be CUDA BF16" in out
    assert "_q2c_publish_history" in out
    assert "_q2c_restore_history_from_cpu" in out
    assert "main_metadata.slot_mapping" in out
    assert "main_metadata.block_table" in out
    assert "scheduler_real_pages" in out
    assert "single_tensor_cpu_backing" in out
    assert "independent resident" not in mod.__doc__.lower()
