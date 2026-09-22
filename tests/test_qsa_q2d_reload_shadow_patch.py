import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCHER = (
    ROOT
    / "tools"
    / "patch_vllm_qwen4_exp"
    / "patch_vllm_qsa_q2d_reload_shadow.py"
)
FIXTURE = (
    ROOT.parent
    / "evidence"
    / "kvmem-k1q2c-runtime-ownership-ccec5bf-live2"
    / "qsa.base.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("q2d_reload_patch", PATCHER)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _target(mod):
    if FIXTURE.exists():
        return FIXTURE
    import vllm

    return Path(vllm.__file__).resolve().parent / mod.TARGET


def test_reload_patch_matches_qsa_and_is_check_only():
    mod = _load()
    target = _target(mod)
    before = target.read_bytes()
    status = mod.patch(target, check_only=True)
    assert target.read_bytes() == before
    assert "anchors/postconditions validated" in status


def test_reload_patch_keeps_stock_cache_as_reference():
    mod = _load()
    src = _target(mod).read_text()
    out = src.replace(mod.IMPORT_ANCHOR, mod.IMPORT_BLOCK, 1)
    out = out.replace(mod.HELPER_ANCHOR, mod.HELPER, 1)
    out = out.replace(mod.INIT_ANCHOR, mod.INIT_BLOCK, 1)
    out = out.replace(mod.RUN_ANCHOR, mod.RUN_BLOCK, 1)
    assert mod.MARKER in out
    assert "_q2d_transfer_cache" in out
    assert "run_reload_shadow" in out
    assert "def get_kv_cache_spec" in out
    assert "qsa_scheduler_owned_transition" not in out
