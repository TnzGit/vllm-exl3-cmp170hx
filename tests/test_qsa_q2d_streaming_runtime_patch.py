import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCHER = (
    ROOT / "tools" / "patch_vllm_qwen4_exp"
    / "patch_vllm_qsa_q2d_streaming_runtime.py"
)
FIXTURE = (
    ROOT.parent / "evidence" / "kvmem-k1q2c-runtime-ownership-ccec5bf-live2"
    / "qsa.base.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("q2d_stream_patch", PATCHER)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _target(mod):
    if FIXTURE.exists():
        return FIXTURE
    import vllm
    return Path(vllm.__file__).resolve().parent / mod.TARGET


def test_streaming_patch_matches_stock_qsa_check_only():
    mod = _load()
    target = _target(mod)
    before = target.read_bytes()
    assert "anchors/postconditions" in mod.patch(target, check_only=True)
    assert target.read_bytes() == before


def test_streaming_patch_has_partitioned_buffers_spec_and_worker():
    mod = _load()
    src = _target(mod).read_text()
    out = src.replace(mod.IMPORT_ANCHOR, mod.IMPORT_BLOCK, 1)
    out = out.replace(mod.HELPER_ANCHOR, mod.HELPER, 1)
    out = out.replace(mod.INIT_ANCHOR, mod.INIT_BLOCK, 1)
    out = out.replace(mod.BIND_ANCHOR, mod.BIND_BLOCK, 1)
    out = out.replace(mod.SPEC_ANCHOR, mod.SPEC_BLOCK, 1)
    out = out.replace(mod.RUN_ANCHOR, mod.RUN_BLOCK, 1)
    assert mod.MARKER in out
    assert "make_qsa_streaming_spec" in out
    assert "_q2d_dedicated_kv_cache" in out
    assert "_q2d_staging" in out
    assert "run_streaming_runtime" in out
