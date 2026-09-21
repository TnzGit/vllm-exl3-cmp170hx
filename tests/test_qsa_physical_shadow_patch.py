import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "patch_vllm_qwen4_exp" / "patch_vllm_qsa_physical_shadow.py"


def _load():
    spec = importlib.util.spec_from_file_location("q2_patch", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def _source():
    return '''from __future__ import annotations

import torch
from .indexer_qsa import QSAIndexer

class Owner:
    def run(self, impl, key, value, output, query, selected, main_metadata, side_metadata, num_tokens):
        impl = cast(Qwen4ExpQSAFlashAttentionImpl, self.impl)
        impl.do_kv_cache_update(
            self,
            key,
            value,
            self.kv_cache,
            main_metadata.slot_mapping,
        )
        impl.forward_qsa(
            self,
            query,
            key,
            value,
            self.kv_cache,
            main_metadata,
            output,
            token_to_req=side_metadata.token_to_req,
        )
'''


def test_q2_patch_default_off_and_dual_attention_contract(tmp_path):
    mod = _load()
    path = tmp_path / "qsa.py"
    path.write_text(_source())
    assert mod.patch(path) == "patched"
    out = path.read_text()
    assert mod.MARKER in out
    assert "VLLM_QWEN_KVMEM_PHYSICAL_PLAN" in out
    assert "_kvmem_bootstrap_resident" in out
    assert "_kvmem_active_slot_mapping" in out
    assert "torch.equal(ref_rows, resident_out)" in out
    assert "qsa_sparse_paged_attention" in out
    assert "_kvmem_plan is None" in out
    compile(out, str(path), "exec")


def test_q2_patch_check_only_is_read_only(tmp_path):
    mod = _load()
    path = tmp_path / "qsa.py"
    before = _source()
    path.write_text(before)
    status = mod.patch(path, check_only=True)
    assert "no files changed" in status
    assert path.read_text() == before


def test_q2_patch_does_not_change_topk_or_scheduler():
    src = SCRIPT.read_text()
    assert "persistent_topk" not in src
    assert "scheduler.py" not in src
    assert "cache_manager" not in src
