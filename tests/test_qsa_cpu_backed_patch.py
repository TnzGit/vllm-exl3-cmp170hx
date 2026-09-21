import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "patch_vllm_qwen4_exp" / "patch_vllm_qsa_cpu_backed.py"


def _load():
    spec = importlib.util.spec_from_file_location("q2b_patch", SCRIPT)
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


def test_q2b_patch_uses_real_generic_cpu_offload_contract(tmp_path):
    mod = _load()
    path = tmp_path / "qsa.py"
    path.write_text(_source())
    assert mod.patch(path) == "patched"
    out = path.read_text()
    assert mod.MARKER in out
    assert "single_tensor_cpu_backing" in out
    assert "backing.publish(" in out
    assert "backing.stage_in(" in out
    assert "cpu_backing_all_present" in out
    assert "bootstrap_all_pages_exact" in out
    assert "vllm_generic_cpu_offload" in out
    assert "publication_staging_pages" in out
    assert "transfer_tensor_page_count" in out
    assert "resident_cache[:resident_count].zero_()" in out
    assert "backing.stage_in(" in out
    assert "resident_cache[start:end].copy_(" not in out
    compile(out, str(path), "exec")


def test_q2b_patch_does_not_directly_construct_generic_worker_or_manager():
    src = SCRIPT.read_text()
    assert "CPUOffloadingWorker(" not in src
    assert "CPUOffloadingManager(" not in src
    assert "persistent_topk" not in src
    assert "scheduler.py" not in src


def test_q2b_patch_default_off_check_is_read_only(tmp_path):
    mod = _load()
    path = tmp_path / "qsa.py"
    before = _source()
    path.write_text(before)
    status = mod.patch(path, check_only=True)
    assert "no files changed" in status
    assert path.read_text() == before
