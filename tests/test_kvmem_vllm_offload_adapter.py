from pathlib import Path

from vllm_exl3.kvmem_vllm_offload import (
    TransferObservation,
    stable_page_digest,
)


ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "src" / "vllm_exl3" / "kvmem_vllm_offload.py"
PROBE = ROOT / "tools" / "kvmem_k1t_transfer_probe.py"


def test_stable_page_digest_is_deterministic_and_position_sensitive():
    a = stable_page_digest("req-a", 123)
    b = stable_page_digest("req-a", 123)
    c = stable_page_digest("req-a", 124)
    d = stable_page_digest("req-b", 123)
    assert a == b
    assert a != c
    assert a != d
    assert len(a) == 32


def test_transfer_observation_bandwidth_helpers():
    obs = TransferObservation(
        job_id=7,
        transfer_bytes=1024**3,
        event_seconds=0.25,
        wall_seconds=0.50,
    )
    assert obs.event_gib_s == 4.0
    assert obs.wall_gib_s == 2.0
    assert obs.prepare_seconds == 0.0
    assert obs.submit_seconds == 0.0
    assert obs.wait_seconds == 0.0
    assert obs.finish_seconds == 0.0


def test_adapter_reuses_generic_manager_worker_contract():
    src = ADAPTER.read_text()
    assert "CPUOffloadingManager" in src
    assert "CPUOffloadingWorker" in src
    assert "prepare_store" in src
    assert "prepare_load" in src
    assert "submit_store" in src
    assert "submit_load" in src
    assert "complete_store" in src
    assert "complete_load" in src
    assert "GPULoadStoreSpec" in src
    assert ".copy_(" not in src


def test_probe_routes_through_adapter_not_direct_tensor_copy():
    src = PROBE.read_text()
    assert "VllmCPUPageBacking" in src
    assert "backing.publish(" in src
    assert "backing.stage_in(" in src
    assert "CPUOffloadingWorker(" not in src
    assert "CPUOffloadingManager(" not in src


def test_probe_primary_geometry_is_explicit():
    src = PROBE.read_text()
    assert 'default=240000' in src
    assert 'default=65536' in src
    assert 'default=256' in src
    assert 'default=16' in src
    assert 'default=0.05' in src
    assert 'default=12' in src
    assert 'default=2' in src
    assert 'default=256' in src
    assert 'default=2' in src



def test_single_tensor_factory_builds_real_model_canonical_page_contract():
    src = ADAPTER.read_text()
    assert "def single_tensor_cpu_backing(" in src
    assert "CanonicalKVCaches(" in src
    assert "CanonicalKVCacheTensor(" in src
    assert "CanonicalKVCacheRef(" in src
    assert "tensor.is_contiguous()" in src
    assert "actual_page_bytes" in src
    assert "VllmCPUPageBacking(" in src
