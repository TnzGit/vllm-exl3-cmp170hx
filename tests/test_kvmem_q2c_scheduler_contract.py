from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

from vllm_exl3.kvmem_qsa_scheduler_contract import (
    QSAResidentContractSpec,
    QSAResidentRegistryProbeManager,
    make_qsa_resident_contract_spec,
    q2c_geometry,
    register_qsa_resident_contract,
)


def test_q2c_geometry_matches_q2b_physical_contract():
    g = q2c_geometry(161_000)
    assert g["resident_page_tokens"] == 16
    assert g["page_size_bytes"] == 32_768
    assert g["history_pages"] == 4096
    assert g["active_pages"] == 64
    assert g["physical_page_cap"] == 4160
    assert g["bounded_bytes_per_layer"] == 136_314_880
    assert g["bounded_mib_per_layer"] == 130.0
    assert g["bounded_gib_all_qsa_layers"] == 1.5234375


def test_q2c_logical_table_is_not_capped_to_physical_pages():
    g = q2c_geometry(161_000)
    assert g["logical_table_pages"] == 10063
    assert g["logical_table_pages"] > g["physical_page_cap"]
    assert g["logical_minus_physical_pages"] == 5903


def test_q2c_240k_keeps_same_physical_cap_but_larger_logical_history():
    a = q2c_geometry(161_000)
    b = q2c_geometry(240_000)
    assert b["logical_table_pages"] == 15000
    assert b["physical_page_cap"] == a["physical_page_cap"] == 4160
    assert b["bounded_bytes_per_layer"] == a["bounded_bytes_per_layer"]
    assert b["full_to_bounded_ratio"] > a["full_to_bounded_ratio"]


def test_q2c_custom_spec_registry_contract():
    register_qsa_resident_contract()
    spec = make_qsa_resident_contract_spec()
    assert isinstance(spec, QSAResidentContractSpec)
    assert spec.prefix_cacheable is False
    assert KVCacheSpecRegistry.get_manager_class(spec) is QSAResidentRegistryProbeManager
    assert KVCacheSpecRegistry.get_uniform_type_base_spec(spec) is QSAResidentContractSpec
