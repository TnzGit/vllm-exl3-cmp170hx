from vllm_exl3.kvmem_resident import (
    ResidentGeometry,
    StickyResidentCoordinator,
)
from vllm_exl3.kvmem_transfer import (
    expand_resident_transition,
    page_bytes_per_layer,
    qsa_main_kv_bytes_per_token,
)


def test_expand_transition_reuses_exact_physical_page_span():
    c = StickyResidentCoordinator(
        capacity_blocks=4,
        replacement_fraction=0.25,
    )
    c.bootstrap({0, 1, 2, 3})
    tr = c.update(
        desired_blocks={0, 1, 2, 9},
        mandatory_blocks={0},
        scores={0: 100, 1: 20, 2: 10, 3: 1, 9: 99},
    )
    geom = ResidentGeometry(region_tokens=256, page_tokens=16)
    exp = expand_resident_transition(tr, geom)

    assert len(exp.stage_in_pages) == 16
    assert len(exp.stage_out_pages) == 16
    assert {m.physical_page for m in exp.stage_in_pages} == {
        m.physical_page for m in exp.stage_out_pages
    }

    slot = tr.stage_in[0].physical_slot
    assert [m.physical_page for m in exp.stage_in_pages] == list(
        range(slot * 16, slot * 16 + 16)
    )
    assert [m.logical_page for m in exp.stage_in_pages] == list(
        range(9 * 16, 9 * 16 + 16)
    )


def test_primary_geometry_matches_measured_kv_bytes():
    assert qsa_main_kv_bytes_per_token(
        qsa_layers=12,
        num_kv_heads=2,
        head_dim=256,
        dtype_bytes=2,
    ) == 24576

    assert page_bytes_per_layer(
        page_tokens=16,
        num_kv_heads=2,
        head_dim=256,
        dtype_bytes=2,
    ) == 32768


def test_primary_5pct_transition_is_72_mib():
    c = StickyResidentCoordinator(
        capacity_blocks=256,
        replacement_fraction=0.05,
    )
    c.bootstrap(set(range(256)))

    incoming = set(range(1000, 1012))
    desired = set(range(244)) | incoming
    scores = {block: 1000 - block for block in range(256)}
    scores.update({block: 10000 - block for block in incoming})
    scores[0] = 100000

    tr = c.update(
        desired_blocks=desired,
        mandatory_blocks={0},
        scores=scores,
    )
    assert tr.query_replacement_cap == 12
    assert tr.query_replacements == 12

    geom = ResidentGeometry(region_tokens=256, page_tokens=16)
    exp = expand_resident_transition(tr, geom)
    assert len(exp.stage_in_pages) == 12 * 16

    all_layer_page_bytes = 12 * 32768
    assert exp.stage_in_bytes(all_layer_page_bytes) == 72 * 1024 * 1024
