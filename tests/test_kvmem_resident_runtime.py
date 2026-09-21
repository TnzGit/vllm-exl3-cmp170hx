from vllm_exl3.kvmem_resident import (
    ResidentGeometry,
    StickyResidentCoordinator,
    deterministic_fresh_set,
    logical_blocks_from_selected_tokens,
)


def test_fresh_set_ties_use_logical_id():
    scores = {9: 10, 7: 10, 4: 10, 2: 10, 1: 10}
    got = deterministic_fresh_set(scores, {0}, capacity_blocks=4)
    assert got == (0, 1, 2, 4)


def test_bootstrap_assigns_stable_dense_slots():
    c = StickyResidentCoordinator(capacity_blocks=4)
    tr = c.bootstrap({8, 2, 5, 1})
    assert c.logical_to_slot == {1: 0, 2: 1, 5: 2, 8: 3}
    assert [(m.logical_block, m.physical_slot) for m in tr.stage_in] == [
        (1, 0), (2, 1), (5, 2), (8, 3)
    ]


def test_equal_score_candidate_does_not_churn():
    c = StickyResidentCoordinator(
        capacity_blocks=4,
        replacement_fraction=0.50,
    )
    c.bootstrap({0, 1, 2, 3})
    before = c.logical_to_slot
    tr = c.update(
        desired_blocks={0, 1, 2, 4},
        mandatory_blocks={0},
        scores={0: 100, 1: 20, 2: 10, 3: 5, 4: 5},
    )
    assert tr.query_replacements == 0
    assert not tr.stage_in
    assert c.logical_to_slot == before


def test_strictly_higher_candidate_reuses_victim_slot():
    c = StickyResidentCoordinator(
        capacity_blocks=4,
        replacement_fraction=0.25,
    )
    c.bootstrap({0, 1, 2, 3})
    victim_slot = c.logical_to_slot[3]
    tr = c.update(
        desired_blocks={0, 1, 2, 4},
        mandatory_blocks={0},
        scores={0: 100, 1: 20, 2: 10, 3: 1, 4: 9},
    )
    assert tr.query_replacements == 1
    assert tr.query_replacement_cap == 1
    assert [(m.logical_block, m.physical_slot) for m in tr.stage_out] == [
        (3, victim_slot)
    ]
    assert [(m.logical_block, m.physical_slot) for m in tr.stage_in] == [
        (4, victim_slot)
    ]
    assert c.logical_to_slot[4] == victim_slot


def test_mandatory_admission_is_outside_query_cap():
    c = StickyResidentCoordinator(
        capacity_blocks=4,
        replacement_fraction=0.0,
    )
    c.bootstrap({0, 1, 2, 3})
    tr = c.update(
        desired_blocks={0, 1, 2, 9},
        mandatory_blocks={0, 9},
        scores={0: 100, 1: 20, 2: 10, 3: 1, 9: 0},
    )
    assert tr.query_replacement_cap == 0
    assert tr.query_replacements == 0
    assert tr.mandatory_replacements == 1
    assert 9 in c.resident_blocks


def test_retained_blocks_keep_physical_slots():
    c = StickyResidentCoordinator(
        capacity_blocks=5,
        replacement_fraction=0.20,
    )
    c.bootstrap({0, 1, 2, 3, 4})
    before = c.logical_to_slot
    c.update(
        desired_blocks={0, 1, 2, 3, 9},
        mandatory_blocks={0},
        scores={0: 100, 1: 50, 2: 40, 3: 30, 4: 1, 9: 20},
    )
    after = c.logical_to_slot
    for logical in (0, 1, 2, 3):
        assert after[logical] == before[logical]


def test_materialized_block_table_uses_negative_for_holes():
    c = StickyResidentCoordinator(capacity_blocks=3)
    c.bootstrap({0, 2, 5})
    table = c.materialize_block_table(logical_page_count=7)
    assert table == [0, -1, 1, -1, -1, 2, -1]


def test_update_is_deterministic_under_score_ties():
    def run():
        c = StickyResidentCoordinator(
            capacity_blocks=6,
            replacement_fraction=0.50,
        )
        c.bootstrap({0, 1, 2, 3, 4, 5})
        return c.update(
            desired_blocks={0, 1, 6, 7, 8, 9},
            mandatory_blocks={0},
            scores={
                0: 100,
                1: 50,
                2: 10,
                3: 10,
                4: 10,
                5: 10,
                6: 20,
                7: 20,
                8: 20,
                9: 20,
            },
        )

    first = run()
    second = run()
    assert first == second


def test_selected_tokens_to_logical_blocks_filters_future_rows():
    assert logical_blocks_from_selected_tokens(
        [0, 1, 255, 256, 511, -1, 1000],
        block_size=256,
        history_limit=900,
    ) == (0, 1)


def test_region_geometry_expands_to_qsa_page_table():
    c = StickyResidentCoordinator(capacity_blocks=2)
    c.bootstrap({1, 3})
    geom = ResidentGeometry(region_tokens=256, page_tokens=16)
    table = c.materialize_qsa_page_table(
        logical_tokens=1024,
        geometry=geom,
    )
    assert len(table) == 64

    # Sorted bootstrap maps logical region 1 -> resident slot 0,
    # and logical region 3 -> resident slot 1.
    assert table[0:16] == [-1] * 16
    assert table[16:32] == list(range(0, 16))
    assert table[32:48] == [-1] * 16
    assert table[48:64] == list(range(16, 32))


def test_region_geometry_rejects_nonintegral_page_span():
    try:
        ResidentGeometry(region_tokens=256, page_tokens=48)
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:
        raise AssertionError("expected ValueError")
