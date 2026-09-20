from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import struct
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "r0_bench_dense_exl3.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("r0_bench_dense_exl3", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_fake_safetensors(path: Path, entries: dict[str, list[int]]) -> None:
    header = {
        key: {
            "dtype": "I16",
            "shape": shape,
            "data_offsets": [0, 0],
        }
        for key, shape in entries.items()
    }
    raw = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)


def test_catalog_recovers_k_dims_and_codebook_flags(tmp_path):
    mod = _load_module()
    shard = "model-00001-of-00001.safetensors"
    k4 = "language_model.layers.0.linear_attn.in_proj_qkv"
    k5 = "language_model.lm_head"
    entries = {
        k4 + ".trellis": [160, 384, 64],
        k4 + ".suh": [2560],
        k4 + ".svh": [6144],
        k4 + ".mul1": [],
        k5 + ".trellis": [160, 1024, 80],
        k5 + ".suh": [2560],
        k5 + ".svh": [16384],
        k5 + ".mcg": [],
        k5 + ".mul1": [],
        # Routed tensors must not appear in the dense catalog.
        "language_model.layers.0.mlp.experts.0.gate_proj.trellis": [160, 40, 48],
        "language_model.mtp.layers.0.linear_attn.in_proj_qkv.trellis": [160, 384, 64],
    }
    _write_fake_safetensors(tmp_path / shard, entries)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: shard for key in entries}}),
        encoding="utf-8",
    )

    rows = {row.base: row for row in mod.catalog(tmp_path)}
    assert set(rows) == {k4, k5}
    assert rows[k4].k == 4
    assert rows[k4].in_features == 2560
    assert rows[k4].out_features == 6144
    assert rows[k4].has_mul1 is True
    assert rows[k4].has_mcg is False
    assert rows[k5].k == 5
    assert rows[k5].has_mul1 is True
    assert rows[k5].has_mcg is True


def test_choose_families_uses_requested_order_and_deduplicates(tmp_path):
    mod = _load_module()
    families = [
        mod.TensorFamily("a.linear_attn.in_proj_qkv", "x", 4, 2560, 6144, False, True),
        mod.TensorFamily("b.lm_head", "x", 5, 2560, 16384, True, True),
    ]
    chosen = mod.choose_families(
        families,
        ["lm_head", "linear_attn.in_proj_qkv", "lm_head"],
    )
    assert [row.base for row in chosen] == [
        "b.lm_head",
        "a.linear_attn.in_proj_qkv",
    ]


def test_variant_matrix_keeps_current_ampere_dispatch_as_control():
    mod = _load_module()

    assert mod.VARIANTS["current"]["EXL3_INT8_GEMV"] == "2"
    assert mod.VARIANTS["current"]["EXL3_INT8_GEMV_MAX_K"] == "5"
    assert mod.VARIANTS["current"]["EXL3_GEMV"] == "1"
    assert mod.VARIANTS["int8_k4_cap"]["EXL3_INT8_GEMV_MAX_K"] == "4"
    assert mod.VARIANTS["fp16_force_shuffle"]["EXL3_GEMV_SMEM"] == "0"
    assert mod.VARIANTS["fp16_force_smem"]["EXL3_GEMV_SMEM"] == "1"


def test_k4_projection_gate_uses_post_coop_calibration(capsys):
    mod = _load_module()
    results = []
    for family, current, candidate in (
        ("linear_attn.in_proj_qkv", 10.0, 7.0),
        ("linear_attn.in_proj_z", 20.0, 14.0),
        ("linear_attn.out_proj", 30.0, 21.0),
    ):
        results.append(
            {
                "base": "language_model.layers.0." + family,
                "k": 4,
                "variant": "current",
                "median_us": current,
            }
        )
        results.append(
            {
                "base": "language_model.layers.0." + family,
                "k": 4,
                "variant": "fp16_force_smem",
                "median_us": candidate,
            }
        )

    mod._print_k4_projection(results, 19.053, 3.659)
    out = capsys.readouterr().out
    assert "fp16_force_smem" in out
    assert "PASS" in out
    # 30% of 3.659 ms is about 1.098 ms = 5.76% of 19.053 ms.
    assert "5.76" in out
