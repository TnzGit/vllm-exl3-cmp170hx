import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "kvmem_q2c_dual_pool_boot_summarize.py"


def _load():
    spec = importlib.util.spec_from_file_location("q2c_boot", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_dual_pool_boot_summary_go():
    mod = _load()
    private = 4161 * 12 * 32768
    text = "\n".join(
        [
            mod.PLATFORM,
            (
                "Q2C dual-pool KV config: private_qsa_blocks=4161 "
                f"private_qsa_bytes={private} regular_blocks=99 "
                "regular_bytes=9000000000"
            ),
            (
                "Q2C dual-pool worker allocation: stock_bytes=9000000000 "
                f"private_qsa_bytes={private} private_qsa_blocks=4161"
            ),
            "Available KV cache memory: 10.07 GiB",
            "GPU KV cache size: 161,000 tokens, "
            "Maximum concurrency for 161,000 tokens per request: 1.00x",
        ]
    )
    out = mod.summarize(text)
    assert out["classification"] == "Q2C_DUAL_POOL_BOOT_GO"
    assert out["dual_pool_boot_gate"] is True
    assert all(out["checks"].values())


def test_dual_pool_boot_summary_rejects_legacy_alignment():
    mod = _load()
    private = 4161 * 12 * 32768
    text = "\n".join(
        [
            mod.PLATFORM,
            (
                "Q2C dual-pool KV config: private_qsa_blocks=4161 "
                f"private_qsa_bytes={private} regular_blocks=99 "
                "regular_bytes=9000000000"
            ),
            (
                "Q2C dual-pool worker allocation: stock_bytes=9000000000 "
                f"private_qsa_bytes={private} private_qsa_blocks=4161"
            ),
            "Maximum concurrency for 161,000 tokens per request: 1.00x",
            (
                "Setting attention block size to 1568 tokens to ensure that "
                "attention page size is >= mamba page size."
            ),
        ]
    )
    out = mod.summarize(text)
    assert out["classification"] == "Q2C_DUAL_POOL_BOOT_NO_GO"
    assert out["dual_pool_boot_gate"] is False
    assert out["checks"]["legacy_1568_alignment_absent"] is False
