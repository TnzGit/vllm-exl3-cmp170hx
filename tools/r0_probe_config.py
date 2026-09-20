#!/usr/bin/env python3
"""Read-only probe of the Qwen3.8-Flash-Next EXL3 pack config.json.

Prints the top-level config minus quantization_config, then the quantization
block, then text_config keys. Never rewrites anything.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

pack = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
cfg = json.loads((pack / "config.json").read_text(encoding="utf-8"))

print("=== top-level (minus quantization_config) ===")
print(json.dumps({k: v for k, v in cfg.items() if k != "quantization_config"}, indent=1)[:4000])

print("=== quantization_config ===")
print(json.dumps(cfg.get("quantization_config"), indent=1))

tc = cfg.get("text_config")
if isinstance(tc, dict):
    print("=== text_config keys ===")
    print(sorted(tc.keys()))
    print("=== text_config quantization_config ===")
    print(json.dumps(tc.get("quantization_config"), indent=1))
