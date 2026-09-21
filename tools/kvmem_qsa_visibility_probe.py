#!/usr/bin/env python3
"""Submit one exact-token K1-Q1 semantic replay case."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path


def _post_json(url: str, payload: dict, timeout: int = 7200) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def _codes_in_order(text: str, codes: list[str]) -> bool:
    pos = 0
    for code in codes:
        found = text.find(code, pos)
        if found < 0:
            return False
        pos = found + len(code)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--case", type=Path, required=True)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    case = json.loads(args.case.read_text())
    token_ids = case["prompt_token_ids"]
    expected_codes = [
        str(row["code"]) for row in case.get("target_facts", [])
        if row.get("code") is not None
    ]

    base = f"http://127.0.0.1:{args.port}"
    model = json.load(
        urllib.request.urlopen(f"{base}/v1/models", timeout=30)
    )["data"][0]["id"]

    payload = {
        "model": model,
        "prompt": token_ids,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "seed": 0,
        "stream": False,
        "logprobs": 1,
    }

    t0 = time.time()
    response = _post_json(f"{base}/v1/completions", payload)
    wall = time.time() - t0
    choice = (response.get("choices") or [{}])[0]
    logprobs = choice.get("logprobs") or {}
    text = str(choice.get("text") or "")

    result = {
        "schema": 1,
        "case": case["name"],
        "query_span": case["query_span"],
        "target_markers": case.get("target_markers", []),
        "expected_codes": expected_codes,
        "target_codes_in_order": _codes_in_order(text, expected_codes),
        "wall_s": round(wall, 6),
        "usage": response.get("usage") or {},
        "finish_reason": choice.get("finish_reason"),
        "text": text,
        "logprob_tokens": logprobs.get("tokens"),
        "token_logprobs": logprobs.get("token_logprobs"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
