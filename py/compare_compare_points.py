#!/usr/bin/env python3
"""Compare additional HLS/Python compare-point traces."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from csiir_common import load_img_info, resolve_repo_path


def load_jsonl(path_like: str):
    path = resolve_repo_path(path_like)
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def compare_entries(name: str, lhs_path: str, rhs_path: str) -> int:
    lhs = load_jsonl(lhs_path)
    rhs = load_jsonl(rhs_path)
    if len(lhs) != len(rhs):
        print(f"FAIL {name}: entry count differs py={len(lhs)} hls={len(rhs)}")
        return 1
    for idx, (lhs_entry, rhs_entry) in enumerate(zip(lhs, rhs)):
        if lhs_entry != rhs_entry:
            print(f"FAIL {name}: first mismatch at entry {idx}")
            print(f"  py : {json.dumps(lhs_entry, ensure_ascii=True)}")
            print(f"  hls: {json.dumps(rhs_entry, ensure_ascii=True)}")
            return 1
    print(f"PASS {name}: {len(lhs)} entries match")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare HLS/Python compare-point traces")
    parser.add_argument("--img-info", default="config/img_info.json")
    args = parser.parse_args()

    img_info = load_img_info(args.img_info)
    outputs = img_info.get("outputs", {})

    checks = [
        (
            "stage4_input_trace",
            outputs.get("py_stage4_input_trace_path", ""),
            outputs.get("hls_stage4_input_trace_path", ""),
        ),
        (
            "feedback_commit_trace",
            outputs.get("py_feedback_commit_trace_path", ""),
            outputs.get("hls_feedback_commit_trace_path", ""),
        ),
    ]

    ran = False
    failed = 0
    for name, py_path, hls_path in checks:
        if not py_path and not hls_path:
            continue
        if not py_path or not hls_path:
            print(f"FAIL {name}: missing py/hls path in img_info.json")
            failed += 1
            ran = True
            continue
        failed += compare_entries(name, py_path, hls_path)
        ran = True

    if not ran:
        print("No compare-point trace paths configured in img_info.json")
        return 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
