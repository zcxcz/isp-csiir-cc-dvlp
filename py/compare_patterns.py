#!/usr/bin/env python3
"""Compare HLS and Python output patterns described by img_info.json."""

import argparse
import sys

from csiir_common import compare_hex_images, load_img_info, normalize_pattern_role


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare ISP-CSIIR output patterns")
    parser.add_argument("--img-info", default="config/img_info.json", help="img_info JSON path")
    args = parser.parse_args()

    img_info = load_img_info(args.img_info)
    width = int(img_info["image"]["width"])
    height = int(img_info["image"]["height"])
    py_path = img_info["outputs"]["py_pattern_path"]
    hls_path = img_info["outputs"]["hls_pattern_path"]

    if normalize_pattern_role(py_path) != normalize_pattern_role(hls_path):
        print("FAIL: py/hls pattern names do not match after normalizing `_py` and `_hls`")
        print(f"  py : {py_path}")
        print(f"  hls: {hls_path}")
        return 1

    report = compare_hex_images(py_path, hls_path, width, height)
    if report["mismatch_count"] == 0:
        print(
            f"PASS: {py_path} matches {hls_path} "
            f"({report['total_pixels']} pixels checked)"
        )
        return 0

    print(
        f"FAIL: {report['mismatch_count']} / {report['total_pixels']} pixels differ, "
        f"max abs diff = {report['max_abs_diff']}"
    )
    for mismatch in report["mismatches"]:
        print(
            "  "
            f"(y={mismatch['y']}, x={mismatch['x']}) "
            f"py={mismatch['lhs']} hls={mismatch['rhs']} delta={mismatch['delta']}"
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
