#!/usr/bin/env python3
"""Run randomized py-vs-hls regressions for ISP-CSIIR."""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from csiir_common import save_json


PATTERNS = ["random", "ramp", "checkerboard", "gradient", "max", "zeros"]


def run_cmd(cmd: List[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )


def monotonic_sample(rng: random.Random, low: int, high: int, count: int) -> List[int]:
    return sorted(rng.sample(range(low, high + 1), count))


def build_reg_cfg(seed: int, width: int, height: int) -> Dict[str, int]:
    rng = random.Random(seed)

    thresh = monotonic_sample(rng, 16, 40, 4)
    clip_y = monotonic_sample(rng, 16, 40, 4)
    clip_sft = [rng.randint(0, 3) for _ in range(4)]
    blend = [rng.randint(0, 64) for _ in range(4)]

    return {
        "reg_image_width": width,
        "reg_image_height": height,
        "reg_win_size_thresh_0": thresh[0],
        "reg_win_size_thresh_1": thresh[1],
        "reg_win_size_thresh_2": thresh[2],
        "reg_win_size_thresh_3": thresh[3],
        "reg_win_size_clip_y_0": clip_y[0],
        "reg_win_size_clip_y_1": clip_y[1],
        "reg_win_size_clip_y_2": clip_y[2],
        "reg_win_size_clip_y_3": clip_y[3],
        "reg_win_size_clip_sft_0": clip_sft[0],
        "reg_win_size_clip_sft_1": clip_sft[1],
        "reg_win_size_clip_sft_2": clip_sft[2],
        "reg_win_size_clip_sft_3": clip_sft[3],
        "reg_blending_ratio_0": blend[0],
        "reg_blending_ratio_1": blend[1],
        "reg_blending_ratio_2": blend[2],
        "reg_blending_ratio_3": blend[3],
        "reg_edge_protect": rng.randint(0, 64),
    }


def build_img_info(seed: int, width: int, height: int, pattern: str, base_dir: Path) -> Dict[str, object]:
    return {
        "seed": seed,
        "image": {
            "width": width,
            "height": height,
            "bitwidth": 10,
            "source": "generate",
            "pattern": pattern,
            "input_path": str(base_dir / f"input_seed{seed}.hex"),
        },
        "generator": {
            "force_regenerate": True,
        },
        "outputs": {
            "py_pattern_path": str(base_dir / f"output_py_seed{seed}.hex"),
            "hls_pattern_path": str(base_dir / f"output_hls_seed{seed}.hex"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Randomized ISP-CSIIR py-vs-hls regression")
    parser.add_argument("--count", type=int, default=20, help="number of regression seeds")
    parser.add_argument("--seed-base", type=int, default=1000, help="base random seed")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument(
        "--work-dir",
        default="tmp/random_regression",
        help="relative workspace for generated configs/patterns",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="continue after failures and summarize all mismatches",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent
    work_dir = repo / args.work_dir
    cfg_dir = work_dir / "config"
    pat_dir = work_dir / "patterns"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    pat_dir.mkdir(parents=True, exist_ok=True)

    build = run_cmd(["make", "-C", "hls"], repo)
    if build.returncode != 0:
        sys.stdout.write(build.stdout)
        sys.stderr.write(build.stderr)
        return build.returncode

    failures = []
    for idx in range(args.count):
        seed = args.seed_base + idx
        pattern = PATTERNS[seed % len(PATTERNS)]
        reg_cfg_path = cfg_dir / f"reg_cfg_seed{seed}.json"
        img_info_path = cfg_dir / f"img_info_seed{seed}.json"

        save_json(reg_cfg_path, build_reg_cfg(seed, args.width, args.height))
        save_json(img_info_path, build_img_info(seed, args.width, args.height, pattern, pat_dir))

        steps = [
            ["python3", "py/gen_pattern.py", "--img-info", str(reg_cfg_path.parent.relative_to(repo) / img_info_path.name)],
            ["python3", "py/isp_csiir_fixed_model.py", "--reg-cfg", str(reg_cfg_path.relative_to(repo)), "--img-info", str(img_info_path.relative_to(repo))],
            ["./hls/hls_top_tb", str(reg_cfg_path.relative_to(repo)), str(img_info_path.relative_to(repo))],
            ["python3", "py/compare_patterns.py", "--img-info", str(img_info_path.relative_to(repo))],
        ]

        for step in steps:
            result = run_cmd(step, repo)
            if result.returncode != 0:
                failures.append(
                    {
                        "seed": seed,
                        "pattern": pattern,
                        "cmd": " ".join(step),
                        "stdout": result.stdout.strip(),
                        "stderr": result.stderr.strip(),
                    }
                )
                print(f"FAIL seed={seed} pattern={pattern} cmd={' '.join(step)}")
                if result.stdout.strip():
                    print(result.stdout.strip())
                if result.stderr.strip():
                    print(result.stderr.strip())
                if not args.keep_going:
                    break
                else:
                    break
        else:
            print(f"PASS seed={seed} pattern={pattern}")
            continue

        if failures and not args.keep_going:
            break

    if failures:
        print(f"SUMMARY: {len(failures)} / {args.count} seeds failed")
        return 1

    print(f"SUMMARY: all {args.count} seeds passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
