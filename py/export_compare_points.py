#!/usr/bin/env python3
"""Export additional compare points for ISP-CSIIR fixed model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from csiir_common import load_img_info, load_reg_config, prepare_input_image, reg_config_to_model_config
from isp_csiir_fixed_model import HORIZONTAL_TAP_STEP, FixedPointConfig, ISPCSIIRFixedModel


def save_jsonl(path_like: str, entries: list[dict]) -> None:
    path = Path(path_like)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, separators=(",", ":")))
            handle.write("\n")


def build_write_xs(model: ISPCSIIRFixedModel, center_x: int, width: int) -> list[int]:
    return [
        int(model._clip(center_x + (col - 2) * HORIZONTAL_TAP_STEP, 0, width - 1))
        for col in range(5)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Export py compare-point traces")
    parser.add_argument("--reg-cfg", default="config/reg_cfg_seed0.json")
    parser.add_argument("--img-info", default="config/img_info.json")
    parser.add_argument("--reg-table", default="config/register_table.csv")
    parser.add_argument("--stage4-out", default=None)
    parser.add_argument("--feedback-out", default=None)
    args = parser.parse_args()

    regs = load_reg_config(args.reg_cfg, args.reg_table)
    model_cfg = reg_config_to_model_config(regs)
    img_info = load_img_info(args.img_info)
    width = int(img_info["image"]["width"])
    height = int(img_info["image"]["height"])
    if width != int(model_cfg["IMG_WIDTH"]) or height != int(model_cfg["IMG_HEIGHT"]):
        raise ValueError("image size in img_info.json must match reg_cfg.json")

    stage4_out = args.stage4_out or img_info.get("outputs", {}).get("py_stage4_input_trace_path", "")
    feedback_out = args.feedback_out or img_info.get("outputs", {}).get("py_feedback_commit_trace_path", "")

    if not stage4_out and not feedback_out:
        raise ValueError("no py trace output path provided in args or img_info.json")

    model = ISPCSIIRFixedModel(FixedPointConfig(**model_cfg))
    input_img, _ = prepare_input_image(img_info)
    src = input_img.astype(np.int32).copy()
    filt = input_img.astype(np.int32).copy()

    grad_row_buf = np.zeros((2, width), dtype=np.int32)
    grad_shift = [0, 0, 0]
    stage4_entries: list[dict] = []
    feedback_entries: list[dict] = []

    for j in range(height):
        for i in range(width):
            grad_h, grad_v, grad_c, win_size, center_win = model._grad_triplet_win_size(src, i, j)
            stage2 = model._stage2_directional_avg(center_win, win_size)

            grad_u = grad_shift[1] if j > 0 else grad_c
            grad_d = grad_row_buf[0, i] if j < height - 1 else grad_c
            grad_l = model._stage1_gradient(model._get_window(src, i - HORIZONTAL_TAP_STEP, j))[2] if i > 0 else grad_c
            grad_r = model._stage1_gradient(model._get_window(src, i + HORIZONTAL_TAP_STEP, j))[2] if i < width - 1 else grad_c
            grads = {"u": grad_u, "d": grad_d, "l": grad_l, "r": grad_r, "c": grad_c}

            grad_shift[0] = grad_shift[1]
            grad_shift[1] = grad_shift[2]
            grad_shift[2] = grad_c
            grad_row_buf[0, i] = grad_c
            grad_row_buf[1, i] = grad_row_buf[0, i]

            blend0_grad, blend1_grad = model._stage3_fusion(stage2["avg0"], stage2["avg1"], grads)

            stage4_win = np.zeros((5, 5), dtype=np.int32)
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    x = model._clip(i + dx * HORIZONTAL_TAP_STEP, 0, width - 1)
                    y = model._clip(j + dy, 0, height - 1)
                    stage4_win[dy + 2, dx + 2] = filt[y, x] if dy < 0 else src[y, x]

            stage4_entries.append(
                {
                    "idx": len(stage4_entries),
                    "center_x": int(i),
                    "center_y": int(j),
                    "win_size": int(win_size),
                    "grad_h": int(abs(grad_h)),
                    "grad_v": int(abs(grad_v)),
                    "blend0": int(blend0_grad),
                    "blend1": int(blend1_grad),
                    "avg0_u": int(stage2["avg0"]["u"]),
                    "avg1_u": int(stage2["avg1"]["u"]),
                    "src_patch_u10": stage4_win.astype(np.int32).tolist(),
                }
            )

            stage4 = model._stage4_window_blend(
                stage4_win,
                win_size,
                blend0_grad,
                blend1_grad,
                stage2["avg0"]["u"],
                stage2["avg1"]["u"],
                grad_h,
                grad_v,
            )
            final_patch_u10 = np.vectorize(model._s11_to_u10)(stage4["final_patch"]).astype(np.int32)
            feedback_entries.append(
                {
                    "idx": len(feedback_entries),
                    "center_x": int(i),
                    "center_y": int(j),
                    "write_xs": build_write_xs(model, i, width),
                    "columns_u10": [final_patch_u10[:, col].astype(np.int32).tolist() for col in range(5)],
                }
            )
            filt[j, i] = int(final_patch_u10[2, 2])

    if stage4_out:
        save_jsonl(stage4_out, stage4_entries)
        print(f"Python stage4 trace written to {stage4_out}")
    if feedback_out:
        save_jsonl(feedback_out, feedback_entries)
        print(f"Python feedback trace written to {feedback_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
