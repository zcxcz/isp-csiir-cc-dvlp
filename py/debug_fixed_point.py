#!/usr/bin/env python3

import argparse
import numpy as np

from csiir_common import load_hex_image, load_img_info, load_reg_config, reg_config_to_model_config
from isp_csiir_fixed_model import HORIZONTAL_TAP_STEP, FixedPointConfig, ISPCSIIRFixedModel


def parse_args():
    parser = argparse.ArgumentParser(description="Dump fixed-model internals at one coordinate")
    parser.add_argument("--reg-cfg", default="config/reg_cfg_seed0.json")
    parser.add_argument("--img-info", default="config/img_info.json")
    parser.add_argument("--reg-table", default="config/register_table.csv")
    parser.add_argument("--x", type=int, required=True)
    parser.add_argument("--y", type=int, required=True)
    return parser.parse_args()


def dump_matrix(tag: str, matrix: np.ndarray):
    print(tag)
    for row in matrix:
        print("  " + " ".join(str(int(v)) for v in row))


def main():
    args = parse_args()
    regs = load_reg_config(args.reg_cfg, args.reg_table)
    model_cfg = reg_config_to_model_config(regs)
    img_info = load_img_info(args.img_info)
    width = int(img_info["image"]["width"])
    height = int(img_info["image"]["height"])
    image = load_hex_image(img_info["image"]["input_path"], width, height)

    model = ISPCSIIRFixedModel(FixedPointConfig(**model_cfg))
    src = image.astype(np.int32).copy()
    filt = image.astype(np.int32).copy()
    grad_row_buf = np.zeros((2, width), dtype=np.int32)
    grad_shift = [0, 0, 0]

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
            filt[j, i] = model._s11_to_u10(int(stage4["final_patch"][2, 2]))

            if i == args.x and j == args.y:
                print(f"PY_DEBUG coord=({j},{i})")
                print(
                    f"  grad_h={grad_h} grad_v={grad_v} "
                    f"grad_u={grad_u} grad_d={grad_d} grad_l={grad_l} grad_c={grad_c} grad_r={grad_r} "
                    f"win_size={win_size}"
                )
                print(
                    "  avg0(c/u/d/l/r)="
                    f"{stage2['avg0']['c']}/{stage2['avg0']['u']}/{stage2['avg0']['d']}/{stage2['avg0']['l']}/{stage2['avg0']['r']}"
                )
                print(
                    "  avg1(c/u/d/l/r)="
                    f"{stage2['avg1']['c']}/{stage2['avg1']['u']}/{stage2['avg1']['d']}/{stage2['avg1']['l']}/{stage2['avg1']['r']}"
                )
                print(
                    f"  blend0_grad={blend0_grad} blend1_grad={blend1_grad} "
                    f"final_center={int(stage4['final_patch'][2,2])} final_u10={int(filt[j, i])} "
                    f"orientation={stage4['orientation']} ratio={stage4['ratio']}"
                )
                dump_matrix("  center_patch_u10", center_win)
                dump_matrix("  stage4_patch_u10", stage4_win)
                dump_matrix("  stage4_final_patch_s11", stage4["final_patch"])
                return

    raise RuntimeError(f"Coordinate ({args.x}, {args.y}) not visited")


if __name__ == "__main__":
    main()
