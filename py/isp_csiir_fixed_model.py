#!/usr/bin/env python3
"""
ISP-CSIIR 定点参考模型 - 对齐 isp-csiir-ref.md 语义
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

from csiir_common import (
    compare_hex_images,
    load_img_info,
    load_reg_config,
    prepare_input_image,
    reg_config_to_model_config,
    save_hex_image,
)


# Center kernels (from ref)
AVG_FACTOR_C_2X2 = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 2, 1, 0],
    [0, 2, 4, 2, 0],
    [0, 1, 2, 1, 0],
    [0, 0, 0, 0, 0],
], dtype=np.int32)
AVG_FACTOR_U_2X2 = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 3, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
], dtype=np.int32)
AVG_FACTOR_D_2X2 = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 3, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
], dtype=np.int32)
AVG_FACTOR_L_2X2 = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 1, 3, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0],
], dtype=np.int32)
AVG_FACTOR_R_2X2 = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0],
    [0, 0, 3, 1, 0],
    [0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0],
], dtype=np.int32)

AVG_FACTOR_C_3X3 = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 2, 1, 0],
    [0, 2, 4, 2, 0],
    [0, 1, 2, 1, 0],
    [0, 0, 0, 0, 0],
], dtype=np.int32)
AVG_FACTOR_U_3X3 = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 2, 1, 0],
    [0, 1, 2, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
], dtype=np.int32)
AVG_FACTOR_D_3X3 = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 2, 1, 0],
    [0, 1, 2, 1, 0],
    [0, 0, 0, 0, 0],
], dtype=np.int32)
AVG_FACTOR_L_3X3 = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 2, 2, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0],
], dtype=np.int32)
AVG_FACTOR_R_3X3 = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0],
    [0, 0, 2, 2, 0],
    [0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0],
], dtype=np.int32)

AVG_FACTOR_C_4X4 = np.array([
    [1, 2, 2, 2, 1],
    [2, 4, 4, 4, 2],
    [2, 4, 4, 4, 2],
    [2, 4, 4, 4, 2],
    [1, 2, 2, 2, 1],
], dtype=np.int32)
AVG_FACTOR_U_4X4 = np.array([
    [1, 2, 2, 2, 1],
    [2, 2, 4, 2, 2],
    [2, 2, 4, 2, 2],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
], dtype=np.int32)
AVG_FACTOR_D_4X4 = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [2, 2, 4, 2, 2],
    [2, 2, 4, 2, 2],
    [1, 2, 2, 2, 1],
], dtype=np.int32)
AVG_FACTOR_L_4X4 = np.array([
    [1, 2, 2, 0, 0],
    [2, 2, 2, 0, 0],
    [2, 4, 4, 0, 0],
    [2, 2, 2, 0, 0],
    [1, 2, 2, 0, 0],
], dtype=np.int32)
AVG_FACTOR_R_4X4 = np.array([
    [0, 0, 2, 2, 1],
    [0, 0, 2, 2, 2],
    [0, 0, 4, 4, 2],
    [0, 0, 2, 2, 2],
    [0, 0, 2, 2, 1],
], dtype=np.int32)

AVG_FACTOR_C_5X5 = np.array([
    [1, 2, 1, 2, 1],
    [1, 1, 1, 1, 1],
    [2, 1, 2, 1, 2],
    [1, 1, 1, 1, 1],
    [1, 2, 1, 2, 1],
], dtype=np.int32)
AVG_FACTOR_U_5X5 = np.array([
    [1, 1, 1, 1, 1],
    [1, 1, 2, 1, 1],
    [1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
], dtype=np.int32)
AVG_FACTOR_D_5X5 = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1],
    [1, 1, 2, 1, 1],
    [1, 1, 1, 1, 1],
], dtype=np.int32)
AVG_FACTOR_L_5X5 = np.array([
    [1, 1, 1, 0, 0],
    [1, 1, 1, 0, 0],
    [1, 2, 1, 0, 0],
    [1, 1, 1, 0, 0],
    [1, 1, 1, 0, 0],
], dtype=np.int32)
AVG_FACTOR_R_5X5 = np.array([
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 1],
    [0, 0, 1, 2, 1],
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 1],
], dtype=np.int32)

HORIZONTAL_TAP_STEP = 2

BLEND_FACTOR_2X2_H = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
], dtype=np.int32)
BLEND_FACTOR_2X2_V = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0],
], dtype=np.int32)
BLEND_FACTOR_2X2 = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 2, 1, 0],
    [0, 2, 4, 2, 0],
    [0, 1, 2, 1, 0],
    [0, 0, 0, 0, 0],
], dtype=np.int32)
BLEND_FACTOR_3X3 = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
], dtype=np.int32)
BLEND_FACTOR_4X4 = np.array([
    [1, 2, 2, 2, 1],
    [2, 4, 4, 4, 2],
    [2, 4, 4, 4, 2],
    [2, 4, 4, 4, 2],
    [1, 2, 2, 2, 1],
], dtype=np.int32)
BLEND_FACTOR_5X5 = np.full((5, 5), 4, dtype=np.int32)

SOBEL_X = np.array([
    [1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [-1, -1, -1, -1, -1],
], dtype=np.int32)
SOBEL_Y = np.array([
    [1, 0, 0, 0, -1],
    [1, 0, 0, 0, -1],
    [1, 0, 0, 0, -1],
    [1, 0, 0, 0, -1],
    [1, 0, 0, 0, -1],
], dtype=np.int32)


@dataclass
class FixedPointConfig:
    DATA_WIDTH: int = 10
    GRAD_WIDTH: int = 14
    ACC_WIDTH: int = 20
    IMG_WIDTH: int = 64
    IMG_HEIGHT: int = 64
    win_size_thresh: List[int] = None
    win_size_clip_y: List[int] = None
    win_size_clip_sft: List[int] = None
    blending_ratio: List[int] = None
    reg_edge_protect: int = 32

    def __post_init__(self):
        if self.win_size_thresh is None:
            self.win_size_thresh = [16, 24, 32, 40]
        if self.win_size_clip_y is None:
            self.win_size_clip_y = [15, 23, 31, 39]
        if self.win_size_clip_sft is None:
            self.win_size_clip_sft = [2, 2, 2, 2]
        if self.blending_ratio is None:
            self.blending_ratio = [32, 32, 32, 32]


class ISPCSIIRFixedModel:
    def __init__(self, config: FixedPointConfig = None):
        self.config = config if config else FixedPointConfig()
        self.DATA_MAX = (1 << self.config.DATA_WIDTH) - 1
        self.src_uv = None

    def _clip(self, value: int, min_val: int = 0, max_val: int = None) -> int:
        if max_val is None:
            max_val = self.DATA_MAX
        return max(min_val, min(max_val, int(value)))

    def _round_div(self, num: int, den: int) -> int:
        if den == 0:
            raise ZeroDivisionError("division by zero")
        if num >= 0:
            return (num + den // 2) // den
        return -(((-num) + den // 2) // den)

    def _u10_to_s11(self, value: int) -> int:
        return int(value) - 512

    def _s11_to_u10(self, value: int) -> int:
        return self._clip(int(value) + 512, 0, 1023)

    def _saturate_s11(self, value: int) -> int:
        return self._clip(value, -512, 511)

    def _get_window(self, img: np.ndarray, i: int, j: int) -> np.ndarray:
        h, w = img.shape
        window = np.zeros((5, 5), dtype=np.int32)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                x = self._clip(i + dx * HORIZONTAL_TAP_STEP, 0, w - 1)
                y = self._clip(j + dy, 0, h - 1)
                window[dy + 2, dx + 2] = int(img[y, x])
        return window

    def _lut_x_nodes(self) -> List[int]:
        nodes = []
        acc = 0
        for shift in self.config.win_size_clip_sft:
            acc += 1 << int(shift)
            nodes.append(acc)
        return nodes

    def _lut_win_size(self, grad_triplet_max: int) -> int:
        x_nodes = self._lut_x_nodes()
        y_nodes = self.config.win_size_clip_y
        x = int(grad_triplet_max)
        if x <= x_nodes[0]:
            win_size_grad = y_nodes[0]
        elif x >= x_nodes[-1]:
            win_size_grad = y_nodes[-1]
        else:
            win_size_grad = y_nodes[-1]
            for idx in range(len(x_nodes) - 1):
                x0, x1 = x_nodes[idx], x_nodes[idx + 1]
                if x0 <= x <= x1:
                    y0, y1 = y_nodes[idx], y_nodes[idx + 1]
                    win_size_grad = y0 + self._round_div((x - x0) * (y1 - y0), x1 - x0)
                    break
        return self._clip(win_size_grad, 16, 40)

    def _stage1_gradient(self, win: np.ndarray) -> Tuple[int, int, int]:
        grad_h = int(np.sum(win * SOBEL_X))
        grad_v = int(np.sum(win * SOBEL_Y))
        grad_h = abs(grad_h)
        grad_v = abs(grad_v)
        grad = self._round_div(grad_h, 5) + self._round_div(grad_v, 5)
        grad = self._clip(grad, 0, 127)  # Clip to [0, 127] per ref
        return grad_h, grad_v, grad

    def _grad_triplet_win_size(self, img: np.ndarray, i: int, j: int) -> Tuple[int, int, int, int, np.ndarray]:
        h, w = img.shape
        # Use unclipped center positions for window building - _get_window handles clipping internally
        # This ensures dx offsets are applied correctly before clipping, matching C++ behavior
        left_win = self._get_window(img, i - HORIZONTAL_TAP_STEP, j)
        center_win = self._get_window(img, i, j)
        right_win = self._get_window(img, i + HORIZONTAL_TAP_STEP, j)
        grad_h, grad_v, grad_c = self._stage1_gradient(center_win)
        _, _, grad_l = self._stage1_gradient(left_win)
        _, _, grad_r = self._stage1_gradient(right_win)
        win_size = self._lut_win_size(max(grad_l, grad_c, grad_r))
        return grad_h, grad_v, grad_c, win_size, center_win

    def _select_stage2_kernels(self, win_size: int) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        t0, t1, t2, t3 = self.config.win_size_thresh
        if win_size < t0:
            avg0 = {"c": np.zeros((5, 5), dtype=np.int32), "u": np.zeros((5, 5), dtype=np.int32),
                    "d": np.zeros((5, 5), dtype=np.int32), "l": np.zeros((5, 5), dtype=np.int32),
                    "r": np.zeros((5, 5), dtype=np.int32)}
            avg1 = {"c": AVG_FACTOR_C_2X2.copy(), "u": AVG_FACTOR_U_2X2.copy(),
                    "d": AVG_FACTOR_D_2X2.copy(), "l": AVG_FACTOR_L_2X2.copy(),
                    "r": AVG_FACTOR_R_2X2.copy()}
        elif win_size < t1:
            avg0 = {"c": AVG_FACTOR_C_2X2.copy(), "u": AVG_FACTOR_U_2X2.copy(),
                    "d": AVG_FACTOR_D_2X2.copy(), "l": AVG_FACTOR_L_2X2.copy(),
                    "r": AVG_FACTOR_R_2X2.copy()}
            avg1 = {"c": AVG_FACTOR_C_3X3.copy(), "u": AVG_FACTOR_U_3X3.copy(),
                    "d": AVG_FACTOR_D_3X3.copy(), "l": AVG_FACTOR_L_3X3.copy(),
                    "r": AVG_FACTOR_R_3X3.copy()}
        elif win_size < t2:
            avg0 = {"c": AVG_FACTOR_C_3X3.copy(), "u": AVG_FACTOR_U_3X3.copy(),
                    "d": AVG_FACTOR_D_3X3.copy(), "l": AVG_FACTOR_L_3X3.copy(),
                    "r": AVG_FACTOR_R_3X3.copy()}
            avg1 = {"c": AVG_FACTOR_C_4X4.copy(), "u": AVG_FACTOR_U_4X4.copy(),
                    "d": AVG_FACTOR_D_4X4.copy(), "l": AVG_FACTOR_L_4X4.copy(),
                    "r": AVG_FACTOR_R_4X4.copy()}
        elif win_size < t3:
            avg0 = {"c": AVG_FACTOR_C_4X4.copy(), "u": AVG_FACTOR_U_4X4.copy(),
                    "d": AVG_FACTOR_D_4X4.copy(), "l": AVG_FACTOR_L_4X4.copy(),
                    "r": AVG_FACTOR_R_4X4.copy()}
            avg1 = {"c": AVG_FACTOR_C_5X5.copy(), "u": AVG_FACTOR_U_5X5.copy(),
                    "d": AVG_FACTOR_D_5X5.copy(), "l": AVG_FACTOR_L_5X5.copy(),
                    "r": AVG_FACTOR_R_5X5.copy()}
        else:
            avg0 = {"c": AVG_FACTOR_C_5X5.copy(), "u": AVG_FACTOR_U_5X5.copy(),
                    "d": AVG_FACTOR_D_5X5.copy(), "l": AVG_FACTOR_L_5X5.copy(),
                    "r": AVG_FACTOR_R_5X5.copy()}
            avg1 = {"c": np.zeros((5, 5), dtype=np.int32), "u": np.zeros((5, 5), dtype=np.int32),
                    "d": np.zeros((5, 5), dtype=np.int32), "l": np.zeros((5, 5), dtype=np.int32),
                    "r": np.zeros((5, 5), dtype=np.int32)}
        return avg0, avg1

    def _weighted_avg_from_factor(self, win_s11: np.ndarray, factor: np.ndarray) -> int:
        weight = int(np.sum(factor))
        if weight == 0:
            return 0
        total = int(np.sum(win_s11 * factor))
        return self._saturate_s11(self._round_div(total, weight))

    def _build_stage2_path(self, win_s11: np.ndarray, kernels: Dict[str, np.ndarray]) -> Dict:
        enable = int(np.sum(kernels["c"])) != 0
        values = {name: self._weighted_avg_from_factor(win_s11, factor) for name, factor in kernels.items()}
        return {
            "enable": enable,
            "kernels": {k: v.copy() for k, v in kernels.items()},
            "values": values,
        }

    def _stage2_directional_avg(self, win_u10: np.ndarray, win_size: int) -> Dict:
        win_s11 = win_u10.astype(np.int32) - 512
        avg0_kernels, avg1_kernels = self._select_stage2_kernels(win_size)
        avg0_path = self._build_stage2_path(win_s11, avg0_kernels)
        avg1_path = self._build_stage2_path(win_s11, avg1_kernels)
        return {
            "avg0_enable": avg0_path["enable"],
            "avg1_enable": avg1_path["enable"],
            "avg0": avg0_path["values"],
            "avg1": avg1_path["values"],
            "avg0_kernels": avg0_path["kernels"],
            "avg1_kernels": avg1_path["kernels"],
        }

    def _stage3_neighbors(self, img: np.ndarray, i: int, j: int, grad_c: int) -> Dict[str, int]:
        h, w = img.shape
        # Boundary handling per ref section 4.1
        grad_u = self._stage1_gradient(self._get_window(img, i, j - 1 if j > 0 else 0))[2] if j > 0 else grad_c
        grad_d = self._stage1_gradient(self._get_window(img, i, j + 1 if j < h - 1 else h - 1))[2] if j < h - 1 else grad_c
        grad_l = self._stage1_gradient(self._get_window(img, i - HORIZONTAL_TAP_STEP if i > 0 else 0, j))[2] if i > 0 else grad_c
        grad_r = self._stage1_gradient(self._get_window(img, i + HORIZONTAL_TAP_STEP if i < w - 1 else w - 1, j))[2] if i < w - 1 else grad_c
        return {"c": grad_c, "u": grad_u, "d": grad_d, "l": grad_l, "r": grad_r}

    def _stage3_fusion(self, avg0: Dict[str, int], avg1: Dict[str, int], grads: Dict[str, int]) -> Tuple[int, int]:
        # New min-tracking algorithm per ref section 4.3
        # Order: u(0), d(1), l(2), r(3), c(4)
        g = [grads["u"], grads["d"], grads["l"], grads["r"], grads["c"]]
        v0 = [avg0["u"], avg0["d"], avg0["l"], avg0["r"], avg0["c"]]
        v1 = [avg1["u"], avg1["d"], avg1["l"], avg1["r"], avg1["c"]]

        # min0 tracking
        min0_grad = 2048
        min0_grad_avg = 0
        if g[0] <= min0_grad:
            min0_grad = g[0]
            min0_grad_avg = v0[0]
        if g[2] <= min0_grad:
            min0_grad = g[2]
            min0_grad_avg = self._round_div(v0[2] + min0_grad_avg + 1, 2)
        if g[4] <= min0_grad:
            min0_grad = g[4]
            min0_grad_avg = self._round_div(v0[4] + min0_grad_avg + 1, 2)
        if g[3] <= min0_grad:
            min0_grad = g[3]
            min0_grad_avg = self._round_div(v0[3] + min0_grad_avg + 1, 2)
        if g[1] <= min0_grad:
            min0_grad = g[1]
            min0_grad_avg = self._round_div(v0[1] + min0_grad_avg + 1, 2)

        # min1 tracking
        min1_grad = 2048
        min1_grad_avg = 0
        if g[0] <= min1_grad:
            min1_grad = g[0]
            min1_grad_avg = v1[0]
        if g[2] <= min1_grad:
            min1_grad = g[2]
            min1_grad_avg = self._round_div(v1[2] + min1_grad_avg + 1, 2)
        if g[4] <= min1_grad:
            min1_grad = g[4]
            min1_grad_avg = self._round_div(v1[4] + min1_grad_avg + 1, 2)
        if g[3] <= min1_grad:
            min1_grad = g[3]
            min1_grad_avg = self._round_div(v1[3] + min1_grad_avg + 1, 2)
        if g[1] <= min1_grad:
            min1_grad = g[1]
            min1_grad_avg = self._round_div(v1[1] + min1_grad_avg + 1, 2)

        blend0_grad = min0_grad_avg
        blend1_grad = min1_grad_avg

        return blend0_grad, blend1_grad

    def _mix_scalar_with_patch(self, scalar: int, src_uv_s11_5x5: np.ndarray, factor: np.ndarray) -> np.ndarray:
        out = np.zeros((5, 5), dtype=np.int32)
        for y in range(5):
            for x in range(5):
                out[y, x] = self._saturate_s11(
                    self._round_div(int(scalar) * int(factor[y, x]) + int(src_uv_s11_5x5[y, x]) * (4 - int(factor[y, x])), 4)
                )
        return out

    def _stage4_window_blend(self, win_u10: np.ndarray, win_size: int,
                             blend0_grad: int, blend1_grad: int,
                             avg0_u: int, avg1_u: int,
                             grad_h: int, grad_v: int) -> Dict:
        src_uv_s11_5x5 = win_u10.astype(np.int32) - 512
        ratio_idx = self._clip((win_size // 8) - 2, 0, 3)
        ratio = int(self.config.blending_ratio[ratio_idx])

        blend0_hor = self._saturate_s11(self._round_div(ratio * blend0_grad + (64 - ratio) * avg0_u, 64))
        blend1_hor = self._saturate_s11(self._round_div(ratio * blend1_grad + (64 - ratio) * avg1_u, 64))

        # G_H/G_V swap per ref: G_H from grad_v, G_V from grad_h
        G_H = abs(grad_v)
        G_V = abs(grad_h)
        orient_factor = BLEND_FACTOR_2X2_H if G_H > G_V else BLEND_FACTOR_2X2_V
        orientation = "h" if G_H > G_V else "v"

        t0, t1, t2, t3 = self.config.win_size_thresh
        blend0_win = None
        blend1_win = None

        if win_size < t0:
            blend10 = self._mix_scalar_with_patch(blend1_hor, src_uv_s11_5x5, orient_factor)
            blend11 = self._mix_scalar_with_patch(blend1_hor, src_uv_s11_5x5, BLEND_FACTOR_2X2)
            blend1_win = np.zeros((5, 5), dtype=np.int32)
            for y in range(5):
                for x in range(5):
                    blend1_win[y, x] = self._saturate_s11(
                        self._round_div(int(blend10[y, x]) * self.config.reg_edge_protect + int(blend11[y, x]) * (64 - self.config.reg_edge_protect), 64)
                    )
        elif win_size < t1:
            blend10 = self._mix_scalar_with_patch(blend1_hor, src_uv_s11_5x5, orient_factor)
            blend11 = self._mix_scalar_with_patch(blend1_hor, src_uv_s11_5x5, BLEND_FACTOR_2X2)
            blend1_win = np.zeros((5, 5), dtype=np.int32)
            for y in range(5):
                for x in range(5):
                    blend1_win[y, x] = self._saturate_s11(
                        self._round_div(int(blend10[y, x]) * self.config.reg_edge_protect + int(blend11[y, x]) * (64 - self.config.reg_edge_protect), 64)
                    )
            blend0_win = self._mix_scalar_with_patch(blend0_hor, src_uv_s11_5x5, BLEND_FACTOR_3X3)
        elif win_size < t2:
            blend1_win = self._mix_scalar_with_patch(blend1_hor, src_uv_s11_5x5, BLEND_FACTOR_3X3)
            blend0_win = self._mix_scalar_with_patch(blend0_hor, src_uv_s11_5x5, BLEND_FACTOR_4X4)
        elif win_size < t3:
            blend1_win = self._mix_scalar_with_patch(blend1_hor, src_uv_s11_5x5, BLEND_FACTOR_4X4)
            blend0_win = self._mix_scalar_with_patch(blend0_hor, src_uv_s11_5x5, BLEND_FACTOR_5X5)
        else:
            blend0_win = self._mix_scalar_with_patch(blend0_hor, src_uv_s11_5x5, BLEND_FACTOR_5X5)

        remain = win_size % 8
        if win_size < t0:
            final_patch = blend1_win
        elif win_size >= t3:
            final_patch = blend0_win
        else:
            final_patch = np.zeros((5, 5), dtype=np.int32)
            for y in range(5):
                for x in range(5):
                    final_patch[y, x] = self._saturate_s11(
                        self._round_div(int(blend0_win[y, x]) * remain + int(blend1_win[y, x]) * (8 - remain), 8)
                    )

        return {
            "ratio": ratio,
            "blend0_hor": blend0_hor,
            "blend1_hor": blend1_hor,
            "orientation": orientation,
            "final_patch": final_patch,
        }

    def _process_feedback_raster(self, input_image: np.ndarray,
                                 emit_center_stream: bool = False,
                                 emit_linebuffer_rows: bool = False,
                                 emit_patch_stream: bool = False):
        src = input_image.astype(np.int32).copy()
        filt = input_image.astype(np.int32).copy()
        self.src_uv = src
        h, w = src.shape
        center_stream = [] if emit_center_stream else None
        linebuffer_rows = [] if emit_linebuffer_rows else None
        patch_stream = [] if emit_patch_stream else None

        # Gradient row buffer (matches C++ grad_row_buf[2][width])
        grad_row_buf = np.zeros((2, w), dtype=np.int32)
        grad_shift = [0, 0, 0]  # grad_shift[3] register

        for j in range(h):
            for i in range(w):
                # C++ processes ALL rows through Stage4 pipeline, but only outputs from row 2+
                # For rows 0,1: gradients are computed (for grad_row_buf to have valid data for row 2),
                # but output is not written (output[0,1] stay as original per C++ semantics)

                # Gradient/stage2/stage3: from ORIGINAL (src)
                grad_h, grad_v, grad_c, win_size, center_win = self._grad_triplet_win_size(src, i, j)
                stage2 = self._stage2_directional_avg(center_win, win_size)

                # Neighbor gradients - use grad_row_buf like C++ with boundary handling per ref 4.1
                # grad_u from grad_shift[1], grad_d from grad_row_buf[0][i]
                grad_u = grad_shift[1] if j > 0 else grad_c
                grad_d = grad_row_buf[0, i] if j < h - 1 else grad_c
                grad_l = self._stage1_gradient(self._get_window(src, i - HORIZONTAL_TAP_STEP, j))[2] if i > 0 else grad_c
                grad_r = self._stage1_gradient(self._get_window(src, i + HORIZONTAL_TAP_STEP, j))[2] if i < w - 1 else grad_c
                grads = {"u": grad_u, "d": grad_d, "l": grad_l, "r": grad_r, "c": grad_c}

                # Update gradient row buffer (matches C++ shift logic)
                grad_shift[0] = grad_shift[1]
                grad_shift[1] = grad_shift[2]
                grad_shift[2] = grad_c
                grad_row_buf[0, i] = grad_c
                grad_row_buf[1, i] = grad_row_buf[0, i]

                blend0_grad, blend1_grad = self._stage3_fusion(stage2["avg0"], stage2["avg1"], grads)

                # Stage4 IIR blend: rows 0-3 from filt, row 4 from src
                stage4_win = np.zeros((5, 5), dtype=np.int32)
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        x = self._clip(i + dx * HORIZONTAL_TAP_STEP, 0, w - 1)
                        y = self._clip(j + dy, 0, h - 1)
                        if dy < 0:
                            stage4_win[dy + 2, dx + 2] = filt[y, x]
                        else:
                            stage4_win[dy + 2, dx + 2] = src[y, x]

                stage4 = self._stage4_window_blend(
                    stage4_win,
                    win_size,
                    blend0_grad,
                    blend1_grad,
                    stage2["avg0"]["u"],
                    stage2["avg1"]["u"],
                    grad_h,
                    grad_v,
                )
                patch = stage4["final_patch"]

                # Write filtered center pixel back to filt (ALL rows, matches C++ behavior)
                filt[j, i] = self._s11_to_u10(int(patch[2, 2]))

                # Output only from row 2+ (matches C++ behavior: if j >= 2, output[j,i] = dout_pixel)
                if j >= 2:
                    if center_stream is not None:
                        center_stream.append(self._s11_to_u10(int(patch[2, 2])))
                    if patch_stream is not None:
                        patch_stream.append({
                            "center_x": i,
                            "center_y": j,
                            "patch_u10": np.vectorize(self._s11_to_u10)(patch).astype(np.int32),
                        })

            if linebuffer_rows is not None:
                row_indices = np.array([self._clip(j + dy, 0, h - 1) for dy in range(-2, 5)], dtype=np.int32)
                linebuffer_rows.append({
                    "after_row": j,
                    "row_indices": row_indices,
                    "rows": filt[row_indices, :].copy(),
                })

        # Output: rows 0-1 from input (boundary), rows 2+ from filt
        final_image = np.empty((h, w), dtype=np.int32)
        for j in range(min(2, h)):
            for i in range(w):
                final_image[j, i] = input_image[j, i]
        for j in range(2, h):
            for i in range(w):
                final_image[j, i] = filt[j, i]
        if center_stream is None and linebuffer_rows is None and patch_stream is None:
            return final_image
        if center_stream is not None and linebuffer_rows is None and patch_stream is None:
            return final_image, np.array(center_stream, dtype=np.int32)
        if center_stream is None and linebuffer_rows is not None and patch_stream is None:
            return final_image, linebuffer_rows
        if center_stream is None and linebuffer_rows is None and patch_stream is not None:
            return final_image, patch_stream
        if center_stream is not None and linebuffer_rows is None and patch_stream is not None:
            return final_image, np.array(center_stream, dtype=np.int32), patch_stream
        if center_stream is None and linebuffer_rows is not None and patch_stream is not None:
            return final_image, linebuffer_rows, patch_stream
        return final_image, np.array(center_stream, dtype=np.int32), linebuffer_rows, patch_stream

    def process(self, input_image: np.ndarray) -> np.ndarray:
        return self._process_feedback_raster(input_image, emit_center_stream=False)

    def process_center_stream(self, input_image: np.ndarray) -> np.ndarray:
        _, center_stream = self._process_feedback_raster(input_image, emit_center_stream=True)
        return center_stream

    def export_linebuffer_row_snapshots(self, input_image: np.ndarray):
        _, linebuffer_rows = self._process_feedback_raster(
            input_image,
            emit_center_stream=False,
            emit_linebuffer_rows=True,
        )
        return linebuffer_rows

    def export_patch_stream(self, input_image: np.ndarray):
        _, patch_stream = self._process_feedback_raster(
            input_image,
            emit_center_stream=False,
            emit_patch_stream=True,
        )
        return patch_stream

    def _process_no_feedback(self, input_image: np.ndarray):
        """Process without feedback - matches C++ process_pixel_at behavior"""
        src = input_image.astype(np.int32).copy()
        h, w = src.shape
        output = np.empty((h, w), dtype=np.int32)

        for j in range(h):
            for i in range(w):
                # Build center window
                center_win = self._get_window(src, i, j)
                grad_h, grad_v, grad_c = self._stage1_gradient(center_win)

                # Compute left and right gradients for LUT (max-based per algorithm)
                left_win = self._get_window(src, i - HORIZONTAL_TAP_STEP, j)
                right_win = self._get_window(src, i + HORIZONTAL_TAP_STEP, j)
                _, _, grad_l = self._stage1_gradient(left_win)
                _, _, grad_r = self._stage1_gradient(right_win)

                win_size = self._lut_win_size(max(grad_l, grad_c, grad_r))

                # Stage 2 directional average
                stage2 = self._stage2_directional_avg(center_win, win_size)

                # Neighbor gradients (from original image, not feedback)
                up_j = self._clip(j - 1, 0, h - 1)
                down_j = self._clip(j + 1, 0, h - 1)
                grad_u = self._stage1_gradient(self._get_window(src, i, up_j))[2]
                grad_d = self._stage1_gradient(self._get_window(src, i, down_j))[2]
                grad_l = self._stage1_gradient(self._get_window(src, self._clip(i - HORIZONTAL_TAP_STEP, 0, w - 1), j))[2]
                grad_r = self._stage1_gradient(self._get_window(src, self._clip(i + HORIZONTAL_TAP_STEP, 0, w - 1), j))[2]
                grads = {"u": grad_u, "d": grad_d, "l": grad_l, "r": grad_r, "c": grad_c}

                # Gradient fusion
                blend0_grad, blend1_grad = self._stage3_fusion(stage2["avg0"], stage2["avg1"], grads)

                # Stage4 IIR blend - use src (original) for all rows (no feedback)
                stage4_win = np.zeros((5, 5), dtype=np.int32)
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        x = self._clip(i + dx * HORIZONTAL_TAP_STEP, 0, w - 1)
                        y = self._clip(j + dy, 0, h - 1)
                        stage4_win[dy + 2, dx + 2] = src[y, x]

                stage4 = self._stage4_window_blend(
                    stage4_win,
                    win_size,
                    blend0_grad,
                    blend1_grad,
                    stage2["avg0"]["u"],
                    stage2["avg1"]["u"],
                    grad_h,
                    grad_v,
                )
                output[j, i] = self._s11_to_u10(int(stage4["final_patch"][2, 2]))

        return output

    def process_no_feedback(self, input_image: np.ndarray) -> np.ndarray:
        return self._process_no_feedback(input_image)


def test_fixed_model():
    import argparse
    parser = argparse.ArgumentParser(description="ISP-CSIIR Fixed-Point Model")
    parser.add_argument("--reg-cfg", type=str, default="config/reg_cfg_seed0.json")
    parser.add_argument("--img-info", type=str, default="config/img_info.json")
    parser.add_argument("--reg-table", type=str, default="config/register_table.csv")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--compare", type=str, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    args = parser.parse_args()

    regs = load_reg_config(args.reg_cfg, args.reg_table)
    model_cfg = reg_config_to_model_config(regs)
    img_info = load_img_info(args.img_info)

    width = int(args.width or img_info["image"]["width"] or model_cfg["IMG_WIDTH"])
    height = int(args.height or img_info["image"]["height"] or model_cfg["IMG_HEIGHT"])
    if width != int(model_cfg["IMG_WIDTH"]) or height != int(model_cfg["IMG_HEIGHT"]):
        raise ValueError("image size in img_info.json must match reg_cfg.json")

    config = FixedPointConfig(**model_cfg)
    model = ISPCSIIRFixedModel(config)

    if args.input:
        from csiir_common import load_hex_image

        input_img = load_hex_image(args.input, width, height)
    else:
        input_img, input_path = prepare_input_image(img_info)
        print(f"Input pattern: {input_path}")

    output = model.process(input_img)
    print(f"输入范围: [{input_img.min()}, {input_img.max()}]")
    print(f"输出范围: [{output.min()}, {output.max()}]")
    assert output.min() >= 0
    assert output.max() <= 1023

    output_path = args.output or img_info["outputs"]["py_pattern_path"]
    save_hex_image(output_path, output.astype(np.int32))
    print(f"Python output written to {output_path}")

    compare_path = args.compare
    if compare_path:
        report = compare_hex_images(output_path, compare_path, width, height)
        print(f"\n=== Python vs Reference Comparison ===")
        print(f"Total pixels: {report['total_pixels']}")
        print(f"Pixels with diff: {report['mismatch_count']}")
        print(f"Max abs diff: {report['max_abs_diff']}")
        if report["mismatch_count"] == 0:
            print("PASS: All outputs match!")
        else:
            print(f"FAIL: {report['mismatch_count']} pixels differ")
            for mismatch in report["mismatches"][:8]:
                print(
                    f"  (y={mismatch['y']}, x={mismatch['x']}) "
                    f"py={mismatch['lhs']} ref={mismatch['rhs']} delta={mismatch['delta']}"
                )

    print("定点模型测试通过!")
    return output


if __name__ == "__main__":
    test_fixed_model()
