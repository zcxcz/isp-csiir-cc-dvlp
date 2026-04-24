#!/usr/bin/env python3
"""
ISP-CSIIR 浮点参考模型

基于 isp-csiir-ref.md 算法参考文档实现的完整浮点模型。
包含四阶段流水线处理逻辑：
  - Stage 1: 梯度计算与窗口大小确定
  - Stage 2: 多尺度方向性平均
  - Stage 3: 梯度加权方向融合
  - Stage 4: IIR滤波与混合输出

IIR 反馈机制 (v1.1):
  - 输出写回像素存储 (src_uv)，作为下一轮迭代输入
  - 常规像素: 写回 5x1 (当前列的上下 5 行)
  - 尾列像素 (最后 3 列): 写回 5x3

作者: rtl-algo 职能代理
日期: 2026-03-22
版本: v1.1
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional
import copy

HORIZONTAL_TAP_STEP = 2


@dataclass
class ISPConfig:
    """ISP-CSIIR 配置参数"""
    width: int = 64
    height: int = 64

    # 窗口大小阈值
    win_size_thresh0: int = 16
    win_size_thresh1: int = 24
    win_size_thresh2: int = 32
    win_size_thresh3: int = 40

    # 梯度裁剪阈值
    win_size_clip_y: List[int] = None
    win_size_clip_sft: List[int] = None

    # IIR 混合比例
    blending_ratio: List[int] = None

    # 边缘保护参数
    edge_protect: int = 32  # reg_edge_protect

    def __post_init__(self):
        if self.win_size_clip_y is None:
            self.win_size_clip_y = [15, 23, 31, 39]
        if self.win_size_clip_sft is None:
            self.win_size_clip_sft = [2, 2, 2, 2]
        if self.blending_ratio is None:
            self.blending_ratio = [32, 32, 32, 32]


class ISPCSIIRFloatModel:
    """
    ISP-CSIIR 浮点参考模型

    实现完整的四阶段处理流程，用于算法验证和 RTL 对比。
    """

    def __init__(self, config: ISPConfig = None):
        """初始化模型"""
        self.config = config if config else ISPConfig()

        # Sobel 滤波器定义
        self.sobel_x = np.array([
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [-1, -1, -1, -1, -1]
        ], dtype=np.float64)

        self.sobel_y = np.array([
            [1, 0, 0, 0, -1],
            [1, 0, 0, 0, -1],
            [1, 0, 0, 0, -1],
            [1, 0, 0, 0, -1],
            [1, 0, 0, 0, -1]
        ], dtype=np.float64)

        # 平均因子核定义
        self._init_avg_factor_kernels()

        # 方向掩码定义
        self._init_direction_masks()

        # 混合因子核定义
        self._init_blend_factor_kernels()

        # IIR 状态存储
        self.src_uv = None  # 反馈更新的源图像

    def _init_avg_factor_kernels(self):
        """初始化平均因子核"""
        # 2x2 核
        self.avg_factor_c_2x2 = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 2, 1, 0],
            [0, 2, 4, 2, 0],
            [0, 1, 2, 1, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.float64)

        # 3x3 核
        self.avg_factor_c_3x3 = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.float64)

        # 4x4 核
        self.avg_factor_c_4x4 = np.array([
            [1, 1, 2, 1, 1],
            [1, 2, 4, 2, 1],
            [2, 4, 8, 4, 2],
            [1, 2, 4, 2, 1],
            [1, 1, 2, 1, 1]
        ], dtype=np.float64)

        # 5x5 核
        self.avg_factor_c_5x5 = np.array([
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1]
        ], dtype=np.float64)

    def _init_direction_masks(self):
        """初始化方向掩码"""
        # 上方向掩码
        self.avg_factor_u_mask = np.array([
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.float64)

        # 下方向掩码
        self.avg_factor_d_mask = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1]
        ], dtype=np.float64)

        # 左方向掩码
        self.avg_factor_l_mask = np.array([
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0]
        ], dtype=np.float64)

        # 右方向掩码
        self.avg_factor_r_mask = np.array([
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1]
        ], dtype=np.float64)

    def _init_blend_factor_kernels(self):
        """初始化混合因子核"""
        # 2x2 混合因子 (水平)
        self.blend_factor_2x2_h = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.float64)

        # 2x2 混合因子 (垂直)
        self.blend_factor_2x2_v = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.float64)

        # 2x2 混合因子 (中心)
        self.blend_factor_2x2 = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 2, 1, 0],
            [0, 2, 4, 2, 0],
            [0, 1, 2, 1, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.float64)

        # 3x3 混合因子
        self.blend_factor_3x3 = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.float64)

        # 4x4 混合因子
        self.blend_factor_4x4 = np.array([
            [1, 2, 2, 2, 1],
            [2, 4, 4, 4, 2],
            [2, 4, 4, 4, 2],
            [2, 4, 4, 4, 2],
            [1, 2, 2, 2, 1]
        ], dtype=np.float64)

        # 5x5 混合因子
        self.blend_factor_5x5 = np.array([
            [4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4]
        ], dtype=np.float64)

    def _clip(self, value: float, min_val: float, max_val: float) -> float:
        """限幅函数"""
        return max(min_val, min(max_val, value))

    def _get_5x5_window(self, img: np.ndarray, i: int, j: int) -> np.ndarray:
        """
        获取以 (i, j) 为中心的 5x5 窗口
        边界处理: 复制边界像素
        """
        height, width = img.shape
        window = np.zeros((5, 5), dtype=np.float64)

        for h in range(-2, 3):
            for w in range(-2, 3):
                # 边界复制
                y = self._clip(j + h, 0, height - 1)
                x = self._clip(i + w * HORIZONTAL_TAP_STEP, 0, width - 1)
                window[h + 2, w + 2] = img[int(y), int(x)]

        return window

    def _stage1_gradient(self, src_uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stage 1: 梯度计算与窗口大小确定

        Returns:
            grad: 综合梯度图
            win_size_clip: 窗口大小图
        """
        height, width = src_uv.shape
        grad_h = np.zeros((height, width), dtype=np.float64)
        grad_v = np.zeros((height, width), dtype=np.float64)
        grad = np.zeros((height, width), dtype=np.float64)
        win_size_clip = np.zeros((height, width), dtype=np.float64)

        # 计算梯度
        for j in range(height):
            for i in range(width):
                win = self._get_5x5_window(src_uv, i, j)

                # 水平梯度
                grad_h[j, i] = np.sum(win * self.sobel_x)
                # 垂直梯度
                grad_v[j, i] = np.sum(win * self.sobel_y)
                # 综合梯度
                grad[j, i] = abs(grad_h[j, i]) / 5 + abs(grad_v[j, i]) / 5

        # 计算窗口大小 (使用三邻域最大梯度)
        for j in range(height):
            for i in range(width):
                # 获取左右梯度
                grad_left = grad[j, self._clip(i - HORIZONTAL_TAP_STEP, 0, width - 1)]
                grad_center = grad[j, i]
                grad_right = grad[j, self._clip(i + HORIZONTAL_TAP_STEP, 0, width - 1)]

                max_grad = max(grad_left, grad_center, grad_right)

                # LUT 查表确定窗口大小
                win_size = self._lut_win_size(max_grad)

                # 裁剪到 [16, 40]
                win_size_clip[j, i] = self._clip(win_size, 16, 40)

        return grad, win_size_clip

    def _lut_win_size(self, max_grad: float) -> float:
        """
        窗口大小 LUT 查表

        根据 win_size_clip_y 和 win_size_clip_sft 计算
        """
        clip_y = self.config.win_size_clip_y
        clip_sft = self.config.win_size_clip_sft
        x_nodes = []
        acc = 0.0
        for shift in clip_sft:
            acc += float(1 << int(shift))
            x_nodes.append(acc)

        if max_grad <= x_nodes[0]:
            win_size = float(clip_y[0])
        elif max_grad >= x_nodes[-1]:
            win_size = float(clip_y[-1])
        else:
            win_size = float(clip_y[-1])
            for idx in range(len(x_nodes) - 1):
                x0 = x_nodes[idx]
                x1 = x_nodes[idx + 1]
                if x0 <= max_grad <= x1:
                    y0 = float(clip_y[idx])
                    y1 = float(clip_y[idx + 1])
                    win_size = y0 + (max_grad - x0) * (y1 - y0) / (x1 - x0)
                    break

        return self._clip(win_size, 16, 40)

    def _stage2_directional_avg(self, src_uv: np.ndarray,
                                 win_size_clip: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stage 2: 多尺度方向性平均

        Returns:
            avg0: 尺度0方向平均值 (5方向)
            avg1: 尺度1方向平均值 (5方向)
        """
        height, width = src_uv.shape

        # 5个方向: c(中心), u(上), d(下), l(左), r(右)
        avg0 = np.zeros((height, width, 5), dtype=np.float64)
        avg1 = np.zeros((height, width, 5), dtype=np.float64)

        for j in range(height):
            for i in range(width):
                win = self._get_5x5_window(src_uv, i, j)
                ws = win_size_clip[j, i]

                # 核选择
                avg0_kernel, avg1_kernel = self._select_kernels(ws)

                # 计算各方向平均值
                # 方向索引: 0=c, 1=u, 2=d, 3=l, 4=r
                for d, mask in enumerate([None, self.avg_factor_u_mask,
                                          self.avg_factor_d_mask,
                                          self.avg_factor_l_mask,
                                          self.avg_factor_r_mask]):
                    if d == 0:
                        # 中心方向
                        avg0[j, i, 0] = self._compute_avg(win, avg0_kernel)
                        avg1[j, i, 0] = self._compute_avg(win, avg1_kernel)
                    else:
                        # 其他方向
                        avg0_kernel_d = avg0_kernel * mask
                        avg1_kernel_d = avg1_kernel * mask
                        avg0[j, i, d] = self._compute_avg(win, avg0_kernel_d)
                        avg1[j, i, d] = self._compute_avg(win, avg1_kernel_d)

        return avg0, avg1

    def _select_kernels(self, win_size: float) -> Tuple[np.ndarray, np.ndarray]:
        """根据窗口大小选择核"""
        t0, t1, t2, t3 = (self.config.win_size_thresh0, self.config.win_size_thresh1,
                          self.config.win_size_thresh2, self.config.win_size_thresh3)

        if win_size < t0:
            return np.zeros((5, 5)), self.avg_factor_c_2x2.copy()
        elif win_size < t1:
            return self.avg_factor_c_3x3.copy(), self.avg_factor_c_2x2.copy()
        elif win_size < t2:
            return self.avg_factor_c_4x4.copy(), self.avg_factor_c_3x3.copy()
        elif win_size < t3:
            return self.avg_factor_c_5x5.copy(), self.avg_factor_c_4x4.copy()
        else:
            return self.avg_factor_c_5x5.copy(), np.zeros((5, 5))

    def _compute_avg(self, win: np.ndarray, kernel: np.ndarray) -> float:
        """计算加权平均值"""
        weight_sum = np.sum(kernel)
        if weight_sum == 0:
            return 0.0

        weighted_sum = np.sum(win * kernel)
        return weighted_sum / weight_sum

    def _stage3_gradient_fusion(self, avg0: np.ndarray, avg1: np.ndarray,
                                 grad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stage 3: 梯度加权方向融合

        Returns:
            blend0_grad: 尺度0融合结果
            blend1_grad: 尺度1融合结果
        """
        height, width = grad.shape
        blend0_grad = np.zeros((height, width), dtype=np.float64)
        blend1_grad = np.zeros((height, width), dtype=np.float64)

        for j in range(height):
            for i in range(width):
                # 获取各方向梯度
                grad_c = grad[j, i]
                grad_u = grad[self._clip(j - 1, 0, height - 1), i] if j > 0 else grad_c
                grad_d = grad[self._clip(j + 1, 0, height - 1), i] if j < height - 1 else grad_c
                grad_l = grad[j, self._clip(i - HORIZONTAL_TAP_STEP, 0, width - 1)] if i >= HORIZONTAL_TAP_STEP else grad_c
                grad_r = grad[j, self._clip(i + HORIZONTAL_TAP_STEP, 0, width - 1)] if i + HORIZONTAL_TAP_STEP < width else grad_c

                # 逆序排序
                grads = [grad_u, grad_d, grad_l, grad_r, grad_c]
                grads_sorted = sorted(grads, reverse=True)

                grad_sum = sum(grads_sorted)

                # 获取各方向平均值
                avg0_c, avg0_u, avg0_d, avg0_l, avg0_r = avg0[j, i]
                avg1_c, avg1_u, avg1_d, avg1_l, avg1_r = avg1[j, i]

                if grad_sum == 0:
                    # 简单平均
                    blend0_grad[j, i] = (avg0_c + avg0_u + avg0_d + avg0_l + avg0_r) / 5
                    blend1_grad[j, i] = (avg1_c + avg1_u + avg1_d + avg1_l + avg1_r) / 5
                else:
                    # 梯度加权融合
                    blend0_grad[j, i] = (avg0_c * grads_sorted[4] + avg0_u * grads_sorted[0] +
                                         avg0_d * grads_sorted[1] + avg0_l * grads_sorted[2] +
                                         avg0_r * grads_sorted[3]) / grad_sum
                    blend1_grad[j, i] = (avg1_c * grads_sorted[4] + avg1_u * grads_sorted[0] +
                                         avg1_d * grads_sorted[1] + avg1_l * grads_sorted[2] +
                                         avg1_r * grads_sorted[3]) / grad_sum

        return blend0_grad, blend1_grad

    def _stage4_iir_blend(self, src_uv: np.ndarray, blend0_grad: np.ndarray,
                          blend1_grad: np.ndarray, avg0: np.ndarray, avg1: np.ndarray,
                          win_size_clip: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """
        Stage 4: IIR滤波与混合输出

        关键特性: 包含 IIR 反馈路径
        - 输出写回像素存储 (src_uv)，作为下一轮迭代输入
        - 常规像素: 写回 5x1 (当前列的上下 5 行)
        - 尾列像素: 写回 5x3 (最后 3 列的特殊处理)

        Returns:
            output: 输出图像
        """
        height, width = src_uv.shape
        output = np.zeros((height, width), dtype=np.float64)

        # IIR 行缓存 (存储上一行的 avg0_u 和 avg1_u)
        avg0_u_prev_row = np.zeros(width, dtype=np.float64)
        avg1_u_prev_row = np.zeros(width, dtype=np.float64)

        for j in range(height):
            # 当前行的 avg_u 值 (用于下一行 IIR 混合)
            avg0_u_curr_row = avg0[j, :, 1].copy()  # avg0[:, :, 1] 是 avg0_u
            avg1_u_curr_row = avg1[j, :, 1].copy()  # avg1[:, :, 1] 是 avg1_u

            for i in range(width):
                ws = win_size_clip[j, i]

                # 获取混合比例
                blend_ratio_idx = int(ws / 8) - 2
                blend_ratio_idx = self._clip(blend_ratio_idx, 0, 3)
                ratio = self.config.blending_ratio[blend_ratio_idx]

                # 水平混合 (IIR 特性: 使用上一行的 avg_u)
                blend0_hor = (ratio * blend0_grad[j, i] +
                             (64 - ratio) * avg0_u_prev_row[i]) / 64
                blend1_hor = (ratio * blend1_grad[j, i] +
                             (64 - ratio) * avg1_u_prev_row[i]) / 64

                # 窗混合
                win = self._get_5x5_window(self.src_uv, i, j)
                center_pixel = self.src_uv[j, i]

                blend0_win, blend1_win = self._compute_window_blend(
                    win, blend0_hor, blend1_hor, center_pixel, ws, grad[j, i]
                )

                # 最终混合
                win_size_remain_8 = ws % 8

                if ws < self.config.win_size_thresh0:
                    blend_uv = blend0_win
                elif ws >= self.config.win_size_thresh3:
                    blend_uv = blend1_win
                else:
                    blend_uv = (blend0_win * win_size_remain_8 +
                               blend1_win * (8 - win_size_remain_8)) / 8

                # 输出限幅
                output[j, i] = self._clip(blend_uv, 0, 1023)

                # IIR 反馈: 输出写回 src_uv
                # 这是真正的 IIR 特性：输出反馈为输入
                self._iir_writeback(i, j, output[j, i], height, width)

            # 更新 IIR 行缓存
            avg0_u_prev_row = avg0_u_curr_row
            avg1_u_prev_row = avg1_u_curr_row

        return output

    def _iir_writeback(self, i: int, j: int, blend_uv: float, height: int, width: int):
        """
        IIR 写回逻辑

        根据位置执行不同的写回策略:
        - 常规像素: 写回 5x1 (当前列的上下 5 行)
        - 尾列像素 (最后 3 列): 写回 5x3

        Args:
            i: 当前列索引
            j: 当前行索引
            blend_uv: 输出值
            height: 图像高度
            width: 图像宽度
        """
        # 判断是否为尾列
        is_tail_col = (i >= width - 3)

        if is_tail_col:
            # 尾列处理: 写回 5x3 (3 列 x 5 行)
            for h in range(-2, 3):
                for w in range(-2, 3):
                    row_idx = j + h
                    col_idx = i + w
                    if 0 <= row_idx < height and 0 <= col_idx < width:
                        self.src_uv[row_idx, col_idx] = blend_uv
        else:
            # 常规像素: 写回 5x1 (当前列的上下 5 行)
            for h in range(-2, 3):
                row_idx = j + h
                if 0 <= row_idx < height:
                    self.src_uv[row_idx, i] = blend_uv

    def _compute_window_blend(self, win: np.ndarray, blend0_hor: float,
                               blend1_hor: float, center_pixel: float,
                               win_size: float, grad_val: float) -> Tuple[float, float]:
        """计算窗混合"""
        t0, t1, t2, t3 = (self.config.win_size_thresh0, self.config.win_size_thresh1,
                          self.config.win_size_thresh2, self.config.win_size_thresh3)

        # 选择混合因子核
        # 根据梯度方向选择水平或垂直
        blend_factor_2x2_hv = self.blend_factor_2x2_h  # 简化: 默认使用水平

        if win_size < t0:
            blend00 = self._blend_window(win, blend0_hor, blend_factor_2x2_hv)
            blend01 = self._blend_window(win, blend0_hor, self.blend_factor_2x2)
            blend0_win = (blend00 * self.config.edge_protect +
                         blend01 * (64 - self.config.edge_protect)) / 64
            blend1_win = blend0_win  # 同样的处理
        elif win_size < t1:
            blend00 = self._blend_window(win, blend0_hor, blend_factor_2x2_hv)
            blend01 = self._blend_window(win, blend0_hor, self.blend_factor_2x2)
            blend0_win = (blend00 * self.config.edge_protect +
                         blend01 * (64 - self.config.edge_protect)) / 64
            blend1_win = self._blend_window(win, blend1_hor, self.blend_factor_3x3)
        elif win_size < t2:
            blend0_win = self._blend_window(win, blend0_hor, self.blend_factor_3x3)
            blend1_win = self._blend_window(win, blend1_hor, self.blend_factor_4x4)
        elif win_size < t3:
            blend0_win = self._blend_window(win, blend0_hor, self.blend_factor_4x4)
            blend1_win = self._blend_window(win, blend1_hor, self.blend_factor_5x5)
        else:
            blend0_win = self._blend_window(win, blend0_hor, self.blend_factor_5x5)
            blend1_win = blend0_win

        return blend0_win, blend1_win

    def _blend_window(self, win: np.ndarray, blend_val: float,
                       blend_factor: np.ndarray) -> float:
        """窗口混合计算"""
        factor_sum = np.sum(blend_factor)
        if factor_sum == 0:
            return blend_val

        result = np.sum(blend_val * blend_factor + win * (4 - blend_factor)) / (factor_sum * 4 / factor_sum)
        # 简化计算
        blend_part = np.sum(blend_val * blend_factor)
        src_part = np.sum(win * (4 - blend_factor))
        total_factor = np.sum(blend_factor) + np.sum(4 - blend_factor)

        return (blend_part + src_part) / total_factor

    def process(self, input_image: np.ndarray) -> np.ndarray:
        """
        处理输入图像

        Args:
            input_image: 输入图像 (height x width), 值范围 [0, 1023]

        Returns:
            output_image: 输出图像 (height x width), 值范围 [0, 1023]
        """
        # 初始化 src_uv (IIR 反馈需要)
        self.src_uv = input_image.astype(np.float64).copy()

        # Stage 1: 梯度计算与窗口大小确定
        grad, win_size_clip = self._stage1_gradient(self.src_uv)

        # Stage 2: 多尺度方向性平均
        avg0, avg1 = self._stage2_directional_avg(self.src_uv, win_size_clip)

        # Stage 3: 梯度加权方向融合
        blend0_grad, blend1_grad = self._stage3_gradient_fusion(avg0, avg1, grad)

        # Stage 4: IIR滤波与混合输出
        output = self._stage4_iir_blend(
            self.src_uv, blend0_grad, blend1_grad,
            avg0, avg1, win_size_clip, grad
        )

        return output


def test_float_model():
    """测试浮点模型"""
    # 创建配置
    config = ISPConfig(width=64, height=64)

    # 创建模型
    model = ISPCSIIRFloatModel(config)

    # 创建测试图像
    np.random.seed(42)
    input_image = np.random.randint(0, 1024, (64, 64), dtype=np.int32)

    # 处理
    output = model.process(input_image.astype(np.float64))

    print(f"输入图像范围: [{input_image.min()}, {input_image.max()}]")
    print(f"输出图像范围: [{output.min():.2f}, {output.max():.2f}]")
    print(f"输出图像平均值: {output.mean():.2f}")
    print(f"输出图像标准差: {output.std():.2f}")

    return output


if __name__ == "__main__":
    test_float_model()
