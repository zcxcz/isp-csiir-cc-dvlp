#!/usr/bin/env python3
"""
ISP-CSIIR realtime fixed-point model aligned to the current RTL frontend order.
"""

import time
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

import numpy as np

from isp_csiir_fixed_model import FixedPointConfig, ISPCSIIRFixedModel


class ISPCSIIRRealtimeModel(ISPCSIIRFixedModel):
    STAGE1_LATENCY_CYCLES = 5
    STAGE2_LATENCY_CYCLES = 4
    STAGE3_LATENCY_CYCLES = 8
    STAGE4_LATENCY_CYCLES = 5

    def __init__(self, config: FixedPointConfig = None, timeout_sec: Optional[float] = None):
        super().__init__(config)
        self.timeout_sec = timeout_sec
        self._active_timeout_sec: Optional[float] = None
        self._deadline: Optional[float] = None

    def _begin_timeout_scope(self, timeout_sec: Optional[float]) -> None:
        effective_timeout = self.timeout_sec if timeout_sec is None else timeout_sec
        if effective_timeout is None or effective_timeout <= 0:
            self._active_timeout_sec = None
            self._deadline = None
            return
        self._active_timeout_sec = float(effective_timeout)
        self._deadline = time.monotonic() + self._active_timeout_sec

    def _end_timeout_scope(self) -> None:
        self._active_timeout_sec = None
        self._deadline = None

    def _check_timeout(self, description: str, *, cycle: Optional[int] = None) -> None:
        if self._deadline is None or time.monotonic() <= self._deadline:
            return
        cycle_text = f" (cycle={int(cycle)})" if cycle is not None else ""
        raise TimeoutError(
            f"{description} timed out after {self._active_timeout_sec:.1f}s{cycle_text}"
        )

    def _load_assembler_patch(
        self,
        row_mem,
        center_abs_x: int,
        *,
        live_col: Optional[np.ndarray] = None,
        live_center_x: Optional[int] = None,
    ) -> np.ndarray:
        taps = [
            self._clip(center_abs_x - 4, 0, self.config.IMG_WIDTH - 1),
            self._clip(center_abs_x - 2, 0, self.config.IMG_WIDTH - 1),
            center_abs_x,
            self._clip(center_abs_x + 2, 0, self.config.IMG_WIDTH - 1),
            self._clip(center_abs_x + 4, 0, self.config.IMG_WIDTH - 1),
        ]
        patch = np.zeros((5, 5), dtype=np.int32)
        for row_idx in range(5):
            for col_idx, tap_x in enumerate(taps):
                if live_col is not None and live_center_x is not None and int(tap_x) == int(live_center_x):
                    patch[row_idx, col_idx] = int(live_col[row_idx])
                else:
                    patch[row_idx, col_idx] = int(row_mem[row_idx][tap_x])
        return patch

    def _row_trace_filter_match(self, row_filter: Optional[set[int]], center_y: int) -> bool:
        return row_filter is None or int(center_y) in row_filter

    def _build_final_patch_and_meta(self, accepted_rows, center_y: int, center_x: int):
        center_patch = accepted_rows[center_y][center_x]
        left_patch = accepted_rows[center_y][self._clip(center_x - 2, 0, self.config.IMG_WIDTH - 1)]
        right_patch = accepted_rows[center_y][self._clip(center_x + 2, 0, self.config.IMG_WIDTH - 1)]
        up_patch = accepted_rows[self._clip(center_y - 1, 0, self.config.IMG_HEIGHT - 1)][center_x]
        down_patch = accepted_rows[self._clip(center_y + 1, 0, self.config.IMG_HEIGHT - 1)][center_x]

        grad_h, grad_v, grad_c = self._stage1_gradient(center_patch)
        _, _, grad_l = self._stage1_gradient(left_patch)
        _, _, grad_r = self._stage1_gradient(right_patch)
        _, _, grad_u = self._stage1_gradient(up_patch)
        _, _, grad_d = self._stage1_gradient(down_patch)

        win_size = self._lut_win_size(max(grad_l, grad_c, grad_r))
        stage2 = self._stage2_directional_avg(center_patch, win_size)
        stage3_state = self._stage3_blend_state(
            stage2["avg0"],
            stage2["avg1"],
            {
                "c": int(grad_c),
                "u": int(grad_u),
                "d": int(grad_d),
                "l": int(grad_l),
                "r": int(grad_r),
            },
        )
        stage4 = self._stage4_window_blend(
            center_patch,
            win_size,
            int(stage3_state["blend0"]),
            int(stage3_state["blend1"]),
            int(stage2["avg0"]["u"]),
            int(stage2["avg1"]["u"]),
            grad_h,
            grad_v,
        )
        final_patch = np.vectorize(self._s11_to_u10)(stage4["final_patch"]).astype(np.int32)
        meta = {
            "center_x": int(center_x),
            "center_y": int(center_y),
            "win_size": int(win_size),
            "grad_h": int(abs(grad_h)),
            "grad_v": int(abs(grad_v)),
            "blend0": int(stage3_state["blend0"]),
            "blend1": int(stage3_state["blend1"]),
            "avg0_u": int(stage2["avg0"]["u"]),
            "avg1_u": int(stage2["avg1"]["u"]),
            "src_patch_u10": center_patch.copy(),
            "g_raw": {
                "c": int(grad_c),
                "u": int(grad_u),
                "d": int(grad_d),
                "l": int(grad_l),
                "r": int(grad_r),
            },
            "grad_inv": {
                "c": int(stage3_state["grad_inv"]["c"]),
                "u": int(stage3_state["grad_inv"]["u"]),
                "d": int(stage3_state["grad_inv"]["d"]),
                "l": int(stage3_state["grad_inv"]["l"]),
                "r": int(stage3_state["grad_inv"]["r"]),
            },
            "avg0": {key: int(stage2["avg0"][key]) for key in ("c", "u", "d", "l", "r")},
            "avg1": {key: int(stage2["avg1"][key]) for key in ("c", "u", "d", "l", "r")},
            "grad_sum": int(stage3_state["grad_sum"]),
            "blend0_total": int(stage3_state["blend0_total"]),
            "blend1_total": int(stage3_state["blend1_total"]),
        }
        return meta, final_patch

    def _build_stage12_meta(self, accepted_rows, center_y: int, center_x: int) -> Dict:
        center_patch = accepted_rows[center_y][center_x]
        left_patch = accepted_rows[center_y][self._clip(center_x - 2, 0, self.config.IMG_WIDTH - 1)]
        right_patch = accepted_rows[center_y][self._clip(center_x + 2, 0, self.config.IMG_WIDTH - 1)]
        grad_h, grad_v, grad_c = self._stage1_gradient(center_patch)
        _, _, grad_l = self._stage1_gradient(left_patch)
        _, _, grad_r = self._stage1_gradient(right_patch)
        win_size = self._lut_win_size(max(grad_l, grad_c, grad_r))
        stage2 = self._stage2_directional_avg(center_patch, win_size)
        return {
            "patch_u10": center_patch.copy(),
            "center_x": int(center_x),
            "center_y": int(center_y),
            "grad_h": int(grad_h),
            "grad_v": int(grad_v),
            "grad": int(grad_c),
            "grad_l": int(grad_l),
            "grad_r": int(grad_r),
            "win_size": int(win_size),
            "avg0": {key: int(stage2["avg0"][key]) for key in ("c", "u", "d", "l", "r")},
            "avg1": {key: int(stage2["avg1"][key]) for key in ("c", "u", "d", "l", "r")},
        }

    def _simulate_realtime_pipeline(
        self,
        input_image: np.ndarray,
        *,
        emit_center_stream: bool = False,
        emit_patch_stream: bool = False,
        emit_linebuffer_columns: bool = False,
        emit_stage1_patch_trace: bool = False,
        emit_stage3_raw_trace: bool = False,
        emit_stage3_input_trace: bool = False,
        emit_stage4_input_trace: bool = False,
        emit_feedback_commit_stream: bool = False,
        emit_frontend_column_trace: bool = False,
        emit_feedback_visible_trace: bool = False,
        emit_capture_read_trace: bool = False,
        center_rows: Optional[Iterable[int]] = None,
        timeout_sec: Optional[float] = None,
    ):
        src_visible = input_image.astype(np.int32).copy()
        h, w = src_visible.shape
        row_filter = self._normalize_row_filter(center_rows)

        accepted_rows = {row_idx: [] for row_idx in range(h)}
        stage1_patch_trace = [] if emit_stage1_patch_trace else None
        center_stream = [] if emit_center_stream else None
        patch_stream = [] if emit_patch_stream else None
        linebuffer_columns = [] if emit_linebuffer_columns else None
        stage3_raw_trace = [] if emit_stage3_raw_trace else None
        stage3_input_trace = [] if emit_stage3_input_trace else None
        stage4_input_trace = [] if emit_stage4_input_trace else None
        feedback_commit_stream = [] if emit_feedback_commit_stream else None
        frontend_column_trace = [] if emit_frontend_column_trace else None
        feedback_visible_trace = [] if emit_feedback_visible_trace else None
        capture_read_trace = [] if emit_capture_read_trace else None

        stage12_meta_cache: Dict[tuple[int, int], Dict] = {}

        stage1_due = defaultdict(list)
        stage2_due = defaultdict(list)
        stage3_due = defaultdict(list)
        stage4_due = defaultdict(list)
        feedback_visible_due = defaultdict(list)
        row_commit_due = defaultdict(list)

        def row_minus_wrap(base: int, delta: int) -> int:
            return int(base - delta) if base >= delta else int(base + 5 - delta)

        def stream_capture_rows(cur_wr_row_ptr: int, cur_row_cnt: int) -> tuple[int, int, int, int, int]:
            wr_row_prev = row_minus_wrap(cur_wr_row_ptr, 1)
            wr_row_prev2 = row_minus_wrap(cur_wr_row_ptr, 2)
            wr_row_prev3 = row_minus_wrap(cur_wr_row_ptr, 3)
            wr_row_prev4 = row_minus_wrap(cur_wr_row_ptr, 4)
            row0 = wr_row_prev2 if cur_row_cnt == 2 else (wr_row_prev3 if cur_row_cnt == 3 else wr_row_prev4)
            row1 = wr_row_prev2 if cur_row_cnt == 2 else (wr_row_prev3 if cur_row_cnt == 3 else wr_row_prev3)
            return (int(row0), int(row1), int(wr_row_prev2), int(wr_row_prev), int(cur_wr_row_ptr))

        def tail_capture_rows(cur_tail_base_ptr: int, cur_tail_center_y: int) -> tuple[int, int, int, int, int]:
            tail_row_prev1 = row_minus_wrap(cur_tail_base_ptr, 1)
            tail_row_prev2 = row_minus_wrap(cur_tail_base_ptr, 2)
            tail_row_prev3 = row_minus_wrap(cur_tail_base_ptr, 3)
            tail_row_prev4 = row_minus_wrap(cur_tail_base_ptr, 4)
            is_last_center = cur_tail_center_y == (h - 1)
            row0 = tail_row_prev3 if is_last_center else tail_row_prev4
            row1 = tail_row_prev2 if is_last_center else tail_row_prev3
            row2 = tail_row_prev1 if is_last_center else tail_row_prev2
            row3 = tail_row_prev1
            row4 = tail_row_prev1
            return (int(row0), int(row1), int(row2), int(row3), int(row4))

        s4_meta_patch_buf = [[None for _ in range(w)] for _ in range(2)]
        s4_meta_grad_h_buf = [[0 for _ in range(w)] for _ in range(2)]
        s4_meta_grad_v_buf = [[0 for _ in range(w)] for _ in range(2)]

        stage2_grad_buf = [[0 for _ in range(w)] for _ in range(2)]
        stage2_payload_buf: List[List[Optional[Dict]]] = [[None for _ in range(w)] for _ in range(2)]
        stage2_write_buf = 0
        stage2_col_counter = 0
        stage2_row_counter = 0
        stage2_row_valid = False
        stage1_prev_grad_valid = False
        stage1_prev_grad = 0
        flush_start_cycle = None
        flush_row_buf = None

        line_mem = [np.zeros(w, dtype=np.int32) for _ in range(5)]
        wr_row_ptr = 0
        wr_col_ptr = 0
        in_row = 0
        in_col = 0
        producer_state = "sof"
        gap_left = 0

        frame_started = False
        row_cnt = 0
        capture_pending = None
        column_valid = False
        col_reg = np.zeros(5, dtype=np.int32)
        center_x_reg = 0
        center_y_reg = 0
        tail_pending = False
        tail_active = False
        tail_base_ptr = 0
        tail_col_ptr = 0
        tail_center_y = 0
        tail_row_turnaround = False

        row_mem = [np.zeros(w, dtype=np.int32) for _ in range(5)]
        flush_active = False
        flush_remaining = 0
        flush_center_x = 0
        flush_center_y = 0
        patch_center_x_reg = 0
        patch_center_y_reg = 0
        patch_reg = np.zeros((5, 5), dtype=np.int32)
        window_valid = False

        feedback_committed_row = None
        cycle = 0

        def get_stage12_meta(center_y: int, center_x: int) -> Dict:
            key = (int(center_y), int(center_x))
            cached = stage12_meta_cache.get(key)
            if cached is None:
                cached = self._build_stage12_meta(accepted_rows, center_y, center_x)
                stage12_meta_cache[key] = cached
            return cached

        def schedule_stage3_from_read(
            read_cycle: int,
            read_buf: int,
            grad_u_buf: int,
            read_col: int,
            read_entry: Dict,
            grad_d_value: int,
            *,
            flush_mode: bool,
        ) -> None:
            grad_c = int(stage2_grad_buf[read_buf][read_col])
            grad_u = int(stage2_grad_buf[grad_u_buf][read_col])
            grad_l = int(stage2_grad_buf[read_buf][max(0, read_col - 2)])
            grad_r = int(stage2_grad_buf[read_buf][min(w - 1, read_col + 2)])
            if int(read_entry["center_y"]) == 0:
                grad_u = grad_c
            if flush_mode or int(read_entry["center_y"]) >= (h - 1):
                grad_d_value = grad_c

            grads = {
                "c": int(grad_c),
                "u": int(grad_u),
                "d": int(grad_d_value),
                "l": int(grad_l),
                "r": int(grad_r),
            }
            stage3_state = self._stage3_blend_state(read_entry["avg0"], read_entry["avg1"], grads)

            if self._row_trace_filter_match(row_filter, int(read_entry["center_y"])):
                if stage3_raw_trace is not None:
                    stage3_raw_trace.append({
                        "cycle": int(read_cycle),
                        "idx": len(stage3_raw_trace),
                        "center_x": int(read_entry["center_x"]),
                        "center_y": int(read_entry["center_y"]),
                        "win_size": int(read_entry["win_size"]),
                        "g_raw": dict(grads),
                    })
                if stage3_input_trace is not None:
                    stage3_input_trace.append({
                        "idx": len(stage3_input_trace),
                        "center_x": int(read_entry["center_x"]),
                        "center_y": int(read_entry["center_y"]),
                        "win_size": int(read_entry["win_size"]),
                        "grad_inv": dict(stage3_state["grad_inv"]),
                        "avg0": dict(read_entry["avg0"]),
                        "avg1": dict(read_entry["avg1"]),
                        "grad_sum": int(stage3_state["grad_sum"]),
                        "blend0_total": int(stage3_state["blend0_total"]),
                        "blend1_total": int(stage3_state["blend1_total"]),
                        "blend0": int(stage3_state["blend0"]),
                        "blend1": int(stage3_state["blend1"]),
                    })

            stage3_due[int(read_cycle + self.STAGE3_LATENCY_CYCLES)].append({
                "center_x": int(read_entry["center_x"]),
                "center_y": int(read_entry["center_y"]),
                "win_size": int(read_entry["win_size"]),
                "avg0_u": int(read_entry["avg0"]["u"]),
                "avg1_u": int(read_entry["avg1"]["u"]),
                "blend0": int(stage3_state["blend0"]),
                "blend1": int(stage3_state["blend1"]),
            })

        self._begin_timeout_scope(timeout_sec)
        try:
            while True:
                self._check_timeout("realtime integrated simulation", cycle=cycle)

                for committed_row in row_commit_due.pop(cycle, []):
                    feedback_committed_row = int(committed_row)

                for event in feedback_visible_due.pop(cycle, []):
                    patch_u10 = event["patch_u10"]
                    center_x = int(event["center_x"])
                    center_y = int(event["center_y"])
                    for patch_col_idx, raw_x in self._feedback_writeback_x(center_x):
                        if not self._feedback_column_is_safe(center_x, raw_x):
                            continue
                        column_u10 = patch_u10[:, patch_col_idx].astype(np.int32)
                        for dy in range(-2, 3):
                            write_y = self._clip(center_y + dy, 0, h - 1)
                            value_u10 = int(column_u10[dy + 2])
                            src_visible[write_y, raw_x] = value_u10
                            line_mem[write_y % 5][raw_x] = value_u10
                        if self._row_trace_filter_match(row_filter, center_y) and feedback_visible_trace is not None:
                            feedback_visible_trace.append({
                                "cycle": int(cycle),
                                "idx": len(feedback_visible_trace),
                                "center_x": center_x,
                                "center_y": center_y,
                                "raw_x": int(raw_x),
                                "column_u10": column_u10.copy(),
                                "phys_snapshot_u10": np.array(
                                    [int(line_mem[row_phys][raw_x]) for row_phys in range(5)],
                                    dtype=np.int32,
                                ),
                            })

                for event in stage4_due.pop(cycle, []):
                    center_x = int(event["center_x"])
                    center_y = int(event["center_y"])
                    final_patch_u10 = event["final_patch_u10"]

                    if center_stream is not None:
                        center_stream.append(int(final_patch_u10[2, 2]))

                    if self._row_trace_filter_match(row_filter, center_y):
                        if patch_stream is not None:
                            patch_stream.append({
                                "idx": len(patch_stream),
                                "center_x": center_x,
                                "center_y": center_y,
                                "patch_u10": final_patch_u10.copy(),
                            })
                        if feedback_commit_stream is not None:
                            commit_entries = []
                            for patch_col_idx, raw_x in self._feedback_writeback_x(center_x):
                                if not self._feedback_column_is_safe(center_x, raw_x):
                                    continue
                                commit_entries.append({
                                    "write_x": int(raw_x),
                                    "column_u10": final_patch_u10[:, patch_col_idx].astype(np.int32).copy(),
                                })
                            columns_u10 = (
                                np.stack([entry["column_u10"] for entry in commit_entries], axis=1).astype(np.int32)
                                if commit_entries else np.zeros((5, 0), dtype=np.int32)
                            )
                            feedback_commit_stream.append({
                                "cycle": int(cycle),
                                "idx": len(feedback_commit_stream),
                                "center_x": center_x,
                                "center_y": center_y,
                                "write_xs": np.array([entry["write_x"] for entry in commit_entries], dtype=np.int32),
                                "columns_u10": columns_u10,
                            })

                    # Match RTL linebuffer timing: patch feedback written on the
                    # stage4 fire edge is not visible to the capture path until
                    # the following clock edge after the write completes.
                    feedback_visible_due[int(cycle + 2)].append({
                        "center_x": center_x,
                        "center_y": center_y,
                        "patch_u10": final_patch_u10.copy(),
                    })
                    if center_x == (w - 1):
                        row_commit_due[int(cycle + 2)].append(center_y)

                for event in stage3_due.pop(cycle, []):
                    center_x = int(event["center_x"])
                    center_y = int(event["center_y"])
                    parity = center_y & 1
                    src_patch = s4_meta_patch_buf[parity][center_x]
                    if src_patch is None:
                        src_patch = get_stage12_meta(center_y, center_x)["patch_u10"].copy()
                    src_patch = np.array(src_patch, dtype=np.int32, copy=True)
                    grad_h = int(s4_meta_grad_h_buf[parity][center_x])
                    grad_v = int(s4_meta_grad_v_buf[parity][center_x])

                    if self._row_trace_filter_match(row_filter, center_y):
                        if stage4_input_trace is not None:
                            stage4_input_trace.append({
                                "cycle": int(cycle),
                                "idx": len(stage4_input_trace),
                                "center_x": center_x,
                                "center_y": center_y,
                                "win_size": int(event["win_size"]),
                                "grad_h": abs(grad_h),
                                "grad_v": abs(grad_v),
                                "blend0": int(event["blend0"]),
                                "blend1": int(event["blend1"]),
                                "avg0_u": int(event["avg0_u"]),
                                "avg1_u": int(event["avg1_u"]),
                                "src_patch_u10": src_patch.copy(),
                            })
                        if linebuffer_columns is not None:
                            linebuffer_columns.append({
                                "idx": len(linebuffer_columns),
                                "center_x": center_x,
                                "center_y": center_y,
                                "column_u10": src_patch[:, 2].copy(),
                            })

                    stage4 = self._stage4_window_blend(
                        src_patch,
                        int(event["win_size"]),
                        int(event["blend0"]),
                        int(event["blend1"]),
                        int(event["avg0_u"]),
                        int(event["avg1_u"]),
                        grad_h,
                        grad_v,
                    )
                    final_patch_u10 = np.vectorize(self._s11_to_u10)(stage4["final_patch"]).astype(np.int32)
                    stage4_due[int(cycle + self.STAGE4_LATENCY_CYCLES)].append({
                        "center_x": center_x,
                        "center_y": center_y,
                        "final_patch_u10": final_patch_u10,
                    })

                for event in stage2_due.pop(cycle, []):
                    center_x = int(event["center_x"])
                    center_y = int(event["center_y"])
                    if center_x != stage2_col_counter:
                        raise RuntimeError(
                            f"stage2 stream desync at cycle={cycle}: expected x={stage2_col_counter} got x={center_x}"
                        )

                    read_buf = 1 - stage2_write_buf
                    read_entry = stage2_payload_buf[read_buf][stage2_col_counter]
                    if stage2_row_valid and read_entry is not None:
                        schedule_stage3_from_read(
                            cycle,
                            read_buf,
                            stage2_write_buf,
                            stage2_col_counter,
                            read_entry,
                            int(event["grad"]),
                            flush_mode=False,
                        )

                    stage2_payload_buf[stage2_write_buf][stage2_col_counter] = {
                        "center_x": center_x,
                        "center_y": center_y,
                        "win_size": int(event["win_size"]),
                        "avg0": dict(event["avg0"]),
                        "avg1": dict(event["avg1"]),
                    }
                    stage2_grad_buf[stage2_write_buf][stage2_col_counter] = int(event["grad"])

                    if center_x == (w - 1):
                        last_write_buf = stage2_write_buf
                        stage2_write_buf = 1 - stage2_write_buf
                        stage2_col_counter = 0
                        stage2_row_counter += 1
                        stage2_row_valid = True
                        if center_y == (h - 1):
                            flush_start_cycle = int(cycle + 2)
                            flush_row_buf = int(last_write_buf)
                    else:
                        stage2_col_counter += 1

                if flush_start_cycle is not None and flush_row_buf is not None:
                    flush_col = cycle - flush_start_cycle
                    if 0 <= flush_col < w:
                        read_entry = stage2_payload_buf[flush_row_buf][flush_col]
                        if read_entry is not None:
                            schedule_stage3_from_read(
                                cycle,
                                flush_row_buf,
                                1 - flush_row_buf,
                                flush_col,
                                read_entry,
                                int(stage2_grad_buf[flush_row_buf][flush_col]),
                                flush_mode=True,
                            )

                for event in stage1_due.pop(cycle, []):
                    center_x = int(event["center_x"])
                    center_y = int(event["center_y"])
                    meta = get_stage12_meta(center_y, center_x)
                    if center_x == 0:
                        stage1_prev_grad_valid = False
                    current_grad = int(meta["grad"])
                    if stage1_prev_grad_valid:
                        stream_win_size = self._lut_win_size(max(current_grad, int(stage1_prev_grad)))
                    else:
                        stream_win_size = self._lut_win_size(current_grad)
                    stage2 = self._stage2_directional_avg(meta["patch_u10"], stream_win_size)
                    stage1_prev_grad = current_grad
                    stage1_prev_grad_valid = True
                    parity = center_y & 1
                    s4_meta_patch_buf[parity][center_x] = meta["patch_u10"].copy()
                    s4_meta_grad_h_buf[parity][center_x] = int(meta["grad_h"])
                    s4_meta_grad_v_buf[parity][center_x] = int(meta["grad_v"])
                    stage2_due[int(cycle + self.STAGE2_LATENCY_CYCLES)].append({
                        "center_x": center_x,
                        "center_y": center_y,
                        "win_size": int(stream_win_size),
                        "grad": current_grad,
                        "avg0": {key: int(stage2["avg0"][key]) for key in ("c", "u", "d", "l", "r")},
                        "avg1": {key: int(stage2["avg1"][key]) for key in ("c", "u", "d", "l", "r")},
                    })

                row_allow_limit = 1 if feedback_committed_row is None else int(feedback_committed_row) + 2
                window_ready = int(patch_center_y_reg) <= row_allow_limit
                column_ready = bool(window_ready) and (not flush_active)
                din_ready = bool(frame_started) and bool(column_ready)

                window_fire = bool(window_valid) and bool(window_ready)
                column_fire = bool(column_valid) and bool(column_ready)

                if window_fire:
                    patch_copy = patch_reg.copy()
                    accepted_rows[int(patch_center_y_reg)].append(patch_copy)
                    if self._row_trace_filter_match(row_filter, int(patch_center_y_reg)) and stage1_patch_trace is not None:
                        stage1_patch_trace.append({
                            "cycle": int(cycle),
                            "idx": len(stage1_patch_trace),
                            "center_x": int(patch_center_x_reg),
                            "center_y": int(patch_center_y_reg),
                            "patch_u10": patch_copy.copy(),
                        })
                    stage1_due[int(cycle + self.STAGE1_LATENCY_CYCLES)].append({
                        "center_x": int(patch_center_x_reg),
                        "center_y": int(patch_center_y_reg),
                    })

                sof = False
                din_valid = False
                normal_capture_fire = False
                if producer_state == "sof":
                    sof = True
                elif producer_state == "pixel" and in_row < h:
                    din_valid = True

                next_frame_started = frame_started
                next_row_cnt = row_cnt
                next_capture_pending = capture_pending
                next_column_valid = column_valid
                next_col_reg = col_reg.copy()
                next_center_x_reg = center_x_reg
                next_center_y_reg = center_y_reg
                next_tail_pending = tail_pending
                next_tail_active = tail_active
                next_tail_base_ptr = tail_base_ptr
                next_tail_col_ptr = tail_col_ptr
                next_tail_center_y = tail_center_y
                next_tail_row_turnaround = tail_row_turnaround
                next_in_row = in_row
                next_in_col = in_col
                next_producer_state = producer_state
                next_gap_left = gap_left
                next_wr_row_ptr = wr_row_ptr
                next_wr_col_ptr = wr_col_ptr

                if sof:
                    next_frame_started = True
                    next_row_cnt = 0
                    next_in_col = 0
                    next_producer_state = "pixel"
                    next_capture_pending = None
                    next_column_valid = False
                    next_tail_pending = False
                    next_tail_active = False
                    next_tail_base_ptr = 0
                    next_tail_row_turnaround = False
                    next_wr_row_ptr = 0
                    next_wr_col_ptr = 0
                else:
                    if column_valid and not column_ready:
                        pass
                    elif not column_ready:
                        next_column_valid = False
                    else:
                        if capture_pending is not None:
                            loaded_column = np.array(
                                [int(line_mem[row_phys][int(capture_pending["center_x"])]) for row_phys in capture_pending["row_phys"]],
                                dtype=np.int32,
                            )
                            if self._row_trace_filter_match(row_filter, int(capture_pending["center_y"])) and capture_read_trace is not None:
                                capture_read_trace.append({
                                    "cycle": int(cycle),
                                    "idx": len(capture_read_trace),
                                    "center_x": int(capture_pending["center_x"]),
                                    "center_y": int(capture_pending["center_y"]),
                                    "row_phys": tuple(int(row_phys) for row_phys in capture_pending["row_phys"]),
                                    "column_u10": loaded_column.copy(),
                                })
                            next_center_x_reg = int(capture_pending["center_x"])
                            next_center_y_reg = int(capture_pending["center_y"])
                            next_col_reg = loaded_column
                            next_column_valid = True
                            next_capture_pending = None
                        else:
                            next_column_valid = False

                    if din_valid and din_ready:
                        normal_capture_fire = row_cnt >= 2 and row_cnt < h
                        if normal_capture_fire:
                            next_capture_pending = {
                                "center_x": int(in_col),
                                "center_y": int(row_cnt - 2),
                                "row_phys": stream_capture_rows(wr_row_ptr, row_cnt),
                            }
                        line_mem[wr_row_ptr][wr_col_ptr] = int(input_image[in_row, in_col])
                        next_wr_col_ptr = wr_col_ptr + 1
                        next_in_col = in_col + 1
                        if next_in_col == w:
                            next_in_col = 0
                            next_producer_state = "eol"
                    elif producer_state == "eol" and column_ready:
                        next_row_cnt = row_cnt + 1
                        next_producer_state = "gap"
                        next_gap_left = 3
                        next_in_row = in_row + 1
                        next_wr_col_ptr = 0
                        next_wr_row_ptr = (wr_row_ptr + 1) % 5
                        if next_row_cnt == h:
                            next_tail_pending = True
                            next_tail_base_ptr = next_wr_row_ptr
                    elif producer_state == "gap" and column_ready:
                        next_gap_left = gap_left - 1
                        if next_gap_left == 0:
                            next_producer_state = "done" if in_row >= h else "pixel"

                    if tail_pending and (not tail_active) and (next_capture_pending is None) and (not normal_capture_fire) and column_ready:
                        next_tail_pending = False
                        next_tail_active = True
                        next_tail_col_ptr = 0
                        next_tail_center_y = h - 2 if h > 1 else 0
                        next_tail_row_turnaround = False
                    if tail_active and (next_capture_pending is None) and column_ready:
                        if tail_row_turnaround:
                            next_tail_row_turnaround = False
                        else:
                            next_capture_pending = {
                                "center_x": int(tail_col_ptr),
                                "center_y": int(tail_center_y),
                                "row_phys": tail_capture_rows(tail_base_ptr, tail_center_y),
                            }
                            next_tail_col_ptr = tail_col_ptr + 1
                            if next_tail_col_ptr >= w:
                                next_tail_col_ptr = 0
                                if tail_center_y >= (h - 1):
                                    next_tail_active = False
                                    next_tail_center_y = h
                                    next_tail_row_turnaround = False
                                else:
                                    next_tail_center_y = tail_center_y + 1
                                    next_tail_row_turnaround = True

                next_row_mem = [row.copy() for row in row_mem]
                next_flush_active = flush_active
                next_flush_remaining = flush_remaining
                next_flush_center_x = flush_center_x
                next_flush_center_y = flush_center_y
                next_patch_center_x_reg = patch_center_x_reg
                next_patch_center_y_reg = patch_center_y_reg
                next_patch_reg = patch_reg.copy()
                next_window_valid = window_valid

                if window_valid and not window_ready:
                    pass
                else:
                    next_window_valid = False
                    if flush_active:
                        next_patch_reg = self._load_assembler_patch(next_row_mem, flush_center_x)
                        next_patch_center_x_reg = flush_center_x
                        next_patch_center_y_reg = flush_center_y
                        next_window_valid = True
                        if flush_remaining == 1:
                            next_flush_active = False
                            next_flush_remaining = 0
                        else:
                            next_flush_remaining = flush_remaining - 1
                            next_flush_center_x = flush_center_x + 1
                    elif column_fire:
                        col_u10 = col_reg.copy()
                        if self._row_trace_filter_match(row_filter, int(center_y_reg)) and frontend_column_trace is not None:
                            frontend_column_trace.append({
                                "cycle": int(cycle),
                                "idx": len(frontend_column_trace),
                                "center_x": int(center_x_reg),
                                "center_y": int(center_y_reg),
                                "column_u10": col_u10.copy(),
                            })
                        for row_idx in range(5):
                            next_row_mem[row_idx][center_x_reg] = int(col_u10[row_idx])
                        if center_x_reg >= 4:
                            next_patch_reg = self._load_assembler_patch(
                                next_row_mem,
                                center_x_reg - 4,
                                live_col=col_u10,
                                live_center_x=center_x_reg,
                            )
                            next_patch_center_x_reg = center_x_reg - 4
                            next_patch_center_y_reg = center_y_reg
                            next_window_valid = True
                        if center_x_reg == (w - 1):
                            next_flush_active = (w > 5) or (w < 5)
                            next_flush_remaining = 4 if w > 4 else w
                            next_flush_center_x = w - 4 if w > 4 else 0
                            next_flush_center_y = center_y_reg

                if din_valid and din_ready and (not sof):
                    src_visible[in_row, in_col] = int(input_image[in_row, in_col])

                frame_started = next_frame_started
                wr_row_ptr = next_wr_row_ptr
                wr_col_ptr = next_wr_col_ptr
                row_cnt = next_row_cnt
                capture_pending = next_capture_pending
                column_valid = next_column_valid
                col_reg = next_col_reg
                center_x_reg = next_center_x_reg
                center_y_reg = next_center_y_reg
                tail_pending = next_tail_pending
                tail_active = next_tail_active
                tail_base_ptr = next_tail_base_ptr
                tail_col_ptr = next_tail_col_ptr
                tail_center_y = next_tail_center_y
                tail_row_turnaround = next_tail_row_turnaround
                row_mem = next_row_mem
                flush_active = next_flush_active
                flush_remaining = next_flush_remaining
                flush_center_x = next_flush_center_x
                flush_center_y = next_flush_center_y
                patch_center_x_reg = next_patch_center_x_reg
                patch_center_y_reg = next_patch_center_y_reg
                patch_reg = next_patch_reg
                window_valid = next_window_valid
                in_row = next_in_row
                in_col = next_in_col
                producer_state = next_producer_state
                gap_left = next_gap_left

                frontend_done = all(len(accepted_rows[row_idx]) == w for row_idx in range(h))
                backend_idle = not any((
                    stage1_due,
                    stage2_due,
                    stage3_due,
                    stage4_due,
                    feedback_visible_due,
                    row_commit_due,
                ))
                flush_done = flush_start_cycle is not None and cycle >= (flush_start_cycle + w - 1)
                if frontend_done and backend_idle and (flush_start_cycle is None or flush_done):
                    break

                cycle += 1
        finally:
            self._end_timeout_scope()

        return {
            "final_image": src_visible.astype(np.int32),
            "center_stream": None if center_stream is None else np.array(center_stream, dtype=np.int32),
            "patch_stream": patch_stream,
            "linebuffer_columns": linebuffer_columns,
            "stage1_patch_trace": stage1_patch_trace,
            "stage3_raw_trace": stage3_raw_trace,
            "stage3_input_trace": stage3_input_trace,
            "stage4_input_trace": stage4_input_trace,
            "feedback_commit_stream": feedback_commit_stream,
            "frontend_column_trace": frontend_column_trace,
            "feedback_visible_trace": feedback_visible_trace,
            "capture_read_trace": capture_read_trace,
        }

    def _simulate_exact_frontend(
        self,
        input_image: np.ndarray,
        *,
        center_rows: Optional[Iterable[int]] = None,
        timeout_sec: Optional[float] = None,
    ):
        src_visible = input_image.astype(np.int32).copy()
        h, w = src_visible.shape
        row_filter = self._normalize_row_filter(center_rows)

        accepted_patches = []
        accepted_rows = {row_idx: [] for row_idx in range(h)}
        feedback_events = []
        scheduled_rows = set()

        in_row = 0
        in_col = 0
        producer_state = "sof"
        gap_left = 0

        frame_started = False
        row_cnt = 0
        capture_pending = None
        column_valid = False
        col_reg = np.zeros(5, dtype=np.int32)
        center_x_reg = 0
        center_y_reg = 0
        tail_pending = False
        tail_active = False
        tail_col_ptr = 0
        tail_center_y = 0

        row_mem = [np.zeros(w, dtype=np.int32) for _ in range(5)]
        flush_active = False
        flush_remaining = 0
        flush_center_x = 0
        flush_center_y = 0
        patch_center_x_reg = 0
        patch_center_y_reg = 0
        patch_reg = np.zeros((5, 5), dtype=np.int32)
        window_valid = False

        feedback_committed_row = None
        cycle = 0

        self._begin_timeout_scope(timeout_sec)
        try:
            while (len(accepted_patches) < (h * w)) or any(event["cycle"] >= cycle for event in feedback_events):
                self._check_timeout("realtime frontend simulation", cycle=cycle)

                for event in [item for item in feedback_events if item["cycle"] == cycle]:
                    patch_u10 = event["patch_u10"]
                    center_x = int(event["center_x"])
                    center_y = int(event["center_y"])
                    for patch_col_idx, raw_x in self._feedback_writeback_x(center_x):
                        if not self._feedback_column_is_safe(center_x, raw_x):
                            continue
                        column_u10 = patch_u10[:, patch_col_idx].astype(np.int32)
                        for dy in range(-2, 3):
                            write_y = self._clip(center_y + dy, 0, h - 1)
                            src_visible[write_y, raw_x] = int(column_u10[dy + 2])
                    if center_x == (w - 1):
                        feedback_committed_row = center_y

                row_allow_limit = 1 if feedback_committed_row is None else int(feedback_committed_row) + 2
                window_ready = int(patch_center_y_reg) <= row_allow_limit
                column_ready = bool(window_ready) and (not flush_active)
                din_ready = bool(frame_started) and bool(column_ready)

                window_fire = bool(window_valid) and bool(window_ready)
                column_fire = bool(column_valid) and bool(column_ready)

                if window_fire:
                    patch_copy = patch_reg.copy()
                    accepted_patches.append({
                        "idx": len(accepted_patches),
                        "center_x": int(patch_center_x_reg),
                        "center_y": int(patch_center_y_reg),
                        "patch_u10": patch_copy,
                    })
                    accepted_rows[int(patch_center_y_reg)].append(patch_copy)

                    for row_idx in range(h):
                        need_next_row = min(h - 1, row_idx + 1)
                        if row_idx in scheduled_rows:
                            continue
                        if len(accepted_rows[row_idx]) != w:
                            continue
                        if len(accepted_rows[need_next_row]) != w:
                            continue

                        launch_cycle = cycle + self._frontend_feedback_delay_cycles()
                        for col_idx in range(w):
                            _, final_patch = self._build_final_patch_and_meta(accepted_rows, row_idx, col_idx)
                            feedback_events.append({
                                "cycle": int(launch_cycle + col_idx),
                                "center_x": int(col_idx),
                                "center_y": int(row_idx),
                                "patch_u10": final_patch,
                            })
                        scheduled_rows.add(row_idx)

                sof = False
                din_valid = False
                normal_capture_fire = False
                if producer_state == "sof":
                    sof = True
                elif producer_state == "pixel" and in_row < h:
                    din_valid = True

                next_frame_started = frame_started
                next_row_cnt = row_cnt
                next_capture_pending = capture_pending
                next_column_valid = column_valid
                next_col_reg = col_reg.copy()
                next_center_x_reg = center_x_reg
                next_center_y_reg = center_y_reg
                next_tail_pending = tail_pending
                next_tail_active = tail_active
                next_tail_col_ptr = tail_col_ptr
                next_tail_center_y = tail_center_y
                next_in_row = in_row
                next_in_col = in_col
                next_producer_state = producer_state
                next_gap_left = gap_left

                if sof:
                    next_frame_started = True
                    next_row_cnt = 0
                    next_in_col = 0
                    next_producer_state = "pixel"
                    next_capture_pending = None
                    next_column_valid = False
                    next_tail_pending = False
                    next_tail_active = False
                else:
                    if column_valid and not column_ready:
                        pass
                    elif not column_ready:
                        next_column_valid = False
                    else:
                        if capture_pending is not None:
                            next_center_x_reg, next_center_y_reg = capture_pending
                            next_col_reg = src_visible[self._linebuffer_row_indices(capture_pending[1], h), capture_pending[0]].copy()
                            next_column_valid = True
                            next_capture_pending = None
                        else:
                            next_column_valid = False

                    if din_valid and din_ready:
                        normal_capture_fire = row_cnt >= 2 and row_cnt < h
                        if normal_capture_fire:
                            next_capture_pending = (in_col, row_cnt - 2)
                        next_in_col = in_col + 1
                        if next_in_col == w:
                            next_in_col = 0
                            next_producer_state = "eol"
                    elif producer_state == "eol" and column_ready:
                        next_row_cnt = row_cnt + 1
                        next_producer_state = "gap"
                        next_gap_left = 3
                        next_in_row = in_row + 1
                        if next_row_cnt == h:
                            next_tail_pending = True
                    elif producer_state == "gap" and column_ready:
                        next_gap_left = gap_left - 1
                        if next_gap_left == 0:
                            next_producer_state = "done" if in_row >= h else "pixel"

                    if tail_pending and (not tail_active) and (next_capture_pending is None) and (not normal_capture_fire) and column_ready:
                        next_tail_pending = False
                        next_tail_active = True
                        next_tail_col_ptr = 0
                        next_tail_center_y = h - 2 if h > 1 else 0
                    if tail_active and (next_capture_pending is None) and column_ready:
                        next_capture_pending = (tail_col_ptr, tail_center_y)
                        next_tail_col_ptr = tail_col_ptr + 1
                        if next_tail_col_ptr >= w:
                            next_tail_col_ptr = 0
                            if tail_center_y >= (h - 1):
                                next_tail_active = False
                                next_tail_center_y = h
                            else:
                                next_tail_center_y = tail_center_y + 1

                next_row_mem = [row.copy() for row in row_mem]
                next_flush_active = flush_active
                next_flush_remaining = flush_remaining
                next_flush_center_x = flush_center_x
                next_flush_center_y = flush_center_y
                next_patch_center_x_reg = patch_center_x_reg
                next_patch_center_y_reg = patch_center_y_reg
                next_patch_reg = patch_reg.copy()
                next_window_valid = window_valid

                if window_valid and not window_ready:
                    pass
                else:
                    next_window_valid = False
                    if flush_active:
                        next_patch_reg = self._load_assembler_patch(next_row_mem, flush_center_x)
                        next_patch_center_x_reg = flush_center_x
                        next_patch_center_y_reg = flush_center_y
                        next_window_valid = True
                        if flush_remaining == 1:
                            next_flush_active = False
                            next_flush_remaining = 0
                        else:
                            next_flush_remaining = flush_remaining - 1
                            next_flush_center_x = flush_center_x + 1
                    elif column_fire:
                        col_u10 = col_reg.copy()
                        for row_idx in range(5):
                            next_row_mem[row_idx][center_x_reg] = int(col_u10[row_idx])
                        if center_x_reg >= 4:
                            next_patch_reg = self._load_assembler_patch(
                                next_row_mem,
                                center_x_reg - 4,
                                live_col=col_u10,
                                live_center_x=center_x_reg,
                            )
                            next_patch_center_x_reg = center_x_reg - 4
                            next_patch_center_y_reg = center_y_reg
                            next_window_valid = True
                        if center_x_reg == (w - 1):
                            next_flush_active = (w > 5) or (w < 5)
                            next_flush_remaining = 4 if w > 4 else w
                            next_flush_center_x = w - 4 if w > 4 else 0
                            next_flush_center_y = center_y_reg

                if din_valid and din_ready and (not sof):
                    src_visible[in_row, in_col] = int(input_image[in_row, in_col])

                frame_started = next_frame_started
                row_cnt = next_row_cnt
                capture_pending = next_capture_pending
                column_valid = next_column_valid
                col_reg = next_col_reg
                center_x_reg = next_center_x_reg
                center_y_reg = next_center_y_reg
                tail_pending = next_tail_pending
                tail_active = next_tail_active
                tail_col_ptr = next_tail_col_ptr
                tail_center_y = next_tail_center_y
                row_mem = next_row_mem
                flush_active = next_flush_active
                flush_remaining = next_flush_remaining
                flush_center_x = next_flush_center_x
                flush_center_y = next_flush_center_y
                patch_center_x_reg = next_patch_center_x_reg
                patch_center_y_reg = next_patch_center_y_reg
                patch_reg = next_patch_reg
                window_valid = next_window_valid
                in_row = next_in_row
                in_col = next_in_col
                producer_state = next_producer_state
                gap_left = next_gap_left
                cycle += 1
        finally:
            self._end_timeout_scope()

        filtered_stage1 = []
        for entry in accepted_patches:
            if self._row_trace_filter_match(row_filter, int(entry["center_y"])):
                filtered_stage1.append({
                    "idx": len(filtered_stage1),
                    "center_x": int(entry["center_x"]),
                    "center_y": int(entry["center_y"]),
                    "patch_u10": entry["patch_u10"].copy(),
                })

        return {
            "final_image": src_visible.astype(np.int32),
            "accepted_rows": accepted_rows,
            "stage1_patch_trace": filtered_stage1,
        }

    def _process_feedback_realtime(
        self,
        input_image: np.ndarray,
        *,
        emit_center_stream: bool = False,
        emit_patch_stream: bool = False,
        emit_linebuffer_columns: bool = False,
        emit_stage1_patch_trace: bool = False,
        emit_stage3_raw_trace: bool = False,
        emit_stage3_input_trace: bool = False,
        emit_stage4_input_trace: bool = False,
        emit_feedback_commit_stream: bool = False,
        emit_frontend_column_trace: bool = False,
        emit_feedback_visible_trace: bool = False,
        emit_capture_read_trace: bool = False,
        center_rows: Optional[Iterable[int]] = None,
        timeout_sec: Optional[float] = None,
    ):
        return self._simulate_realtime_pipeline(
            input_image,
            emit_center_stream=emit_center_stream,
            emit_patch_stream=emit_patch_stream,
            emit_linebuffer_columns=emit_linebuffer_columns,
            emit_stage1_patch_trace=emit_stage1_patch_trace,
            emit_stage3_raw_trace=emit_stage3_raw_trace,
            emit_stage3_input_trace=emit_stage3_input_trace,
            emit_stage4_input_trace=emit_stage4_input_trace,
            emit_feedback_commit_stream=emit_feedback_commit_stream,
            emit_frontend_column_trace=emit_frontend_column_trace,
            emit_feedback_visible_trace=emit_feedback_visible_trace,
            emit_capture_read_trace=emit_capture_read_trace,
            center_rows=center_rows,
            timeout_sec=timeout_sec,
        )

    def process(self, input_image: np.ndarray, timeout_sec: Optional[float] = None) -> np.ndarray:
        return self._process_feedback_realtime(input_image, timeout_sec=timeout_sec)["final_image"]

    def process_center_stream(self, input_image: np.ndarray, timeout_sec: Optional[float] = None) -> np.ndarray:
        behavior = self._process_feedback_realtime(
            input_image,
            emit_center_stream=True,
            timeout_sec=timeout_sec,
        )
        return behavior["center_stream"]

    def export_linebuffer_column_stream(
        self,
        input_image: np.ndarray,
        center_rows: Optional[Iterable[int]] = None,
        timeout_sec: Optional[float] = None,
    ):
        behavior = self._process_feedback_realtime(
            input_image,
            emit_linebuffer_columns=True,
            center_rows=center_rows,
            timeout_sec=timeout_sec,
        )
        return behavior["linebuffer_columns"]

    def export_patch_stream(
        self,
        input_image: np.ndarray,
        center_rows: Optional[Iterable[int]] = None,
        timeout_sec: Optional[float] = None,
    ):
        behavior = self._process_feedback_realtime(
            input_image,
            emit_patch_stream=True,
            center_rows=center_rows,
            timeout_sec=timeout_sec,
        )
        return behavior["patch_stream"]

    def export_stage1_input_patch_trace(
        self,
        input_image: np.ndarray,
        center_rows: Optional[Iterable[int]] = None,
        timeout_sec: Optional[float] = None,
    ):
        behavior = self._process_feedback_realtime(
            input_image,
            emit_stage1_patch_trace=True,
            center_rows=center_rows,
            timeout_sec=timeout_sec,
        )
        return behavior["stage1_patch_trace"]

    def export_stage3_raw_trace(
        self,
        input_image: np.ndarray,
        center_rows: Optional[Iterable[int]] = None,
        timeout_sec: Optional[float] = None,
    ):
        behavior = self._process_feedback_realtime(
            input_image,
            emit_stage3_raw_trace=True,
            center_rows=center_rows,
            timeout_sec=timeout_sec,
        )
        return behavior["stage3_raw_trace"]

    def export_stage3_input_trace(
        self,
        input_image: np.ndarray,
        center_rows: Optional[Iterable[int]] = None,
        timeout_sec: Optional[float] = None,
    ):
        behavior = self._process_feedback_realtime(
            input_image,
            emit_stage3_input_trace=True,
            center_rows=center_rows,
            timeout_sec=timeout_sec,
        )
        return behavior["stage3_input_trace"]

    def export_stage4_input_trace(
        self,
        input_image: np.ndarray,
        center_rows: Optional[Iterable[int]] = None,
        timeout_sec: Optional[float] = None,
    ):
        behavior = self._process_feedback_realtime(
            input_image,
            emit_stage4_input_trace=True,
            center_rows=center_rows,
            timeout_sec=timeout_sec,
        )
        return behavior["stage4_input_trace"]

    def export_feedback_commit_stream(
        self,
        input_image: np.ndarray,
        center_rows: Optional[Iterable[int]] = None,
        timeout_sec: Optional[float] = None,
    ):
        behavior = self._process_feedback_realtime(
            input_image,
            emit_feedback_commit_stream=True,
            center_rows=center_rows,
            timeout_sec=timeout_sec,
        )
        return behavior["feedback_commit_stream"]
