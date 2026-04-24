//==============================================================================
// ISP-CSIIR HLS Top Module - Stream-based Processing with ISPCSIIR Class
//==============================================================================
// Features:
//   - ISPCSIIR class for clean instantiation in HLS tool
//   - 1PSRAM (1-port) memory with doubled bit width for time-multiplexing
//   - AXI-Stream interface for streaming pixel processing
//==============================================================================

#ifndef ISP_CSIIR_HLS_TOP_HPP
#define ISP_CSIIR_HLS_TOP_HPP

#include <ap_fixed.h>
#include <hls_stream.h>
#include <cstring>
#include "isp_csiir_regs.hpp"

//==============================================================================
// Type Definitions - Fixed-Point Precision
//==============================================================================
static const int DATA_WIDTH_I = 10;       // Input pixel width
static const int GRAD_WIDTH_I = 14;       // Gradient width
static const int ACC_WIDTH_I = 20;        // Accumulator width
static const int SIGNED_WIDTH_I = 11;     // Signed intermediate width
static const int PATCH_SIZE = 25;         // 5x5 window

static const int MAX_WIDTH = 4096;
static const int MAX_HEIGHT = 4096;

// Bit width for 1PSRAM time-multiplexing (double width for 2-cycle access)
static const int PSRAM_PIXEL_WIDTH = DATA_WIDTH_I * 2;   // 20-bit for 2 pixels
static const int PSRAM_GRAD_WIDTH = GRAD_WIDTH_I * 2;    // 28-bit for 2 gradients

typedef ap_uint<DATA_WIDTH_I> pixel_t;
typedef ap_int<SIGNED_WIDTH_I> s11_t;
typedef ap_uint<GRAD_WIDTH_I> grad_t;
typedef ap_int<ACC_WIDTH_I> acc_t;

// Packed types for 1PSRAM time-multiplexing
typedef ap_uint<PSRAM_PIXEL_WIDTH> pixel_pack_t;   // 2 pixels packed
typedef ap_uint<PSRAM_GRAD_WIDTH> grad_pack_t;     // 2 gradients packed

//==============================================================================
// AXI-Stream Interface Types
//==============================================================================
struct axis_pixel_t {
    pixel_t data;
    ap_uint<1> last;
    ap_uint<1> user;
};

//==============================================================================
// Configuration Structure
//==============================================================================
struct Config {
    int img_width;
    int img_height;
    int win_size_thresh[4];
    int win_size_clip_y[4];
    int win_size_clip_sft[4];
    int blending_ratio[4];
    int reg_edge_protect;

    Config() {
        img_width = 64;
        img_height = 64;
        win_size_thresh[0] = 16;
        win_size_thresh[1] = 24;
        win_size_thresh[2] = 32;
        win_size_thresh[3] = 40;
        win_size_clip_y[0] = 15;
        win_size_clip_y[1] = 23;
        win_size_clip_y[2] = 31;
        win_size_clip_y[3] = 39;
        win_size_clip_sft[0] = 2;
        win_size_clip_sft[1] = 2;
        win_size_clip_sft[2] = 2;
        win_size_clip_sft[3] = 2;
        blending_ratio[0] = 32;
        blending_ratio[1] = 32;
        blending_ratio[2] = 32;
        blending_ratio[3] = 32;
        reg_edge_protect = 32;
    }
};

//==============================================================================
// Helper Functions
//==============================================================================
inline int round_div(int num, int den) {
    if (den == 0) return 0;
    if (num >= 0) return (num + den / 2) / den;
    return - (((-num) + den / 2) / den);
}

template<typename T>
inline T clip(T value, int min_val, int max_val) {
    if (value < min_val) return min_val;
    if (value > max_val) return max_val;
    return value;
}

inline pixel_t clip_u10(int value) {
    return clip<pixel_t>(value, 0, 1023);
}

inline s11_t u10_to_s11(pixel_t value) {
    return (s11_t)value - 512;
}

inline pixel_t s11_to_u10(s11_t value) {
    return clip_u10((int)value + 512);
}

inline s11_t saturate_s11(s11_t value) {
    return clip<s11_t>(value, -512, 511);
}

inline int clip_win_size(int win_size) {
    return clip<int>(win_size, 16, 40);
}

//==============================================================================
// ISPCSIIR Class - Top-level Processing Engine
//==============================================================================
class ISPCSIIR {
public:
    Config cfg;

    ISPCSIIR() {}
    ISPCSIIR(const Config& c) : cfg(c) {}

    //-------------------------------------------------------------------------
    // Sobel Gradient (5x5)
    //-------------------------------------------------------------------------
    //-------------------------------------------------------------------------
    // Sobel Gradient (5x5)
    //-------------------------------------------------------------------------
    void sobel_gradient_5x5(const pixel_t window[PATCH_SIZE],
                           grad_t& grad_h, grad_t& grad_v, grad_t& grad) {
        int sum_h = window[0] + window[1] + window[2] + window[3] + window[4]
                  - window[20] - window[21] - window[22] - window[23] - window[24];
        int sum_v = window[0] + window[5] + window[10] + window[15] + window[20]
                  - window[4] - window[9] - window[14] - window[19] - window[24];
        grad_h = (grad_t)abs(sum_h);
        grad_v = (grad_t)abs(sum_v);
        grad = (grad_t)(round_div(grad_h, 5) + round_div(grad_v, 5));
    }

    //-------------------------------------------------------------------------
    // Window Size LUT
    //-------------------------------------------------------------------------
    int lut_win_size(int grad_triplet_max) {
        int x_nodes[4];
        int acc = 0;
        for (int i = 0; i < 4; i++) {
            acc += (1 << cfg.win_size_clip_sft[i]);
            x_nodes[i] = acc;
        }

        int x = grad_triplet_max;
        int y0 = cfg.win_size_clip_y[0];
        int y1 = cfg.win_size_clip_y[3];

        if (x <= x_nodes[0]) return clip_win_size(y0);
        if (x >= x_nodes[3]) return clip_win_size(y1);

        int win_size = y1;
        for (int idx = 0; idx < 3; idx++) {
            if (x_nodes[idx] <= x && x <= x_nodes[idx + 1]) {
                int x0 = x_nodes[idx];
                int x1 = x_nodes[idx + 1];
                int y0_i = cfg.win_size_clip_y[idx];
                int y1_i = cfg.win_size_clip_y[idx + 1];
                win_size = y0_i + round_div((x - x0) * (y1_i - y0_i), (x1 - x0));
                break;
            }
        }
        return clip_win_size(win_size);
    }

    //-------------------------------------------------------------------------
    // Kernel Selection
    //-------------------------------------------------------------------------
    int select_kernel_type(int win_size) {
        if (win_size < cfg.win_size_thresh[0]) return 0;
        if (win_size < cfg.win_size_thresh[1]) return 1;
        if (win_size < cfg.win_size_thresh[2]) return 2;
        if (win_size < cfg.win_size_thresh[3]) return 3;
        return 4;
    }

    //-------------------------------------------------------------------------
    // Directional Average
    //-------------------------------------------------------------------------
    struct DirAvgResult {
        s11_t avg0_c, avg0_u, avg0_d, avg0_l, avg0_r;
        s11_t avg1_c, avg1_u, avg1_d, avg1_l, avg1_r;
    };

    s11_t weighted_avg(const s11_t patch_s11[PATCH_SIZE], const int* kernel) {
        acc_t total = 0;
        int weight = 0;
        for (int i = 0; i < PATCH_SIZE; i++) {
            if (kernel[i] != 0) {
                total += (acc_t)patch_s11[i] * kernel[i];
                weight += kernel[i];
            }
        }
        if (weight == 0) return 0;
        return saturate_s11(round_div((int)total, weight));
    }

    DirAvgResult compute_directional_avg(const s11_t patch_s11[PATCH_SIZE], int win_size) {
        DirAvgResult result = {};
        int kt = select_kernel_type(win_size);

        // Kernels
        static const int K2X2[PATCH_SIZE] = {0,0,0,0,0,0,1,2,1,0,0,2,4,2,0,0,1,2,1,0,0,0,0,0,0};
        static const int K3X3[PATCH_SIZE] = {0,0,0,0,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,0,0,0,0};
        static const int K4X4[PATCH_SIZE] = {1,1,2,1,1,1,2,4,2,1,2,4,8,4,2,1,2,4,2,1,1,1,2,1,1};
        static const int K5X5[PATCH_SIZE] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};

        const int* k0 = nullptr;
        const int* k1 = nullptr;
        switch (kt) {
            case 0: k0 = nullptr; k1 = K2X2; break;
            case 1: k0 = K3X3; k1 = K2X2; break;
            case 2: k0 = K4X4; k1 = K3X3; break;
            case 3: k0 = K5X5; k1 = K4X4; break;
            default: k0 = K5X5; k1 = nullptr; break;
        }

        // Direction masks
        static const int MC[PATCH_SIZE] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
        static const int MU[PATCH_SIZE] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0};
        static const int MD[PATCH_SIZE] = {0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
        static const int ML[PATCH_SIZE] = {1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0};
        static const int MR[PATCH_SIZE] = {0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1};

        if (k0) {
            int mk0_c[PATCH_SIZE], mk0_u[PATCH_SIZE], mk0_d[PATCH_SIZE];
            int mk0_l[PATCH_SIZE], mk0_r[PATCH_SIZE];
            for (int i = 0; i < PATCH_SIZE; i++) {
                mk0_c[i] = k0[i] * MC[i];
                mk0_u[i] = k0[i] * MU[i];
                mk0_d[i] = k0[i] * MD[i];
                mk0_l[i] = k0[i] * ML[i];
                mk0_r[i] = k0[i] * MR[i];
            }
            result.avg0_c = weighted_avg(patch_s11, mk0_c);
            result.avg0_u = weighted_avg(patch_s11, mk0_u);
            result.avg0_d = weighted_avg(patch_s11, mk0_d);
            result.avg0_l = weighted_avg(patch_s11, mk0_l);
            result.avg0_r = weighted_avg(patch_s11, mk0_r);
        }

        if (k1) {
            int mk1_c[PATCH_SIZE], mk1_u[PATCH_SIZE], mk1_d[PATCH_SIZE];
            int mk1_l[PATCH_SIZE], mk1_r[PATCH_SIZE];
            for (int i = 0; i < PATCH_SIZE; i++) {
                mk1_c[i] = k1[i] * MC[i];
                mk1_u[i] = k1[i] * MU[i];
                mk1_d[i] = k1[i] * MD[i];
                mk1_l[i] = k1[i] * ML[i];
                mk1_r[i] = k1[i] * MR[i];
            }
            result.avg1_c = weighted_avg(patch_s11, mk1_c);
            result.avg1_u = weighted_avg(patch_s11, mk1_u);
            result.avg1_d = weighted_avg(patch_s11, mk1_d);
            result.avg1_l = weighted_avg(patch_s11, mk1_l);
            result.avg1_r = weighted_avg(patch_s11, mk1_r);
        }

        return result;
    }

    //-------------------------------------------------------------------------
    // Gradient Fusion
    //-------------------------------------------------------------------------
    struct FusionResult {
        s11_t blend0;
        s11_t blend1;
    };

    void grad_inverse_remap(const int g[5], int inv[5]) {
        int idx[5] = {0, 1, 2, 3, 4};
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4 - i; j++) {
                if (g[idx[j]] < g[idx[j + 1]]) {
                    int tmp = idx[j];
                    idx[j] = idx[j + 1];
                    idx[j + 1] = tmp;
                }
            }
        }
        for (int i = 0; i < 5; i++) {
            inv[idx[4 - i]] = g[idx[i]];
        }
    }

    FusionResult compute_gradient_fusion(const DirAvgResult& dir_avg,
                                         int grad_u, int grad_d,
                                         int grad_l, int grad_r, int grad_c) {
        FusionResult result;
        int g[5] = {grad_u, grad_d, grad_l, grad_r, grad_c};
        int inv[5];
        grad_inverse_remap(g, inv);

        int sum = inv[0] + inv[1] + inv[2] + inv[3] + inv[4];

        int v0[5] = {(int)dir_avg.avg0_c, (int)dir_avg.avg0_u, (int)dir_avg.avg0_d,
                     (int)dir_avg.avg0_l, (int)dir_avg.avg0_r};
        int v1[5] = {(int)dir_avg.avg1_c, (int)dir_avg.avg1_u, (int)dir_avg.avg1_d,
                     (int)dir_avg.avg1_l, (int)dir_avg.avg1_r};

        int total0 = 0, total1 = 0;
        for (int i = 0; i < 5; i++) {
            total0 += v0[i] * inv[i];
            total1 += v1[i] * inv[i];
        }

        if (sum == 0) {
            int simple_sum0 = v0[0] + v0[1] + v0[2] + v0[3] + v0[4];
            int simple_sum1 = v1[0] + v1[1] + v1[2] + v1[3] + v1[4];
            result.blend0 = saturate_s11(round_div(simple_sum0, 5));
            result.blend1 = saturate_s11(round_div(simple_sum1, 5));
        } else {
            result.blend0 = saturate_s11(round_div(total0, sum));
            result.blend1 = saturate_s11(round_div(total1, sum));
        }

        return result;
    }

    //-------------------------------------------------------------------------
    // Blending Ratio
    //-------------------------------------------------------------------------
    int get_ratio(int win_size) {
        int idx = clip<int>(win_size / 8 - 2, 0, 3);
        return cfg.blending_ratio[idx];
    }

    //-------------------------------------------------------------------------
    // IIR Blend
    //-------------------------------------------------------------------------
    void mix_scalar(s11_t scalar, const s11_t src[PATCH_SIZE],
                   const int factor[PATCH_SIZE], s11_t out[PATCH_SIZE]) {
        for (int i = 0; i < PATCH_SIZE; i++) {
            int val = round_div((int)scalar * factor[i] + (int)src[i] * (4 - factor[i]), 4);
            out[i] = saturate_s11(val);
        }
    }

    void compute_iir_blend(const s11_t src[PATCH_SIZE], int win_size,
                          s11_t blend0_g, s11_t blend1_g,
                          s11_t avg0_u, s11_t avg1_u,
                          int grad_h, int grad_v,
                          s11_t final_patch[PATCH_SIZE]) {
        int ratio = get_ratio(win_size);

        s11_t b0_hor = saturate_s11(round_div((int)ratio * blend0_g + (64 - ratio) * avg0_u, 64));
        s11_t b1_hor = saturate_s11(round_div((int)ratio * blend1_g + (64 - ratio) * avg1_u, 64));

        bool vert_dom = abs(grad_v) > abs(grad_h);

        // Orientation factor
        static const int F_ORI_V[PATCH_SIZE] = {0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0};
        static const int F_ORI_H[PATCH_SIZE] = {0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0};
        const int* f_orient = vert_dom ? F_ORI_V : F_ORI_H;

        // Blend kernels
        static const int F2X2[PATCH_SIZE] = {0,0,0,0,0,0,1,2,1,0,0,2,4,2,0,0,1,2,1,0,0,0,0,0,0};
        static const int F3X3[PATCH_SIZE] = {0,0,0,0,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,0,0,0,0};
        static const int F4X4[PATCH_SIZE] = {1,1,2,1,1,1,2,4,2,1,2,4,8,4,2,1,2,4,2,1,1,1,2,1,1};
        static const int F5X5[PATCH_SIZE] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};

        s11_t blend0_win[PATCH_SIZE], blend1_win[PATCH_SIZE];

        if (win_size < cfg.win_size_thresh[0]) {
            s11_t tmp1[PATCH_SIZE], tmp2[PATCH_SIZE];
            mix_scalar(b1_hor, src, f_orient, tmp1);
            mix_scalar(b1_hor, src, F2X2, tmp2);
            for (int i = 0; i < PATCH_SIZE; i++) {
                int val = round_div((int)tmp1[i] * cfg.reg_edge_protect +
                                   (int)tmp2[i] * (64 - cfg.reg_edge_protect), 64);
                blend1_win[i] = saturate_s11(val);
                blend0_win[i] = 0;
            }
        } else if (win_size < cfg.win_size_thresh[1]) {
            s11_t tmp1[PATCH_SIZE], tmp2[PATCH_SIZE];
            mix_scalar(b1_hor, src, f_orient, tmp1);
            mix_scalar(b1_hor, src, F2X2, tmp2);
            for (int i = 0; i < PATCH_SIZE; i++) {
                int val = round_div((int)tmp1[i] * cfg.reg_edge_protect +
                                   (int)tmp2[i] * (64 - cfg.reg_edge_protect), 64);
                blend1_win[i] = saturate_s11(val);
            }
            mix_scalar(b0_hor, src, F3X3, blend0_win);
        } else if (win_size < cfg.win_size_thresh[2]) {
            mix_scalar(b1_hor, src, F3X3, blend1_win);
            mix_scalar(b0_hor, src, F4X4, blend0_win);
        } else if (win_size < cfg.win_size_thresh[3]) {
            mix_scalar(b1_hor, src, F4X4, blend1_win);
            mix_scalar(b0_hor, src, F5X5, blend0_win);
        } else {
            mix_scalar(b0_hor, src, F5X5, blend0_win);
            for (int i = 0; i < PATCH_SIZE; i++) blend1_win[i] = 0;
        }

        int remain = win_size % 8;
        if (win_size < cfg.win_size_thresh[0]) {
            for (int i = 0; i < PATCH_SIZE; i++) final_patch[i] = blend1_win[i];
        } else if (win_size >= cfg.win_size_thresh[3]) {
            for (int i = 0; i < PATCH_SIZE; i++) final_patch[i] = blend0_win[i];
        } else {
            for (int i = 0; i < PATCH_SIZE; i++) {
                int val = round_div(blend0_win[i] * remain + blend1_win[i] * (8 - remain), 8);
                final_patch[i] = saturate_s11(val);
            }
        }
    }
};

//==============================================================================
// HLS Top Function - AXI-Stream Interface with Register Struct
//==============================================================================
void isp_csiir_top(
    hls::stream<axis_pixel_t> &din_stream,
    hls::stream<axis_pixel_t> &dout_stream,
    ISPCSIIR_Regs &regs
) {
    #pragma HLS INTERFACE axis port=din_stream
    #pragma HLS INTERFACE axis port=dout_stream
    #pragma HLS INTERFACE s_axilite port=regs bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=return bundle=CTRL
    #pragma HLS INTERFACE ap_ctrl_hs port=return

    // Configure ISPCSIIR instance
    ISPCSIIR isp;
    isp.cfg.img_width = (int)regs.img_width;
    isp.cfg.img_height = (int)regs.img_height;
    for (int i = 0; i < 4; i++) {
        #pragma HLS UNROLL
        isp.cfg.win_size_thresh[i] = (int)regs.win_size_thresh[i];
        isp.cfg.win_size_clip_y[i] = (int)regs.win_size_clip_y[i];
        isp.cfg.win_size_clip_sft[i] = (int)regs.win_size_clip_sft[i];
        isp.cfg.blending_ratio[i] = (int)regs.blending_ratio[i];
    }
    isp.cfg.reg_edge_protect = (int)regs.edge_protect;

    // Local copies of dimensions for use in array indexing
    const int img_width = (int)regs.img_width;
    const int img_height = (int)regs.img_height;

    // Line buffer for ORIGINAL image (4 rows, used by gradient + stage2)
    // Stores raw input pixels - never modified during feedback
    pixel_pack_t line_buf_src[4][MAX_WIDTH];
    #pragma HLS ARRAY_PARTITION variable=line_buf_src dim=1 complete
    #pragma HLS RESOURCE variable=line_buf_src core=RAM_1P_BRAM

    // Line buffer for FILTERED image (5 rows, used by stage4 feedback + subsequent pixels)
    // Updated with filtered values as pixels are processed
    pixel_pack_t line_buf_filt[5][MAX_WIDTH];
    #pragma HLS ARRAY_PARTITION variable=line_buf_filt dim=1 complete
    #pragma HLS RESOURCE variable=line_buf_filt core=RAM_1P_BRAM

    // Column shift register for ORIGINAL data (for gradient window)
    pixel_t col_src[5][5];
    #pragma HLS ARRAY_PARTITION variable=col_src complete

    // Column shift register for FILTERED data (for stage4 IIR blend window)
    pixel_t col_filt[5][5];
    #pragma HLS ARRAY_PARTITION variable=col_filt complete

    grad_pack_t grad_buf_pack[2][MAX_WIDTH];
    #pragma HLS ARRAY_PARTITION variable=grad_buf_pack dim=1 complete
    #pragma HLS RESOURCE variable=grad_buf_pack core=RAM_1P_BRAM

    grad_t grad_shift[3];
    #pragma HLS ARRAY_PARTITION variable=grad_shift complete

    grad_t grad_next_row_delay[MAX_WIDTH];
    #pragma HLS RESOURCE variable=grad_next_row_delay core=RAM_1P_BRAM

    // Gradient window: built from original line buffer
    // Used by stage1 (gradient), stage2 (directional avg)
    // Note: stage4's IIR blend uses col_filt instead (filtered data)
    pixel_t src_5x5[5][5];
    #pragma HLS ARRAY_PARTITION variable=src_5x5 complete

    s11_t src_s11_5x5[PATCH_SIZE];
    #pragma HLS ARRAY_PARTITION variable=src_s11_5x5 complete

    grad_t current_grad = 0;
    grad_t grad_u, grad_d, grad_l, grad_r;

    // 2-cycle delay buffer for grad values (to get grad_r)
    grad_t grad_delay_buf[2][MAX_WIDTH];
    #pragma HLS ARRAY_PARTITION variable=grad_delay_buf dim=1 complete
    #pragma HLS RESOURCE variable=grad_delay_buf core=RAM_2P_BRAM

    // Delayed stage2 results (need to store for 2 cycles)
    ISPCSIIR::DirAvgResult dir_avg_delay[2];
    #pragma HLS ARRAY_PARTITION variable=dir_avg_delay complete

    // Delayed coordinates and win_size
    unsigned int delay_row[2];
    unsigned int delay_col[2];
    int delay_win_size[2];
    #pragma HLS ARRAY_PARTITION variable=delay_row complete
    #pragma HLS ARRAY_PARTITION variable=delay_col complete
    #pragma HLS ARRAY_PARTITION variable=delay_win_size complete

    // Delayed fusion results
    ISPCSIIR::FusionResult delay_fusion[2];
    #pragma HLS ARRAY_PARTITION variable=delay_fusion complete

    // Delayed filt_5x5 for stage4
    s11_t delay_filt_5x5[2][PATCH_SIZE];
    #pragma HLS ARRAY_PARTITION variable=delay_filt_5x5 complete

    // Delayed grad_h, grad_v for stage4
    grad_t delay_grad_h[2];
    grad_t delay_grad_v[2];
    #pragma HLS ARRAY_PARTITION variable=delay_grad_h complete
    #pragma HLS ARRAY_PARTITION variable=delay_grad_v complete

    unsigned int total_pixels = (unsigned int)img_width * (unsigned int)img_height;

    // Initialize original line buffer
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < MAX_WIDTH; j++) {
            #pragma HLS UNROLL factor=4
            line_buf_src[i][j] = 0;
        }
    }
    // Initialize filtered line buffer
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < MAX_WIDTH; j++) {
            #pragma HLS UNROLL factor=4
            line_buf_filt[i][j] = 0;
        }
    }
    // Initialize column shift registers
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            col_src[i][j] = 0;
            col_filt[i][j] = 0;
        }
    }
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            col_src[i][j] = 0;
            col_filt[i][j] = 0;
        }
    }
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < MAX_WIDTH; j++) {
            #pragma HLS UNROLL factor=4
            grad_buf_pack[i][j] = 0;
        }
    }
    for (int j = 0; j < 3; j++) {
        grad_shift[j] = 0;
    }
    for (int j = 0; j < MAX_WIDTH; j++) {
        #pragma HLS UNROLL factor=4
        grad_next_row_delay[j] = 0;
    }
    // Initialize delay buffers
    for (int i = 0; i < 2; i++) {
        delay_row[i] = 0;
        delay_col[i] = 0;
        delay_win_size[i] = 0;
        delay_grad_h[i] = 0;
        delay_grad_v[i] = 0;
        for (int j = 0; j < PATCH_SIZE; j++) {
            delay_filt_5x5[i][j] = 0;
        }
    }
    dir_avg_delay[0].avg0_c = 0; dir_avg_delay[0].avg0_u = 0; dir_avg_delay[0].avg0_d = 0;
    dir_avg_delay[0].avg0_l = 0; dir_avg_delay[0].avg0_r = 0;
    dir_avg_delay[0].avg1_c = 0; dir_avg_delay[0].avg1_u = 0; dir_avg_delay[0].avg1_d = 0;
    dir_avg_delay[0].avg1_l = 0; dir_avg_delay[0].avg1_r = 0;
    dir_avg_delay[1].avg0_c = 0; dir_avg_delay[1].avg0_u = 0; dir_avg_delay[1].avg0_d = 0;
    dir_avg_delay[1].avg0_l = 0; dir_avg_delay[1].avg0_r = 0;
    dir_avg_delay[1].avg1_c = 0; dir_avg_delay[1].avg1_u = 0; dir_avg_delay[1].avg1_d = 0;
    dir_avg_delay[1].avg1_l = 0; dir_avg_delay[1].avg1_r = 0;
    delay_fusion[0].blend0 = 0; delay_fusion[0].blend1 = 0;
    delay_fusion[1].blend0 = 0; delay_fusion[1].blend1 = 0;

    // Main processing loop - streaming pixel-by-pixel
    for (unsigned int pixel_idx = 0; pixel_idx < total_pixels; pixel_idx++) {
        #pragma HLS PIPELINE II=1 rewind

        unsigned int row_val = pixel_idx / (unsigned int)img_width;
        unsigned int col_val = pixel_idx % (unsigned int)img_width;

        // Read input pixel from stream
        axis_pixel_t din = din_stream.read();

        //======================================================================
        // Update ORIGINAL line buffer (4 rows, for gradient/stage2)
        //======================================================================
        // Shift col_src register right (col_src[r][c] <- col_src[r][c+1])
        for (int r = 0; r < 5; r++) {
            #pragma HLS UNROLL
            for (int c = 0; c < 4; c++) {
                #pragma HLS UNROLL
                col_src[r][c] = col_src[r][c+1];
            }
        }

        // Read old rows from original line buffer
        pixel_pack_t src_old_row0_pack = line_buf_src[0][col_val];
        pixel_pack_t src_old_row1_pack = line_buf_src[1][col_val];
        pixel_pack_t src_old_row2_pack = line_buf_src[2][col_val];
        pixel_pack_t src_old_row3_pack = line_buf_src[3][col_val];

        pixel_t src_old_row0 = (pixel_t)src_old_row0_pack.range(9, 0).to_int();
        pixel_t src_old_row1 = (pixel_t)src_old_row1_pack.range(9, 0).to_int();
        pixel_t src_old_row2 = (pixel_t)src_old_row2_pack.range(9, 0).to_int();
        pixel_t src_old_row3 = (pixel_t)src_old_row3_pack.range(9, 0).to_int();

        // Shift original line buffer down: row0<-row1<-row2<-row3<-din
        line_buf_src[0][col_val] = (pixel_pack_t)src_old_row1 << 0 | (pixel_pack_t)src_old_row0 << 10;
        line_buf_src[1][col_val] = (pixel_pack_t)src_old_row2 << 0 | (pixel_pack_t)src_old_row1 << 10;
        line_buf_src[2][col_val] = (pixel_pack_t)src_old_row3 << 0 | (pixel_pack_t)src_old_row2 << 10;
        line_buf_src[3][col_val] = (pixel_pack_t)din.data       << 0 | (pixel_pack_t)src_old_row3 << 10;

        // Load new pixel into col_src (the "5th" row of the shift register)
        col_src[0][4] = src_old_row0;
        col_src[1][4] = src_old_row1;
        col_src[2][4] = src_old_row2;
        col_src[3][4] = src_old_row3;
        col_src[4][4] = din.data;

        //======================================================================
        // Update FILTERED line buffer (5 rows, for stage4 feedback)
        //======================================================================
        // Shift col_filt register right (col_filt[r][c] <- col_filt[r][c+1])
        for (int r = 0; r < 5; r++) {
            #pragma HLS UNROLL
            for (int c = 0; c < 4; c++) {
                #pragma HLS UNROLL
                col_filt[r][c] = col_filt[r][c+1];
            }
        }

        // Read old rows from filtered line buffer
        pixel_pack_t filt_old_row0_pack = line_buf_filt[0][col_val];
        pixel_pack_t filt_old_row1_pack = line_buf_filt[1][col_val];
        pixel_pack_t filt_old_row2_pack = line_buf_filt[2][col_val];
        pixel_pack_t filt_old_row3_pack = line_buf_filt[3][col_val];
        pixel_pack_t filt_old_row4_pack = line_buf_filt[4][col_val];

        pixel_t filt_old_row0 = (pixel_t)filt_old_row0_pack.range(9, 0).to_int();
        pixel_t filt_old_row1 = (pixel_t)filt_old_row1_pack.range(9, 0).to_int();
        pixel_t filt_old_row2 = (pixel_t)filt_old_row2_pack.range(9, 0).to_int();
        pixel_t filt_old_row3 = (pixel_t)filt_old_row3_pack.range(9, 0).to_int();
        pixel_t filt_old_row4 = (pixel_t)filt_old_row4_pack.range(9, 0).to_int();

        // Shift filtered line buffer: rows shift down, current pixel filtered
        // value will be written AFTER computation (see below)
        line_buf_filt[0][col_val] = (pixel_pack_t)filt_old_row1 << 0 | (pixel_pack_t)filt_old_row0 << 10;
        line_buf_filt[1][col_val] = (pixel_pack_t)filt_old_row2 << 0 | (pixel_pack_t)filt_old_row1 << 10;
        line_buf_filt[2][col_val] = (pixel_pack_t)filt_old_row3 << 0 | (pixel_pack_t)filt_old_row2 << 10;
        line_buf_filt[3][col_val] = (pixel_pack_t)filt_old_row4 << 0 | (pixel_pack_t)filt_old_row3 << 10;
        // line_buf_filt[4] will be updated with dout_pixel AFTER computation

        // Load col_filt register: col_filt[r][4] = filtered value at (col_val, row_val-r)
        // Rows 0-3: previously filtered pixels (already shifted from previous col)
        // Row 4: placeholders (current pixel hasn't been processed yet)
        col_filt[0][4] = filt_old_row0;  // (col_val, row-0) = filtered from prev col in same row
        col_filt[1][4] = filt_old_row1;  // (col_val, row-1)
        col_filt[2][4] = filt_old_row2;  // (col_val, row-2)
        col_filt[3][4] = filt_old_row3;  // (col_val, row-3)
        col_filt[4][4] = din.data;       // (col_val, row-4) = orig for now, updated after

        //======================================================================
        // Build gradient window from ORIGINAL line buffer (for stage1/stage2)
        //======================================================================
        for (int r = 0; r < 5; r++) {
            #pragma HLS UNROLL
            for (int c = 0; c < 5; c++) {
                #pragma HLS UNROLL
                int win_col = (int)col_val - 2 + c;
                if (c < 3) {
                    // col_src[r][c+2]: from shift register (columns col_val-1, col_val, col_val+1)
                    src_5x5[r][c] = col_src[r][c + 2];
                } else {
                    // col_src from line buffer
                    if (win_col < 0) {
                        src_5x5[r][c] = line_buf_src[r < 4 ? r : 3][0].range(9, 0);
                    } else if (win_col >= (int)img_width) {
                        src_5x5[r][c] = line_buf_src[r < 4 ? r : 3][(int)img_width - 1].range(9, 0);
                    } else {
                        src_5x5[r][c] = line_buf_src[r < 4 ? r : 3][win_col].range(9, 0);
                    }
                }
            }
        }

        //======================================================================
        // Stage 1: Sobel Gradient (reads ORIGINAL data)
        //======================================================================
        grad_t grad_h, grad_v, grad;
        isp.sobel_gradient_5x5(&src_5x5[0][0], grad_h, grad_v, grad);

        // Convert gradient window to s11 for stage2
        for (int i = 0; i < PATCH_SIZE; i++) {
            src_s11_5x5[i] = u10_to_s11(src_5x5[i / 5][i % 5]);
        }

        // Read grad_l BEFORE computing win_size (grad_l = grad(i-2,j) from shift register)
        // At pixel (i,j), BEFORE shift: grad_shift[0]=grad(i-3), grad_shift[1]=grad(i-2), grad_shift[2]=grad(i-1)
        grad_t grad_l_for_win = grad_shift[1];  // grad(i-2,j)

        // Compute win_size using max(grad_l, grad_c) - grad_r not available yet
        int win_size = isp.lut_win_size((int)((grad_l_for_win > grad) ? grad_l_for_win : grad));

        //======================================================================
        // Stage 2: Directional Average (reads ORIGINAL data)
        //======================================================================
        ISPCSIIR::DirAvgResult dir_avg = isp.compute_directional_avg(src_s11_5x5, win_size);

        //======================================================================
        // Stage 3: Gradient Fusion - neighbor gradients
        //======================================================================
        grad_t grad_next_row = grad_next_row_delay[col_val];

        grad_u = (grad_t)grad_buf_pack[0][col_val].range(13, 0);
        // grad_l from shift register BEFORE update: grad_shift[0] = grad(i-2,j)
        grad_l = grad_l_for_win;
        grad_t grad_c = grad;
        grad_d = grad_next_row;
        // grad_r approximation: use current grad (will be refined in delayed output)
        grad_r = grad;

        // Update shift register AFTER reading grad_l
        grad_shift[0] = grad_shift[1];
        grad_shift[1] = grad_shift[2];
        grad_shift[2] = grad;

        grad_buf_pack[0][col_val] = (grad_pack_t)grad << 0 | (grad_pack_t)grad_buf_pack[1][col_val].range(13, 0).to_int();
        grad_buf_pack[1][col_val] = (grad_pack_t)grad << 0 | (grad_pack_t)grad_next_row << 14;

        grad_next_row_delay[col_val] = current_grad;

        ISPCSIIR::FusionResult fusion = isp.compute_gradient_fusion(dir_avg,
            (int)grad_u, (int)grad_d, (int)grad_l, (int)grad_r, (int)grad_c);

        //======================================================================
        // Stage 4: IIR Blend
        // - grad/stage2 read src_s11_5x5 (ORIGINAL data)
        // - stage4 reads col_filt for filtered neighborhood
        //======================================================================
        // Build stage4's 5x5 window from filtered line buffer:
        //   Rows 0-3: from col_filt (filtered data from previous columns)
        //   Row 4: from col_src (original data for current row, to be replaced by filtered)
        s11_t filt_5x5[PATCH_SIZE];
        for (int r = 0; r < 5; r++) {
            #pragma HLS UNROLL
            for (int c = 0; c < 5; c++) {
                #pragma HLS UNROLL
                int win_col = (int)col_val - 2 + c;
                int patch_idx = r * 5 + c;
                if (r < 4) {
                    // Rows 0-3: from filtered col_filt
                    if (c < 3) {
                        filt_5x5[patch_idx] = u10_to_s11(col_filt[r][c + 2]);
                    } else {
                        if (win_col < 0) {
                            filt_5x5[patch_idx] = u10_to_s11(line_buf_filt[r][0].range(9, 0));
                        } else if (win_col >= (int)img_width) {
                            filt_5x5[patch_idx] = u10_to_s11(line_buf_filt[r][(int)img_width - 1].range(9, 0));
                        } else {
                            filt_5x5[patch_idx] = u10_to_s11(line_buf_filt[r][win_col].range(9, 0));
                        }
                    }
                } else {
                    // Row 4: original data (for boundary extension logic)
                    if (c < 3) {
                        filt_5x5[patch_idx] = u10_to_s11(col_src[4][c + 2]);
                    } else {
                        if (win_col < 0) {
                            filt_5x5[patch_idx] = u10_to_s11(line_buf_src[3][0].range(9, 0));
                        } else if (win_col >= (int)img_width) {
                            filt_5x5[patch_idx] = u10_to_s11(line_buf_src[3][(int)img_width - 1].range(9, 0));
                        } else {
                            filt_5x5[patch_idx] = u10_to_s11(line_buf_src[3][win_col].range(9, 0));
                        }
                    }
                }
            }
        }

        // Store values for potential delayed output
        delay_grad_h[0] = grad_h;
        delay_grad_v[0] = grad_v;
        delay_fusion[0] = fusion;
        delay_win_size[0] = win_size;
        delay_row[0] = row_val;
        delay_col[0] = col_val;
        dir_avg_delay[0] = dir_avg;

        s11_t final_patch[PATCH_SIZE];
        // Always use current values for now (output at same position as gradient)
        isp.compute_iir_blend(filt_5x5, win_size, fusion.blend0, fusion.blend1,
                              dir_avg.avg0_u, dir_avg.avg1_u,
                              (int)grad_h, (int)grad_v, final_patch);

        pixel_t dout_pixel;

        // Boundary handling: rows 0-1 output original input (passthrough)
        // This matches Python behavior where rows 0-1 are copied from input
        if (row_val < 2) {
            // For boundary rows, output original pixel directly
            dout_pixel = din.data;
        } else {
            // For rows 2+, output filtered result
            dout_pixel = s11_to_u10(final_patch[12]);
        }
        // Update line_buf_filt[4] with filtered value for current position
        // This makes the filtered value available for subsequent pixel processing
        pixel_pack_t filt_row4_pack = line_buf_filt[4][col_val];
        pixel_t filt_row4_even = (pixel_t)filt_row4_pack.range(9, 0).to_int();
        line_buf_filt[4][col_val] = (pixel_pack_t)dout_pixel << 0 | (pixel_pack_t)filt_row4_even << 10;
        // Also update col_filt[4][4] so the next column sees the filtered value
        col_filt[4][4] = dout_pixel;

                //======================================================================
        // Output
        //======================================================================
        axis_pixel_t dout;
        dout.data = dout_pixel;
        dout.last = (row_val == (unsigned int)img_height - 1 && col_val == (unsigned int)img_width - 1) ? 1 : 0;
        dout.user = (row_val == 0 && col_val == 0) ? 1 : 0;

        // Output current pixel (filtered value)
        dout_stream.write(dout);

        current_grad = grad;
    }
}

#endif // ISP_CSIIR_HLS_TOP_HPP
