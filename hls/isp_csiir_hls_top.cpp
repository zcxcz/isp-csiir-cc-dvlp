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

#include <cstring>
#ifndef __SYNTHESIS__
#include <vector>
#endif
#include "csiir_hls_backend.hpp"
#include "isp_csiir_regs.hpp"

//==============================================================================
// Type Definitions - Fixed-Point Precision
//==============================================================================
static const int DATA_WIDTH_I = 10;       // Input pixel width
static const int GRAD_WIDTH_I = 14;       // Gradient width
static const int ACC_WIDTH_I = 20;        // Accumulator width
static const int SIGNED_WIDTH_I = 11;     // Signed intermediate width
static const int PATCH_SIZE = 25;         // 5x5 window
static const int HORIZONTAL_TAP_STEP = 2; // Horizontal sample spacing

static const int MAX_WIDTH = 4096;
static const int MAX_HEIGHT = 4096;

// Bit width for 1PSRAM time-multiplexing (double width for 2-cycle access)
static const int PSRAM_PIXEL_WIDTH = DATA_WIDTH_I * 2;   // 20-bit for 2 pixels
static const int PSRAM_GRAD_WIDTH = GRAD_WIDTH_I * 2;    // 28-bit for 2 gradients

typedef csiir_hls::uint_t<DATA_WIDTH_I> pixel_t;
typedef csiir_hls::int_t<SIGNED_WIDTH_I> s11_t;
typedef csiir_hls::uint_t<GRAD_WIDTH_I> grad_t;
typedef csiir_hls::int_t<ACC_WIDTH_I> acc_t;

// Packed types for 1PSRAM time-multiplexing
typedef csiir_hls::uint_t<PSRAM_PIXEL_WIDTH> pixel_pack_t;   // 2 pixels packed
typedef csiir_hls::uint_t<PSRAM_GRAD_WIDTH> grad_pack_t;     // 2 gradients packed

//==============================================================================
// AXI-Stream Interface Types
//==============================================================================
struct axis_pixel_t {
    pixel_t data;
    csiir_hls::uint_t<1> last;
    csiir_hls::uint_t<1> user;
};

#ifndef __SYNTHESIS__
namespace csiir_debug {

struct Stage4InputTraceEntry {
    int idx;
    int center_x;
    int center_y;
    int win_size;
    int grad_h;
    int grad_v;
    int blend0;
    int blend1;
    int avg0_u;
    int avg1_u;
    int src_patch_u10[PATCH_SIZE];
};

struct FeedbackCommitTraceEntry {
    int idx;
    int center_x;
    int center_y;
    int write_xs[5];
    int columns_u10[5][5];
};

struct TraceStore {
    std::vector<Stage4InputTraceEntry> stage4_input_trace;
    std::vector<FeedbackCommitTraceEntry> feedback_commit_trace;
};

static TraceStore* g_trace_store = nullptr;

inline void set_trace_store(TraceStore* store) {
    g_trace_store = store;
}

inline void clear_trace_store() {
    g_trace_store = nullptr;
}

}  // namespace csiir_debug
#endif

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

inline int abs_i(int value) {
    return (value < 0) ? -value : value;
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
        grad_h = (grad_t)abs_i(sum_h);
        grad_v = (grad_t)abs_i(sum_v);
        int grad_i = round_div(grad_h, 5) + round_div(grad_v, 5);
        grad = (grad_t)clip<int>(grad_i, 0, 127);
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

        static const int K2_C[PATCH_SIZE] = {0,0,0,0,0,0,1,2,1,0,0,2,4,2,0,0,1,2,1,0,0,0,0,0,0};
        static const int K2_U[PATCH_SIZE] = {0,0,0,0,0,0,1,1,1,0,0,1,3,1,0,0,0,0,0,0,0,0,0,0,0};
        static const int K2_D[PATCH_SIZE] = {0,0,0,0,0,0,0,0,0,0,0,1,3,1,0,0,1,1,1,0,0,0,0,0,0};
        static const int K2_L[PATCH_SIZE] = {0,0,0,0,0,0,1,1,0,0,0,1,3,0,0,0,1,1,0,0,0,0,0,0,0};
        static const int K2_R[PATCH_SIZE] = {0,0,0,0,0,0,0,1,1,0,0,0,3,1,0,0,0,1,1,0,0,0,0,0,0};

        static const int K3_C[PATCH_SIZE] = {0,0,0,0,0,0,1,2,1,0,0,2,4,2,0,0,1,2,1,0,0,0,0,0,0};
        static const int K3_U[PATCH_SIZE] = {0,0,0,0,0,0,1,2,1,0,0,1,2,1,0,0,0,0,0,0,0,0,0,0,0};
        static const int K3_D[PATCH_SIZE] = {0,0,0,0,0,0,0,0,0,0,0,1,2,1,0,0,1,2,1,0,0,0,0,0,0};
        static const int K3_L[PATCH_SIZE] = {0,0,0,0,0,0,1,1,0,0,0,2,2,0,0,0,1,1,0,0,0,0,0,0,0};
        static const int K3_R[PATCH_SIZE] = {0,0,0,0,0,0,0,1,1,0,0,0,2,2,0,0,0,1,1,0,0,0,0,0,0};

        static const int K4_C[PATCH_SIZE] = {1,2,2,2,1,2,4,4,4,2,2,4,4,4,2,2,4,4,4,2,1,2,2,2,1};
        static const int K4_U[PATCH_SIZE] = {1,2,2,2,1,2,2,4,2,2,2,2,4,2,2,0,0,0,0,0,0,0,0,0,0};
        static const int K4_D[PATCH_SIZE] = {0,0,0,0,0,0,0,0,0,0,2,2,4,2,2,2,2,4,2,2,1,2,2,2,1};
        static const int K4_L[PATCH_SIZE] = {1,2,2,0,0,2,2,2,0,0,2,4,4,0,0,2,2,2,0,0,1,2,2,0,0};
        static const int K4_R[PATCH_SIZE] = {0,0,2,2,1,0,0,2,2,2,0,0,4,4,2,0,0,2,2,2,0,0,2,2,1};

        static const int K5_C[PATCH_SIZE] = {1,2,1,2,1,1,1,1,1,1,2,1,2,1,2,1,1,1,1,1,1,2,1,2,1};
        static const int K5_U[PATCH_SIZE] = {1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0};
        static const int K5_D[PATCH_SIZE] = {0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1};
        static const int K5_L[PATCH_SIZE] = {1,1,1,0,0,1,1,1,0,0,1,2,1,0,0,1,1,1,0,0,1,1,1,0,0};
        static const int K5_R[PATCH_SIZE] = {0,0,1,1,1,0,0,1,1,1,0,0,1,2,1,0,0,1,1,1,0,0,1,1,1};

        const int *avg0_c = nullptr, *avg0_u = nullptr, *avg0_d = nullptr, *avg0_l = nullptr, *avg0_r = nullptr;
        const int *avg1_c = nullptr, *avg1_u = nullptr, *avg1_d = nullptr, *avg1_l = nullptr, *avg1_r = nullptr;
        switch (kt) {
            case 0:
                avg1_c = K2_C; avg1_u = K2_U; avg1_d = K2_D; avg1_l = K2_L; avg1_r = K2_R;
                break;
            case 1:
                avg0_c = K2_C; avg0_u = K2_U; avg0_d = K2_D; avg0_l = K2_L; avg0_r = K2_R;
                avg1_c = K3_C; avg1_u = K3_U; avg1_d = K3_D; avg1_l = K3_L; avg1_r = K3_R;
                break;
            case 2:
                avg0_c = K3_C; avg0_u = K3_U; avg0_d = K3_D; avg0_l = K3_L; avg0_r = K3_R;
                avg1_c = K4_C; avg1_u = K4_U; avg1_d = K4_D; avg1_l = K4_L; avg1_r = K4_R;
                break;
            case 3:
                avg0_c = K4_C; avg0_u = K4_U; avg0_d = K4_D; avg0_l = K4_L; avg0_r = K4_R;
                avg1_c = K5_C; avg1_u = K5_U; avg1_d = K5_D; avg1_l = K5_L; avg1_r = K5_R;
                break;
            default:
                avg0_c = K5_C; avg0_u = K5_U; avg0_d = K5_D; avg0_l = K5_L; avg0_r = K5_R;
                break;
        }

        if (avg0_c) {
            result.avg0_c = weighted_avg(patch_s11, avg0_c);
            result.avg0_u = weighted_avg(patch_s11, avg0_u);
            result.avg0_d = weighted_avg(patch_s11, avg0_d);
            result.avg0_l = weighted_avg(patch_s11, avg0_l);
            result.avg0_r = weighted_avg(patch_s11, avg0_r);
        }

        if (avg1_c) {
            result.avg1_c = weighted_avg(patch_s11, avg1_c);
            result.avg1_u = weighted_avg(patch_s11, avg1_u);
            result.avg1_d = weighted_avg(patch_s11, avg1_d);
            result.avg1_l = weighted_avg(patch_s11, avg1_l);
            result.avg1_r = weighted_avg(patch_s11, avg1_r);
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

    FusionResult compute_gradient_fusion(const DirAvgResult& dir_avg,
                                         int grad_u, int grad_d,
                                         int grad_l, int grad_r, int grad_c) {
        FusionResult result;
        int g[5] = {grad_u, grad_d, grad_l, grad_r, grad_c};
        int v0[5] = {(int)dir_avg.avg0_u, (int)dir_avg.avg0_d, (int)dir_avg.avg0_l,
                     (int)dir_avg.avg0_r, (int)dir_avg.avg0_c};
        int v1[5] = {(int)dir_avg.avg1_u, (int)dir_avg.avg1_d, (int)dir_avg.avg1_l,
                     (int)dir_avg.avg1_r, (int)dir_avg.avg1_c};

        int min0_grad = 2048;
        int min0_grad_avg = 0;
        if (g[0] <= min0_grad) {
            min0_grad = g[0];
            min0_grad_avg = v0[0];
        }
        if (g[2] <= min0_grad) {
            min0_grad = g[2];
            min0_grad_avg = round_div(v0[2] + min0_grad_avg + 1, 2);
        }
        if (g[4] <= min0_grad) {
            min0_grad = g[4];
            min0_grad_avg = round_div(v0[4] + min0_grad_avg + 1, 2);
        }
        if (g[3] <= min0_grad) {
            min0_grad = g[3];
            min0_grad_avg = round_div(v0[3] + min0_grad_avg + 1, 2);
        }
        if (g[1] <= min0_grad) {
            min0_grad = g[1];
            min0_grad_avg = round_div(v0[1] + min0_grad_avg + 1, 2);
        }

        int min1_grad = 2048;
        int min1_grad_avg = 0;
        if (g[0] <= min1_grad) {
            min1_grad = g[0];
            min1_grad_avg = v1[0];
        }
        if (g[2] <= min1_grad) {
            min1_grad = g[2];
            min1_grad_avg = round_div(v1[2] + min1_grad_avg + 1, 2);
        }
        if (g[4] <= min1_grad) {
            min1_grad = g[4];
            min1_grad_avg = round_div(v1[4] + min1_grad_avg + 1, 2);
        }
        if (g[3] <= min1_grad) {
            min1_grad = g[3];
            min1_grad_avg = round_div(v1[3] + min1_grad_avg + 1, 2);
        }
        if (g[1] <= min1_grad) {
            min1_grad = g[1];
            min1_grad_avg = round_div(v1[1] + min1_grad_avg + 1, 2);
        }

        result.blend0 = saturate_s11(min0_grad_avg);
        result.blend1 = saturate_s11(min1_grad_avg);

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

        bool horiz_dom = (int)grad_v > (int)grad_h;

        // Orientation factor
        static const int F_ORI_V[PATCH_SIZE] = {0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0};
        static const int F_ORI_H[PATCH_SIZE] = {0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0};
        const int* f_orient = horiz_dom ? F_ORI_H : F_ORI_V;

        // Blend kernels
        static const int F2X2[PATCH_SIZE] = {0,0,0,0,0,0,1,2,1,0,0,2,4,2,0,0,1,2,1,0,0,0,0,0,0};
        static const int F3X3[PATCH_SIZE] = {0,0,0,0,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,0,0,0,0};
        static const int F4X4[PATCH_SIZE] = {1,2,2,2,1,2,4,4,4,2,2,4,4,4,2,2,4,4,4,2,1,2,2,2,1};
        static const int F5X5[PATCH_SIZE] = {4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4};

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
    csiir_hls::stream_t<axis_pixel_t> &din_stream,
    csiir_hls::stream_t<axis_pixel_t> &dout_stream,
    ISPCSIIR_Regs &regs
) {
    CSIIR_HLS_INTERFACE_AXIS(din_stream);
    CSIIR_HLS_INTERFACE_AXIS(dout_stream);
    CSIIR_HLS_INTERFACE_AXILITE(regs, CTRL);
    CSIIR_HLS_INTERFACE_AXILITE(return, CTRL);
    CSIIR_HLS_INTERFACE_CTRL_HS(return);

    // Configure ISPCSIIR instance
    ISPCSIIR isp;
    isp.cfg.img_width = (int)regs.img_width;
    isp.cfg.img_height = (int)regs.img_height;
    for (int i = 0; i < 4; i++) {
        CSIIR_HLS_UNROLL;
        isp.cfg.win_size_thresh[i] = (int)regs.win_size_thresh[i];
        isp.cfg.win_size_clip_y[i] = (int)regs.win_size_clip_y[i];
        isp.cfg.win_size_clip_sft[i] = (int)regs.win_size_clip_sft[i];
        isp.cfg.blending_ratio[i] = (int)regs.blending_ratio[i];
    }
    isp.cfg.reg_edge_protect = (int)regs.edge_protect;

    const int img_width = (int)regs.img_width;
    const int img_height = (int)regs.img_height;
    pixel_t orig_rows[5][MAX_WIDTH];
    CSIIR_HLS_ARRAY_PARTITION_COMPLETE(orig_rows, 1);
    pixel_t filt_rows[3][MAX_WIDTH];
    CSIIR_HLS_ARRAY_PARTITION_COMPLETE(filt_rows, 1);
    pixel_t filt_current[MAX_WIDTH];
    pixel_t emit_row[MAX_WIDTH];
    pixel_t patch_u10[PATCH_SIZE];
    s11_t patch_s11[PATCH_SIZE];
    int grad_row_buf[2][MAX_WIDTH];
    int grad_shift[3];
    int orig_row_ids[5];
    int filt_row_ids[3];

    for (int i = 0; i < 5; i++) {
        orig_row_ids[i] = -1;
    }
    for (int i = 0; i < 3; i++) {
        filt_row_ids[i] = -1;
    }
    for (int row = 0; row < 2; row++) {
        for (int x = 0; x < img_width; x++) {
            grad_row_buf[row][x] = 0;
        }
    }
    grad_shift[0] = 0;
    grad_shift[1] = 0;
    grad_shift[2] = 0;

    auto find_orig_slot = [&](int row) -> int {
        for (int i = 0; i < 5; i++) {
            if (orig_row_ids[i] == row) {
                return i;
            }
        }
        return -1;
    };

    auto find_filt_slot = [&](int row) -> int {
        for (int i = 0; i < 3; i++) {
            if (filt_row_ids[i] == row) {
                return i;
            }
        }
        return -1;
    };

    auto get_orig_pixel = [&](int row, int col) -> pixel_t {
        int y = clip<int>(row, 0, img_height - 1);
        int x = clip<int>(col, 0, img_width - 1);
        int slot = find_orig_slot(y);
        return (slot >= 0) ? orig_rows[slot][x] : pixel_t(0);
    };

    auto get_filt_pixel = [&](int row, int col, int current_row) -> pixel_t {
        int y = clip<int>(row, 0, img_height - 1);
        int x = clip<int>(col, 0, img_width - 1);
        if (y == current_row) {
            return filt_current[x];
        }
        int slot = find_filt_slot(y);
        return (slot >= 0) ? filt_rows[slot][x] : get_orig_pixel(y, x);
    };

    auto build_orig_window = [&](int center_x, int center_y, pixel_t patch[PATCH_SIZE]) {
        for (int dy = -2; dy <= 2; dy++) {
            for (int dx = -2; dx <= 2; dx++) {
                int patch_idx = (dy + 2) * 5 + (dx + 2);
                patch[patch_idx] = get_orig_pixel(center_y + dy, center_x + dx * HORIZONTAL_TAP_STEP);
            }
        }
    };

    auto build_stage4_window = [&](int center_x, int center_y, pixel_t patch[PATCH_SIZE]) {
        for (int dy = -2; dy <= 2; dy++) {
            for (int dx = -2; dx <= 2; dx++) {
                int patch_idx = (dy + 2) * 5 + (dx + 2);
                int tap_x = center_x + dx * HORIZONTAL_TAP_STEP;
                patch[patch_idx] = (dy < 0)
                    ? get_filt_pixel(center_y + dy, tap_x, center_y)
                    : get_orig_pixel(center_y + dy, tap_x);
            }
        }
    };

    auto emit_output_row = [&](int row) {
        for (int x = 0; x < img_width; x++) {
            axis_pixel_t dout;
            dout.data = emit_row[x];
            dout.user = (row == 0 && x == 0) ? 1 : 0;
            dout.last = (row == img_height - 1 && x == img_width - 1) ? 1 : 0;
            dout_stream.write(dout);
        }
    };

    auto process_row = [&](int target_row) {
        for (int x = 0; x < img_width; x++) {
            filt_current[x] = get_orig_pixel(target_row, x);
        }

        for (int x = 0; x < img_width; x++) {
            pixel_t center_patch[PATCH_SIZE];
            pixel_t left_patch[PATCH_SIZE];
            pixel_t right_patch[PATCH_SIZE];

            build_orig_window(x, target_row, center_patch);
            build_orig_window(x - HORIZONTAL_TAP_STEP, target_row, left_patch);
            build_orig_window(x + HORIZONTAL_TAP_STEP, target_row, right_patch);

            grad_t grad_h, grad_v, grad_c_u14;
            grad_t grad_tmp_h, grad_tmp_v, grad_l_u14, grad_r_u14;
            isp.sobel_gradient_5x5(center_patch, grad_h, grad_v, grad_c_u14);
            isp.sobel_gradient_5x5(left_patch, grad_tmp_h, grad_tmp_v, grad_l_u14);
            isp.sobel_gradient_5x5(right_patch, grad_tmp_h, grad_tmp_v, grad_r_u14);

            for (int i = 0; i < PATCH_SIZE; i++) {
                patch_s11[i] = u10_to_s11(center_patch[i]);
            }

            int grad_c = (int)grad_c_u14;
            int grad_l = (x > 0) ? (int)grad_l_u14 : grad_c;
            int grad_r = (x < img_width - 1) ? (int)grad_r_u14 : grad_c;
            int grad_u = (target_row > 0) ? grad_shift[1] : grad_c;
            int grad_d = (target_row < img_height - 1) ? grad_row_buf[0][x] : grad_c;

            grad_shift[0] = grad_shift[1];
            grad_shift[1] = grad_shift[2];
            grad_shift[2] = grad_c;
            grad_row_buf[0][x] = grad_c;
            grad_row_buf[1][x] = grad_row_buf[0][x];

            int grad_triplet_max = grad_l;
            if (grad_c > grad_triplet_max) grad_triplet_max = grad_c;
            if (grad_r > grad_triplet_max) grad_triplet_max = grad_r;
            int win_size = isp.lut_win_size(grad_triplet_max);

            ISPCSIIR::DirAvgResult dir_avg = isp.compute_directional_avg(patch_s11, win_size);
            ISPCSIIR::FusionResult fusion = isp.compute_gradient_fusion(
                dir_avg, grad_u, grad_d, grad_l, grad_r, grad_c
            );

            build_stage4_window(x, target_row, patch_u10);
            for (int i = 0; i < PATCH_SIZE; i++) {
                patch_s11[i] = u10_to_s11(patch_u10[i]);
            }

#ifndef __SYNTHESIS__
            if (csiir_debug::g_trace_store != nullptr) {
                csiir_debug::Stage4InputTraceEntry entry;
                entry.idx = (int)csiir_debug::g_trace_store->stage4_input_trace.size();
                entry.center_x = x;
                entry.center_y = target_row;
                entry.win_size = win_size;
                entry.grad_h = (int)grad_h;
                entry.grad_v = (int)grad_v;
                entry.blend0 = fusion.blend0;
                entry.blend1 = fusion.blend1;
                entry.avg0_u = dir_avg.avg0_u;
                entry.avg1_u = dir_avg.avg1_u;
                for (int i = 0; i < PATCH_SIZE; i++) {
                    entry.src_patch_u10[i] = (int)patch_u10[i];
                }
                csiir_debug::g_trace_store->stage4_input_trace.push_back(entry);
            }
#endif

            s11_t final_patch[PATCH_SIZE];
            isp.compute_iir_blend(
                patch_s11,
                win_size,
                fusion.blend0,
                fusion.blend1,
                dir_avg.avg0_u,
                dir_avg.avg1_u,
                (int)grad_h,
                (int)grad_v,
                final_patch
            );

#ifndef __SYNTHESIS__
            if (csiir_debug::g_trace_store != nullptr) {
                csiir_debug::FeedbackCommitTraceEntry entry;
                entry.idx = (int)csiir_debug::g_trace_store->feedback_commit_trace.size();
                entry.center_x = x;
                entry.center_y = target_row;
                for (int col = 0; col < 5; col++) {
                    entry.write_xs[col] = clip<int>(x + (col - 2) * HORIZONTAL_TAP_STEP, 0, img_width - 1);
                    for (int row = 0; row < 5; row++) {
                        entry.columns_u10[col][row] = (int)s11_to_u10(final_patch[row * 5 + col]);
                    }
                }
                csiir_debug::g_trace_store->feedback_commit_trace.push_back(entry);
            }
#endif
            filt_current[x] = s11_to_u10(final_patch[12]);
        }

        int slot = target_row % 3;
        filt_row_ids[slot] = target_row;
        for (int x = 0; x < img_width; x++) {
            filt_rows[slot][x] = filt_current[x];
        }
    };

    for (int row = 0; row < img_height; row++) {
        int slot = row % 5;
        orig_row_ids[slot] = row;
        for (int x = 0; x < img_width; x++) {
            axis_pixel_t din = din_stream.read();
            orig_rows[slot][x] = din.data;
        }

        if (row >= 2) {
            int target_row = row - 2;
            process_row(target_row);
            if (target_row < 2) {
                int emit_slot = find_orig_slot(target_row);
                for (int x = 0; x < img_width; x++) {
                    emit_row[x] = orig_rows[emit_slot][x];
                }
            } else {
                for (int x = 0; x < img_width; x++) {
                    emit_row[x] = filt_current[x];
                }
            }
            emit_output_row(target_row);
        }
    }

    for (int target_row = img_height - 2; target_row < img_height; target_row++) {
        if (target_row < 0) {
            continue;
        }
        process_row(target_row);
        for (int x = 0; x < img_width; x++) {
            emit_row[x] = filt_current[x];
        }
        emit_output_row(target_row);
    }
}

#endif // ISP_CSIIR_HLS_TOP_HPP
