//==============================================================================
// ISP-CSIIR HLS Top Module - Stream-based Processing with pixel-stage classes
//==============================================================================

#ifndef ISP_CSIIR_HLS_TOP_HPP
#define ISP_CSIIR_HLS_TOP_HPP

#include <cstring>
#ifndef __SYNTHESIS__
#include <vector>
#endif
#include "csiir_hls_backend.hpp"
#include "isp_csiir_regs.hpp"

static const int DATA_WIDTH_I = 10;
static const int GRAD_WIDTH_I = 14;
static const int ACC_WIDTH_I = 20;
static const int SIGNED_WIDTH_I = 11;
static const int PATCH_SIZE = 25;
static const int HORIZONTAL_TAP_STEP = 2;

static const int MAX_WIDTH = 4096;
static const int MAX_HEIGHT = 4096;

static const int PSRAM_PIXEL_WIDTH = DATA_WIDTH_I * 2;
static const int PSRAM_GRAD_WIDTH = GRAD_WIDTH_I * 2;

typedef csiir_hls::uint_t<DATA_WIDTH_I> pixel_t;
typedef csiir_hls::int_t<SIGNED_WIDTH_I> s11_t;
typedef csiir_hls::uint_t<GRAD_WIDTH_I> grad_t;
typedef csiir_hls::int_t<ACC_WIDTH_I> acc_t;

typedef csiir_hls::uint_t<PSRAM_PIXEL_WIDTH> pixel_pack_t;
typedef csiir_hls::uint_t<PSRAM_GRAD_WIDTH> grad_pack_t;

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

inline int round_div(int num, int den) {
    if (den == 0) return 0;
    if (num >= 0) return (num + den / 2) / den;
    return -(((-num) + den / 2) / den);
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

class ISPCSIIR {
public:
    struct DirAvgResult {
        s11_t avg0_c, avg0_u, avg0_d, avg0_l, avg0_r;
        s11_t avg1_c, avg1_u, avg1_d, avg1_l, avg1_r;
    };

    struct FusionResult {
        s11_t blend0;
        s11_t blend1;
    };

    Config cfg;

    ISPCSIIR() {}
    ISPCSIIR(const Config& c) : cfg(c) {}

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

    int select_kernel_type(int win_size) {
        if (win_size < cfg.win_size_thresh[0]) return 0;
        if (win_size < cfg.win_size_thresh[1]) return 1;
        if (win_size < cfg.win_size_thresh[2]) return 2;
        if (win_size < cfg.win_size_thresh[3]) return 3;
        return 4;
    }

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

    int get_ratio(int win_size) {
        int idx = clip<int>(win_size / 8 - 2, 0, 3);
        return cfg.blending_ratio[idx];
    }

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

        static const int F_ORI_V[PATCH_SIZE] = {0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0};
        static const int F_ORI_H[PATCH_SIZE] = {0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0};
        const int* f_orient = horiz_dom ? F_ORI_H : F_ORI_V;

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

struct PixelCoord {
    int x;
    int y;
};

struct GradientStageResult {
    grad_t grad_h;
    grad_t grad_v;
    int grad_c;
    int grad_l;
    int grad_r;
    int grad_u;
    int grad_d;
    int win_size;
    s11_t center_patch_s11[PATCH_SIZE];
};

struct FilterStageResult {
    pixel_t stage4_patch_u10[PATCH_SIZE];
    s11_t stage4_patch_s11[PATCH_SIZE];
    ISPCSIIR::DirAvgResult dir_avg;
    ISPCSIIR::FusionResult fusion;
    s11_t final_patch[PATCH_SIZE];
    pixel_t final_center;
};

static inline void trace_stage4_input(
    const PixelCoord& coord,
    const GradientStageResult& grad_state,
    const FilterStageResult& filt_state
);

static inline void trace_feedback_commit(
    const PixelCoord& coord,
    const FilterStageResult& filt_state,
    int img_width
);

class lb_grad_pixel {
public:
    pixel_t mem[MAX_WIDTH];
    int row_id;

    lb_grad_pixel() { reset(); }

    void reset() {
        row_id = -1;
        for (int x = 0; x < MAX_WIDTH; x++) {
            mem[x] = 0;
        }
    }

    void attach_row(int row) {
        row_id = row;
    }

    void write(int x, pixel_t value) {
        mem[x] = value;
    }

    pixel_t read(int x) const {
        return mem[x];
    }
};

class lb_grad_value {
public:
    int mem[MAX_WIDTH];
    int row_id;

    lb_grad_value() { reset(); }

    void reset() {
        row_id = -1;
        for (int x = 0; x < MAX_WIDTH; x++) {
            mem[x] = 0;
        }
    }

    void attach_row(int row) {
        row_id = row;
    }

    void write(int x, int value) {
        mem[x] = value;
    }

    int read(int x) const {
        return mem[x];
    }
};

class lb_filt_pixel {
public:
    pixel_t mem[MAX_WIDTH];
    int row_id;

    lb_filt_pixel() { reset(); }

    void reset() {
        row_id = -1;
        for (int x = 0; x < MAX_WIDTH; x++) {
            mem[x] = 0;
        }
    }

    void attach_row(int row) {
        row_id = row;
    }

    void write(int x, pixel_t value) {
        mem[x] = value;
    }

    pixel_t read(int x) const {
        return mem[x];
    }
};

class stage_gradient_op {
public:
    int img_width;
    int img_height;
    int win_size_clip_y[4];
    int win_size_clip_sft[4];
    stage_gradient_op() {}

    void init(const Config& cfg) {
        img_width = cfg.img_width;
        img_height = cfg.img_height;
        for (int i = 0; i < 4; i++) {
            win_size_clip_y[i] = cfg.win_size_clip_y[i];
            win_size_clip_sft[i] = cfg.win_size_clip_sft[i];
        }
    }

    void run(const pixel_t window[PATCH_SIZE], grad_t& grad_h, grad_t& grad_v, grad_t& grad) const {
        int sum_h = window[0] + window[1] + window[2] + window[3] + window[4]
                  - window[20] - window[21] - window[22] - window[23] - window[24];
        int sum_v = window[0] + window[5] + window[10] + window[15] + window[20]
                  - window[4] - window[9] - window[14] - window[19] - window[24];
        grad_h = (grad_t)abs_i(sum_h);
        grad_v = (grad_t)abs_i(sum_v);
        int grad_i = round_div(grad_h, 5) + round_div(grad_v, 5);
        grad = (grad_t)clip<int>(grad_i, 0, 127);
    }
};

namespace csiir_pipe {

class stage_gradient {
public:
    stage_gradient_op op_center;
    stage_gradient_op op_left;
    stage_gradient_op op_right;
    stage_gradient_op op_up;
    stage_gradient_op op_down;
    lb_grad_pixel& lb0;
    lb_grad_pixel& lb1;
    lb_grad_pixel& lb2;
    lb_grad_pixel& lb3;
    lb_grad_pixel& lb4;
    lb_grad_value& grad_row;
    int img_width;
    int img_height;
    int win_size_clip_y[4];
    int win_size_clip_sft[4];

    stage_gradient(lb_grad_pixel& a,
                   lb_grad_pixel& b,
                   lb_grad_pixel& c,
                   lb_grad_pixel& d,
                   lb_grad_pixel& e,
                   lb_grad_value& grad_lb)
        : lb0(a), lb1(b), lb2(c), lb3(d), lb4(e), grad_row(grad_lb) {}

    void init(const Config& cfg) {
        img_width = cfg.img_width;
        img_height = cfg.img_height;
        for (int i = 0; i < 4; i++) {
            win_size_clip_y[i] = cfg.win_size_clip_y[i];
            win_size_clip_sft[i] = cfg.win_size_clip_sft[i];
        }
        op_center.init(cfg);
        op_left.init(cfg);
        op_right.init(cfg);
        op_up.init(cfg);
        op_down.init(cfg);
    }

    void reset() {
        grad_row.reset();
    }

    void run(const PixelCoord& coord, GradientStageResult& out) {
        pixel_t center[PATCH_SIZE];
        pixel_t left[PATCH_SIZE];
        pixel_t right[PATCH_SIZE];
        pixel_t up[PATCH_SIZE];
        pixel_t down[PATCH_SIZE];
        grad_t tmp_h, tmp_v;
        grad_t grad_c_u14, grad_l_u14, grad_r_u14, grad_u_u14, grad_d_u14;

        build_window(coord.x, coord.y, center);
        build_window(coord.x - HORIZONTAL_TAP_STEP, coord.y, left);
        build_window(coord.x + HORIZONTAL_TAP_STEP, coord.y, right);
        build_window(coord.x, coord.y - 1, up);
        build_window(coord.x, coord.y + 1, down);

        op_center.run(center, out.grad_h, out.grad_v, grad_c_u14);
        op_left.run(left, tmp_h, tmp_v, grad_l_u14);
        op_right.run(right, tmp_h, tmp_v, grad_r_u14);
        op_up.run(up, tmp_h, tmp_v, grad_u_u14);
        op_down.run(down, tmp_h, tmp_v, grad_d_u14);

        for (int i = 0; i < PATCH_SIZE; i++) {
            out.center_patch_s11[i] = u10_to_s11(center[i]);
        }

        out.grad_c = (int)grad_c_u14;
        out.grad_l = (coord.x > 0) ? (int)grad_l_u14 : out.grad_c;
        out.grad_r = (coord.x < img_width - 1) ? (int)grad_r_u14 : out.grad_c;
        out.grad_u = (coord.y > 0) ? (int)grad_u_u14 : out.grad_c;
        out.grad_d = (coord.y < img_height - 1) ? (int)grad_d_u14 : out.grad_c;

        grad_row.attach_row(coord.y);
        grad_row.write(coord.x, out.grad_c);

        int grad_triplet_max = out.grad_l;
        if (out.grad_c > grad_triplet_max) grad_triplet_max = out.grad_c;
        if (out.grad_r > grad_triplet_max) grad_triplet_max = out.grad_r;
        out.win_size = lut_win_size(grad_triplet_max);
    }

private:
    pixel_t read_orig_pixel(int row, int col) const {
        int y = clip<int>(row, 0, img_height - 1);
        int x = clip<int>(col, 0, img_width - 1);
        const lb_grad_pixel* lb = find_grad_lb(y);
        return (lb != nullptr) ? lb->read(x) : pixel_t(0);
    }

    const lb_grad_pixel* find_grad_lb(int row) const {
        if (lb0.row_id == row) return &lb0;
        if (lb1.row_id == row) return &lb1;
        if (lb2.row_id == row) return &lb2;
        if (lb3.row_id == row) return &lb3;
        if (lb4.row_id == row) return &lb4;
        return nullptr;
    }

    void build_window(int center_x, int center_y, pixel_t patch[PATCH_SIZE]) const {
        for (int dy = -2; dy <= 2; dy++) {
            for (int dx = -2; dx <= 2; dx++) {
                int idx = (dy + 2) * 5 + (dx + 2);
                patch[idx] = read_orig_pixel(center_y + dy, center_x + dx * HORIZONTAL_TAP_STEP);
            }
        }
    }

    int lut_win_size(int grad_triplet_max) const {
        int x_nodes[4];
        int acc = 0;
        for (int i = 0; i < 4; i++) {
            acc += (1 << win_size_clip_sft[i]);
            x_nodes[i] = acc;
        }

        int x = grad_triplet_max;
        int y0 = win_size_clip_y[0];
        int y1 = win_size_clip_y[3];

        if (x <= x_nodes[0]) return clip_win_size(y0);
        if (x >= x_nodes[3]) return clip_win_size(y1);

        int win_size = y1;
        for (int idx = 0; idx < 3; idx++) {
            if (x_nodes[idx] <= x && x <= x_nodes[idx + 1]) {
                int x0 = x_nodes[idx];
                int x1 = x_nodes[idx + 1];
                int y0_i = win_size_clip_y[idx];
                int y1_i = win_size_clip_y[idx + 1];
                win_size = y0_i + round_div((x - x0) * (y1_i - y0_i), (x1 - x0));
                break;
            }
        }
        return clip_win_size(win_size);
    }
};

class stage_filter {
public:
    lb_grad_pixel& grad_lb0;
    lb_grad_pixel& grad_lb1;
    lb_grad_pixel& grad_lb2;
    lb_grad_pixel& grad_lb3;
    lb_grad_pixel& grad_lb4;
    lb_filt_pixel& filt_lb0;
    lb_filt_pixel& filt_lb1;
    lb_filt_pixel& filt_lb2;
    int img_width;
    int img_height;
    int win_size_thresh[4];
    int blending_ratio[4];
    int reg_edge_protect;

    stage_filter(lb_grad_pixel& g0,
                 lb_grad_pixel& g1,
                 lb_grad_pixel& g2,
                 lb_grad_pixel& g3,
                 lb_grad_pixel& g4,
                 lb_filt_pixel& f0,
                 lb_filt_pixel& f1,
                 lb_filt_pixel& f2)
        : grad_lb0(g0), grad_lb1(g1), grad_lb2(g2), grad_lb3(g3), grad_lb4(g4),
          filt_lb0(f0), filt_lb1(f1), filt_lb2(f2) {}

    void init(const Config& cfg) {
        img_width = cfg.img_width;
        img_height = cfg.img_height;
        for (int i = 0; i < 4; i++) {
            win_size_thresh[i] = cfg.win_size_thresh[i];
            blending_ratio[i] = cfg.blending_ratio[i];
        }
        reg_edge_protect = cfg.reg_edge_protect;
    }

    void reset() {}

    void run(const PixelCoord& coord,
             const GradientStageResult& grad_state,
             FilterStageResult& out) const {
        build_stage4_window(coord, out.stage4_patch_u10);
        for (int i = 0; i < PATCH_SIZE; i++) {
            out.stage4_patch_s11[i] = u10_to_s11(out.stage4_patch_u10[i]);
        }

        out.dir_avg = compute_directional_avg(grad_state.center_patch_s11, grad_state.win_size);
        out.fusion = compute_gradient_fusion(
            out.dir_avg,
            grad_state.grad_u,
            grad_state.grad_d,
            grad_state.grad_l,
            grad_state.grad_r,
            grad_state.grad_c
        );

        compute_iir_blend(
            out.stage4_patch_s11,
            grad_state.win_size,
            out.fusion.blend0,
            out.fusion.blend1,
            out.dir_avg.avg0_u,
            out.dir_avg.avg1_u,
            (int)grad_state.grad_h,
            (int)grad_state.grad_v,
            out.final_patch
        );
        out.final_center = s11_to_u10(out.final_patch[12]);
    }

private:
    const lb_grad_pixel* find_grad_lb(int row) const {
        if (grad_lb0.row_id == row) return &grad_lb0;
        if (grad_lb1.row_id == row) return &grad_lb1;
        if (grad_lb2.row_id == row) return &grad_lb2;
        if (grad_lb3.row_id == row) return &grad_lb3;
        if (grad_lb4.row_id == row) return &grad_lb4;
        return nullptr;
    }

    const lb_filt_pixel* find_filt_lb(int row) const {
        if (filt_lb0.row_id == row) return &filt_lb0;
        if (filt_lb1.row_id == row) return &filt_lb1;
        if (filt_lb2.row_id == row) return &filt_lb2;
        return nullptr;
    }

    pixel_t read_grad_pixel(int row, int col) const {
        int y = clip<int>(row, 0, img_height - 1);
        int x = clip<int>(col, 0, img_width - 1);
        const lb_grad_pixel* lb = find_grad_lb(y);
        return (lb != nullptr) ? lb->read(x) : pixel_t(0);
    }

    pixel_t read_filt_pixel(int row, int col) const {
        int y = clip<int>(row, 0, img_height - 1);
        int x = clip<int>(col, 0, img_width - 1);
        const lb_filt_pixel* lb = find_filt_lb(y);
        if (lb != nullptr) {
            return lb->read(x);
        }
        return read_grad_pixel(y, x);
    }

    void build_stage4_window(const PixelCoord& coord, pixel_t patch[PATCH_SIZE]) const {
        for (int dy = -2; dy <= 2; dy++) {
            for (int dx = -2; dx <= 2; dx++) {
                int idx = (dy + 2) * 5 + (dx + 2);
                int tap_x = coord.x + dx * HORIZONTAL_TAP_STEP;
                patch[idx] = (dy < 0) ? read_filt_pixel(coord.y + dy, tap_x)
                                      : read_grad_pixel(coord.y + dy, tap_x);
            }
        }
    }

    int select_kernel_type(int win_size) const {
        if (win_size < win_size_thresh[0]) return 0;
        if (win_size < win_size_thresh[1]) return 1;
        if (win_size < win_size_thresh[2]) return 2;
        if (win_size < win_size_thresh[3]) return 3;
        return 4;
    }

    s11_t weighted_avg(const s11_t patch_s11[PATCH_SIZE], const int* kernel) const {
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

    ISPCSIIR::DirAvgResult compute_directional_avg(const s11_t patch_s11[PATCH_SIZE], int win_size) const {
        ISPCSIIR::DirAvgResult result = {};
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

    ISPCSIIR::FusionResult compute_gradient_fusion(const ISPCSIIR::DirAvgResult& dir_avg,
                                                   int grad_u,
                                                   int grad_d,
                                                   int grad_l,
                                                   int grad_r,
                                                   int grad_c) const {
        ISPCSIIR::FusionResult result;
        int g[5] = {grad_u, grad_d, grad_l, grad_r, grad_c};
        int v0[5] = {(int)dir_avg.avg0_u, (int)dir_avg.avg0_d, (int)dir_avg.avg0_l,
                     (int)dir_avg.avg0_r, (int)dir_avg.avg0_c};
        int v1[5] = {(int)dir_avg.avg1_u, (int)dir_avg.avg1_d, (int)dir_avg.avg1_l,
                     (int)dir_avg.avg1_r, (int)dir_avg.avg1_c};

        int min0_grad = 2048;
        int min0_grad_avg = 0;
        if (g[0] <= min0_grad) { min0_grad = g[0]; min0_grad_avg = v0[0]; }
        if (g[2] <= min0_grad) { min0_grad = g[2]; min0_grad_avg = round_div(v0[2] + min0_grad_avg + 1, 2); }
        if (g[4] <= min0_grad) { min0_grad = g[4]; min0_grad_avg = round_div(v0[4] + min0_grad_avg + 1, 2); }
        if (g[3] <= min0_grad) { min0_grad = g[3]; min0_grad_avg = round_div(v0[3] + min0_grad_avg + 1, 2); }
        if (g[1] <= min0_grad) { min0_grad = g[1]; min0_grad_avg = round_div(v0[1] + min0_grad_avg + 1, 2); }

        int min1_grad = 2048;
        int min1_grad_avg = 0;
        if (g[0] <= min1_grad) { min1_grad = g[0]; min1_grad_avg = v1[0]; }
        if (g[2] <= min1_grad) { min1_grad = g[2]; min1_grad_avg = round_div(v1[2] + min1_grad_avg + 1, 2); }
        if (g[4] <= min1_grad) { min1_grad = g[4]; min1_grad_avg = round_div(v1[4] + min1_grad_avg + 1, 2); }
        if (g[3] <= min1_grad) { min1_grad = g[3]; min1_grad_avg = round_div(v1[3] + min1_grad_avg + 1, 2); }
        if (g[1] <= min1_grad) { min1_grad = g[1]; min1_grad_avg = round_div(v1[1] + min1_grad_avg + 1, 2); }

        result.blend0 = saturate_s11(min0_grad_avg);
        result.blend1 = saturate_s11(min1_grad_avg);
        return result;
    }

    int get_ratio(int win_size) const {
        int idx = clip<int>(win_size / 8 - 2, 0, 3);
        return blending_ratio[idx];
    }

    void mix_scalar(s11_t scalar,
                    const s11_t src[PATCH_SIZE],
                    const int factor[PATCH_SIZE],
                    s11_t out[PATCH_SIZE]) const {
        for (int i = 0; i < PATCH_SIZE; i++) {
            int val = round_div((int)scalar * factor[i] + (int)src[i] * (4 - factor[i]), 4);
            out[i] = saturate_s11(val);
        }
    }

    void compute_iir_blend(const s11_t src[PATCH_SIZE],
                           int win_size,
                           s11_t blend0_g,
                           s11_t blend1_g,
                           s11_t avg0_u,
                           s11_t avg1_u,
                           int grad_h,
                           int grad_v,
                           s11_t final_patch[PATCH_SIZE]) const {
        int ratio = get_ratio(win_size);
        s11_t b0_hor = saturate_s11(round_div((int)ratio * blend0_g + (64 - ratio) * avg0_u, 64));
        s11_t b1_hor = saturate_s11(round_div((int)ratio * blend1_g + (64 - ratio) * avg1_u, 64));
        bool horiz_dom = (int)grad_v > (int)grad_h;

        static const int F_ORI_V[PATCH_SIZE] = {0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0};
        static const int F_ORI_H[PATCH_SIZE] = {0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0};
        const int* f_orient = horiz_dom ? F_ORI_H : F_ORI_V;

        static const int F2X2[PATCH_SIZE] = {0,0,0,0,0,0,1,2,1,0,0,2,4,2,0,0,1,2,1,0,0,0,0,0,0};
        static const int F3X3[PATCH_SIZE] = {0,0,0,0,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1,0,0,0,0,0,0};
        static const int F4X4[PATCH_SIZE] = {1,2,2,2,1,2,4,4,4,2,2,4,4,4,2,2,4,4,4,2,1,2,2,2,1};
        static const int F5X5[PATCH_SIZE] = {4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4};

        s11_t blend0_win[PATCH_SIZE], blend1_win[PATCH_SIZE];
        if (win_size < win_size_thresh[0]) {
            s11_t tmp1[PATCH_SIZE], tmp2[PATCH_SIZE];
            mix_scalar(b1_hor, src, f_orient, tmp1);
            mix_scalar(b1_hor, src, F2X2, tmp2);
            for (int i = 0; i < PATCH_SIZE; i++) {
                int val = round_div((int)tmp1[i] * reg_edge_protect + (int)tmp2[i] * (64 - reg_edge_protect), 64);
                blend1_win[i] = saturate_s11(val);
                blend0_win[i] = 0;
            }
        } else if (win_size < win_size_thresh[1]) {
            s11_t tmp1[PATCH_SIZE], tmp2[PATCH_SIZE];
            mix_scalar(b1_hor, src, f_orient, tmp1);
            mix_scalar(b1_hor, src, F2X2, tmp2);
            for (int i = 0; i < PATCH_SIZE; i++) {
                int val = round_div((int)tmp1[i] * reg_edge_protect + (int)tmp2[i] * (64 - reg_edge_protect), 64);
                blend1_win[i] = saturate_s11(val);
            }
            mix_scalar(b0_hor, src, F3X3, blend0_win);
        } else if (win_size < win_size_thresh[2]) {
            mix_scalar(b1_hor, src, F3X3, blend1_win);
            mix_scalar(b0_hor, src, F4X4, blend0_win);
        } else if (win_size < win_size_thresh[3]) {
            mix_scalar(b1_hor, src, F4X4, blend1_win);
            mix_scalar(b0_hor, src, F5X5, blend0_win);
        } else {
            mix_scalar(b0_hor, src, F5X5, blend0_win);
            for (int i = 0; i < PATCH_SIZE; i++) blend1_win[i] = 0;
        }

        int remain = win_size % 8;
        if (win_size < win_size_thresh[0]) {
            for (int i = 0; i < PATCH_SIZE; i++) final_patch[i] = blend1_win[i];
        } else if (win_size >= win_size_thresh[3]) {
            for (int i = 0; i < PATCH_SIZE; i++) final_patch[i] = blend0_win[i];
        } else {
            for (int i = 0; i < PATCH_SIZE; i++) {
                int val = round_div(blend0_win[i] * remain + blend1_win[i] * (8 - remain), 8);
                final_patch[i] = saturate_s11(val);
            }
        }
    }
};

class isp_csiir_top {
public:
    static const int ROW_DELAY = 2;
    static const int COL_DELAY = 2 * HORIZONTAL_TAP_STEP;

    Config cfg;
    lb_grad_pixel lb_grad_pixel0;
    lb_grad_pixel lb_grad_pixel1;
    lb_grad_pixel lb_grad_pixel2;
    lb_grad_pixel lb_grad_pixel3;
    lb_grad_pixel lb_grad_pixel4;
    lb_grad_value lb_grad_value0;
    lb_filt_pixel lb_filt_pixel0;
    lb_filt_pixel lb_filt_pixel1;
    lb_filt_pixel lb_filt_pixel2;
    stage_gradient stage_gradient0;
    stage_filter stage_filter0;

    isp_csiir_top(const Config& c)
        : cfg(c),
          stage_gradient0(lb_grad_pixel0, lb_grad_pixel1, lb_grad_pixel2, lb_grad_pixel3, lb_grad_pixel4, lb_grad_value0),
          stage_filter0(lb_grad_pixel0, lb_grad_pixel1, lb_grad_pixel2, lb_grad_pixel3, lb_grad_pixel4,
                        lb_filt_pixel0, lb_filt_pixel1, lb_filt_pixel2) {
        reset();
        stage_gradient0.init(cfg);
        stage_filter0.init(cfg);
    }

    void reset() {
        lb_grad_pixel0.reset();
        lb_grad_pixel1.reset();
        lb_grad_pixel2.reset();
        lb_grad_pixel3.reset();
        lb_grad_pixel4.reset();
        lb_grad_value0.reset();
        lb_filt_pixel0.reset();
        lb_filt_pixel1.reset();
        lb_filt_pixel2.reset();
    }

    void run(csiir_hls::stream_t<axis_pixel_t>& din_stream,
             csiir_hls::stream_t<axis_pixel_t>& dout_stream) {
        for (int scan_y = 0; scan_y < cfg.img_height + ROW_DELAY; scan_y++) {
            for (int scan_x = 0; scan_x < cfg.img_width + COL_DELAY; scan_x++) {
                if (scan_y < cfg.img_height && scan_x < cfg.img_width) {
                    axis_pixel_t din = din_stream.read();
                    lb_grad_pixel& wr_lb = select_grad_lb(scan_y);
                    if (scan_x == 0) {
                        wr_lb.attach_row(scan_y);
                    }
                    wr_lb.write(scan_x, din.data);
                }

                PixelCoord coord = {scan_x - COL_DELAY, scan_y - ROW_DELAY};
                if (!output_valid(coord)) {
                    continue;
                }

                if (coord.x == 0) {
                    seed_filter_row(coord.y);
                }

                GradientStageResult grad_state;
                FilterStageResult filt_state;
                stage_gradient0.run(coord, grad_state);
                stage_filter0.run(coord, grad_state, filt_state);
                select_filt_lb(coord.y).write(coord.x, filt_state.final_center);

                trace_stage4_input(coord, grad_state, filt_state);
                trace_feedback_commit(coord, filt_state, cfg.img_width);

                axis_pixel_t dout;
                dout.data = (coord.y < 2) ? read_orig_pixel(coord.y, coord.x) : filt_state.final_center;
                dout.user = (coord.x == 0 && coord.y == 0) ? 1 : 0;
                dout.last = (coord.x == cfg.img_width - 1 && coord.y == cfg.img_height - 1) ? 1 : 0;
                dout_stream.write(dout);
            }
        }
    }

private:
    bool output_valid(const PixelCoord& coord) const {
        return coord.x >= 0 && coord.x < cfg.img_width &&
               coord.y >= 0 && coord.y < cfg.img_height;
    }

    lb_grad_pixel& select_grad_lb(int row) {
        int slot = row % 5;
        if (slot == 0) return lb_grad_pixel0;
        if (slot == 1) return lb_grad_pixel1;
        if (slot == 2) return lb_grad_pixel2;
        if (slot == 3) return lb_grad_pixel3;
        return lb_grad_pixel4;
    }

    lb_filt_pixel& select_filt_lb(int row) {
        int slot = row % 3;
        if (slot == 0) return lb_filt_pixel0;
        if (slot == 1) return lb_filt_pixel1;
        return lb_filt_pixel2;
    }

    pixel_t read_orig_pixel(int row, int col) const {
        int y = clip<int>(row, 0, cfg.img_height - 1);
        int x = clip<int>(col, 0, cfg.img_width - 1);
        if (lb_grad_pixel0.row_id == y) return lb_grad_pixel0.read(x);
        if (lb_grad_pixel1.row_id == y) return lb_grad_pixel1.read(x);
        if (lb_grad_pixel2.row_id == y) return lb_grad_pixel2.read(x);
        if (lb_grad_pixel3.row_id == y) return lb_grad_pixel3.read(x);
        if (lb_grad_pixel4.row_id == y) return lb_grad_pixel4.read(x);
        return 0;
    }

    void seed_filter_row(int row) {
        lb_filt_pixel& wr_lb = select_filt_lb(row);
        wr_lb.attach_row(row);
        for (int x = 0; x < cfg.img_width; x++) {
            wr_lb.write(x, read_orig_pixel(row, x));
        }
    }
};

}  // namespace csiir_pipe

static inline void emit_output_row(
    csiir_hls::stream_t<axis_pixel_t>& dout_stream,
    const pixel_t emit_row[MAX_WIDTH],
    int row,
    int img_width,
    int img_height
) {
    for (int x = 0; x < img_width; x++) {
        axis_pixel_t dout;
        dout.data = emit_row[x];
        dout.user = (row == 0 && x == 0) ? 1 : 0;
        dout.last = (row == img_height - 1 && x == img_width - 1) ? 1 : 0;
        dout_stream.write(dout);
    }
}

static inline void trace_stage4_input(
    const PixelCoord& coord,
    const GradientStageResult& grad_state,
    const FilterStageResult& filt_state
) {
#ifndef __SYNTHESIS__
    if (csiir_debug::g_trace_store != nullptr) {
        csiir_debug::Stage4InputTraceEntry entry;
        entry.idx = (int)csiir_debug::g_trace_store->stage4_input_trace.size();
        entry.center_x = coord.x;
        entry.center_y = coord.y;
        entry.win_size = grad_state.win_size;
        entry.grad_h = (int)grad_state.grad_h;
        entry.grad_v = (int)grad_state.grad_v;
        entry.blend0 = filt_state.fusion.blend0;
        entry.blend1 = filt_state.fusion.blend1;
        entry.avg0_u = filt_state.dir_avg.avg0_u;
        entry.avg1_u = filt_state.dir_avg.avg1_u;
        for (int i = 0; i < PATCH_SIZE; i++) {
            entry.src_patch_u10[i] = (int)filt_state.stage4_patch_u10[i];
        }
        csiir_debug::g_trace_store->stage4_input_trace.push_back(entry);
    }
#else
    (void)coord;
    (void)grad_state;
    (void)filt_state;
#endif
}

static inline void trace_feedback_commit(
    const PixelCoord& coord,
    const FilterStageResult& filt_state,
    int img_width
) {
#ifndef __SYNTHESIS__
    if (csiir_debug::g_trace_store != nullptr) {
        csiir_debug::FeedbackCommitTraceEntry entry;
        entry.idx = (int)csiir_debug::g_trace_store->feedback_commit_trace.size();
        entry.center_x = coord.x;
        entry.center_y = coord.y;
        for (int col = 0; col < 5; col++) {
            entry.write_xs[col] = clip<int>(coord.x + (col - 2) * HORIZONTAL_TAP_STEP, 0, img_width - 1);
            for (int row = 0; row < 5; row++) {
                entry.columns_u10[col][row] = (int)s11_to_u10(filt_state.final_patch[row * 5 + col]);
            }
        }
        csiir_debug::g_trace_store->feedback_commit_trace.push_back(entry);
    }
#else
    (void)coord;
    (void)filt_state;
    (void)img_width;
#endif
}

void isp_csiir_top(
    csiir_hls::stream_t<axis_pixel_t> &din_stream,
    csiir_hls::stream_t<axis_pixel_t> &dout_stream,
    ISPCSIIR_Regs &regs
) {
    Config cfg;
    cfg.img_width = (int)regs.img_width;
    cfg.img_height = (int)regs.img_height;
    for (int i = 0; i < 4; i++) {
        cfg.win_size_thresh[i] = (int)regs.win_size_thresh[i];
        cfg.win_size_clip_y[i] = (int)regs.win_size_clip_y[i];
        cfg.win_size_clip_sft[i] = (int)regs.win_size_clip_sft[i];
        cfg.blending_ratio[i] = (int)regs.blending_ratio[i];
    }
    cfg.reg_edge_protect = (int)regs.edge_protect;
    csiir_pipe::isp_csiir_top dut(cfg);
    dut.run(din_stream, dout_stream);
}

#endif // ISP_CSIIR_HLS_TOP_HPP
