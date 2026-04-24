//==============================================================================
// ISP-CSIIR Register Definition
//==============================================================================
// All registers with HLS-compatible bit-accurate types
// Used by both HLS top and testbench
//==============================================================================

#ifndef ISP_CSIIR_REGS_HPP
#define ISP_CSIIR_REGS_HPP

#include <ap_fixed.h>
#include <ap_int.h>

//==============================================================================
// Register Group Struct
//==============================================================================
struct ISPCSIIR_Regs {
    // Image dimensions
    ap_uint<16> img_width;
    ap_uint<16> img_height;

    // Window size thresholds [4]
    ap_uint<8> win_size_thresh[4];

    // Window size LUT outputs [4]
    ap_uint<8> win_size_clip_y[4];

    // Window size LUT x-node shifts [4]
    ap_uint<4> win_size_clip_sft[4];

    // Blending ratios [4]
    ap_uint<8> blending_ratio[4];

    // Edge protection
    ap_uint<8> edge_protect;

    // Initialize with default values
    void reset() {
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
        edge_protect = 32;
    }
};

#endif // ISP_CSIIR_REGS_HPP
