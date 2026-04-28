#===============================================================================
# Catapult HLS TCL Script for ISP-CSIIR
#===============================================================================
# Usage:
#   catapult -shell -file hls/hls_catapult.tcl
#
# Intent:
#   - Synthesize only the DUT in `isp_csiir_hls_top.cpp`
#   - Force the Catapult backend (`ac_int` / `ac_channel`)
#   - Force synthesis mode so debug-only STL code stays excluded
#   - Centralize HLS implementation constraints in Tcl instead of C++ pragmas
#   - Provide explicit search paths for local headers and AC datatypes
#===============================================================================

#--------------------------
# User configuration
#--------------------------
set script_dir       [file normalize [file dirname [info script]]]
set project_root     [file normalize [file join $script_dir ..]]
set project_name     "isp_csiir_catapult"
set solution_name    "soln_200m"
set top_name         "isp_csiir_top"

set clk_name         "clk"
set rst_name         "rst_n"
set clk_period_ns    5.000
set clk_uncertainty  0.200
set reset_active     "low"
set reset_async      "false"

set src_file         [file join $script_dir "isp_csiir_hls_top.cpp"]
set tb_file          [file join $script_dir "isp_csiir_hls_top_tb.cpp"]
set output_dir       [file join $project_root "catapult_prj"]
set rtl_lang         "verilog"

# This DUT currently builds cleanly with C++17 in host testbench and Catapult
# flow. Keeping one language level avoids drift between simulation and HLS.
set cpp_std          "-std=c++17"

# Required compile definitions:
#   - CSIIR_HLS_BACKEND_CATAPULT: selects ac_int/ac_channel backend
#   - __SYNTHESIS__: strips debug-only std::vector tracing blocks
set compile_defs     "-DCSIIR_HLS_BACKEND_CATAPULT=1 -D__SYNTHESIS__=1"

# Search paths:
#   - $script_dir: local headers and relative include root
#   - ac_types/include: direct access for Catapult's AC datatypes
#   - hls_lib: harmless extra path in case future code pulls shared HLS headers
set include_dirs [list \
    $script_dir \
    [file join $script_dir "third_party" "ac_types" "include"] \
    [file join $script_dir "hls_lib"] \
]

set include_flags ""
foreach inc $include_dirs {
    append include_flags " -I$inc"
}
set compile_flags "$cpp_std $compile_defs$include_flags"

# Optional technology library hook
set techlib_name ""

#--------------------------
# Project / solution setup
#--------------------------
file mkdir $output_dir
cd $output_dir

options set ProjectInit ProjectNamePrefix $project_name
project new
solution new $solution_name

# Keep language / RTL settings explicit.
solution options set /Input/CppStandard c++17
solution options set /Output/OutputLanguage $rtl_lang
solution options set /Output/OutputDirectory [file join $output_dir "${project_name}_${solution_name}_rtl"]

#--------------------------
# Source files
#--------------------------
# Add only the DUT for synthesis. The TB pulls in filesystem and host-side I/O,
# which is useful for C-sim but should not be part of the synthesis compile.
solution file add $src_file -type C++ -cflags $compile_flags

# Keep the TB visible but excluded so it can be enabled later for SCVerify/C-sim
# without rebuilding the script structure.
if {[file exists $tb_file]} {
    solution file add $tb_file -type C++ -exclude true -cflags $compile_flags
}

# Make sure Catapult knows the design entry point explicitly.
solution design set $top_name

#--------------------------
# Clock / reset
#--------------------------
directive set /$top_name -CLOCKS "{$clk_name {-CLOCK_PERIOD $clk_period_ns -CLOCK_EDGE rising -CLOCK_HIGH_TIME [expr {$clk_period_ns / 2.0}] -CLOCK_OFFSET 0.0 -CLOCK_UNCERTAINTY $clk_uncertainty}}"
directive set /$top_name -RESETS "{$rst_name {-RESET_ACTIVE $reset_active -RESET_ASYNC $reset_async}}"

#--------------------------
# Interface notes
#--------------------------
# The C top uses:
#   - ac_channel<axis_pixel_t>& din_stream
#   - ac_channel<axis_pixel_t>& dout_stream
#   - ISPCSIIR_Regs& regs
#
# Catapult can synthesize these directly as channel / struct ports. Do not force
# guessed bus mapping syntax here; keep packaging decisions version-local.

#--------------------------
# Architecture directives
#--------------------------
# Keep all implementation constraints here rather than inside C++ source.
# The current source is intentionally free of tool-specific interface/unroll
# pragmas. Add Catapult directives below as timing/area data justifies them.
#
# Examples to enable when needed:
# directive set /$top_name -PIPELINE_INIT_INTERVAL 1
# directive set /$top_name/grad_pxl_lb:rsc -MAP_TO_MEMORY {2P}
# directive set /$top_name/filt_pxl_lb:rsc -MAP_TO_MEMORY {2P}
# directive set /$top_name/process_row -UNROLL no

#--------------------------
# Optional technology library
#--------------------------
if {$techlib_name ne ""} {
    solution library add $techlib_name
}

#--------------------------
# Main flow
#--------------------------
go analyze
go compile
go libraries
go assembly
go architect
go allocate
go extract

report design
project save

puts ""
puts "Catapult flow completed."
puts "ProjectDir : $output_dir"
puts "Solution   : $solution_name"
puts "Top        : $top_name"
puts "CppFlags   : $compile_flags"
puts ""
