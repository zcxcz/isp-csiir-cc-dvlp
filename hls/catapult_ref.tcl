#===============================================================================
# Catapult HLS Reference Script for ISP-CSIIR
#===============================================================================
# Usage example:
#   catapult -shell -file hls/catapult_ref.tcl
#
# Scope:
#   - Synthesizes the C++ core with the AC backend enabled
#   - Uses ac_channel/ac_int through CSIIR_HLS_BACKEND_CATAPULT
#   - Leaves technology library and interface packaging as user-configurable
#
# Notes:
#   - Command details may vary slightly across Catapult versions.
#   - This script is intended as a project-local starting point, not a locked flow.
#===============================================================================

#--------------------------
# User configuration
#--------------------------
set script_dir [file normalize [file dirname [info script]]]
set project_root [file normalize [file join $script_dir ..]]
set project_name "isp_csiir_catapult"
set solution_name "core_200m"
set top_name "isp_csiir_top"

# Clock / reset
set clk_name "clk"
set rst_name "rst_n"
set clk_period_ns 5.000
set clk_uncertainty_ns 0.200
set reset_active "low"
set reset_async "false"

# Source setup
set src_file [file join $script_dir "isp_csiir_hls_top.cpp"]
set include_dir $script_dir
set cpp_std "-std=c++17"
set compile_defs "-DCSIIR_HLS_BACKEND_CATAPULT=1"
set compile_flags "$cpp_std $compile_defs -I$include_dir"

# RTL / report setup
set rtl_lang "verilog"
set output_dir [file join $project_root "catapult_prj"]

# Technology selection
# Replace these with your actual Catapult library / technology target.
# Common practice is to maintain one block for ASIC and one for FPGA.
set target_kind "generic"
set techlib_name ""

# Example placeholders:
# set target_kind "asic"
# set techlib_name "nangate45_typ"
#
# set target_kind "fpga"
# set techlib_name "xilinx_ultrascale_plus"

#--------------------------
# Project setup
#--------------------------
file mkdir $output_dir
project new $project_name
project saveas [file join $output_dir $project_name]

solution new -state initial $solution_name
solution options set /Input/CppStandard c++17

#--------------------------
# Source files
#--------------------------
solution file add $src_file -type C++ -cflags $compile_flags

# If you later want SCVerify / C simulation in Catapult, add the TB explicitly:
# set tb_file [file join $script_dir "isp_csiir_hls_top_tb.cpp"]
# solution file add $tb_file -type C++ -exclude true -cflags $compile_flags

#--------------------------
# Top / clock / reset
#--------------------------
directive set /$top_name -CLOCKS "{$clk_name {-CLOCK_PERIOD $clk_period_ns -CLOCK_EDGE rising -CLOCK_HIGH_TIME [expr {$clk_period_ns / 2.0}] -CLOCK_OFFSET 0.0 -CLOCK_UNCERTAINTY $clk_uncertainty_ns}}"
directive set /$top_name -RESETS "{$rst_name {-RESET_ACTIVE $reset_active -RESET_ASYNC $reset_async}}"

#--------------------------
# Interface intent
#--------------------------
# Current C model intent under Catapult:
#   din_stream  : streaming input  (ac_channel<axis_pixel_t>)
#   dout_stream : streaming output (ac_channel<axis_pixel_t>)
#   regs        : config struct
#
# For pure core synthesis, let Catapult keep channel semantics internally.
# If your downstream flow requires AXI-Stream / AXI-Lite packaging, it is safer
# to add a thin RTL wrapper after synthesis rather than force bus protocols here.
#
# If your Catapult version supports explicit interface directives for these ports,
# add them below. Keep them version-local instead of hardcoding guessed syntax.
#
# Example skeleton only:
# directive set /$top_name/din_stream  -INTERFACE some_stream_mapping
# directive set /$top_name/dout_stream -INTERFACE some_stream_mapping
# directive set /$top_name/regs        -INTERFACE some_config_mapping

#--------------------------
# Technology / library
#--------------------------
if {$techlib_name ne ""} {
  solution library add $techlib_name
}

#--------------------------
# Optional architecture directives
#--------------------------
# Start conservative. The source already contains the key local-array partition
# and loop-unroll intent via abstraction macros on the Vivado side. Under
# Catapult, it is usually better to add further scheduling/binding directives
# after the first compile based on the reports.
#
# Examples you may enable after the first pass:
# directive set /$top_name -PIPELINE_INIT_INTERVAL 1
# directive set /$top_name/orig_rows:rsc -MAP_TO_MEMORY {2P}
# directive set /$top_name/filt_rows:rsc -MAP_TO_MEMORY {2P}

#--------------------------
# Main flow
#--------------------------
go analyze
go compile

# Library binding and scheduling
go libraries
go assembly
go architect
go allocate

# RTL generation
go extract

#--------------------------
# Output controls
#--------------------------
solution options set /Output/OutputLanguage $rtl_lang
solution options set /Output/OutputDirectory [file join $output_dir "${project_name}_${solution_name}_rtl"]

report design
project save

puts ""
puts "Catapult reference flow completed."
puts "Project   : [file join $output_dir $project_name]"
puts "Solution  : $solution_name"
puts "Top       : $top_name"
puts "Clock(ns) : $clk_period_ns"
puts ""
