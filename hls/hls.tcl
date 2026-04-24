#===============================================================================
# Vivado HLS TCL Script for ISP-CSIIR HLS Synthesis
#===============================================================================

set project_name "isp_csiir_hls"
set top_function "isp_csiir_top"
set solution_name "solution_1"

# Device settings (customize for your target)
set device "xcu200-fsgd2104-2-e"

# Create project
hls::create_project $project_name $solution_name -part $device

# Add source files
hls::add_files "isp_csiir_hls_top.cpp" -cflags "-std=c++17 -O2"

# Set top function
hls::set_top $top_function

# Synthesis settings
hls::set Clock 5.0  ;# 200 MHz clock
hls::set Flow target

# Performance directives
# - UNROLL: Unroll loops for parallel processing
# - PIPELINE: Pipeline loops for throughput
# - ARRAY_PARTITION: Partition arrays for better memory access
# - INLINE: Inline small functions

# Run synthesis
hls::run

# Generate RTL
hls::export_design -format ip_catalog

puts "HLS synthesis complete!"
puts "Results in: $project_name/$solution_name/syn/report/"
