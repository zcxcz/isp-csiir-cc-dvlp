# ISP-CSIIR HLS Model Makefile
# Supports both g++ compilation with HLS library and Vivado HLS synthesis

CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -I./hls_lib
HLS_DIR = .

# Source files
HLS_SRC = $(HLS_DIR)/isp_csiir_hls_top.cpp
HLS_TB = $(HLS_DIR)/isp_csiir_hls_top_tb.cpp

# Output executables
TB_EXE = $(HLS_DIR)/hls_top_tb

# Default target: compile testbench
all: $(TB_EXE)

# Compile testbench with HLS library
$(TB_EXE): $(HLS_TB) $(HLS_SRC)
	$(CXX) $(CXXFLAGS) -o $@ $(HLS_TB)

# Run testbench
run_tb: $(TB_EXE)
	./$(TB_EXE)

# Clean
clean:
	rm -f $(HLS_DIR)/hls_top_tb $(HLS_DIR)/*.out

# Clean all (including Vivado HLS)
distclean: clean
	rm -rf isp_csiir_hls/ solution_1/

# Vivado HLS synthesis (requires Vivado HLS)
VIVADO_HLS ?= vivado_hls

hls_synth:
	cd $(HLS_DIR) && $(VIVADO_HLS) -f hls.tcl

hls_csim:
	cd $(HLS_DIR) && $(VIVADO_HLS) -f hls.tcl -csim

# Show help
help:
	@echo "ISP-CSIIR HLS Top Module Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all            - Compile testbench (default)"
	@echo "  run_tb         - Run testbench"
	@echo "  clean          - Remove compiled files"
	@echo "  distclean      - Remove all generated files"
	@echo "  hls_synth      - Run Vivado HLS synthesis"
	@echo "  hls_csim       - Run Vivado HLS C simulation"
	@echo ""
	@echo "For full HLS synthesis with Vivado HLS:"
	@echo "  make hls_synth"
