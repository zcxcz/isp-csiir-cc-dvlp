# ISP-CSIIR HLS Model Makefile
# Host testbench build plus Catapult entrypoints

CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -DCSIIR_HLS_BACKEND_CATAPULT=1 -I. -I./third_party/ac_types/include
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

# Clean all generated directories
distclean: clean
	rm -rf catapult_prj/

# Catapult synthesis (requires Catapult)
CATAPULT ?= catapult

catapult:
	cd $(HLS_DIR) && $(CATAPULT) -shell -file hls_catapult.tcl

catapult_ref:
	cd $(HLS_DIR) && $(CATAPULT) -shell -file catapult_ref.tcl

# Show help
help:
	@echo "ISP-CSIIR HLS Top Module Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all            - Compile host testbench with Catapult datatype backend"
	@echo "  run_tb         - Run host testbench"
	@echo "  clean          - Remove compiled files"
	@echo "  distclean      - Remove generated directories"
	@echo "  catapult       - Run Catapult synthesis flow"
	@echo "  catapult_ref   - Run Catapult reference flow"
	@echo ""
	@echo "For Catapult synthesis:"
	@echo "  make catapult"
