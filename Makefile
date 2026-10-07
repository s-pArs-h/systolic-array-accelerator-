# INT8 systolic-array GEMM accelerator: top-level flows
#
#   make lint      Verilator -Wall, N = 2, 4, 8, 16 (with and without FORMAL)
#   make sim       cocotb regression for N = 2, 4, 8
#   make formal    SymbiYosys: unbounded proof + bounded data checks + covers
#   make synth     Yosys area sweep for Xilinx 7-series, N = 2, 4, 8, 16
#   make all       everything above
#
# Tools: iverilog, verilator, yosys, sby + yices, python3 with cocotb >= 2.0 and numpy

RTL = rtl/sa_pe.sv rtl/sa_skew.sv rtl/systolic_array.sv

.PHONY: all lint sim formal synth clean

all: lint sim formal synth

lint:
	@for n in 2 4 8 16; do \
	  verilator --lint-only -Wall -GN=$$n $(RTL) --top-module systolic_array || exit 1; \
	  verilator --lint-only -Wall -DFORMAL -GN=$$n $(RTL) --top-module systolic_array || exit 1; \
	done
	@echo "Verilator -Wall: clean for N = 2, 4, 8, 16"

sim:
	@for n in 2 4 8; do $(MAKE) -C tb N=$$n || exit 1; done

formal:
	cd formal && sby -f sa.sby

synth:
	python3 synth/sweep.py 2 4 8 16

clean:
	rm -rf tb/sim_build_* tb/results.xml tb/__pycache__ tb/coverage_N*.txt formal/sa_*/ synth/reports
