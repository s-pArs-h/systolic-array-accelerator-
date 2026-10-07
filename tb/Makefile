# cocotb testbench for systolic_array (Icarus Verilog)
#   make            N = 4
#   make N=8 SEED=7

SIM          ?= icarus
TOPLEVEL_LANG = verilog
N            ?= 4
SEED         ?= 1

RTL = $(abspath ../rtl)
VERILOG_SOURCES     = $(RTL)/sa_pe.sv $(RTL)/sa_skew.sv $(RTL)/systolic_array.sv
COCOTB_TOPLEVEL     = systolic_array
COCOTB_TEST_MODULES = test_systolic
SIM_BUILD           = sim_build_N$(N)
COMPILE_ARGS       += -Psystolic_array.N=$(N)

export SA_N = $(N)
export SEED

include $(shell cocotb-config --makefiles)/Makefile.sim
