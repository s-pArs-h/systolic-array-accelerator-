# INT8 Systolic-Array GEMM Accelerator

A parameterised N x N output-stationary systolic array for INT8 matrix
multiplication with INT32 accumulation, in SystemVerilog. It streams one
k-slice per cycle, overlaps each tile's result drain with the next tile's
compute, and sustains **N² multiply-accumulates per cycle** (95-96% measured
utilisation over 20 back-to-back tiles, with the only overhead being the
initial fill and final drain).

Verified with a constrained-random cocotb testbench against NumPy, functional
coverage, an unbounded formal proof of the controller's safety property, and
bounded formal data-integrity checks; lint-clean under Verilator -Wall.
A PPA iteration moved the accumulator into the DSP slice: **-48% LUTs and
-36% flip-flops** at the same throughput. Design notes and trade-offs:
[docs/DESIGN.md](docs/DESIGN.md).

## Architecture

```
            s_b: B[k][0..N-1] (row k of the B tile)
                 |        |        |        |
              skew 1   skew 2   skew 3   skew 4        (column j delayed j+1)
                 v        v        v        v
 s_a: A[0][k] -skew 1-> [PE00] -> [PE01] -> [PE02] -> [PE03]
      A[1][k] -skew 2-> [PE10] -> [PE11] -> [PE12] -> [PE13]     A moves right,
      A[2][k] -skew 3-> [PE20] -> [PE21] -> [PE22] -> [PE23]     B moves down,
      A[3][k] -skew 4-> [PE30] -> [PE31] -> [PE32] -> [PE33]     one PE per cycle
                          ^ drain   ^ drain  ^ drain   ^ drain
                          |         |        |         |
              results move up each column; row 0 leaves on m_c (N x INT32)
```

* **PE**: INT8 x INT8 multiply, INT32 accumulate (one DSP48E1 per PE).
  On the tile's last slice the finished sum is copied into the PE's result
  register, which is also a link of its column's drain chain.
* **Skew**: row i of A and column j of B are delayed so that A[i][k] and
  B[k][j] meet in PE(i, j) in the same cycle. Bubbles (cycles without input)
  flow through the array consistently, so no global stall is needed.
* **Controller**: counts down until every PE has captured, then drains the
  tile row by row (row 0 first) under valid/ready back-pressure. Only a
  tile's last slice is ever held back, when its captures would collide with
  a drain still in progress.

## Interface

| Port | Dir | Width | Description |
|---|---|---|---|
| `s_valid`, `s_ready` | in / out | 1 | k-slice handshake |
| `s_a` | in | N x 8 | column k of the A tile, lane i = row i |
| `s_b` | in | N x 8 | row k of the B tile, lane j = column j |
| `s_last` | in | 1 | marks the last slice (k = K-1) of a tile |
| `m_valid`, `m_ready` | out / in | 1 | result-row handshake |
| `m_c` | out | N x 32 | one row of the C tile, rows 0..N-1 in order |
| `m_last` | out | 1 | marks row N-1 |
| `busy` | out | 1 | slices or results still in flight |

K can be anything from 1 to 2^16 (INT32 cannot overflow: 128 x 128 x 2^16 <
2^31). Larger matrices are computed as a sequence of N x N output tiles;
`tb/model.py` has the tiler used by the GEMM test.

## Verification

### Simulation: cocotb + Icarus Verilog vs. NumPy (`make sim`)

| Test | Checks |
|---|---|
| `test_identity_and_zero` | A x I, I x A, A x 0 |
| `test_random_tiles` | 40 tiles, random K from 1 to 4N, random input gaps (75 %) and output back-pressure (60 %) |
| `test_extreme_accumulation` | (-128) x (-128) and 127 x (-128) summed over K = 4096; extreme-value mix |
| `test_small_k_back_to_back` | K = 1 .. 3N back to back: the last-slice stall is exercised |
| `test_full_rate` | 20 tiles at K = 3N+1: zero stalls, measured MAC utilisation |
| `test_gemm` | a 3N x 37 by 37 x 2N GEMM computed tile by tile |
| `test_coverage_closure` | fails unless every coverage bin was hit |

Runs for N = 2, 4 and 8; every result element is compared with NumPy int64.
Mutation check: zero-extending the product instead of sign-extending it fails every test.

### Formal: SymbiYosys + Yices (`make formal`)

| Check | Kind | Configuration | Result |
|---|---|---|---|
| No PE ever captures a result in the same cycle its drain chain shifts; controller counters stay in range | unbounded proof (k-induction) | N = 4, INT8 | proven |
| A stalled output beat holds its data | unbounded proof | N = 4, INT8 | proven |
| `m_last` on every N-th row; at most one finished tile waiting | bounded, 16 cycles | N = 2 and N = 3 | pass |
| The first tile's results equal sum_k A[i][k] B[k][j] for any inputs, gaps and stalls | bounded, 16 cycles | N = 2 and N = 3, 2-bit operands | pass |
| Cover: a held last slice, a stalled output, two complete tiles | cover | N = 2 | reached |

## Implementation

### Area sweep: Yosys `synth_xilinx` (Artix-7), `make synth`

| N | PEs | LUTs | FFs | DSP48E1 | MACs / cycle | LUTs per PE | stall-free K |
|---|---|---|---|---|---|---|---|
| 2 | 4 | 156 | 198 | 4 | 4 | 39 | >= 7 |
| 4 | 16 | 562 | 854 | 16 | 16 | 35 | >= 13 |
| 8 | 64 | 2200 | 3556 | 64 | 64 | 34 | >= 25 |
| 16 | 256 | 8731 | 14526 | 256 | 256 | 34 | >= 49 |

Area grows linearly with the number of PEs (34 LUTs and one DSP each), and
throughput grows with it. At N = 16 the array needs 256 DSP48E1 slices, more
than an Artix-7 100T has (240), so N = 8 is the largest power-of-two size
for the Nexys A7.

### PPA iteration: accumulator into the DSP

The first version captured each PE's result straight from the adder output.
Because the adder output was then needed outside the multiply-accumulate,
the accumulator could not live in the DSP slice's internal register.
Capturing from the accumulator register one cycle later instead:

| N = 16 | LUTs | FFs | DSP48E1 | Cost |
|---|---|---|---|---|
| capture from adder | 16,923 | 22,702 | 256 | |
| capture from accumulator | **8,731** (-48%) | **14,526** (-36%) | 256 | +1 cycle latency; stall-free K from 3N to 3N+1 |

### Vivado

Vivado 2025.1, xc7a100tcsg324-1, N = 8, out of context (the array alone,
no pins) with a 4 ns clock target; maximum frequency estimated as
1 / (period - worst slack):

| N = 8 | Vivado default | `use_dsp` on the PE |
|---|---|---|
| LUTs | 7,436 | **1,175** (-84%) |
| Flip-flops | 5,753 | 3,455 |
| DSP48E1 | 0 | 128 |
| Fmax (estimate) | 128 MHz | 143 MHz |

By default Vivado builds an 8 x 8 multiplier from LUTs because it is small,
and the accumulator then stays in fabric too. The `use_dsp` attribute on
`sa_pe` moves the multiply-accumulate into DSP slices. Vivado still uses
two DSPs per PE (multiplier and accumulator separately), so the next step
is a PE written to Xilinx's multiply-accumulate template, with registered
inputs and product, to fit one DSP per PE and raise the clock, at the cost
of two more cycles of latency.

An OpenLane Sky130 run (`openlane/config.json`) is next.

## Running it

```
make lint      # Verilator -Wall, N = 2, 4, 8, 16
make sim       # cocotb regression, N = 2, 4, 8 (make -C tb N=8 SEED=7)
make formal    # SymbiYosys
make synth     # Yosys area sweep
```

Requires Icarus Verilog 12, Verilator 5, Yosys, SymbiYosys with Yices, and
Python 3 with `cocotb >= 2.0` and NumPy. CI runs all of it on every push.

## Repository layout

```
rtl/        systolic_array.sv (array, skew, controller), sa_pe.sv, sa_skew.sv
tb/         test_systolic.py (cocotb), model.py (NumPy reference + tiler), coverage.py
formal/     sa_formal.sv, sa.sby
synth/      sweep.py (Yosys area sweep)
openlane/   config.json (Sky130)
docs/       DESIGN.md
```
