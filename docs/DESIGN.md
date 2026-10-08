# Design notes

## 1. What it computes

C = A x B for INT8 A (M x K) and B (K x N), INT32 C. The array computes one
N x N output tile at a time; a larger C is a sequence of tiles, each needing
row-panel `A[r:r+N, :]` and column-panel `B[:, c:c+N]`. Each tile streams K
"k-slices": column k of the A panel and row k of the B panel.

## 2. Why output-stationary

In an output-stationary (OS) array each PE owns one element of C and keeps
its partial sum locally; A and B flow through.

| | Output-stationary (this design) | Weight-stationary (e.g. TPU v1) |
|---|---|---|
| What stays in the PE | one C element (INT32) | one B element (INT8) |
| What moves | A right, B down | A right, partial sums down |
| Inner dimension K | any length, streamed | limited by array height per pass |
| Partial sums leave the array | once per tile | every cycle |
| Reloading | none (results drain) | weights preloaded per tile |

OS suits a stream with arbitrary K: partial sums never leave the PE until
they are final, so no wide partial-sum traffic and no accumulation buffer
outside the array. The price is a result drain per tile, which this design
overlaps with the next tile's compute.

## 3. Skew and bubbles

A[i][k] must meet B[k][j] in PE(i, j). A takes j hops to reach column j and
B takes i hops to reach row i, so row i of A is delayed by i and column j of
B by j (plus one register each, so the array inputs are registered). Then
slice k reaches PE(i, j) at cycle k + i + j + 1 along both paths.

If the input has no valid slice in a cycle, that "bubble" enters both skews
at the same time and stays aligned through the array, so the array never
needs a global stall: every PE simply skips cycles where its valid bit is
low. Data registers only load when valid is high (lower switching power, and
the enables map onto flip-flop and DSP clock enables).

## 4. Results and the drain

On the last slice a PE copies its finished sum into `res`. The `res`
registers of each column form a shift chain toward row 0: on every accepted
output beat, every column shifts up by one, so the output is row 0, then
row 1, and so on, N beats per tile.

Captures happen on a diagonal wavefront (PE(i, j) one cycle after
PE(i, j-1) and PE(i-1, j)), so the drain may only start once PE(N-1, N-1)
has captured: `CAP_LAT = 2N - 1` cycles after the last slice is accepted.

## 5. Overlapping tiles: the one stall rule

The next tile's slices can enter as soon as the previous tile's last slice
has: they only touch the accumulators, which the previous tile no longer
needs once it has been captured. The danger is the next tile's **captures**:
if PE(i, j) captured while its column was still draining, it would either
overwrite a value not yet sent or be shifted to the wrong row.

Rule: **a last slice is accepted only when no drain is pending or in
progress** (`s_ready = !(s_last && drain_busy)`). Nothing else ever stalls.
The earliest the next capture can then happen is after the drain completes.

With an always-ready output, the drain of a tile ends 3N cycles after its
last slice was accepted, so a tile with K >= 3N + 1 never sees the stall:
the array runs at full rate. For smaller K the last slice waits a few
cycles. Accepting it one cycle earlier would be possible only by making
`s_ready` depend combinationally on `m_ready`; the simpler rule was chosen.

This rule is the controller's key safety property. Each PE asserts
`!(capture && drain)` under `` `ifdef FORMAL ``, and k-induction proves it for
all time at N = 4 with INT8 data.

## 6. Widths

INT8 x INT8 fits in 16 bits (largest magnitude (-128)(-128) = 16384). The
INT32 accumulator holds up to 2^31 / 2^14 = 131,072 such products, so any K
up to 2^16 is safe. `test_extreme_accumulation` drives (-128)(-128) for
K = 4096 (result 67,108,864).

## 7. PPA iteration: putting the accumulator in the DSP

A DSP48E1 contains a multiplier, an adder and an output register (P) that
can act as the accumulator. The first version captured `res <= sum`, the
adder output, in the same cycle as the last accumulation. Because the sum was
needed outside the multiply-accumulate, synthesis kept the accumulator and
the restart multiplexer in fabric: 66 LUTs and 83 flip-flops per PE.

Capturing one cycle later from the accumulator register (`res <= acc`)
removes that need, and the accumulator moves into the DSP: 34 LUTs and 51
flip-flops per PE (-48% LUTs, -39% flip-flops per PE; -48% / -36% for the
whole N = 16 array). The cost is one cycle of capture latency, which moves
the stall-free threshold from K >= 3N to K >= 3N + 1. Throughput for
realistic K is unchanged.

What remains per PE: the INT32 result register and its 2:1 multiplexer
(capture or shift), the A and B pass-through registers, and the flags.
Further options:

* drop the separate result register by draining directly out of the
  accumulators, at the cost of stalling compute during the drain;
* narrow the accumulator if K is known to be small;
* time-multiplex one output port across columns instead of N x 32 bits.

## 8. Verification strategy

| Layer | What it shows |
|---|---|
| Verilator `-Wall`, N = 2 / 4 / 8 / 16 | no width, latch or unused-signal issues |
| cocotb vs. NumPy (int64), N = 2 / 4 / 8 | every element of every tile is exact, under random input gaps and output back-pressure, for K from 1 to 4096, extreme values, and full GEMMs tiled by `model.gemm_tiles` |
| Functional coverage | K = 1, the last-slice stall, input gaps, output stalls, -128 and 127 operands, results beyond 2^24, negative results, back-to-back tiles, full rate, multi-tile GEMM, identity, zero |
| Formal, unbounded (k-induction) | the capture/drain exclusion in every PE and the controller invariants, at the real size and width |
| Formal, bounded | output protocol and end-to-end data integrity of the first tile, for any input and back-pressure sequence |

As in the K-means project, proving multipliers equal is very hard for SAT
solvers, so the end-to-end data check uses 2-bit operands; full-width
arithmetic is covered by simulation, including the extremes. The
reference model in the formal harness computes products at their exact
width: a 32-bit multiply there made the solver stall even at tiny sizes.
