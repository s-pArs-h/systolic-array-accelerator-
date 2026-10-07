`default_nettype none

// One processing element of the output-stationary systolic array.
//
// Each PE owns one element of the result tile C. Every cycle a slice of the
// computation may pass through it: a value of A arrives from the left, a
// value of B from the top, and the PE accumulates a * b. Both values (and
// the tile flags, which travel with A) are passed on to the right and down
// one cycle later.
//
//   first_in  first k-slice of a tile: the sum restarts at a * b
//   last_in   last k-slice: one cycle later the finished sum is copied from
//             the accumulator into `res`
//
// Capturing from the accumulator REGISTER (rather than from the adder
// output in the same cycle) means nothing outside the multiply-accumulate
// needs the adder output, so synthesis can keep the accumulator inside the
// DSP slice (its P register). That halves the fabric logic per PE
// (Yosys, Artix-7: 66 -> 34 LUTs, 83 -> 51 flip-flops) for one extra cycle
// of latency.
//
// `res` is also one link of its column's drain chain: when `drain` is high,
// it takes the value of the PE below, so finished rows move up and leave the
// array at row 0. The controller guarantees a capture and a drain never
// happen in the same PE in the same cycle.
module sa_pe #(
    parameter DW   = 8,                  // signed operand width
    parameter ACCW = 32                  // accumulator width
) (
    input  wire                     clk,
    input  wire                     rst_n,

    input  wire                     vld_in,
    input  wire                     first_in,
    input  wire                     last_in,
    input  wire signed [DW-1:0]     a_in,
    input  wire signed [DW-1:0]     b_in,

    output logic                    vld_out,
    output logic                    first_out,
    output logic                    last_out,
    output logic signed [DW-1:0]    a_out,
    output logic signed [DW-1:0]    b_out,

    input  wire                     drain,
    input  wire signed [ACCW-1:0]   res_below,
    output logic signed [ACCW-1:0]  res
);
    logic signed [ACCW-1:0] acc;

    wire signed [2*DW-1:0] prod     = a_in * b_in;
    wire signed [ACCW-1:0] prod_ext = {{(ACCW-2*DW){prod[2*DW-1]}}, prod};
    wire signed [ACCW-1:0] sum      = (first_in ? {ACCW{1'b0}} : acc) + prod_ext;

    logic cap;                           // the accumulator holds a finished sum

    // control: valid, tile flags and capture (reset)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            vld_out   <= 1'b0;
            first_out <= 1'b0;
            last_out  <= 1'b0;
            cap       <= 1'b0;
        end else begin
            vld_out   <= vld_in;
            first_out <= vld_in && first_in;
            last_out  <= vld_in && last_in;
            cap       <= vld_in && last_in;
        end
    end

    // datapath: only loads when a slice is passing (no reset needed)
    always_ff @(posedge clk) begin
        if (vld_in) begin
            a_out <= a_in;
            b_out <= b_in;
            acc   <= sum;
        end
        if (cap)        res <= acc;
        else if (drain) res <= res_below;
    end

`ifdef FORMAL
    // The controller must never let a capture and a drain shift meet here.
    always @(*) if (rst_n) assert(!(cap && drain));
`endif

endmodule

`default_nettype wire
