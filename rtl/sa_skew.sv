`default_nettype none

// Delay line used to skew the array inputs: lane i of A (and of B) must
// enter the array i cycles after lane 0, so that A[i][k] and B[k][j] meet
// in PE(i, j) on the same cycle. Data registers only load when the stage
// before them holds a valid value.
module sa_skew #(
    parameter DEPTH = 1,                 // >= 1
    parameter W     = 8
) (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         vld_in,
    input  wire [W-1:0] d_in,
    output wire         vld_out,
    output wire [W-1:0] d_out
);
    logic [DEPTH-1:0] v;
    logic [W-1:0]     d [0:DEPTH-1];

    generate
        if (DEPTH == 1) begin : g_one
            always_ff @(posedge clk or negedge rst_n)
                if (!rst_n) v <= 1'b0;
                else        v <= vld_in;
        end else begin : g_many
            always_ff @(posedge clk or negedge rst_n)
                if (!rst_n) v <= '0;
                else        v <= {v[DEPTH-2:0], vld_in};
        end
    endgenerate

    always_ff @(posedge clk)
        if (vld_in) d[0] <= d_in;

    genvar s;
    generate
        for (s = 1; s < DEPTH; s = s + 1) begin : g_stage
            always_ff @(posedge clk)
                if (v[s-1]) d[s] <= d[s-1];
        end
    endgenerate

    assign vld_out = v[DEPTH-1];
    assign d_out   = d[DEPTH-1];

endmodule

`default_nettype wire
