`default_nettype none

// N x N output-stationary systolic array for INT8 matrix multiplication.
//
// Computes one N x N tile of C = A * B, with K (the inner dimension)
// streamed in one slice per cycle:
//
//   slice k:  s_a = A[0..N-1][k]   (column k of the A tile, lane i = row i)
//             s_b = B[k][0..N-1]   (row k of the B tile,    lane j = column j)
//             s_last marks k = K-1
//
// A values flow right through the array and B values flow down; skew delay
// lines make A[i][k] and B[k][j] meet in PE(i, j). After the last slice the
// finished tile leaves through the result drain chains, row 0 first, one
// row of N INT32 values per m_* beat, with m_last on row N-1.
//
// Tiles stream back to back: the next tile's slices enter while the
// previous tile is still being drained. Only a tile's LAST slice can be held
// back (s_ready low), when its results would otherwise be captured while
// the drain chains are still busy. With K >= 3N + 1 and an always-ready
// output, the array never stalls: N*N multiply-accumulates every cycle.
module systolic_array #(
    parameter N    = 4,
    parameter DW   = 8,
    parameter ACCW = 32
) (
    input  wire               clk,
    input  wire               rst_n,

    input  wire               s_valid,
    output wire               s_ready,
    input  wire               s_last,
    input  wire [N*DW-1:0]    s_a,
    input  wire [N*DW-1:0]    s_b,

    output wire               m_valid,
    input  wire               m_ready,
    output wire               m_last,
    output wire [N*ACCW-1:0]  m_c,

    output wire               busy
);
    // ------------------------------------------------------------------
    // Control
    // ------------------------------------------------------------------
    localparam CAP_LAT = 2*N - 1;          // cycles from accepting a last slice to the
                                           // cycle before PE(N-1, N-1) captures it
    localparam CW = $clog2(CAP_LAT + 1) + 1;
    localparam RW = $clog2(N + 1);

    logic          first_q;                // the next accepted slice starts a tile
    logic          cap_pending;            // last slice accepted, captures in progress
    logic [CW-1:0] cap_cnt;
    logic [RW-1:0] rows_left;              // rows of the finished tile still to send

    wire drain_busy = cap_pending || (rows_left != '0);
    assign s_ready  = !(s_last && drain_busy);
    wire in_fire    = s_valid && s_ready;

    assign m_valid  = (rows_left != '0);
    assign m_last   = (rows_left == RW'(1));
    wire out_fire   = m_valid && m_ready;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            first_q     <= 1'b1;
            cap_pending <= 1'b0;
            cap_cnt     <= '0;
            rows_left   <= '0;
        end else begin
            if (in_fire) first_q <= s_last;

            if (in_fire && s_last) begin
                cap_pending <= 1'b1;
                cap_cnt     <= CW'(CAP_LAT);
            end else if (cap_pending) begin
                if (cap_cnt == '0) begin
                    cap_pending <= 1'b0;
                    rows_left   <= RW'(N);
                end else begin
                    cap_cnt <= cap_cnt - 1'b1;
                end
            end

            if (out_fire) rows_left <= rows_left - 1'b1;
        end
    end

    // track whether any slice is still travelling through the array
    logic [2*N:0] in_flight;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) in_flight <= '0;
        else        in_flight <= {in_flight[2*N-1:0], in_fire};
    end
    assign busy = drain_busy || (in_flight != '0);

    // ------------------------------------------------------------------
    // Array wiring (flat buses)
    //   horizontal signals entering PE(i, j): position (i, j), j = 0..N
    //   vertical   signals entering PE(i, j): position (i, j), i = 0..N
    // ------------------------------------------------------------------
    wire [N*(N+1)*DW-1:0] a_h;
    wire [N*(N+1)-1:0]    vld_h, first_h, last_h;
    wire [(N+1)*N*DW-1:0] b_v;
    wire [N*N*ACCW-1:0]   res;

    genvar i, j;
    generate
        // input skew: row i of A (with its flags) is delayed i+1 cycles
        for (i = 0; i < N; i = i + 1) begin : g_row_skew
            wire [DW+1:0] d_out;
            sa_skew #(.DEPTH(i + 1), .W(DW + 2)) u_skew (
                .clk(clk), .rst_n(rst_n),
                .vld_in  (in_fire),
                .d_in    ({first_q, s_last, s_a[i*DW +: DW]}),
                .vld_out (vld_h[i*(N+1)]),
                .d_out   (d_out)
            );
            assign first_h[i*(N+1)]           = d_out[DW+1];
            assign last_h[i*(N+1)]            = d_out[DW];
            assign a_h[(i*(N+1))*DW +: DW]    = d_out[DW-1:0];
        end

        // input skew: column j of B is delayed j+1 cycles
        for (j = 0; j < N; j = j + 1) begin : g_col_skew
            /* verilator lint_off PINCONNECTEMPTY */
            sa_skew #(.DEPTH(j + 1), .W(DW)) u_skew (
                .clk(clk), .rst_n(rst_n),
                .vld_in  (in_fire),
                .d_in    (s_b[j*DW +: DW]),
                .vld_out (),
                .d_out   (b_v[j*DW +: DW])
            );
            /* verilator lint_on PINCONNECTEMPTY */
        end

        for (i = 0; i < N; i = i + 1) begin : g_r
            for (j = 0; j < N; j = j + 1) begin : g_c
                localparam H_IN  = i*(N+1) + j;
                localparam H_OUT = i*(N+1) + j + 1;
                localparam V_IN  = i*N + j;
                localparam V_OUT = (i+1)*N + j;

                wire signed [ACCW-1:0] below;
                if (i == N - 1) begin : g_bottom
                    assign below = {ACCW{1'b0}};
                end else begin : g_inner
                    assign below = res[((i+1)*N + j)*ACCW +: ACCW];
                end

                sa_pe #(.DW(DW), .ACCW(ACCW)) u_pe (
                    .clk       (clk),
                    .rst_n     (rst_n),
                    .vld_in    (vld_h[H_IN]),
                    .first_in  (first_h[H_IN]),
                    .last_in   (last_h[H_IN]),
                    .a_in      (a_h[H_IN*DW +: DW]),
                    .b_in      (b_v[V_IN*DW +: DW]),
                    .vld_out   (vld_h[H_OUT]),
                    .first_out (first_h[H_OUT]),
                    .last_out  (last_h[H_OUT]),
                    .a_out     (a_h[H_OUT*DW +: DW]),
                    .b_out     (b_v[V_OUT*DW +: DW]),
                    .drain     (out_fire),
                    .res_below (below),
                    .res       (res[(i*N + j)*ACCW +: ACCW])
                );
            end
        end
    endgenerate

    // row 0 of the drain chains is the output
    assign m_c = res[N*ACCW-1:0];

`ifdef FORMAL
    // Inductive invariants of the controller (see formal/sa.sby, task prove)
    always @(*) begin
        if (rst_n) begin
            assert(cap_cnt <= CW'(CAP_LAT));
            assert(rows_left <= RW'(N));
            assert(!(cap_pending && rows_left != '0));
        end
    end
`endif

    // unused: the last column's A outputs and the bottom row's B outputs
    /* verilator lint_off UNUSEDSIGNAL */
    wire unused = &{1'b0, a_h, b_v, vld_h, first_h, last_h};
    /* verilator lint_on UNUSEDSIGNAL */

endmodule

`default_nettype wire
