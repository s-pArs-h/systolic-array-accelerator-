`default_nettype none

// Formal harness for systolic_array (SymbiYosys, see sa.sby).
//
// Inputs are unconstrained apart from reset in the first cycle.
//   P1  a stalled output beat holds its data and m_last
//   P2  m_last marks exactly every N-th result row       (SCOREBOARD)
//   P3  at most one finished tile is ever waiting to drain (SCOREBOARD)
//   P4  the first tile's results equal sum_k A[i][k] * B[k][j], computed
//       here at full precision                           (CHECK_DATA)
// The PEs additionally assert that a capture never meets a drain shift, and
// the controller asserts its own invariants (both under `ifdef FORMAL).
module sa_formal #(
    parameter N          = 2,
    parameter DW         = 3,
    parameter ACCW       = 12,
    parameter KMAX       = 3,            // longest first tile the data check tracks
    parameter SCOREBOARD = 1,
    parameter CHECK_DATA = 1
) (
    input wire              clk,
    input wire              rst_n,
    input wire              s_valid,
    input wire              s_last,
    input wire [N*DW-1:0]   s_a,
    input wire [N*DW-1:0]   s_b,
    input wire              m_ready
);
    wire              s_ready, m_valid, m_last, busy;
    wire [N*ACCW-1:0] m_c;

    systolic_array #(.N(N), .DW(DW), .ACCW(ACCW)) dut (
        .clk(clk), .rst_n(rst_n),
        .s_valid(s_valid), .s_ready(s_ready), .s_last(s_last), .s_a(s_a), .s_b(s_b),
        .m_valid(m_valid), .m_ready(m_ready), .m_last(m_last), .m_c(m_c),
        .busy(busy)
    );

    reg f_past_valid = 1'b0;
    always @(posedge clk) f_past_valid <= 1'b1;
    always @(*) assume(rst_n == f_past_valid);

    wire in_fire  = s_valid && s_ready;
    wire out_fire = m_valid && m_ready;

    // ---------------- P1 ----------------
    always @(posedge clk)
        if (f_past_valid && rst_n && $past(rst_n) && $past(m_valid && !m_ready)) begin
            assert(m_valid);
            assert($stable(m_c));
            assert($stable(m_last));
        end

    // ---------------- P2, P3 ----------------
    reg [7:0] row_idx;
    reg [7:0] tiles_in, tiles_out;
    always @(posedge clk) begin
        if (!rst_n) begin
            row_idx   <= 0;
            tiles_in  <= 0;
            tiles_out <= 0;
        end else begin
            if (in_fire && s_last) tiles_in <= tiles_in + 1'b1;
            if (out_fire) begin
                row_idx <= m_last ? 8'd0 : row_idx + 1'b1;
                if (m_last) tiles_out <= tiles_out + 1'b1;
            end
        end
    end
    always @(posedge clk)
        if (SCOREBOARD && f_past_valid && rst_n) begin
            if (out_fire) assert(m_last == (row_idx == N - 1));
            assert(tiles_out <= tiles_in);
            assert(tiles_in - tiles_out <= 1);
        end

    // ---------------- P4: data integrity of the first tile ----------------
    reg signed [DW-1:0] fa [0:N-1][0:KMAX-1];
    reg signed [DW-1:0] fb [0:KMAX-1][0:N-1];
    reg [7:0] k0;                        // slices of tile 0 accepted so far
    reg       t0_done;                   // tile 0's last slice accepted
    integer i, j, k;
    always @(posedge clk) begin
        if (!rst_n) begin
            k0      <= 0;
            t0_done <= 1'b0;
        end else if (in_fire && !t0_done) begin
            for (i = 0; i < N; i = i + 1) begin
                fa[i][k0] <= s_a[i*DW +: DW];
                fb[k0][i] <= s_b[i*DW +: DW];
            end
            k0 <= k0 + 1'b1;
            if (s_last) t0_done <= 1'b1;
        end
    end
    // keep tile 0 short enough to track
    always @(*) if (CHECK_DATA && rst_n && in_fire && !t0_done && k0 == KMAX - 1) assume(s_last);

    // Products at their exact width (2*DW bits). A 32-bit multiply here would
    // make the solver prove two very different multiplier circuits equal.
    reg signed [ACCW-1:0] exp_c [0:N-1][0:N-1];
    reg signed [2*DW-1:0] f_prod;
    always @(*) begin
        for (i = 0; i < N; i = i + 1)
            for (j = 0; j < N; j = j + 1) begin
                exp_c[i][j] = 0;
                for (k = 0; k < KMAX; k = k + 1)
                    if (k < k0) begin
                        f_prod      = fa[i][k] * fb[k][j];
                        exp_c[i][j] = exp_c[i][j] + f_prod;
                    end
            end
    end

    always @(posedge clk)
        if (CHECK_DATA && f_past_valid && rst_n && out_fire && tiles_out == 0)
            for (j = 0; j < N; j = j + 1)
                assert(m_c[j*ACCW +: ACCW] == exp_c[row_idx][j]);

    // ---------------- cover ----------------
    always @(posedge clk)
        if (f_past_valid && rst_n) begin
            cover(out_fire && m_last && tiles_out == 1);               // two tiles done
            cover(s_valid && s_last && !s_ready);                      // last slice held
            cover(m_valid && !m_ready && busy);                        // output stalled
        end

endmodule

`default_nettype wire
