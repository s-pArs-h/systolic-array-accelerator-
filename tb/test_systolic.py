"""cocotb testbench for systolic_array.

Every result tile is compared with NumPy (int64). Stimulus covers random,
extreme and structured matrices, K from 1 upward, random gaps on the input,
random back-pressure on the output, back-to-back tiles, and complete GEMMs
larger than the array computed tile by tile. Functional coverage must close.

    make            N = 4
    make N=8 SEED=3
"""
import os
import random

import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, ReadOnly, RisingEdge

from coverage import cov
import model

N = int(os.environ.get("SA_N", "4"))
DW, ACCW = 8, 32
SEED = int(os.environ.get("SEED", "1"))
FULL_RATE_K = 3 * N + 1          # smallest K that never stalls (see docs/DESIGN.md)

for name in ("k_equals_1", "small_k_last_slice_stalled", "input_gap", "output_stall",
             "operand_minus128", "operand_127", "acc_beyond_2pow24", "negative_result",
             "back_to_back_tiles", "full_rate_no_stall", "gemm_multi_tile", "identity",
             "zero_matrix"):
    cov.define(name)


def bit(h):
    return str(h.value) == "1"


def uint(h):
    return int(str(h.value), 2)


def rand_mat(rng, rows, cols, kind="random"):
    if kind == "extreme":
        return rng.choice(np.array([model.I8_MIN, model.I8_MAX, -1, 0, 1], dtype=np.int64),
                          size=(rows, cols))
    if kind == "small":
        return rng.integers(-3, 4, size=(rows, cols))
    return rng.integers(model.I8_MIN, model.I8_MAX + 1, size=(rows, cols))


async def reset(dut):
    Clock(dut.clk, 10, unit="ns").start()
    dut.s_valid.value = 0
    dut.s_last.value = 0
    dut.s_a.value = 0
    dut.s_b.value = 0
    dut.m_ready.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 3)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def drive(dut, tiles, pyrng, valid_prob, stats):
    """Stream every k-slice of every tile; once valid is raised it is held
    with stable data until accepted."""
    holding = False
    for a_tile, b_tile in tiles:
        for a_col, b_row, last in model.tile_slices(a_tile, b_tile):
            while True:
                valid = holding or pyrng.random() < valid_prob
                dut.s_valid.value = int(valid)
                if valid:
                    dut.s_a.value = model.pack(a_col, DW)
                    dut.s_b.value = model.pack(b_row, DW)
                    dut.s_last.value = int(last)
                else:                                   # junk must be ignored
                    dut.s_a.value = pyrng.getrandbits(N * DW)
                    dut.s_b.value = pyrng.getrandbits(N * DW)
                    dut.s_last.value = pyrng.getrandbits(1)
                await ReadOnly()
                ready = bit(dut.s_ready)
                if valid and not ready:
                    stats["stalls"] += 1
                    cov.hit("small_k_last_slice_stalled")
                if not valid:
                    cov.hit("input_gap")
                holding = valid and not ready
                await RisingEdge(dut.clk)
                if valid and ready:
                    break
    dut.s_valid.value = 0


async def collect(dut, n_tiles, pyrng, ready_prob, out):
    rows = []
    while len(out) < n_tiles:
        ready = pyrng.random() < ready_prob
        dut.m_ready.value = int(ready)
        await ReadOnly()
        if bit(dut.m_valid):
            if ready:
                rows.append(model.unpack(uint(dut.m_c), N, ACCW))
                expect_last = len(rows) == N
                assert bit(dut.m_last) == expect_last, f"m_last wrong on row {len(rows) - 1}"
                if expect_last:
                    out.append(np.array(rows, dtype=np.int64))
                    rows = []
            else:
                cov.hit("output_stall")
        await RisingEdge(dut.clk)
    dut.m_ready.value = 0


async def run_tiles(dut, tiles, pyrng, valid_prob=1.0, ready_prob=1.0):
    """Send tiles, collect results, check each against NumPy.
    Returns (cycles, input stalls)."""
    stats = {"stalls": 0}
    out = []
    start = cocotb.utils.get_sim_time("ns")
    cocotb.start_soon(drive(dut, tiles, pyrng, valid_prob, stats))
    await collect(dut, len(tiles), pyrng, ready_prob, out)
    cycles = int((cocotb.utils.get_sim_time("ns") - start) // 10)

    for t, ((a_tile, b_tile), got) in enumerate(zip(tiles, out)):
        exp = model.matmul(a_tile, b_tile)
        assert np.array_equal(got, exp), (
            f"tile {t} (K={a_tile.shape[1]}): mismatch at "
            f"{np.argwhere(got != exp)[:4].tolist()}\ngot\n{got}\nexpected\n{exp}")
        if (a_tile == model.I8_MIN).any() or (b_tile == model.I8_MIN).any():
            cov.hit("operand_minus128")
        if (a_tile == model.I8_MAX).any() or (b_tile == model.I8_MAX).any():
            cov.hit("operand_127")
        if np.abs(exp).max() >= 1 << 24:
            cov.hit("acc_beyond_2pow24")
        if (exp < 0).any():
            cov.hit("negative_result")
        if a_tile.shape[1] == 1:
            cov.hit("k_equals_1")
    if len(tiles) > 1:
        cov.hit("back_to_back_tiles")

    # the array must go idle once everything has drained
    await ClockCycles(dut.clk, 2 * N + 2)
    await ReadOnly()
    assert not bit(dut.busy), "busy still high after all tiles finished"
    await RisingEdge(dut.clk)
    return cycles, stats["stalls"]


@cocotb.test()
async def test_identity_and_zero(dut):
    rng, pyrng = np.random.default_rng(SEED), random.Random(SEED)
    await reset(dut)
    a = rand_mat(rng, N, N)
    tiles = [(a, np.eye(N, dtype=np.int64)), (np.eye(N, dtype=np.int64), a),
             (a, np.zeros((N, N), dtype=np.int64))]
    await run_tiles(dut, tiles, pyrng)
    cov.hit("identity")
    cov.hit("zero_matrix")


@cocotb.test()
async def test_random_tiles(dut):
    """Random K and values, random input gaps and output back-pressure."""
    rng, pyrng = np.random.default_rng(SEED + 1), random.Random(SEED + 1)
    await reset(dut)
    tiles = []
    for _ in range(40):
        k = int(rng.integers(1, 4 * N))
        tiles.append((rand_mat(rng, N, k), rand_mat(rng, k, N)))
    await run_tiles(dut, tiles, pyrng, valid_prob=0.75, ready_prob=0.6)


@cocotb.test()
async def test_extreme_accumulation(dut):
    """-128 x -128 summed over a long K: 16384 * K must not overflow or wrap."""
    rng, pyrng = np.random.default_rng(SEED + 2), random.Random(SEED + 2)
    await reset(dut)
    k = 4096
    tiles = [(np.full((N, k), -128, dtype=np.int64), np.full((k, N), -128, dtype=np.int64)),
             (np.full((N, k), 127, dtype=np.int64), np.full((k, N), -128, dtype=np.int64)),
             (rand_mat(rng, N, 300, "extreme"), rand_mat(rng, 300, N, "extreme"))]
    await run_tiles(dut, tiles, pyrng)


@cocotb.test()
async def test_small_k_back_to_back(dut):
    """K = 1 .. 3N: the last slice has to wait for the previous drain."""
    rng, pyrng = np.random.default_rng(SEED + 3), random.Random(SEED + 3)
    await reset(dut)
    tiles = []
    for k in list(range(1, FULL_RATE_K)) * 3:
        tiles.append((rand_mat(rng, N, k, "small"), rand_mat(rng, k, N, "small")))
    _, stalls = await run_tiles(dut, tiles, pyrng)
    assert stalls > 0, "expected the last-slice stall to be exercised"


@cocotb.test()
async def test_full_rate(dut):
    """With K >= 3N+1 and no back-pressure, slices are accepted every cycle:
    N*N multiply-accumulates per cycle."""
    rng, pyrng = np.random.default_rng(SEED + 4), random.Random(SEED + 4)
    await reset(dut)
    n_tiles, k = 20, FULL_RATE_K
    tiles = [(rand_mat(rng, N, k), rand_mat(rng, k, N)) for _ in range(n_tiles)]
    cycles, stalls = await run_tiles(dut, tiles, pyrng)
    assert stalls == 0, f"{stalls} stall cycles at K = {k}"
    ideal = n_tiles * k
    dut._log.info("%d tiles x K=%d: %d cycles (ideal %d + fill/drain %d), %.1f%% MAC utilisation",
                  n_tiles, k, cycles, ideal, cycles - ideal, 100.0 * ideal / cycles)
    cov.hit("full_rate_no_stall")


@cocotb.test()
async def test_gemm(dut):
    """A complete GEMM larger than the array, computed tile by tile."""
    rng, pyrng = np.random.default_rng(SEED + 5), random.Random(SEED + 5)
    await reset(dut)
    m_dim, k_dim, n_dim = 3 * N, 37, 2 * N
    a = rand_mat(rng, m_dim, k_dim)
    b = rand_mat(rng, k_dim, n_dim)
    plan = model.gemm_tiles(a, b, N)
    tiles = [(ap, bp) for _, ap, bp in plan]
    await run_tiles(dut, tiles, pyrng, valid_prob=0.9, ready_prob=0.8)
    cov.hit("gemm_multi_tile")
    dut._log.info("GEMM %dx%d @ %dx%d done in %d tiles", m_dim, k_dim, k_dim, n_dim, len(tiles))


@cocotb.test()
async def test_coverage_closure(dut):
    dut._log.info("\n%s", cov.report())
    with open(f"coverage_N{N}.txt", "w") as f:
        f.write(cov.report() + "\n")
    assert not cov.missing(), f"uncovered bins: {cov.missing()}"
