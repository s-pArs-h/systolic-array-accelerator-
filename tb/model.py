"""Reference model and helpers for the systolic-array testbench (NumPy).

All arithmetic is done in int64, so the reference never overflows; any
truncation in the hardware shows up as a mismatch.
"""
import numpy as np

I8_MIN, I8_MAX = -128, 127


def pack(values, width):
    """Pack signed lane values into one bus integer, lane 0 in the low bits."""
    mask = (1 << width) - 1
    word = 0
    for lane, v in enumerate(values):
        word |= (int(v) & mask) << (lane * width)
    return word


def unpack(word, lanes, width):
    """Split a bus integer into signed lane values."""
    out = []
    for lane in range(lanes):
        v = (word >> (lane * width)) & ((1 << width) - 1)
        out.append(v - (1 << width) if v >> (width - 1) else v)
    return out


def tile_slices(a_tile, b_tile):
    """The k-slices of one tile: (column k of A, row k of B, last flag)."""
    k_dim = a_tile.shape[1]
    return [(a_tile[:, k], b_tile[k, :], k == k_dim - 1) for k in range(k_dim)]


def gemm_tiles(a, b, n):
    """Split C = A @ B into N x N output tiles. Returns a list of
    ((row, col), A row-panel, B column-panel). M and N must be multiples of n."""
    m_dim, k_dim = a.shape
    _, n_dim = b.shape
    assert m_dim % n == 0 and n_dim % n == 0
    return [((r, c), a[r:r + n, :], b[:, c:c + n])
            for r in range(0, m_dim, n) for c in range(0, n_dim, n)]


def matmul(a, b):
    return a.astype(np.int64) @ b.astype(np.int64)
