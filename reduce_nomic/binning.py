"""Stages 7 and 8, pre-binning and replacement: bin frames in time and fill interior NaNs.

These are the ``do_pre_binning`` and ``do_replace`` steps of the IDL predecessor
(reduce_nomic.pro lines 6974-7421). Within each chunk, consecutive groups of ``prebin`` frames
are combined; the binned frames of all chunks are then concatenated, and NaNs inside each binned
frame's outermost good rows and columns are filled with ``fixpix``.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from . import idl


def median_rows(cube: np.ndarray, index: np.ndarray, block: int = 128) -> np.ndarray:
    """Take the IDL median with ``/EVEN`` over some frames of a cube, a block of rows at a time.

    Parameters
    ----------
    cube : ndarray, shape (m, ny, nx)
        Frames, where m is their number and ny, nx their size.
    index : ndarray of int, shape (g,)
        The frames to take the median over, where g is their number.
    block : int, default 128
        Rows per block.

    Returns
    -------
    med : ndarray, shape (ny, nx)
        Float64, equal to ``idl.median(cube[index], axis=0, even=True)``. Taking it by blocks of
        rows keeps the float64 copy and sort of ``idl.median`` to ``block / ny`` of the group.
    """
    ny = cube.shape[1]
    med = np.empty(cube.shape[1:], dtype=np.float64)
    for y0 in range(0, ny, block):
        med[y0:y0 + block] = idl.median(cube[index, y0:y0 + block], axis=0, even=True)
    return med


def prebin(cube: np.ndarray, angles: Sequence[float], exptimes: Sequence[float], st: Dict,
           index: Optional[Sequence[int]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin one chunk's frames in groups of ``prebin`` (IDL pre-binning).

    Parameters
    ----------
    cube : ndarray, shape (m, ny, nx)
        Frames of one chunk, where m is their number.
    angles : sequence of float
        Parallactic angle of each binned frame in degrees, in the order of ``index``.
    exptimes : sequence of float
        Effective exposure time of each binned frame in seconds, in the order of ``index``.
    st : dict
        Settings ``prebin`` (group size) and ``prebin_type`` (``median`` or ``mean``).
    index : sequence of int, optional
        The frames of ``cube`` to bin, in order (the frames the pre-clean keeps); default every
        frame. Passing the indices saves copying the kept frames.

    Returns
    -------
    binned : ndarray, shape (b, ny, nx)
        One frame per group, where b = ceil(k / prebin) and k is the number of frames binned:
        the IDL median with ``/EVEN`` (``median_rows``) or the NaN-ignoring mean; the last group
        may be shorter, and a one-frame group is that frame.
    angles_b : ndarray, shape (b,)
        Mean angle of each group.
    exptimes_b : ndarray, shape (b,)
        Summed exposure time of each group.
    """
    n = int(round(st["prebin"]))
    a = np.asarray(angles, dtype=float)
    e = np.asarray(exptimes, dtype=float)
    idx = np.arange(cube.shape[0]) if index is None else np.asarray(index, dtype=int)
    if n <= 1:
        return (cube if index is None else cube[idx]), a, e
    out, aa, ee = [], [], []
    for s in range(0, idx.size, n):
        gi = idx[s:s + n]
        if gi.size == 1:
            out.append(cube[gi[0]])
        elif st["prebin_type"] == "median":
            out.append(median_rows(cube, gi))
        else:
            out.append(np.nanmean(cube[gi], axis=0))
        aa.append(a[s:s + n].mean())
        ee.append(e[s:s + n].sum())
    return np.asarray(out), np.asarray(aa), np.asarray(ee)


def replace_interior_nans(frame: np.ndarray) -> np.ndarray:
    """Fill the NaNs inside a frame's outermost good rows and columns (IDL replace stage).

    Parameters
    ----------
    frame : ndarray, shape (ny, nx)
        Binned frame.

    Returns
    -------
    out : ndarray, shape (ny, nx)
        ``fixpix`` applied with every NaN marked bad, except those left of the last leading
        all-NaN column, below the last leading all-NaN row (row 0 when none), from the last
        column holding a finite value to the right, and from the last row holding one upward;
        those stay NaN, as in IDL.
    """
    good = np.isfinite(frame).astype(np.float64)
    col_any = good.sum(axis=0) > 0
    row_any = good.sum(axis=1) > 0
    if col_any[0]:
        raise ValueError("column 0 holds a finite pixel; the IDL replace step fails here")
    leftmost = int(np.argmax(col_any)) - 1
    bottommost = int(np.argmax(row_any)) - 1 if not row_any[0] else 0
    rightmost = int(np.nonzero(col_any)[0][-1])
    topmost = int(np.nonzero(row_any)[0][-1])
    bm = good.copy()
    bm[:, 0:leftmost + 1] = 1.0
    bm[0:bottommost + 1, :] = 1.0
    bm[:, rightmost:] = 1.0
    bm[topmost:, :] = 1.0
    return idl.FixPix(bm).apply(frame)
