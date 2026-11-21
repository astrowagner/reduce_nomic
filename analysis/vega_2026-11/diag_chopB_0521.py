#!/usr/bin/env python3
"""Run the port on a few hundred frames of one night and save what the split and align stages see.

usage: python3 diag_chopB_0521.py PORT_DIR NIGHT SIDE FIRST LAST OUT_DIR [--set NAME=VALUE ...]

A diagnostic for the chop-B failures of 2026-11-20 (Vega 2022-05-21 left: every chop-B chunk's
aligned median had its star 5-60 px from the center, and the pre-clean rejected 49,682 of
49,686 frames). The port (``PORT_DIR``, an unpacked reduce_nomic) is run through its own command
line on frames FIRST-LAST, with no sky rounds (``n_sky=0``, ``skysub=0``) unless ``--set`` says
otherwise, and with ``geometry.align_cube`` and ``geometry.center_cube`` wrapped so that every
call saves the median of the cut cube it is given, the median after alignment and the shifts it
applied; and the center found. The frame range is short enough to make one chunk, so the split
stage's edge check does not change the cut position (it acts only from the second chunk).

Parameters
----------
PORT_DIR : str
    Folder holding the ``reduce_nomic`` package.
NIGHT : str
    Night folder name, current or old.
SIDE : {"left", "right"}
    Aperture.
FIRST, LAST : int
    Raw frame numbers, inclusive.
OUT_DIR : str
    Folder for the port's products (``run/``), its memory-mapped intermediates (``work/``) and the
    diagnostics; it must not exist yet.
--set : list of str, default none
    Further port settings, as for the port's command line.

Returns
-------
OUT_DIR/diag_align_<k>.npz : files
    One per ``align_cube`` call, in call order: ``cut_median`` and ``aligned_median`` (shape
    (2 radius, 2 radius), pixels), ``shifts`` (shape (m, 2), the (x, y) shifts applied in pixels,
    where m is the number of frames), ``n`` (m) and ``peak_cut`` (x, y of the cut median's maximum
    within 60 px of the center).
OUT_DIR/diag_center_<k>.json : files
    One per ``center_cube`` call: the center returned, or the error, and the estimators' centers.
OUT_DIR/diag.png : file
    Cut and aligned medians (central 240 x 240 pixels) and the shift histories, one row per call.
OUT_DIR/run/, OUT_DIR/work/ : folders
    The port's products and intermediates for the frame range.
"""
from __future__ import annotations

import json
import os
import sys
from typing import List


def main(argv: List[str]) -> int:
    """Run the instrumented port and draw the diagnostics.

    Parameters
    ----------
    argv : list of str
        Command-line arguments without the program name (see the module docstring).

    Returns
    -------
    status : int
        The port's exit status.
    """
    if len(argv) < 6:
        raise SystemExit(__doc__.split("\n\n")[1])
    port, night, side, first, last, out = argv[:6]
    extra = argv[6:]
    os.makedirs(out, exist_ok=False)
    sys.path.insert(0, port)
    import numpy as np
    from reduce_nomic import cli, geometry

    calls = {"align": 0, "center": 0}
    orig_align, orig_center = geometry.align_cube, geometry.center_cube

    def align_wrapped(cube, st, ref=None):
        """Save the cut median, call ``align_cube``, save the aligned median and shifts.

        Parameters
        ----------
        cube : ndarray, shape (m, 2 radius, 2 radius)
            Cut frames (overwritten in place by ``align_cube`` when float32).
        st : dict
            Settings.
        ref : ndarray, optional
            Reference image, passed on.

        Returns
        -------
        aligned, shifts : ndarray
            As ``align_cube`` returns them.
        """
        r = int(st["radius"])
        with np.errstate(all="ignore"):
            cut_med = np.nanmedian(cube, axis=0)
        win = cut_med[r - 60:r + 60, r - 60:r + 60]
        iy, ix = divmod(int(np.nanargmax(np.nan_to_num(win, nan=-np.inf))), win.shape[1])
        aligned, shifts = orig_align(cube, st, ref)
        with np.errstate(all="ignore"):
            al_med = np.nanmedian(aligned, axis=0)
        k = calls["align"]
        calls["align"] += 1
        np.savez(os.path.join(out, f"diag_align_{k}.npz"), cut_median=cut_med.astype(np.float32),
                 aligned_median=al_med.astype(np.float32), shifts=np.asarray(shifts), n=cube.shape[0],
                 peak_cut=np.array([ix + r - 60, iy + r - 60]))
        return aligned, shifts

    def center_wrapped(aligned, st, rng, diag=None):
        """Call ``center_cube`` and save its result or error.

        Parameters
        ----------
        aligned : ndarray
            Aligned cube.
        st : dict
            Settings.
        rng : numpy.random.Generator
            Passed on.
        diag : dict, optional
            Filled by ``center_cube`` with the estimators' centers.

        Returns
        -------
        centered, center
            As ``center_cube`` returns them.
        """
        d = {} if diag is None else diag
        k = calls["center"]
        calls["center"] += 1
        rec = {}
        try:
            res = orig_center(aligned, st, rng, diag=d)
            rec["center"] = [float(x) for x in res[1]]
            return res
        except Exception as err:            # record and re-raise
            rec["error"] = repr(err)
            raise
        finally:
            rec.update({key: d.get(key) for key in ("center_idl", "center_gauss", "center_used")})
            with open(os.path.join(out, f"diag_center_{k}.json"), "w") as fh:
                json.dump(rec, fh, indent=1, default=str)

    geometry.align_cube, geometry.center_cube = align_wrapped, center_wrapped
    args = [night, "--side", side, "--workers", "1", "--out", os.path.join(out, "run"),
            "--work", os.path.join(out, "work"), "--frames", str(first), str(last),
            "--set", "n_sky=0", "--set", "skysub=0"] + extra
    archive = os.environ.get("NOMIC_ARCHIVE")
    if archive:
        args += ["--archive", archive]
    status = cli.main(args)
    draw(out)
    return status


def draw(out: str) -> None:
    """Draw the saved medians and shift histories.

    Parameters
    ----------
    out : str
        Folder holding the ``diag_align_<k>.npz`` files.

    Returns
    -------
    None
        Writes ``out/diag.png``.
    """
    import glob
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = sorted(glob.glob(os.path.join(out, "diag_align_*.npz")),
                   key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
    if not files:
        return
    fig, axes = plt.subplots(len(files), 3, figsize=(13, 4.2 * len(files)), squeeze=False)
    for row, p in zip(axes, files):
        z = np.load(p)
        cut, al, sh = z["cut_median"], z["aligned_median"], z["shifts"]
        r = cut.shape[0] // 2
        k = p.rsplit("_", 1)[1].split(".")[0]
        for ax, img, title in ((row[0], cut, f"Cut Median, Call {k} ({int(z['n'])} Frames)"),
                               (row[1], al, f"Aligned Median, Call {k}")):
            sub = img[r - 120:r + 120, r - 120:r + 120]
            lo, hi = np.nanpercentile(sub, [1, 99.7])
            ax.imshow(sub, origin="lower", cmap="magma", vmin=lo, vmax=hi,
                      extent=(r - 120, r + 120, r - 120, r + 120))
            ax.axhline(r, color="c", lw=0.5)
            ax.axvline(r, color="c", lw=0.5)
            ax.set_title(title)
            ax.set_xlabel("X (Pixels)")
            ax.set_ylabel("Y (Pixels)")
        row[0].plot(*z["peak_cut"], "x", color="w", ms=8)
        row[2].plot(sh[:, 0], label="X Shift")
        row[2].plot(sh[:, 1], label="Y Shift")
        row[2].set_title(f"Align Shifts, Call {k}")
        row[2].set_xlabel("Frame in Chunk")
        row[2].set_ylabel("Shift (Pixels)")
        row[2].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "diag.png"), dpi=80)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
