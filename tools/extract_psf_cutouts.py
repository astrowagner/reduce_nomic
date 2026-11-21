"""Cut the PSF core out of the port's binned cubes, for the PSF-shape fit.

usage: python tools/extract_psf_cutouts.py REDUCED_DIR NIGHT SIDE OUT_DIR [--half 48] [--product clean|all]

For chop A and B (cube 1 and 2) of one night and side, reads the port's binned, registered frames
(``<night>_<side>_cube<n>_align_cen_clean_bin.fits`` with ``--product clean``, the IDL clean
selection; ``..._align_cen_bin.fits`` with ``--product all``, every binned frame) and their
parallactic angles, and writes the central (2 half) × (2 half) pixels of every frame with the
angles, so the core can be analyzed away from the archive machine. The port puts the star at
(radius - 0.5, radius - 0.5) of frames 2 radius wide (header ``XCEN``, ``YCEN``); the cut keeps it
at (half - 0.5, half - 0.5).

Parameters
----------
REDUCED_DIR : str
    The night's ``reduced_py`` folder.
NIGHT : str
    Night folder name, e.g. ``Vega_20210227``.
SIDE : str
    ``left`` or ``right``.
OUT_DIR : str
    Output folder, created if missing.
--half : int, default 48
    Half-width of the cut, pixels (48 gives 96 × 96 px, 1.72 arcsec at 0.0179 arcsec/px).
--product : {"clean", "all"}, default "clean"
    Which binned cube to cut.

Returns
-------
<OUT_DIR>/<night>_<side>_cube<n>_<product>_core<2 half>.fits : file
    Primary HDU: the cut frames, float32, shape (n_frames, 2 half, 2 half), with ``SRCFILE``,
    ``SRCSHA``, ``XCEN``, ``YCEN`` (star position in the cut, 0-based pixels), ``PXSCALE``,
    ``CHOP`` and ``SIDE`` in the header; HDU ``ANGLES``: parallactic angle of each frame,
    degrees, float64, shape (n_frames,). One file for each cube that exists.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from typing import Sequence

import numpy as np
from astropy.io import fits


def sha256_of(path: str, block: int = 1 << 24) -> str:
    """SHA-256 of a file, read in blocks.

    Parameters
    ----------
    path : str
        File to hash.
    block : int, default 16777216
        Read size, bytes.

    Returns
    -------
    digest : str
        Hexadecimal SHA-256.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(block)
            if not b:
                return h.hexdigest()
            h.update(b)


def cut_one(cube_path: str, angles_path: str, out_path: str, half: int) -> int:
    """Cut the central pixels of every frame of one cube and write them with the angles.

    Parameters
    ----------
    cube_path : str
        Binned cube, shape (n_frames, 2 radius, 2 radius), star at (radius - 0.5, radius - 0.5).
    angles_path : str
        Parallactic angles of its frames, degrees, shape (n_frames,).
    out_path : str
        Output FITS file; must not exist.
    half : int
        Half-width of the cut, pixels.

    Returns
    -------
    n_frames : int
        Number of frames written. ``out_path`` is written.
    """
    with fits.open(cube_path, memmap=True) as hd:
        head = hd[0].header
        n, ny, nx = hd[0].data.shape
        xc = float(head.get("XCEN", nx / 2 - 0.5))
        yc = float(head.get("YCEN", ny / 2 - 0.5))
        x0, y0 = int(round(xc + 0.5)) - half, int(round(yc + 0.5)) - half
        if x0 < 0 or y0 < 0 or x0 + 2 * half > nx or y0 + 2 * half > ny:
            raise ValueError(f"a {2 * half} px cut around ({xc}, {yc}) does not fit in {nx} x {ny}")
        cut = np.array(hd[0].data[:, y0:y0 + 2 * half, x0:x0 + 2 * half], dtype=np.float32)
        chop, side, pxscale = head.get("CHOP", ""), head.get("SIDE", ""), head.get("PXSCALE", 0.0179)
    ang = np.asarray(fits.getdata(angles_path), dtype=np.float64).ravel()
    if len(ang) != n:
        raise ValueError(f"{angles_path} has {len(ang)} angles for {n} frames")
    h = fits.Header()
    h["SRCFILE"] = os.path.basename(cube_path)
    h["SRCSHA"] = sha256_of(cube_path)
    h["XCEN"] = (xc - x0, "star x in the cut, 0-based px")
    h["YCEN"] = (yc - y0, "star y in the cut, 0-based px")
    h["CUTX0"], h["CUTY0"] = x0, y0
    h["PXSCALE"], h["CHOP"], h["SIDE"] = pxscale, chop, side
    fits.HDUList([fits.PrimaryHDU(cut, h), fits.ImageHDU(ang, name="ANGLES")]).writeto(out_path, overwrite=False)
    return n


def main(argv: Sequence[str]) -> int:
    """Run the extraction from the command line.

    Parameters
    ----------
    argv : sequence of str
        Command-line arguments without the program name (see the module docstring).

    Returns
    -------
    status : int
        0 when at least one cube was cut, 1 when none was found.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("reduced_dir")
    ap.add_argument("night")
    ap.add_argument("side", choices=["left", "right"])
    ap.add_argument("out_dir")
    ap.add_argument("--half", type=int, default=48)
    ap.add_argument("--product", choices=["clean", "all"], default="clean")
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    pre = f"{a.night}_{a.side}"
    tag = "_clean" if a.product == "clean" else ""
    done = 0
    for cn in (1, 2):
        cube = os.path.join(a.reduced_dir, f"{pre}_cube{cn}_align_cen{tag}_bin.fits")
        angles = os.path.join(a.reduced_dir, f"{pre}_angles{cn}{tag}_bin.fits")
        if not (os.path.exists(cube) and os.path.exists(angles)):
            print(f"{pre} cube{cn}: no {os.path.basename(cube)} or its angles; skipped")
            continue
        out = os.path.join(a.out_dir, f"{pre}_cube{cn}_{a.product}_core{2 * a.half}.fits")
        n = cut_one(cube, angles, out, a.half)
        print(f"{pre} cube{cn}: {n} frames -> {out}")
        done += 1
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
