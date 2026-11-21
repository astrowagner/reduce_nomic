"""Cross-check one night's KLIP-TPE input set against the port products it was cut from.

usage: python analysis/procyon4_2026-11/qa_klip_tpe_inputs.py REDUCED_PY INPUTS_DIR PREFIX HALF

For each chop this checks, independently of tools/klip_tpe_inputs.py, that the input cube
``<PREFIX>_cube<n>_all_c<2 HALF>.fits`` is finite and equal, frame by frame and bit for bit, to the
centered (2 HALF)² region of ``<PREFIX>_cube<n>_align_cen_bin.fits``; that its angles match the
binned angles; where the star's flux-weighted centroid falls; and that the PSF template is finite
and peaked at its center. It applies to an input set built with ``--product all`` and no frame
left out (the manifest's ``dropped_*`` lists empty).

Parameters
----------
REDUCED_PY : str
    The night's ``reduced_py`` folder.
INPUTS_DIR : str
    The input set, ``<night>/klip_tpe/inputs_<tag>``.
PREFIX : str
    File prefix of the side, e.g. ``Procyon_20211220_left``.
HALF : int
    Half-width of the cut, pixels (``--half`` of the build).

Returns
-------
stdout : text
    One line per chop: shape and dtype, finiteness, equality with the source region, the largest
    angle difference (degrees), the median frame's centroid against the array center (pixels), and
    the PSF template's shape, finiteness, peak pixel and sum.
"""
import os
import sys

import numpy as np
from astropy.io import fits


def centroid(img: np.ndarray, radius: float = 8.0):
    """Return the flux-weighted centroid of the positive pixels within ``radius`` of the peak.

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        Image with the star as its brightest pixel.
    radius : float, default 8.0
        Radius around the peak, pixels.

    Returns
    -------
    xc, yc : float
        The centroid, pixels.
    """
    iy, ix = np.unravel_index(np.argmax(img), img.shape)
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    m = (np.hypot(xx - ix, yy - iy) < radius) & (img > 0)
    w = img[m]
    return float((w * xx[m]).sum() / w.sum()), float((w * yy[m]).sum() / w.sum())


def main(argv=None) -> int:
    """Run the checks on both chops.

    Parameters
    ----------
    argv : list of str or None, default None
        REDUCED_PY INPUTS_DIR PREFIX HALF; None reads ``sys.argv[1:]``.

    Returns
    -------
    status : int
        0 when every check passes, 1 otherwise.
    """
    red, inp, pre, half = (argv or sys.argv[1:])[:4]
    half = int(half)
    ok = True
    for cn in (1, 2):
        cut = fits.getdata(os.path.join(inp, f"{pre}_cube{cn}_all_c{2 * half}.fits"))
        ang = np.asarray(fits.getdata(os.path.join(inp, f"{pre}_angles{cn}_all.fits")), float).ravel()
        psf = fits.getdata(os.path.join(inp, f"{pre}_cube{cn}_psf60.fits"))
        sang = np.asarray(fits.getdata(os.path.join(red, f"{pre}_angles{cn}_bin.fits")), float).ravel()
        with fits.open(os.path.join(red, f"{pre}_cube{cn}_align_cen_bin.fits"), memmap=True) as hd:
            n, ny, nx = hd[0].data.shape
            cy, cx = ny // 2, nx // 2
            same = n == cut.shape[0] and all(
                np.array_equal(np.asarray(hd[0].data[k, cy - half:cy + half, cx - half:cx + half]), cut[k],
                               equal_nan=True) for k in range(n))
        finite = bool(np.isfinite(cut).all())
        dang = float(np.max(np.abs(ang - sang))) if ang.size == sang.size else np.inf
        xc, yc = centroid(np.median(cut, axis=0))
        piy, pix = np.unravel_index(np.argmax(psf), psf.shape)
        ok &= finite and same and dang < 1e-9 and bool(np.isfinite(psf).all())
        print(f"cube{cn}: {cut.shape} {cut.dtype}; finite {finite}; equal to the source region {same}; "
              f"largest angle difference {dang:.2g} deg over {ang.size}; median-frame centroid "
              f"({xc:.2f}, {yc:.2f}) against the center {(cut.shape[2] - 1) / 2:.1f}; PSF {psf.shape} finite "
              f"{bool(np.isfinite(psf).all())}, peak ({pix}, {piy}), sum {float(psf.sum()):.4g}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
