"""Independent 5σ contrast check on KLIP-TPE inputs: classical ADI and derotation only, with IDL's S/N.

usage: python adi_crosscheck.py INPUTS_DIR [--seps 0.54 0.71 0.90 1.10] [--n-pa 8] [--c-inj 0.003 0.01]
       [--crop 160] [--out CSV]

For each partition in ``INPUTS_DIR/manifest.json`` (a ``tools/klip_tpe_inputs.py`` folder), fake
companions, copies of the partition's PSF template scaled to a flux ratio c (the frames are
peak-scaled, so the template's own scale is the star's), are injected at separation r and sky
angle θ, at detector angle θ - q in each frame of parallactic angle q. Two reductions follow:

* classical ADI: the median of all frames is subtracted from each frame; then derotation by +q
  and the mean over frames;
* derotation only: derotation by +q and the mean, without subtraction.

The partitions' images are averaged. The S/N follows the IDL predecessor's contrastcurve_nomic.pro:
the flux in an aperture of radius FWHM/2 (FWHM = λ/D = 15.2 px) at the source, minus the mean of
apertures around the ring spaced one aperture diameter apart (leaving out the two on each side of
the source), over their standard deviation times sqrt(1 + 1/n) (Mawet et al. 2014). The 5σ flux
ratio is 5 c / S/N(c), the median over ``--n-pa`` angles, for each injected flux ratio c. It holds
only where S/N grows in proportion to c: a bright source's own ADI residuals (negative arcs from the
median) enter the ring apertures and flatten S/N, so injections near the limit are what count; the
two values of ``--c-inj`` show whether the S/N is still proportional.

Parameters
----------
INPUTS_DIR : str
    A ``klip_tpe/inputs_<tag>/`` folder with ``manifest.json``.
--seps : float list, default 0.54 0.71 0.90 1.10
    Separations, arcsec.
--n-pa : int, default 8
    Sky angles per separation, evenly spaced.
--pxscale : float, default 0.0179
    Pixel scale, arcsec per pixel.
--c-inj : float list, default 0.003 0.01
    Injected flux ratios.
--out : str, optional
    CSV to write (separation, method, injected flux ratio, median and spread of the 5σ flux
    ratio over angles).
--crop : int, default 160
    Size of the central region kept, pixels (even, so the star stays at the center).

Returns
-------
stdout : text
    The 5σ flux ratio per separation and method.
<out> : file
    The same as CSV [--out].
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
from astropy.io import fits
from scipy import ndimage as ndi

FWHM_PX = 11.1e-6 / 8.4 * 206264.806 / 0.0179     # λ/D at 11.1 µm on 8.4 m, in 0.0179" pixels (15.2)


def rotate(img: np.ndarray, deg: float) -> np.ndarray:
    """Rotate an image counterclockwise about its center.

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        Image, NaN allowed (NaN pixels become 0 before the bilinear rotation).
    deg : float
        Angle, degrees, counterclockwise with x along columns and y along rows pointing up.

    Returns
    -------
    out : ndarray, shape (ny, nx)
        Rotated image; pixels from outside become NaN.
    """
    good = np.isfinite(img)
    # scipy's rotate turns clockwise for a positive angle when y (rows) points up, hence -deg
    a = ndi.rotate(np.where(good, img, 0.0), -deg, reshape=False, order=1, mode="constant", cval=np.nan)
    m = ndi.rotate(good.astype(float), -deg, reshape=False, order=1, mode="constant", cval=0.0)
    a[m < 0.5] = np.nan
    return a


def inject(cube: np.ndarray, angles: np.ndarray, psf: np.ndarray, c: float, r_px: float, theta_deg: float) -> np.ndarray:
    """Add a fake companion to every frame.

    Parameters
    ----------
    cube : ndarray, shape (n, ny, nx)
        Peak-scaled frames with the star at ((nx - 1) / 2, (ny - 1) / 2).
    angles : ndarray, shape (n,)
        Parallactic angles, degrees.
    psf : ndarray, shape (m, m)
        PSF template on the same scale as the star, centered at ((m - 1) / 2, (m - 1) / 2).
    c : float
        Flux ratio.
    r_px : float
        Separation, pixels.
    theta_deg : float
        Sky angle, degrees counterclockwise from +x.

    Returns
    -------
    out : ndarray, shape (n, ny, nx)
        The frames with the companion at detector angle theta - q.
    """
    n, ny, nx = cube.shape
    m = psf.shape[0]
    canvas = np.zeros((ny, nx))
    y0, x0 = (ny - m) // 2, (nx - m) // 2            # template center lands on the frame center
    canvas[y0:y0 + m, x0:x0 + m] = np.nan_to_num(psf)
    out = cube.copy()
    for i in range(n):
        a = np.deg2rad(theta_deg - angles[i])
        out[i] += c * ndi.shift(canvas, (r_px * np.sin(a), r_px * np.cos(a)), order=1, mode="constant", cval=0.0)
    return out


def reduce(cube: np.ndarray, angles: np.ndarray, subtract: bool) -> np.ndarray:
    """Classical ADI (or derotation only) of one partition.

    Parameters
    ----------
    cube : ndarray, shape (n, ny, nx)
        Frames.
    angles : ndarray, shape (n,)
        Parallactic angles, degrees.
    subtract : bool
        Subtract the median of all frames before derotation (classical ADI).

    Returns
    -------
    image : ndarray, shape (ny, nx)
        Mean of the derotated frames (NaN-ignoring).
    """
    res = cube - np.nanmedian(cube, axis=0)[None] if subtract else cube
    der = np.array([rotate(res[i], angles[i]) for i in range(len(angles))])
    return np.nanmean(der, axis=0)


def aperture_sum(img: np.ndarray, x: float, y: float, r: float) -> float:
    """Sum of an image in a circular aperture (pixel centers inside; NaN as 0).

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        Image.
    x, y : float
        Aperture center, pixels.
    r : float
        Radius, pixels.

    Returns
    -------
    flux : float
        The sum.
    """
    ny, nx = img.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    return float(np.nansum(np.where(np.hypot(xx - x, yy - y) <= r, img, 0.0)))


def snr_idl(img: np.ndarray, r_px: float, theta_deg: float) -> float:
    """S/N of a source as contrastcurve_nomic.pro measures it.

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        Final image, star at the center.
    r_px : float
        Separation, pixels.
    theta_deg : float
        Sky angle of the source, degrees.

    Returns
    -------
    snr : float
        (aperture flux - mean of the ring apertures) / (their standard deviation x sqrt(1 + 1/n)).
    """
    ny, nx = img.shape
    xc, yc = (nx - 1) / 2.0, (ny - 1) / 2.0
    ra = FWHM_PX / 2.0
    t0 = np.deg2rad(theta_deg)
    sig = aperture_sum(img, xc + r_px * np.cos(t0), yc + r_px * np.sin(t0), ra)
    nth = int(round(2 * np.pi * r_px / (2 * ra) - 1))
    lo, hi = (1, nth - 1) if nth < 6 else (2, nth - 2)
    ring = [aperture_sum(img, xc + r_px * np.cos(t0 + 2 * np.pi * k / nth), yc + r_px * np.sin(t0 + 2 * np.pi * k / nth), ra)
            for k in range(lo, hi + 1)]
    ring = np.array([f for f in ring if f != 0.0])
    noise = np.std(ring, ddof=0) * np.sqrt(1 + 1.0 / len(ring))
    return float((sig - ring.mean()) / noise)


def main(argv: Sequence[str]) -> int:
    """Run the check from the command line.

    Parameters
    ----------
    argv : sequence of str
        Command-line arguments without the program name (see the module docstring).

    Returns
    -------
    status : int
        0 on success.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("inputs_dir")
    ap.add_argument("--seps", type=float, nargs="+", default=[0.54, 0.71, 0.90, 1.10])
    ap.add_argument("--n-pa", type=int, default=8)
    ap.add_argument("--pxscale", type=float, default=0.0179)
    ap.add_argument("--out", default=None)
    ap.add_argument("--crop", type=int, default=160)
    ap.add_argument("--c-inj", type=float, nargs="+", default=[0.003, 0.01])
    a = ap.parse_args(argv)
    m = json.load(open(os.path.join(a.inputs_dir, "manifest.json")))
    parts = []
    for p in m["partitions"]:
        loc = lambda k: os.path.join(a.inputs_dir, os.path.basename(p[k]))  # noqa: E731
        cube = np.asarray(fits.getdata(loc("cube")), dtype=np.float64)
        ny = cube.shape[-1]
        lo = (ny - a.crop) // 2
        cube = cube[:, lo:lo + a.crop, lo:lo + a.crop]          # the star stays at the center
        ang = np.asarray(fits.getdata(loc("angles")), dtype=np.float64).ravel()
        psf = np.asarray(fits.getdata(loc("psf")), dtype=np.float64)
        psf = psf / np.nanmax(psf)                     # peak 1, as the peak-scaled star
        parts.append((p["name"], cube, ang, psf))
    rows = []
    for sep in a.seps:
        r_px = sep / a.pxscale
        for method, sub in (("classical ADI", True), ("derotation only", False)):
            for c_inj in a.c_inj:
                c5 = []
                for k in range(a.n_pa):
                    th = 360.0 * k / a.n_pa + 11.25
                    img = np.nanmean([reduce(inject(cu, an, ps, c_inj, r_px, th), an, sub) for _, cu, an, ps in parts], axis=0)
                    s = snr_idl(img, r_px, th)
                    c5.append(5.0 * c_inj / s if s > 0 else np.inf)
                c5 = np.array(c5)
                rows.append({"sep_arcsec": sep, "method": method, "c_inj": c_inj, "c5_median": float(np.median(c5)),
                             "c5_p16": float(np.percentile(c5, 16)), "c5_p84": float(np.percentile(c5, 84))})
                print(f"{sep:.2f}\" {method:16s} c_inj {c_inj:.2f}: 5-sigma flux ratio {np.median(c5):.2e} "
                      f"(16-84%: {np.percentile(c5, 16):.2e}-{np.percentile(c5, 84):.2e})", flush=True)
    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
