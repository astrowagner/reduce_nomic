#!/usr/bin/env python3
"""Compare a night's Python products with the IDL predecessor's products.

Reads, for each side and chop n (1 for chop A, 2 for chop B), from PORT_DIR (``reduced_py``) and
IDL_DIR (``reduced``):

* the frame counts of the binned cubes ``<night>_<side>_cube<n>_align_cen_bin.fits`` and the
  clean cubes ``..._cube<n>_align_cen_clean_bin.fits`` (FITS headers);
* the binned angles and exposure times: port ``..._angles<n>_bin.fits`` and
  ``..._frametags<n>_bin.fits`` (column ``exptime_s`` when present), IDL ``..._angles<n>_bin.sav``
  (``angles<n>``, ``coadds<n>`` for the kept unbinned frames, ``exptimes<n>``); the clean
  angles: port ``..._angles<n>_clean_bin.fits``, IDL ``..._angles<n>_clean_bin.sav``;
* the median PSFs ``..._cube<n>_align_cen_median_PSF.fits`` (all bins) and
  ``..._cube<n>_align_cen_clean_median_PSF.fits`` (clean bins);
* the central 128 x 128 pixels of every binned frame: the port's binned cube and the IDL
  ``..._cube<n>_align_cen_bin_replace.fits`` (or ``..._bin.fits`` when that is missing).

It prints, per side and chop:

* bins, kept unbinned frames, summed exposure time in seconds and clean bins, port and IDL;
* the angle match: for each port bin, the IDL bin nearest in parallactic angle, against the
  IDL chop with the same number and with the other number (to catch swapped chop labels):
  median and maximum absolute difference in degrees;
* for each median PSF: the Pearson correlation of the central 64 x 64 pixels, and the FWHM and
  center of a circular Gaussian plus constant fitted to the 41 x 41 pixels around the maximum,
  in pixels (the star is placed at (radius - 0.5, radius - 0.5));
* the frame comparison: for the port bins whose nearest IDL bin (same chop number) is within
  0.05 degrees, the correlation of their central 128 x 128 pixels: median, 5th percentile,
  minimum and count;
* the clean selection: the share of IDL clean bins with a port clean bin within 0.01 degrees,
  and the reverse;
* the centering consistency of each version: the core center of every binned frame (a
  Gaussian fit to its central 128 x 128 pixels, ``core_centroids``), its median, its robust
  scatter (1.4826 x the median absolute deviation) and the share of bins more than 0.5 pixel
  from the median, in x and y;
* with ``--worst K``, the K frame pairs of lowest correlation: port and IDL bin index, angle,
  correlation, whether each version's clean keeps the bin (a clean bin within 0.01 degrees),
  and the port's frame tags ``corrs`` and ``noises`` for the bin.

Usage: compare_idl_products.py PORT_DIR IDL_DIR NIGHT [--sides left right] [--worst K]
"""
import argparse
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
from astropy.io import fits

HALF = 64       # half size of the central region compared frame by frame, pixels


def nframes(path: str) -> int:
    """Return the number of frames of a FITS cube from its header.

    Parameters
    ----------
    path : str
        FITS file with a 3-D primary image.

    Returns
    -------
    n : int
        NAXIS3, or -1 when the file is missing.
    """
    if not os.path.exists(path):
        return -1
    return int(fits.getheader(path).get("NAXIS3", 1))


def central(path: str, half: int = HALF) -> np.ndarray:
    """Read the central pixels of every frame of a FITS cube without loading the whole cube.

    Parameters
    ----------
    path : str
        FITS file with a 3-D primary image of n frames of ny x nx pixels.
    half : int, default 64
        Half size of the region, pixels.

    Returns
    -------
    cut : ndarray, shape (n, 2 half, 2 half)
        Float64 pixels ``[ny//2 - half : ny//2 + half, nx//2 - half : nx//2 + half]`` of each
        frame.
    """
    with fits.open(path, memmap=True) as h:
        _, ny, nx = h[0].shape
        y0, x0 = ny // 2 - half, nx // 2 - half
        return np.asarray(h[0].section[:, y0:y0 + 2 * half, x0:x0 + 2 * half], dtype=np.float64)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two images over the pixels finite in both.

    Parameters
    ----------
    a, b : ndarray, shape (ny, nx)
        Images, where ny and nx are their size.

    Returns
    -------
    r : float
        Correlation coefficient; NaN with fewer than 10 common finite pixels.
    """
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def gauss_fit(img: np.ndarray, half: int = 20) -> Tuple[float, float, float]:
    """Fit a circular Gaussian plus constant around the maximum of the central 100 x 100 pixels.

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        PSF image, where ny and nx are its size.
    half : int, default 20
        Half size of the fitted box, pixels; the box is clipped at the image's edges.

    Returns
    -------
    fwhm : float
        Full width at half maximum, pixels.
    xc, yc : float
        Fitted center in the image's pixel coordinates.
    """
    from scipy.optimize import curve_fit
    ny, nx = img.shape
    cy, cx = ny // 2, nx // 2
    box = np.nan_to_num(img[cy - 50:cy + 50, cx - 50:cx + 50], nan=-np.inf)
    iy, ix = np.unravel_index(int(np.argmax(box)), box.shape)
    iy, ix = iy + cy - 50, ix + cx - 50
    y0, y1 = max(iy - half, 0), min(iy + half + 1, ny)
    x0, x1 = max(ix - half, 0), min(ix + half + 1, nx)
    sub = img[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    ok = np.isfinite(sub)

    def model(xy, amp, x0, y0, sig, off):
        """Circular Gaussian plus constant at the points xy = (x, y)."""
        return amp * np.exp(-((xy[0] - x0) ** 2 + (xy[1] - y0) ** 2) / (2 * sig ** 2)) + off

    p0 = (float(np.nanmax(sub)), float(ix), float(iy), 8.0, 0.0)
    p, _ = curve_fit(model, (xx[ok], yy[ok]), sub[ok], p0=p0, maxfev=20000)
    return 2.3548 * abs(p[3]), p[1], p[2]


def core_centroids(cut: np.ndarray) -> np.ndarray:
    """Find the core center of every frame of a central cut with a Gaussian fit.

    Parameters
    ----------
    cut : ndarray, shape (n, 2 half, 2 half)
        Central pixels of n frames, the star near the middle (``central``).

    Returns
    -------
    xy : ndarray, shape (n, 2)
        Center (x, y) of each frame in the cut's pixels, from ``gauss_fit`` (a circular Gaussian
        plus constant over the 41 x 41 pixels around the maximum); -1 where the fit fails.
    """
    xy = np.full((cut.shape[0], 2), -1.0)
    for i, f in enumerate(cut):
        try:
            _, xy[i, 0], xy[i, 1] = gauss_fit(f)
        except (RuntimeError, ValueError):
            pass
    return xy


def scatter_line(name: str, xy: np.ndarray) -> str:
    """Summarize the centroids of one version's binned frames.

    Parameters
    ----------
    name : str
        Version name for the line.
    xy : ndarray, shape (n, 2)
        Centers from ``core_centroids``, pixels.

    Returns
    -------
    line : str
        Median, robust scatter and share of bins more than 0.5 pixel from the median, x and y.
    """
    ok = (xy[:, 0] >= 0) & (xy[:, 1] >= 0)
    v = xy[ok]
    med = np.median(v, axis=0)
    mad = 1.4826 * np.median(np.abs(v - med), axis=0)
    far = np.mean(np.abs(v - med) > 0.5, axis=0)
    return (f"{name} median ({med[0]:.2f}, {med[1]:.2f}), scatter ({mad[0]:.2f}, {mad[1]:.2f}) px, "
            f"beyond 0.5 px ({far[0]:.1%}, {far[1]:.1%}), {ok.sum()} bins")


def nearest(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Find, for each value of a, the nearest value of b.

    Parameters
    ----------
    a : ndarray, shape (m,)
        Query values, where m is their number.
    b : ndarray, shape (k,)
        Reference values, where k is their number (k > 0).

    Returns
    -------
    idx : ndarray of int, shape (m,)
        Index in b of the nearest value.
    diff : ndarray, shape (m,)
        Absolute difference to it.
    """
    order = np.argsort(b)
    bs = b[order]
    pos = np.clip(np.searchsorted(bs, a), 1, max(bs.size - 1, 1))
    lo, hi = bs[pos - 1], bs[np.minimum(pos, bs.size - 1)]
    pick = np.where(np.abs(a - lo) <= np.abs(a - hi), pos - 1, np.minimum(pos, bs.size - 1))
    return order[pick], np.abs(a - bs[pick])


def side_chop(port: str, idl: str, night: str, side: str, cn: int, worst: int = 0) -> List[str]:
    """Compare one side and chop; see the module docstring.

    Parameters
    ----------
    port, idl : str
        Folders of the Python and IDL products.
    night : str
        Night name, the products' prefix.
    side : str
        ``left`` or ``right``.
    cn : int
        Chop number, 1 or 2.
    worst : int, default 0
        Number of lowest-correlation frame pairs to list.

    Returns
    -------
    lines : list of str
        Report lines.
    """
    from scipy.io import readsav
    pre = f"{night}_{side}"
    other = 3 - cn
    out = [f"== {side} chop {'AB'[cn - 1]} (cube{cn})"]
    pa = np.asarray(fits.getdata(os.path.join(port, f"{pre}_angles{cn}_bin.fits")), dtype=float)
    sv = readsav(os.path.join(idl, f"{pre}_angles{cn}_bin.sav"))
    ia = np.asarray(sv[f"angles{cn}"], dtype=float)
    ia_other = np.asarray(readsav(os.path.join(idl, f"{pre}_angles{other}_bin.sav"))[f"angles{other}"], dtype=float)
    pcl = np.asarray(fits.getdata(os.path.join(port, f"{pre}_angles{cn}_clean_bin.fits")), dtype=float)
    icl = np.asarray(readsav(os.path.join(idl, f"{pre}_angles{cn}_clean_bin.sav"))[f"angles{cn}"], dtype=float)
    tags = fits.getdata(os.path.join(port, f"{pre}_frametags{cn}_bin.fits"))
    pexp = float(np.sum(tags["exptime_s"])) if "exptime_s" in tags.names else float("nan")
    pcoadd = sum(1 for _ in open(os.path.join(port, f"{pre}_coadds{cn}_bin.csv"))) - 1
    out.append(f"  bins: port {nframes(os.path.join(port, f'{pre}_cube{cn}_align_cen_bin.fits'))} ({pa.size} angles), "
               f"IDL {nframes(os.path.join(idl, f'{pre}_cube{cn}_align_cen_bin.fits'))} ({ia.size} angles)")
    out.append(f"  kept unbinned frames: port {pcoadd}, IDL {np.asarray(sv[f'coadds{cn}']).size}; "
               f"exposure s: port {pexp:.1f}, IDL {float(np.sum(sv[f'exptimes{cn}'])):.1f}")
    out.append(f"  clean bins: port {nframes(os.path.join(port, f'{pre}_cube{cn}_align_cen_clean_bin.fits'))} "
               f"({pcl.size} angles), IDL {nframes(os.path.join(idl, f'{pre}_cube{cn}_align_cen_clean_bin.fits'))} "
               f"({icl.size} angles)")
    j_same, d_same = nearest(pa, ia)
    _, d_other = nearest(pa, ia_other)
    out.append(f"  angle match, port bin to nearest IDL bin: same chop median {np.median(d_same):.4f} deg "
               f"(max {d_same.max():.3f}); other chop median {np.median(d_other):.4f} deg (max {d_other.max():.3f})")
    out.append(f"  angle ranges: port {pa.min():.3f}..{pa.max():.3f}, IDL {ia.min():.3f}..{ia.max():.3f}")
    for kind in ("median_PSF", "clean_median_PSF"):
        fp = fits.getdata(os.path.join(port, f"{pre}_cube{cn}_align_cen_{kind}.fits")).astype(float)
        fi = fits.getdata(os.path.join(idl, f"{pre}_cube{cn}_align_cen_{kind}.fits")).astype(float)
        c = fp.shape[0] // 2
        r64 = corr(fp[c - 32:c + 32, c - 32:c + 32], fi[c - 32:c + 32, c - 32:c + 32])
        try:
            wp, xp, yp = gauss_fit(fp)
            wi, xi, yi = gauss_fit(fi)
            fit = (f"FWHM port {wp:.2f} px, IDL {wi:.2f} px; center port ({xp:.2f}, {yp:.2f}), "
                   f"IDL ({xi:.2f}, {yi:.2f})")
        except (RuntimeError, ValueError) as err:
            fit = f"Gaussian fit failed: {err}"
        out.append(f"  {kind}: central 64x64 correlation {r64:.4f}; {fit}")
    ipath = os.path.join(idl, f"{pre}_cube{cn}_align_cen_bin_replace.fits")
    if not os.path.exists(ipath):
        ipath = os.path.join(idl, f"{pre}_cube{cn}_align_cen_bin.fits")
    cp = central(os.path.join(port, f"{pre}_cube{cn}_align_cen_bin.fits"))
    ci = central(ipath)
    pairs = [(i, int(j)) for i, (j, d) in enumerate(zip(j_same, d_same)) if d < 0.05 and i < cp.shape[0] and j < ci.shape[0]]
    rall = np.array([corr(cp[i], ci[j]) for i, j in pairs])
    rs = rall[np.isfinite(rall)]
    if rs.size:
        out.append(f"  frames (central {2 * HALF}x{2 * HALF}, vs {os.path.basename(ipath)}): correlation median "
                   f"{np.median(rs):.4f}, 5th percentile {np.percentile(rs, 5):.4f}, min {rs.min():.4f}, "
                   f"{rs.size} pairs within 0.05 deg")
    out.append("  centering: " + scatter_line("port", core_centroids(cp)) + "; "
               + scatter_line("IDL", core_centroids(ci)))
    if worst and rs.size:
        for k in np.argsort(np.nan_to_num(rall, nan=-2.0))[:worst]:
            i, j = pairs[k]
            pk = bool(pcl.size and np.min(np.abs(pcl - pa[i])) < 0.01)
            ik = bool(icl.size and np.min(np.abs(icl - ia[j])) < 0.01)
            out.append(f"    port bin {i} / IDL bin {j}, angle {pa[i]:.3f}: r = {rall[k]:.4f}; clean keeps: port {pk}, "
                       f"IDL {ik}; port tags corrs {tags['corrs'][i]:.4f}, noises {tags['noises'][i]:.3f}")
    if icl.size and pcl.size:
        _, d1 = nearest(icl, pcl)
        _, d2 = nearest(pcl, icl)
        out.append(f"  clean selection: {np.mean(d1 < 0.01):.1%} of IDL clean bins have a port clean bin within "
                   f"0.01 deg; {np.mean(d2 < 0.01):.1%} of port clean bins have an IDL one")
    return out


def main(argv: Optional[List[str]] = None) -> None:
    """Run the comparison; see the module docstring.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments; default ``sys.argv[1:]``.

    Returns
    -------
    None
        Prints the report.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("port_dir")
    ap.add_argument("idl_dir")
    ap.add_argument("night")
    ap.add_argument("--sides", nargs="+", default=["left", "right"])
    ap.add_argument("--worst", type=int, default=0, help="list this many lowest-correlation frame pairs")
    a = ap.parse_args(argv)
    for side in a.sides:
        for cn in (1, 2):
            print("\n".join(side_chop(a.port_dir, a.idl_dir, a.night, side, cn, a.worst)), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
