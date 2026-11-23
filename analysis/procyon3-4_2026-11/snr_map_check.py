"""Check a KLIP-TPE final S/N map for candidates, on a ring, and against its injections.

usage: python analysis/procyon3-4_2026-11/snr_map_check.py SNR_MAP [--rmin RMIN] [--rmax RMAX]
       [--thresh THRESH] [--ring-r RING_R] [--ring-hw RING_HW] [--ring-pa PA [PA ...]]
       [--fwhm FWHM] [--pxscale PXSCALE]
       [--inj-map INJ_MAP --verify-report REPORT] [--search-px SEARCH_PX]

The star sits at ``((nx - 1) / 2, (ny - 1) / 2)``, KLIP-TPE's convention, and position angles are
east of north with north up and east left, so a pixel at offset (dx, dy) from the star has
PA = atan2(-dx, dy). With ``--inj-map`` and ``--verify-report`` the injected sources the stitched
verify lists are looked up in the injected S/N map, which checks that convention: each source's
peak should fall within a few pixels of where its separation and PA put it.

Parameters
----------
SNR_MAP : str
    The final S/N map (``klip_stitched_snr.fits``), 2-D, NaN outside the reduced region.
--rmin : float, default 0.30
    Inner separation of the region searched for candidates, in arcsec.
--rmax : float, default 2.45
    Outer separation of that region, in arcsec.
--thresh : float, default 4.0
    S/N above which a pixel counts as a candidate pixel (and below minus which as a negative one).
--ring-r : float, default 25.5
    Radius of the ring checked, in pixels (25.5 px is the Procyon_3-4 position in Kevin's log).
--ring-hw : float, default 1.5
    Half width of that ring, in pixels (1.5 px gives Procyon_4's earlier ring check, max 1.77).
--ring-pa : float, optional
    Position angles on the ring to report, in degrees east of north: the maximum and mean S/N
    of the ring's pixels within half a FWHM of arc of each.
--fwhm : float, default 15.65
    PSF FWHM, in pixels (the run's ``fwhm(px)``); sets the arc of ``--ring-pa``.
--pxscale : float, default 0.0179
    Pixel scale, in arcsec per pixel (NOMIC).
--inj-map : str, optional
    The injected S/N map (``klip_stitched_snr_inj.fits``).
--verify-report : str, optional
    The stitched verify report (``stitched_verify_report.txt``) whose injection calibration lists
    the sources (``rho=... PA=... c=...``).
--search-px : float, default 8.0
    Radius around each source's expected position searched for its peak, in pixels (about half
    a FWHM at NOMIC's 15.7 px).

Returns
-------
stdout : text
    The region's pixel count, maximum (with its separation and PA), minimum, standard deviation and
    the number of pixels beyond the threshold; the ring's maximum and minimum with their PA, and
    its maximum and mean near each ``--ring-pa``; and,
    with the injected map, each source's peak S/N and its offset from the expected position.
"""
import argparse
import re
import sys
from typing import List, Optional, Sequence, Tuple

import numpy as np
from astropy.io import fits


def polar_grid(shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Give each pixel's separation from the star and its position angle.

    Parameters
    ----------
    shape : tuple of int
        Image shape (ny, nx).

    Returns
    -------
    r : ndarray, shape (ny, nx)
        Separation from the star at ((nx - 1) / 2, (ny - 1) / 2), in pixels.
    pa : ndarray, shape (ny, nx)
        Position angle east of north (north up, east left), in degrees in [0, 360).
    """
    ny, nx = shape
    y, x = np.indices(shape, dtype=float)
    dx = x - (nx - 1) / 2.0
    dy = y - (ny - 1) / 2.0
    r = np.hypot(dx, dy)
    pa = np.degrees(np.arctan2(-dx, dy)) % 360.0
    return r, pa


def region_stats(snr: np.ndarray, r: np.ndarray, pa: np.ndarray, rmin_px: float, rmax_px: float,
                 thresh: float) -> dict:
    """Summarize the S/N map between two separations.

    Parameters
    ----------
    snr : ndarray, shape (ny, nx)
        The S/N map.
    r : ndarray, shape (ny, nx)
        Separation of each pixel, in pixels.
    pa : ndarray, shape (ny, nx)
        Position angle of each pixel, in degrees.
    rmin_px : float
        Inner separation, in pixels.
    rmax_px : float
        Outer separation, in pixels.
    thresh : float
        S/N beyond which a pixel is counted (above thresh, and below -thresh).

    Returns
    -------
    stats : dict
        ``n`` finite pixels; ``max``, ``r_max`` (px), ``pa_max`` (deg); ``min``, ``r_min``,
        ``pa_min``; ``std``; ``n_above`` and ``n_below``.
    """
    sel = (r >= rmin_px) & (r <= rmax_px) & np.isfinite(snr)
    v = snr[sel]
    i_max = np.argmax(v)
    i_min = np.argmin(v)
    return {"n": int(sel.sum()), "max": float(v[i_max]), "r_max": float(r[sel][i_max]),
            "pa_max": float(pa[sel][i_max]), "min": float(v[i_min]), "r_min": float(r[sel][i_min]),
            "pa_min": float(pa[sel][i_min]), "std": float(np.std(v)),
            "n_above": int((v > thresh).sum()), "n_below": int((v < -thresh).sum())}


def read_sources(report: str) -> List[Tuple[float, float, float]]:
    """Read the injected sources from a stitched verify report.

    Parameters
    ----------
    report : str
        Path of ``stitched_verify_report.txt``.

    Returns
    -------
    sources : list of tuple of float
        (separation in arcsec, PA in degrees, contrast) of each source in the injection
        calibration, in the report's order.
    """
    pat = re.compile(r'rho=\s*([0-9.]+)"\s+PA=\s*([0-9.]+)\s+c=\s*([0-9.Ee+-]+)')
    out = []
    with open(report) as f:
        for line in f:
            m = pat.search(line)
            if m:
                out.append((float(m.group(1)), float(m.group(2)), float(m.group(3))))
    return out


def find_peaks(inj: np.ndarray, sources: Sequence[Tuple[float, float, float]], pxscale: float,
               search_px: float) -> List[Tuple[float, float, float, float, float]]:
    """Find each injected source's peak in the injected S/N map.

    Parameters
    ----------
    inj : ndarray, shape (ny, nx)
        The injected S/N map.
    sources : sequence of tuple of float
        (separation in arcsec, PA in degrees, contrast) of each source.
    pxscale : float
        Pixel scale, in arcsec per pixel.
    search_px : float
        Radius around the expected position searched, in pixels.

    Returns
    -------
    peaks : list of tuple of float
        (separation in arcsec, PA in degrees, contrast, peak S/N, offset of the peak from the
        expected position in pixels) per source.
    """
    ny, nx = inj.shape
    y, x = np.indices(inj.shape, dtype=float)
    out = []
    for rho, pa, c in sources:
        rp = rho / pxscale
        ex = (nx - 1) / 2.0 - rp * np.sin(np.radians(pa))
        ey = (ny - 1) / 2.0 + rp * np.cos(np.radians(pa))
        near = (np.hypot(x - ex, y - ey) <= search_px) & np.isfinite(inj)
        vals = np.where(near, inj, -np.inf)
        k = np.unravel_index(np.argmax(vals), inj.shape)
        out.append((rho, pa, c, float(inj[k]), float(np.hypot(x[k] - ex, y[k] - ey))))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Read the maps and print the checks.

    Parameters
    ----------
    argv : sequence of str or None, default None
        The command-line arguments (see the module's usage); None reads ``sys.argv``.

    Returns
    -------
    status : int
        0 on success.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("snr_map")
    ap.add_argument("--rmin", type=float, default=0.30)
    ap.add_argument("--rmax", type=float, default=2.45)
    ap.add_argument("--thresh", type=float, default=4.0)
    ap.add_argument("--ring-r", type=float, default=25.5)
    ap.add_argument("--ring-hw", type=float, default=1.5)
    ap.add_argument("--ring-pa", type=float, nargs="+", default=[])
    ap.add_argument("--fwhm", type=float, default=15.65)
    ap.add_argument("--pxscale", type=float, default=0.0179)
    ap.add_argument("--inj-map")
    ap.add_argument("--verify-report")
    ap.add_argument("--search-px", type=float, default=8.0)
    a = ap.parse_args(argv)
    snr = fits.getdata(a.snr_map).astype(float)
    r, pa = polar_grid(snr.shape)
    s = region_stats(snr, r, pa, a.rmin / a.pxscale, a.rmax / a.pxscale, a.thresh)
    print(f"S/N map: {a.snr_map}")
    print(f"star at ({(snr.shape[1] - 1) / 2:.1f}, {(snr.shape[0] - 1) / 2:.1f}); PA east of north, "
          f"north up, east left; {a.pxscale} arcsec per pixel")
    print(f"region {a.rmin:.2f}-{a.rmax:.2f}\": {s['n']} finite pixels; max {s['max']:.2f} at "
          f"{s['r_max'] * a.pxscale:.3f}\" PA {s['pa_max']:.0f}; min {s['min']:.2f} at "
          f"{s['r_min'] * a.pxscale:.3f}\" PA {s['pa_min']:.0f}; std {s['std']:.2f}; "
          f"{s['n_above']} pixels above {a.thresh:g}, {s['n_below']} below -{a.thresh:g}")
    ring = (np.abs(r - a.ring_r) <= a.ring_hw) & np.isfinite(snr)
    v = snr[ring]
    print(f"ring {a.ring_r:g} +/- {a.ring_hw:g} px ({a.ring_r * a.pxscale:.3f}\"): {int(ring.sum())} "
          f"pixels; max {v.max():.2f} at PA {pa[ring][np.argmax(v)]:.0f}; "
          f"min {v.min():.2f} at PA {pa[ring][np.argmin(v)]:.0f}")
    half_arc = np.degrees(0.5 * a.fwhm / a.ring_r)
    for p0 in a.ring_pa:
        d = np.abs((pa[ring] - p0 + 180.0) % 360.0 - 180.0)
        arc = v[d <= half_arc]
        print(f"ring near PA {p0:g} (+/- {half_arc:.1f} deg, half a FWHM of arc): {arc.size} pixels; "
              f"max {arc.max():.2f}, mean {arc.mean():.2f}")
    if a.inj_map and a.verify_report:
        inj = fits.getdata(a.inj_map).astype(float)
        if inj.ndim > 2:
            inj = inj[0]
        for rho, p, c, peak, off in find_peaks(inj, read_sources(a.verify_report), a.pxscale,
                                               a.search_px):
            print(f"injected rho {rho:.3f}\" PA {p:.1f} c {c:.3e}: peak S/N {peak:.2f}, "
                  f"{off:.1f} px from the expected position")
    return 0


if __name__ == "__main__":
    sys.exit(main())
