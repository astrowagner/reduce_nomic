"""Measure suspected point sources at given positions in KLIP-TPE and IDL-predecessor images.

usage: python analysis/procyon3-4_2026-11/feature_check.py OUT_PREFIX --pos RHO PA [--pos RHO PA ...]
       IMAGE [IMAGE ...] [--inj LABEL=PATH[:CONTRAST] ...] [--center RHO PA] [--rmin RMIN]
       [--rmax RMAX] [--search-fwhm SEARCH] [--fwhm FWHM] [--pxscale PXSCALE] [--ncol NCOL]
       [--half-width HW] [--title TITLE] [--smooth-fwhm SMOOTH] [--fakes N] [--seed SEED]

Each image gets a per-pixel Mawet S/N map over the annulus RMIN-RMAX, built the way the IDL
predecessor's ``contrastcurve_nomic.pro`` measures one position: the flux in an aperture of radius
FWHM/2 at the position, against the apertures of radius FWHM/2 spaced one FWHM apart on the ring
at its separation, the position's own aperture and its two neighbors left out, with Mawet et al.
(2014)'s small-sample factor,

    S/N = (F - mean(F_ring)) / (std(F_ring, ddof=1) * sqrt(1 + 1/n)),

where F is the position's aperture flux, F_ring the ring's aperture fluxes and n their number.
Ring apertures that touch a NaN pixel are left out. A suspect's S/N is the map's maximum within
SEARCH FWHM of its position. Its rank counts the map's local maxima in the annulus (at least one
FWHM apart) at or above it, which says how unusual it is among the image's other bumps.

Each IMAGE is ``[idl:]LABEL=PATH[:PLANE]``, or ``-`` for an empty panel in the figure. LABEL
names it in the outputs. PLANE picks a plane of a cube, by the value of a ``NIGHTn`` header keyword
(``left_A``, ``P3_left_B``) or by its index from 0. The star sits at
``((nx - 1) / 2, (ny - 1) / 2)``, KLIP-TPE's convention, or, with the ``idl:`` prefix, at
``(nx / 2, ny / 2)``, the IDL predecessor's (500, 500) in its 1000 px frames. Either way north is up
and east left, so a pixel at offset (dx, dy) from the star has PA = atan2(-dx, dy), east of north.

With ``--inj LABEL=PATH[:CONTRAST]`` two more things are measured in image LABEL. PATH is its
injected version (``klip_stitched_inj.fits``), whose header lists the injected sources (``INJRHOk``,
``INJPAk``, in arcsec and degrees, and ``INJCONk``, their contrasts, where the run wrote them);
CONTRAST is the contrast they were injected at, needed only where the header has no ``INJCONk``.
The sources within RMIN-RMAX, measured in the injected image minus the clean one, give
1. the flux per unit contrast at those separations, and each suspect's contrast is its aperture
   flux over that;
2. the shape of a point source after the reduction: the radius where the azimuthal mean of each
   source's profile first drops below zero (the high-pass filter's negative ring), averaged over
   the sources, beside the same radius for each suspect. Each profile is centered on the peak of
   the image smoothed by half a FWHM (within 3 px of a source's listed position, within a quarter
   FWHM of a suspect's S/N peak).

Parameters
----------
OUT_PREFIX : str
    Path and stem of the outputs; ``.csv``, ``.txt`` and ``.png`` are added. Existing outputs are
    not overwritten.
IMAGE : str
    The images to measure, as ``[idl:]LABEL=PATH[:PLANE]``, in the order the figure shows them;
    ``-`` leaves a panel empty.
--pos : two floats, repeatable
    Separation (arcsec) and position angle (degrees east of north) of a suspect; at least one.
    Suspects are numbered from 1 in the order given.
--inj : str, optional
    ``LABEL=PATH[:CONTRAST]`` for an image whose injected version gives the flux scale and the
    point-source shape; repeatable.
--center : two floats, optional
    Separation (arcsec) and position angle (degrees) of the cutouts' center; by default the mean
    offset of the suspects.
--rmin : float, default 1.65
    Inner separation of the annulus the S/N maps and the ranks cover, in arcsec. The default keeps
    half a FWHM inside the 1.50-2.50" zone of the runs' annulus 3, whose edges carry KLIP's edge
    effects.
--rmax : float, default 2.35
    Outer separation of that annulus, in arcsec.
--search-fwhm : float, default 0.5
    Radius around each suspect's position searched for its peak, in FWHM.
--fwhm : float, default 15.653
    PSF FWHM, in pixels (KLIP-TPE's ``fwhm(px)`` for NOMIC at 11.1 um).
--pxscale : float, default 0.0179
    Pixel scale, in arcsec per pixel.
--ncol : int, default 4
    Panels per row of the figure.
--half-width : float, default 0.9
    Half width of each panel's cutout, in arcsec.
--title : str, default ""
    Figure title.
--smooth-fwhm : float, default 0.5
    FWHM of the Gaussian each cutout is smoothed with, in units of FWHM; 0 shows the images as they
    are. The measurements use the images as they are either way.
--fakes : int, default 0
    Fake point sources per suspect for the shape check, in each image with ``--inj``: copies of the
    reduction's own point source at the suspect's S/N put into the clean image one at a time (see
    :func:`fake_zero_radii`), whose zero-crossing radii show the spread a real source that faint
    would have. 0 skips the check.
--seed : int, default 1
    Seed of the fakes' random positions.

Returns
-------
OUT_PREFIX.csv : file
    One row per image and suspect: label, file, plane, the suspect's number and position, its peak
    separation, PA and offset from the position, its S/N and that at the position itself, its
    aperture flux, the annulus' robust pixel sigma, the rank and the number of local maxima, the
    annulus' highest local maximum with its position, and, where ``--inj`` applies, the contrast,
    the number of injected sources used, the zero-crossing radii of the suspect's and the sources'
    profiles and, with ``--fakes``, the number of fakes, the median and 16th and 84th percentiles
    of their zero-crossing radii, and the fraction at or below the suspect's.
OUT_PREFIX.txt : file
    The same as tables, one per suspect, with each image's five highest local maxima.
OUT_PREFIX.png : file
    One cutout per image, smoothed (``--smooth-fwhm``) and in units of the smoothed annulus'
    robust sigma, with numbered ticks around each suspect's peak.
"""
import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from astropy.io import fits
from scipy import ndimage as ndi

#: tick colors of suspects 1, 2, 3, ... in the figure
TICK_COLORS = ("#4dd0e1", "#b2ff59", "#ffffff", "#ff80ab")


@dataclass
class ImageSpec:
    """One image to measure, as given on the command line.

    Attributes
    ----------
    label : str
        Name of the image in the outputs.
    path : str
        FITS file.
    plane : str or None
        Plane of a cube, by ``NIGHTn`` header value or index from 0; None for a 2-D image.
    idl : bool
        True for the IDL predecessor's centering (star at nx/2, ny/2), False for KLIP-TPE's.
    """

    label: str
    path: str
    plane: Optional[str]
    idl: bool


def parse_image_spec(spec: str) -> ImageSpec:
    """Read ``[idl:]LABEL=PATH[:PLANE]``.

    Parameters
    ----------
    spec : str
        The command-line item.

    Returns
    -------
    ImageSpec
        The parsed item.

    Raises
    ------
    ValueError
        If there is no ``=`` or the label or path is empty.
    """
    idl = spec.startswith("idl:")
    body = spec[4:] if idl else spec
    label, sep, rest = body.partition("=")
    if not sep or not label or not rest:
        raise ValueError(f"expected [idl:]LABEL=PATH[:PLANE], got {spec!r}")
    path, plane = rest, None
    if ":" in rest:
        head, _, tail = rest.rpartition(":")
        if head.lower().endswith((".fits", ".fits.gz", ".fit")):
            path, plane = head, tail
    return ImageSpec(label, path, plane, idl)


def load_image(spec: ImageSpec) -> Tuple[np.ndarray, str]:
    """Read one image, picking the plane of a cube.

    Parameters
    ----------
    spec : ImageSpec
        The image.

    Returns
    -------
    img : ndarray, shape (ny, nx)
        The image as float, NaN where there is no data.
    plane : str
        The plane read ("" for a 2-D image).

    Raises
    ------
    ValueError
        If a cube has no plane given, the plane is not found, or the data is not 2-D or 3-D.
    """
    with fits.open(spec.path) as h:
        data = np.asarray(h[0].data, dtype=float)
        hdr = h[0].header
    if data.ndim == 2:
        if spec.plane is not None:
            raise ValueError(f"{spec.label}: a plane was given for a 2-D image")
        return data, ""
    if data.ndim != 3:
        raise ValueError(f"{spec.label}: expected a 2-D image or a cube, got shape {data.shape}")
    if spec.plane is None:
        raise ValueError(f"{spec.label}: a cube needs a plane")
    names = [str(hdr.get(f"NIGHT{k + 1}", "")) for k in range(data.shape[0])]
    if spec.plane in names:
        k = names.index(spec.plane)
    elif spec.plane.isdigit() and int(spec.plane) < data.shape[0]:
        k = int(spec.plane)
    else:
        raise ValueError(f"{spec.label}: plane {spec.plane!r} not in {names}")
    return data[k], names[k] or str(k)


def star_center(shape: Tuple[int, int], idl: bool) -> Tuple[float, float]:
    """Return the star's pixel position.

    Parameters
    ----------
    shape : tuple of int
        Image shape (ny, nx).
    idl : bool
        True for the IDL predecessor's convention (nx/2, ny/2), False for KLIP-TPE's
        ((nx - 1)/2, (ny - 1)/2).

    Returns
    -------
    cy, cx : float
        Row and column of the star, in pixels from 0.
    """
    ny, nx = shape
    return (ny / 2.0, nx / 2.0) if idl else ((ny - 1) / 2.0, (nx - 1) / 2.0)


def offset_of(rho_px: float, pa_deg: float) -> Tuple[float, float]:
    """Return the pixel offset of a position, north up and east left.

    Parameters
    ----------
    rho_px : float
        Separation, in pixels.
    pa_deg : float
        Position angle, in degrees east of north.

    Returns
    -------
    dy, dx : float
        Row and column offsets from the star, in pixels.
    """
    t = np.deg2rad(pa_deg)
    return rho_px * np.cos(t), -rho_px * np.sin(t)


def aperture_kernel(radius: float, sub: int = 8) -> np.ndarray:
    """Make a disk of fractional pixel weights.

    Parameters
    ----------
    radius : float
        Disk radius, in pixels.
    sub : int, default 8
        Subpixels per pixel side used to compute each pixel's covered fraction.

    Returns
    -------
    ndarray, shape (m, m)
        Covered fraction of each pixel, with m = 2 * ceil(radius) + 1, centered on the middle pixel.
    """
    h = int(np.ceil(radius))
    off = (np.arange(sub) + 0.5) / sub - 0.5
    g = np.arange(-h, h + 1)
    yy = (g[:, None, None, None] + off[None, None, :, None])
    xx = (g[None, :, None, None] + off[None, None, None, :])
    inside = (yy ** 2 + xx ** 2) <= radius ** 2
    return inside.mean(axis=(2, 3))


def aperture_maps(img: np.ndarray, radius: float) -> Tuple[np.ndarray, np.ndarray]:
    """Aperture flux and NaN contact for an aperture centered on every pixel.

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        The image, NaN where there is no data.
    radius : float
        Aperture radius, in pixels.

    Returns
    -------
    flux : ndarray, shape (ny, nx)
        Sum of the image over the aperture centered on each pixel, NaN pixels counted as 0.
    bad : ndarray of bool, shape (ny, nx)
        True where that aperture touches a NaN pixel or the image edge.
    """
    k = aperture_kernel(radius)
    fin = np.isfinite(img)
    flux = ndi.convolve(np.where(fin, img, 0.0), k, mode="constant", cval=0.0)
    touch = ndi.convolve((~fin).astype(float), (k > 0).astype(float), mode="constant", cval=1.0)
    return flux, touch > 0


def snr_map(img: np.ndarray, center: Tuple[float, float], fwhm: float, rmin_px: float,
            rmax_px: float) -> Tuple[np.ndarray, np.ndarray]:
    """Per-pixel Mawet S/N over an annulus, with the IDL predecessor's apertures and ring.

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        The image, NaN where there is no data.
    center : tuple of float
        Star position (cy, cx), in pixels.
    fwhm : float
        PSF FWHM, in pixels; apertures have radius fwhm/2 and are spaced fwhm apart on each ring.
    rmin_px, rmax_px : float
        Separations covered, in pixels.

    Returns
    -------
    snr : ndarray, shape (ny, nx)
        S/N of an aperture centered on each pixel; NaN outside the annulus, where the pixel's own
        aperture touches a NaN pixel, or where fewer than 3 ring apertures are clean.
    flux : ndarray, shape (ny, nx)
        Aperture flux of each pixel (see :func:`aperture_maps`).
    """
    flux, bad = aperture_maps(img, fwhm / 2.0)
    cy, cx = center
    ny, nx = img.shape
    yy, xx = np.mgrid[:ny, :nx]
    r = np.hypot(yy - cy, xx - cx)
    th = np.arctan2(xx - cx, yy - cy)                  # any fixed origin works on a full ring
    snr = np.full(img.shape, np.nan)
    sel = (r >= rmin_px) & (r <= rmax_px) & ~bad
    n_ring = np.floor(2 * np.pi * r / fwhm).astype(int)
    for n in np.unique(n_ring[sel]):
        if n < 6:
            continue
        m = sel & (n_ring == n)
        ys, xs = np.nonzero(m)
        rr, tt = r[m], th[m]
        steps = np.arange(2, n - 1)                       # leave out the aperture and both neighbors
        ang = tt[:, None] + 2 * np.pi * steps[None, :] / n
        py = cy + rr[:, None] * np.cos(ang)
        px = cx + rr[:, None] * np.sin(ang)
        f = ndi.map_coordinates(flux, [py.ravel(), px.ravel()], order=1, mode="constant",
                                cval=np.nan).reshape(py.shape)
        b = ndi.map_coordinates(bad.astype(float), [py.ravel(), px.ravel()], order=1,
                                mode="constant", cval=1.0).reshape(py.shape) > 0
        f = np.where(b, np.nan, f)
        cnt = np.isfinite(f).sum(axis=1)
        mu = np.nanmean(f, axis=1)
        sd = np.nanstd(f, axis=1, ddof=1)
        val = (flux[ys, xs] - mu) / (sd * np.sqrt(1.0 + 1.0 / np.maximum(cnt, 1)))
        val[cnt < 3] = np.nan
        snr[ys, xs] = val
    return snr, flux


def local_maxima(snr: np.ndarray, fwhm: float) -> List[Tuple[float, int, int]]:
    """Local maxima of an S/N map, at least about one FWHM apart.

    Parameters
    ----------
    snr : ndarray, shape (ny, nx)
        S/N map, NaN where undefined.
    fwhm : float
        PSF FWHM, in pixels; a local maximum is the highest pixel within about one FWHM
        (a square of side 2 * floor(fwhm) + 1 pixels).

    Returns
    -------
    list of (float, int, int)
        (S/N, row, column) of each local maximum, highest first.
    """
    s = np.where(np.isfinite(snr), snr, -np.inf)
    size = int(2 * np.floor(fwhm) + 1)
    peak = (s == ndi.maximum_filter(s, size=size, mode="constant", cval=-np.inf)) & np.isfinite(snr)
    ys, xs = np.nonzero(peak)
    out = sorted(zip(snr[ys, xs].tolist(), ys.tolist(), xs.tolist()), reverse=True)
    return out


def polar(y: float, x: float, center: Tuple[float, float], pxscale: float) -> Tuple[float, float]:
    """Separation and PA of a pixel, north up and east left.

    Parameters
    ----------
    y, x : float
        Row and column, in pixels.
    center : tuple of float
        Star position (cy, cx), in pixels.
    pxscale : float
        Pixel scale, in arcsec per pixel.

    Returns
    -------
    rho : float
        Separation, in arcsec.
    pa : float
        Position angle, in degrees east of north, in [0, 360).
    """
    dy, dx = y - center[0], x - center[1]
    return float(np.hypot(dy, dx) * pxscale), float(np.rad2deg(np.arctan2(-dx, dy)) % 360.0)


def robust_sigma(v: np.ndarray) -> float:
    """1.4826 times the median absolute deviation of the finite values.

    Parameters
    ----------
    v : ndarray
        Values.

    Returns
    -------
    float
        The robust standard deviation.
    """
    v = v[np.isfinite(v)]
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def parse_inj(item: str) -> Tuple[str, str, Optional[float]]:
    """Read ``LABEL=PATH[:CONTRAST]``.

    Parameters
    ----------
    item : str
        The ``--inj`` value.

    Returns
    -------
    label : str
        Image the injected version belongs to.
    path : str
        The injected image.
    contrast : float or None
        Contrast of the injections, or None if not given.

    Raises
    ------
    ValueError
        If there is no ``=`` or the label or path is empty.
    """
    label, sep, rest = item.partition("=")
    if not sep or not label or not rest:
        raise ValueError(f"expected LABEL=PATH[:CONTRAST], got {item!r}")
    head, colon, tail = rest.rpartition(":")
    if colon:
        try:
            return label, head, float(tail)
        except ValueError:
            pass
    return label, rest, None


def flux_per_contrast(clean: np.ndarray, inj_path: str, contrast: Optional[float],
                      center: Tuple[float, float], fwhm: float, pxscale: float, rmin_px: float,
                      rmax_px: float) -> Tuple[float, int]:
    """Aperture flux per unit contrast from the injected sources in an annulus.

    Parameters
    ----------
    clean : ndarray, shape (ny, nx)
        The clean image.
    inj_path : str
        Its injected version, whose header lists the sources as ``INJRHOk`` (arcsec) and
        ``INJPAk`` (degrees east of north), k = 1 .. ``INJNSRC``, and their contrasts as
        ``INJCONk`` where the run wrote them.
    contrast : float or None
        Contrast of the sources whose header has no ``INJCONk``.
    center : tuple of float
        Star position (cy, cx), in pixels.
    fwhm : float
        PSF FWHM, in pixels; the aperture radius is fwhm/2.
    pxscale : float
        Pixel scale, in arcsec per pixel.
    rmin_px, rmax_px : float
        Separations of the sources used, in pixels.

    Returns
    -------
    per_contrast : float
        Mean over the sources of the aperture flux of (injected - clean) at each, divided by its
        contrast.
    n : int
        Number of sources used.

    Raises
    ------
    ValueError
        If the injected image's shape differs, no source falls in the annulus, or a source has
        neither ``INJCONk`` nor ``contrast``.
    """
    with fits.open(inj_path) as h:
        inj = np.asarray(h[0].data, dtype=float)
        hdr = h[0].header
    if inj.shape != clean.shape:
        raise ValueError(f"{inj_path}: shape {inj.shape} differs from the clean image's {clean.shape}")
    diff, _ = aperture_maps(inj - clean, fwhm / 2.0)
    vals = []
    for k in range(1, int(hdr.get("INJNSRC", 0)) + 1):
        rho_px = float(hdr[f"INJRHO{k}"]) / pxscale
        if not rmin_px <= rho_px <= rmax_px:
            continue
        c = hdr.get(f"INJCON{k}", contrast)
        if c is None:
            raise ValueError(f"{inj_path}: source {k} has no INJCON{k} and no contrast was given")
        dy, dx = offset_of(rho_px, float(hdr[f"INJPA{k}"]))
        f = float(ndi.map_coordinates(diff, [[center[0] + dy], [center[1] + dx]], order=1)[0])
        vals.append(f / float(c))
    if not vals:
        raise ValueError(f"{inj_path}: no injected source between {rmin_px:.1f} and {rmax_px:.1f} px")
    return float(np.mean(vals)), len(vals)


def smooth_nan(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-smooth an image, ignoring its NaN pixels.

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        The image, NaN where there is no data.
    sigma : float
        Gaussian sigma, in pixels; 0 returns a copy.

    Returns
    -------
    ndarray, shape (ny, nx)
        The smoothed image (finite pixels weighted by the kernel and renormalized), NaN where the
        input is NaN.
    """
    if sigma <= 0:
        return img.copy()
    fin = np.isfinite(img)
    num = ndi.gaussian_filter(np.where(fin, img, 0.0), sigma, mode="constant")
    den = ndi.gaussian_filter(fin.astype(float), sigma, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[~fin] = np.nan
    return out


def profile_zero_radius(img: np.ndarray, y: int, x: int, rmax_px: float) -> float:
    """Radius where the azimuthal mean of an image around a pixel first drops below zero.

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        The image, NaN where there is no data.
    y, x : int
        Row and column of the profile's center.
    rmax_px : float
        Largest radius examined, in pixels.

    Returns
    -------
    float
        The radius, in pixels, interpolated linearly between the means of the 1 px rings
        [k, k + 1) taken at their mid-radii k + 0.5; NaN if the mean stays at or above zero.
    """
    h = int(np.ceil(rmax_px)) + 1
    y0, y1 = max(y - h, 0), min(y + h + 1, img.shape[0])
    x0, x1 = max(x - h, 0), min(x + h + 1, img.shape[1])
    sub = img[y0:y1, x0:x1]
    gy, gx = np.mgrid[y0:y1, x0:x1]
    rr = np.hypot(gy - y, gx - x)
    prev = None
    for k in range(int(np.floor(rmax_px))):
        m = (rr >= k) & (rr < k + 1) & np.isfinite(sub)
        if not m.any():
            continue
        v = float(np.mean(sub[m]))
        if v < 0:
            if prev is None:
                return k + 0.5
            pk, pv = prev
            return pk + (k + 0.5 - pk) * pv / (pv - v)
        prev = (k + 0.5, v)
    return float("nan")


def profile_center(sm: np.ndarray, y: float, x: float, search_px: float) -> Tuple[int, int]:
    """Pixel of the highest smoothed value near a position, where a profile is centered.

    Parameters
    ----------
    sm : ndarray, shape (ny, nx)
        The image smoothed (see :func:`smooth_nan`), NaN where there is no data.
    y, x : float
        Row and column of the position.
    search_px : float
        Search radius, in pixels.

    Returns
    -------
    py, px : int
        Row and column of the highest finite value within ``search_px`` of (y, x), or the rounded
        position itself if there is none.
    """
    h = int(np.ceil(search_px))
    y0, x0 = int(round(y)), int(round(x))
    best, out = -np.inf, (y0, x0)
    for dy in range(-h, h + 1):
        for dx in range(-h, h + 1):
            yy, xx = y0 + dy, x0 + dx
            if (dy * dy + dx * dx > search_px * search_px or not 0 <= yy < sm.shape[0]
                    or not 0 <= xx < sm.shape[1] or not np.isfinite(sm[yy, xx])):
                continue
            if sm[yy, xx] > best:
                best, out = sm[yy, xx], (yy, xx)
    return out


def psf_zero_radius(clean: np.ndarray, inj_path: str, center: Tuple[float, float], fwhm: float,
                    pxscale: float, rmin_px: float, rmax_px: float) -> Tuple[float, int]:
    """Mean zero-crossing radius of the injected sources' profiles after the reduction.

    Parameters
    ----------
    clean : ndarray, shape (ny, nx)
        The clean image.
    inj_path : str
        Its injected version, with the sources in its header (see :func:`flux_per_contrast`).
    center : tuple of float
        Star position (cy, cx), in pixels.
    fwhm : float
        PSF FWHM, in pixels; profiles are examined out to 1.5 FWHM.
    pxscale : float
        Pixel scale, in arcsec per pixel.
    rmin_px, rmax_px : float
        Separations of the sources used, in pixels.

    Returns
    -------
    radius : float
        Mean over the sources of :func:`profile_zero_radius` of (injected - clean), each centered
        on the peak of (injected - clean) smoothed by half a FWHM within 3 px of its listed
        position, in pixels; NaN if no source has one.
    n : int
        Number of sources used.
    """
    with fits.open(inj_path) as h:
        inj = np.asarray(h[0].data, dtype=float)
        hdr = h[0].header
    diff = inj - clean
    sm = smooth_nan(diff, 0.5 * fwhm / (2 * np.sqrt(2 * np.log(2))))
    out = []
    for k in range(1, int(hdr.get("INJNSRC", 0)) + 1):
        rho_px = float(hdr[f"INJRHO{k}"]) / pxscale
        if not rmin_px <= rho_px <= rmax_px:
            continue
        dy, dx = offset_of(rho_px, float(hdr[f"INJPA{k}"]))
        py, px = profile_center(sm, center[0] + dy, center[1] + dx, 3.0)
        z = profile_zero_radius(diff, py, px, 1.5 * fwhm)
        if np.isfinite(z):
            out.append(z)
    return (float(np.mean(out)) if out else float("nan")), len(out)


def fake_zero_radii(clean: np.ndarray, inj_path: str, center: Tuple[float, float], fwhm: float,
                    pxscale: float, rmin_px: float, rmax_px: float, snr: float,
                    avoid_yx: Sequence[Tuple[float, float]], n: int, seed: int = 1
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Zero-crossing radii of fake faint point sources put into the clean image.

    The fake is the reduction's own point source: the mean of (injected - clean) over the injected
    sources within ``rmin_px``-``rmax_px``, each cut out to 1.5 FWHM around the peak of
    (injected - clean) smoothed by half a FWHM. Each copy is scaled so that its aperture flux is
    ``snr`` times the standard deviation of the clean image's aperture fluxes around the ring at its
    separation, added at a random position at least half a FWHM inside the annulus and two FWHM from
    every injected source and position in ``avoid_yx``, found again at the S/N map's peak within half
    a FWHM, and its
    profile measured as a suspect's is (:func:`profile_zero_radius`, centered on the image smoothed
    by half a FWHM, within a quarter FWHM of that peak).

    Parameters
    ----------
    clean : ndarray, shape (ny, nx)
        The clean image.
    inj_path : str
        Its injected version, with the sources in its header (see :func:`flux_per_contrast`).
    center : tuple of float
        Star position (cy, cx), in pixels.
    fwhm : float
        PSF FWHM, in pixels.
    pxscale : float
        Pixel scale, in arcsec per pixel.
    rmin_px, rmax_px : float
        Annulus of the injected sources used and of the fakes, in pixels.
    snr : float
        S/N each fake is scaled to before the noise (the suspect's measured S/N; a measured peak
        comes out higher than its true S/N, so the fakes are, if anything, clearer than the suspect).
    avoid_yx : sequence of (float, float)
        Pixel positions (row, column) the fakes keep two FWHM away from, besides the injected
        sources, which they always avoid: the suspects.
    n : int
        Number of fakes, each put in alone.
    seed : int, default 1
        Seed of the random positions.

    Returns
    -------
    zero : ndarray, shape (n,)
        Each fake's zero-crossing radius, in pixels (NaN where its profile stays positive).
    snr_found : ndarray, shape (n,)
        Each fake's S/N where it was found.

    Raises
    ------
    ValueError
        If no injected source falls in the annulus or no position is left for the fakes.
    """
    with fits.open(inj_path) as h:
        inj = np.asarray(h[0].data, dtype=float)
        hdr = h[0].header
    diff = inj - clean
    sig = 0.5 * fwhm / (2 * np.sqrt(2 * np.log(2)))
    smd = smooth_nan(diff, sig)
    hw = int(np.ceil(1.5 * fwhm))
    cuts = []
    for k in range(1, int(hdr.get("INJNSRC", 0)) + 1):
        rho_px = float(hdr[f"INJRHO{k}"]) / pxscale
        if not rmin_px <= rho_px <= rmax_px:
            continue
        dy, dx = offset_of(rho_px, float(hdr[f"INJPA{k}"]))
        py, px = profile_center(smd, center[0] + dy, center[1] + dx, 3.0)
        cut = diff[py - hw:py + hw + 1, px - hw:px + hw + 1]
        if cut.shape == (2 * hw + 1, 2 * hw + 1):
            cuts.append(cut)
    if not cuts:
        raise ValueError(f"{inj_path}: no injected source between {rmin_px:.1f} and {rmax_px:.1f} px")
    tmpl = np.nan_to_num(np.nanmean(cuts, axis=0))
    avoid = list(avoid_yx)
    for k in range(1, int(hdr.get("INJNSRC", 0)) + 1):
        dy, dx = offset_of(float(hdr[f"INJRHO{k}"]) / pxscale, float(hdr[f"INJPA{k}"]))
        avoid.append((center[0] + dy, center[1] + dx))
    kern = aperture_kernel(fwhm / 2.0)
    kh = kern.shape[0] // 2
    tflux = float((tmpl[hw - kh:hw + kh + 1, hw - kh:hw + kh + 1] * kern).sum())
    flux, bad = aperture_maps(clean, fwhm / 2.0)
    rng = np.random.default_rng(seed)
    lo, hi = rmin_px + 0.5 * fwhm, rmax_px - 0.5 * fwhm
    yy, xx = np.mgrid[:clean.shape[0], :clean.shape[1]]
    zero, found = [], []
    tries = 0
    while len(zero) < n:
        tries += 1
        if tries > 200 * n:
            raise ValueError("no room left for the fakes")
        r = float(rng.uniform(lo, hi))
        dy, dx = offset_of(r, float(rng.uniform(0.0, 360.0)))
        y0, x0 = center[0] + dy, center[1] + dx
        if any(np.hypot(y0 - ay, x0 - ax) < 2.0 * fwhm for ay, ax in avoid):
            continue
        nr = int(np.floor(2 * np.pi * r / fwhm))
        ang = 2 * np.pi * np.arange(nr) / nr
        ry, rx = center[0] + r * np.cos(ang), center[1] + r * np.sin(ang)
        f = ndi.map_coordinates(flux, [ry, rx], order=1)
        b = ndi.map_coordinates(bad.astype(float), [ry, rx], order=1) > 0
        sd = float(np.std(f[~b], ddof=1))
        iy, ix = int(round(y0)), int(round(x0))
        img = clean.copy()
        img[iy - hw:iy + hw + 1, ix - hw:ix + hw + 1] += snr * sd / tflux * tmpl
        s, _ = snr_map(img, center, fwhm, r - 0.5 * fwhm, r + 0.5 * fwhm)
        near = (np.hypot(yy - iy, xx - ix) <= 0.5 * fwhm) & np.isfinite(s)
        if not near.any():
            continue
        j = np.argmax(np.where(near, s, -np.inf))
        py, px = np.unravel_index(j, s.shape)
        zy, zx = profile_center(smooth_nan(img, sig), py, px, 0.25 * fwhm)
        zero.append(profile_zero_radius(img, zy, zx, 1.5 * fwhm))
        found.append(float(s[py, px]))
    return np.asarray(zero), np.asarray(found)


def measure_image(img: np.ndarray, center: Tuple[float, float], positions: Sequence[Tuple[float, float]],
                  fwhm: float, pxscale: float, rmin: float, rmax: float, search_fwhm: float
                  ) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """Measure the suspects in one image.

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        The image, NaN where there is no data.
    center : tuple of float
        Star position (cy, cx), in pixels.
    positions : sequence of (float, float)
        Each suspect's separation (arcsec) and position angle (degrees east of north).
    fwhm : float
        PSF FWHM, in pixels.
    pxscale : float
        Pixel scale, in arcsec per pixel.
    rmin, rmax : float
        Annulus of the S/N map and of the ranks, in arcsec.
    search_fwhm : float
        Search radius around each position, in FWHM.

    Returns
    -------
    common : dict
        ``snr_map``, ``sigma`` (robust pixel sigma of the annulus), ``n_max`` (local maxima in the
        annulus) and ``maxima`` (the five highest as (S/N, rho, PA)).
    per : list of dict
        For each suspect: ``peak_rho``, ``peak_pa`` (arcsec, degrees), ``offset_px`` from its
        position, ``snr`` at the peak, ``snr_nominal`` at the position, ``flux`` (aperture flux at
        the peak), ``rank`` (local maxima at or above the peak's S/N), ``peak_yx`` and
        ``nominal_yx``.

    Raises
    ------
    ValueError
        If the S/N map has no finite value within a suspect's search radius.
    """
    s, flux = snr_map(img, center, fwhm, rmin / pxscale, rmax / pxscale)
    yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
    maxima = local_maxima(s, fwhm)
    r = np.hypot(yy - center[0], xx - center[1]) * pxscale
    common = {"snr_map": s, "sigma": robust_sigma(img[(r >= rmin) & (r <= rmax)]),
              "n_max": len(maxima), "maxima": [(v,) + polar(y, x, center, pxscale) for v, y, x in maxima[:5]]}
    per = []
    for rho, pa in positions:
        dy, dx = offset_of(rho / pxscale, pa)
        ny0, nx0 = center[0] + dy, center[1] + dx
        near = (np.hypot(yy - ny0, xx - nx0) <= search_fwhm * fwhm) & np.isfinite(s)
        if not near.any():
            raise ValueError(f"no finite S/N within the search radius of ({rho}\", {pa} deg)")
        j = np.argmax(np.where(near, s, -np.inf))
        py, px = np.unravel_index(j, s.shape)
        peak = float(s[py, px])
        prho, ppa = polar(py, px, center, pxscale)
        per.append({"peak_rho": prho, "peak_pa": ppa, "offset_px": float(np.hypot(py - ny0, px - nx0)),
                    "snr": peak,
                    "snr_nominal": float(ndi.map_coordinates(np.nan_to_num(s, nan=0.0), [[ny0], [nx0]], order=1)[0]),
                    "flux": float(flux[py, px]), "rank": sum(1 for v, _, _ in maxima if v >= peak - 1e-9),
                    "peak_yx": (int(py), int(px)), "nominal_yx": (ny0, nx0)})
    return common, per


def plot_cutouts(items: Sequence[Optional[Tuple[ImageSpec, np.ndarray, Tuple[float, float], Dict[str, object],
                                                List[Dict[str, object]]]]],
                 png: str, cut_center: Tuple[float, float], fwhm: float, pxscale: float, ncol: int,
                 half_width: float, title: str, smooth_fwhm: float, rmin: float, rmax: float) -> None:
    """Draw one cutout per image around the suspects.

    Each cutout is the image smoothed with a Gaussian of FWHM ``smooth_fwhm`` * ``fwhm``, NaN-aware,
    over the robust sigma of the smoothed image's pixels in the annulus ``rmin``-``rmax``. Numbered
    ticks mark each suspect's peak, from 0.6 to 1.1 FWHM out, so that the peak itself stays visible.

    Parameters
    ----------
    items : sequence of (ImageSpec, ndarray, tuple, dict, list of dict) or None
        Each image's spec, data, star position (cy, cx) and :func:`measure_image` results; None
        leaves that panel empty.
    png : str
        Output file.
    cut_center : tuple of float
        Separation (arcsec) and position angle (degrees east of north) of the cutouts' center.
    fwhm : float
        PSF FWHM, in pixels.
    pxscale : float
        Pixel scale, in arcsec per pixel.
    ncol : int
        Panels per row.
    half_width : float
        Half width of each cutout, in arcsec.
    title : str
        Figure title.
    smooth_fwhm : float
        FWHM of the smoothing Gaussian, in units of ``fwhm``; 0 shows the image as it is.
    rmin, rmax : float
        Annulus whose smoothed pixels set each panel's sigma, in arcsec.

    Returns
    -------
    png : file
        The figure, written to ``png``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(items)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 3.45 * nrow + 0.6), squeeze=False)
    hw = half_width / pxscale
    dy, dx = offset_of(cut_center[0] / pxscale, cut_center[1])
    sig = smooth_fwhm * fwhm / (2 * np.sqrt(2 * np.log(2)))
    cmap = plt.get_cmap("magma").with_extremes(bad="0.75")
    im = None
    for ax, item in zip(axes.ravel(), items):
        if item is None:
            ax.axis("off")
            continue
        spec, img, center, common, per = item
        sm = smooth_nan(img, sig)
        yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
        r = np.hypot(yy - center[0], xx - center[1]) * pxscale
        norm = robust_sigma(sm[(r >= rmin) & (r <= rmax)])
        cy, cx = center[0] + dy, center[1] + dx
        y0, y1 = max(int(np.floor(cy - hw)), 0), int(np.ceil(cy + hw)) + 1
        x0, x1 = max(int(np.floor(cx - hw)), 0), int(np.ceil(cx + hw)) + 1
        cut = sm[y0:y1, x0:x1] / norm

        def east(x):
            return -(x - center[1]) * pxscale

        def north(y):
            return (y - center[0]) * pxscale

        ext = (east(x0 - 0.5), east(x1 - 0.5), north(y0 - 0.5), north(y1 - 0.5))
        im = ax.imshow(cut, origin="lower", extent=ext, cmap=cmap, vmin=-2, vmax=4,
                       interpolation="nearest")
        a, b = 0.6 * fwhm * pxscale, 1.1 * fwhm * pxscale
        for i, p in enumerate(per):
            col = TICK_COLORS[i % len(TICK_COLORS)]
            py, px = p["peak_yx"]
            ex, nn = east(px), north(py)
            for ux, uy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ax.plot([ex + ux * a, ex + ux * b], [nn + uy * a, nn + uy * b], color=col, lw=1.2)
            ax.text(ex - 0.8 * b, nn + 0.8 * b, str(i + 1), color=col, fontsize=8, ha="center",
                    va="center", fontweight="bold")
        snrs = " · ".join(f"{p['snr']:.1f}" for p in per)
        ranks = " · ".join(str(p["rank"]) for p in per)
        word = "Rank" if len(per) == 1 else "Ranks"
        ax.set_title(f"{spec.label}\nS/N {snrs} ({word} {ranks} of {common['n_max']})", fontsize=9)
        ax.set_xlabel("East Offset (arcsec)", fontsize=8)
        ax.set_ylabel("North Offset (arcsec)", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.92, 0.97 if title else 1.0))
    cax = fig.add_axes((0.935, 0.15, 0.015, 0.7))
    cb = fig.colorbar(im, cax=cax)
    what = f"Smoothed Image ({smooth_fwhm:g} FWHM)" if smooth_fwhm > 0 else "Image"
    cb.set_label(f"{what} / Annulus σ (Robust)", fontsize=9)
    fig.savefig(png, dpi=130)
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the command line (see the module docstring).

    Parameters
    ----------
    argv : sequence of str or None, default None
        Arguments; None reads ``sys.argv``.

    Returns
    -------
    int
        0 on success.
    """
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("out_prefix")
    p.add_argument("images", nargs="+")
    p.add_argument("--pos", type=float, nargs=2, action="append", required=True, metavar=("RHO", "PA"))
    p.add_argument("--inj", action="append", default=[])
    p.add_argument("--center", type=float, nargs=2, default=None, metavar=("RHO", "PA"))
    p.add_argument("--rmin", type=float, default=1.65)
    p.add_argument("--rmax", type=float, default=2.35)
    p.add_argument("--search-fwhm", type=float, default=0.5)
    p.add_argument("--fwhm", type=float, default=15.653)
    p.add_argument("--pxscale", type=float, default=0.0179)
    p.add_argument("--ncol", type=int, default=4)
    p.add_argument("--half-width", type=float, default=0.9)
    p.add_argument("--title", default="")
    p.add_argument("--smooth-fwhm", type=float, default=0.5)
    p.add_argument("--fakes", type=int, default=0)
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args(argv)

    outs = [a.out_prefix + ext for ext in (".csv", ".txt", ".png")]
    for o in outs:
        if os.path.exists(o):
            raise SystemExit(f"refusing to overwrite {o}")
    positions = [tuple(x) for x in a.pos]
    if a.center is None:
        offs = np.array([offset_of(rho, pa) for rho, pa in positions]).mean(axis=0)
        cut_center = (float(np.hypot(*offs)), float(np.rad2deg(np.arctan2(-offs[1], offs[0])) % 360.0))
    else:
        cut_center = tuple(a.center)
    specs = [None if s == "-" else parse_image_spec(s) for s in a.images]
    labels = [s.label for s in specs if s is not None]
    if not labels:
        raise SystemExit("no image given")
    if len(set(labels)) != len(labels):
        raise SystemExit("labels must be unique")
    inj: Dict[str, Tuple[str, Optional[float]]] = {}
    for item in a.inj:
        label, path, c = parse_inj(item)
        if label not in labels:
            raise SystemExit(f"--inj {item!r}: no image labeled {label!r}")
        inj[label] = (path, c)

    rmin_px, rmax_px = a.rmin / a.pxscale, a.rmax / a.pxscale
    rows, items = [], []
    text: List[List[str]] = [[] for _ in positions]
    for spec in specs:
        if spec is None:
            items.append(None)
            continue
        img, plane = load_image(spec)
        center = star_center(img.shape, spec.idl)
        common, per = measure_image(img, center, positions, a.fwhm, a.pxscale, a.rmin, a.rmax, a.search_fwhm)
        per_c, n_inj, z_psf = np.nan, 0, np.nan
        if spec.label in inj:
            per_c, n_inj = flux_per_contrast(img, inj[spec.label][0], inj[spec.label][1], center,
                                             a.fwhm, a.pxscale, rmin_px, rmax_px)
            z_psf, _ = psf_zero_radius(img, inj[spec.label][0], center, a.fwhm, a.pxscale, rmin_px, rmax_px)
        top = common["maxima"][0]
        tops = ", ".join(f"{v:.2f} at {r:.2f}\"/{t:.0f}" for v, r, t in common["maxima"])
        sm = smooth_nan(img, 0.5 * a.fwhm / (2 * np.sqrt(2 * np.log(2)))) if n_inj else None
        for i, ((rho, pa), q) in enumerate(zip(positions, per)):
            contrast = q["flux"] / per_c if np.isfinite(per_c) else np.nan
            z = np.nan
            fz = np.array([])
            if sm is not None:
                zy, zx = profile_center(sm, q["peak_yx"][0], q["peak_yx"][1], 0.25 * a.fwhm)
                z = profile_zero_radius(img, zy, zx, 1.5 * a.fwhm)
                if a.fakes > 0:
                    fz, _ = fake_zero_radii(img, inj[spec.label][0], center, a.fwhm, a.pxscale, rmin_px, rmax_px,
                                            q["snr"], [p_["nominal_yx"] for p_ in per], a.fakes, a.seed)
            fin_fz = fz[np.isfinite(fz)]
            rows.append({"label": spec.label, "file": spec.path, "plane": plane, "suspect": i + 1,
                         "rho_arcsec": f"{rho:.3f}", "pa_deg": f"{pa:.1f}",
                         "peak_rho_arcsec": f"{q['peak_rho']:.4f}", "peak_pa_deg": f"{q['peak_pa']:.2f}",
                         "offset_px": f"{q['offset_px']:.2f}", "snr_peak": f"{q['snr']:.3f}",
                         "snr_at_position": f"{q['snr_nominal']:.3f}", "aperture_flux": f"{q['flux']:.6e}",
                         "annulus_sigma": f"{common['sigma']:.6e}", "rank": q["rank"],
                         "n_maxima": common["n_max"], "top_snr": f"{top[0]:.3f}",
                         "top_rho_arcsec": f"{top[1]:.4f}", "top_pa_deg": f"{top[2]:.2f}",
                         "contrast": "" if not np.isfinite(contrast) else f"{contrast:.4e}", "n_inj": n_inj,
                         "zero_radius_px": "" if not np.isfinite(z) else f"{z:.2f}",
                         "psf_zero_radius_px": "" if not np.isfinite(z_psf) else f"{z_psf:.2f}",
                         "n_fakes": int(fz.size),
                         "fake_zero_median_px": f"{np.median(fin_fz):.2f}" if fin_fz.size else "",
                         "fake_zero_p16_px": f"{np.percentile(fin_fz, 16):.2f}" if fin_fz.size else "",
                         "fake_zero_p84_px": f"{np.percentile(fin_fz, 84):.2f}" if fin_fz.size else "",
                         "fake_frac_at_or_below": (f"{np.mean(fin_fz <= z):.3f}"
                                                   if fin_fz.size and np.isfinite(z) else "")})
            ctxt = "" if not np.isfinite(contrast) else f"{contrast:.2e}"
            ztxt = "" if not np.isfinite(z) else f"{z:.1f}/{z_psf:.1f}"
            if fin_fz.size and np.isfinite(z):
                ztxt += (f" fakes {np.median(fin_fz):.1f} ({np.percentile(fin_fz, 16):.1f}-"
                         f"{np.percentile(fin_fz, 84):.1f}), {100 * np.mean(fin_fz <= z):.0f}% <= it")
            text[i].append(f"{spec.label:38s} {q['peak_rho']:6.3f} {q['peak_pa']:7.1f} {q['offset_px']:5.1f}"
                           f" {q['snr']:6.2f} {q['snr_nominal']:6.2f} {q['rank']:4d}/{common['n_max']:<4d}"
                           f" {ctxt:>9s}   top 5: {tops}" + (f"\n{'':40s}zero radius (px) {ztxt}" if ztxt else ""))
        items.append((spec, img, center, common, per))

    with open(outs[0], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    head = (f"# feature_check: S/N maps over {a.rmin:.2f}-{a.rmax:.2f}\"; each peak searched within "
            f"{a.search_fwhm:.2f} FWHM ({a.search_fwhm * a.fwhm:.1f} px) of its position; FWHM {a.fwhm:.3f} px, "
            f"{a.pxscale} \"/px\n"
            f"# rank = local maxima (>= 1 FWHM apart) in the annulus at or above the suspect's S/N; "
            f"zero radius = where the profile first drops below 0, the suspect's / the injected sources', and the "
            f"fakes' median (16-84%) at the suspect's S/N")
    cols = (f"# {'label':36s} {'rho':>6s} {'PA':>7s} {'off':>5s} {'S/N':>6s} {'S/Nnom':>6s} "
            f"{'rank':>9s} {'contrast':>9s}")
    blocks = [f"## suspect {i + 1}: rho={rho:.3f}\" PA={pa:.1f} deg\n{cols}\n" + "\n".join(t)
              for i, ((rho, pa), t) in enumerate(zip(positions, text))]
    with open(outs[1], "w") as f:
        f.write(head + "\n" + "\n\n".join(blocks) + "\n")
    plot_cutouts(items, outs[2], cut_center, a.fwhm, a.pxscale, a.ncol, a.half_width, a.title,
                 a.smooth_fwhm, a.rmin, a.rmax)
    print(head)
    print("\n\n".join(blocks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
