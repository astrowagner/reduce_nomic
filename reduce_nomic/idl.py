"""IDL-compatible primitives used by the IDL predecessor, reduce_nomic.pro.

Each function reproduces the IDL routine of the same name, as it is found first on Kevin's IDL
path (``~/idl``), closely enough that the port's stages give the same results.  Arrays here are
numpy arrays indexed ``[y, x]``; IDL indexes ``[x, y]``.  Where an IDL routine has a quirk that
changes results (edge handling, tie rules, an indexing bug), the quirk is reproduced and named in
the function's Notes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import ndimage as ndi
from scipy import optimize


def idl_round(x):
    """Round half away from zero, as IDL ``ROUND`` does.

    Parameters
    ----------
    x : float or ndarray
        Values to round.

    Returns
    -------
    r : int or ndarray of int
        Rounded values.
    """
    r = np.sign(x) * np.floor(np.abs(x) + 0.5)
    return r.astype(int) if isinstance(r, np.ndarray) else int(r)


def smooth(a: np.ndarray, width: int, nan: bool = False) -> np.ndarray:
    """Boxcar-smooth an image as IDL ``SMOOTH(a, width[, /NAN])`` does.

    Parameters
    ----------
    a : ndarray, shape (ny, nx)
        Image.
    width : int
        Box width in pixels. An even width is increased by one, as in IDL.
    nan : bool, default False
        Skip NaN pixels in each box mean (IDL ``/NAN``). A box with no finite pixel gives NaN.
        Without it, a NaN anywhere in a box makes that box's mean NaN.

    Returns
    -------
    out : ndarray, shape (ny, nx)
        The smoothed image, float64. Pixels within (w - 1) / 2 of an edge are copied from ``a``
        unchanged, where w is the width used, as IDL does without an ``/EDGE_*`` keyword.
    """
    w = int(width)
    if w % 2 == 0:
        w += 1
    a = np.asarray(a, dtype=np.float64)
    h = (w - 1) // 2
    if w <= 1 or min(a.shape) <= 2 * h:
        return a.copy()
    if nan:
        good = np.isfinite(a)
        s = ndi.uniform_filter(np.where(good, a, 0.0), size=w, mode="constant") * w * w
        c = ndi.uniform_filter(good.astype(np.float64), size=w, mode="constant") * w * w
        with np.errstate(invalid="ignore", divide="ignore"):
            m = s / np.round(c)
        m[np.round(c) == 0] = np.nan
    else:
        bad = ~np.isfinite(a)
        m = ndi.uniform_filter(np.where(bad, 0.0, a), size=w, mode="constant")
        if bad.any():
            nb = ndi.uniform_filter(bad.astype(np.float64), size=w, mode="constant")
            m[nb > 0.5 / (w * w)] = np.nan
    out = a.copy()
    out[h:-h, h:-h] = m[h:-h, h:-h]
    return out


def median(a: np.ndarray, axis: Optional[int] = None, even: bool = False):
    """Take a median as IDL ``MEDIAN`` does, ignoring NaN.

    Parameters
    ----------
    a : ndarray
        Values.
    axis : int, optional
        Axis to take the median along; None takes it over all values.
    even : bool, default False
        For an even number of finite values, average the two middle values (IDL ``/EVEN``).
        Otherwise take the upper of the two, as IDL does by default.

    Returns
    -------
    m : float or ndarray
        The median, NaN where no value is finite.
    """
    a = np.asarray(a, dtype=np.float64)
    if axis is None:
        v = np.sort(a[np.isfinite(a)])
        n = v.size
        if n == 0:
            return np.nan
        if n % 2 == 1 or not even:
            return float(v[n // 2])
        return float(0.5 * (v[n // 2 - 1] + v[n // 2]))
    s = np.sort(a, axis=axis)
    n = np.isfinite(a).sum(axis=axis)
    hi = np.expand_dims(np.clip(n // 2, 0, a.shape[axis] - 1), axis)
    upper = np.take_along_axis(s, hi, axis=axis).squeeze(axis)
    if even:
        lo = np.expand_dims(np.clip(n // 2 - 1, 0, a.shape[axis] - 1), axis)
        lower = np.take_along_axis(s, lo, axis=axis).squeeze(axis)
        m = np.where(n % 2 == 0, 0.5 * (lower + upper), upper)
    else:
        m = upper
    return np.where(n > 0, m, np.nan)


def fshift(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Shift an image by a fraction of a pixel as ``~/idl/fshift.pro`` does.

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        Image.
    dx, dy : float
        Shift in pixels along x and y; positive moves the image toward higher indices.

    Returns
    -------
    out : ndarray, shape (ny, nx)
        The shifted image: a circular shift by the whole-pixel part, then bilinear
        interpolation from four circular shifts.

    Notes
    -----
    The whole-pixel part is IDL ``fix`` (truncation toward zero), with a negative fraction
    borrowing one pixel. The shifts wrap around, and a NaN spreads to the pixels that
    interpolate from it, even with zero weight, as in IDL.
    """
    intx, inty = int(dx), int(dy)
    fx, fy = dx - intx, dy - inty
    if fx < 0:
        fx += 1.0
        intx -= 1
    if fy < 0:
        fy += 1.0
        inty -= 1
    x = np.roll(np.asarray(img, dtype=np.float64), (inty, intx), axis=(0, 1))
    if dx == 0.0 and dy == 0.0:
        return x
    return (x * ((1 - fx) * (1 - fy)) + np.roll(x, 1, axis=0) * ((1 - fx) * fy)
            + np.roll(x, 1, axis=1) * (fx * (1 - fy)) + np.roll(x, (1, 1), axis=(0, 1)) * (fx * fy))


def _taper(n_el: int) -> np.ndarray:
    """Return the rising sine taper ``crosscorr`` applies to the first tenth of an axis.

    Parameters
    ----------
    n_el : int
        Axis length minus one (IDL ``nel``).

    Returns
    -------
    t : ndarray, shape (n + 1,)
        ``sin(k pi / 2 / n)`` for k = 0..n, where n = n_el // 10.
    """
    n = n_el // 10
    return np.sin(np.arange(n + 1) * np.pi / 2.0 / n)


def polyfit2d_max(patch: np.ndarray, degree: int = 4) -> Tuple[float, float]:
    """Find the maximum of a polynomial surface through a small patch, as ``polyfit2d(/max)`` does.

    Parameters
    ----------
    patch : ndarray, shape (m, m)
        Values on a square grid, normally the 5x5 around a correlation peak.
    degree : int, default 4
        Degree in each of x and y of the tensor-product polynomial; (degree + 1)^2 terms.

    Returns
    -------
    x, y : float
        Position of the surface's maximum in patch pixels, found by conjugate gradients from
        the highest grid point.

    Notes
    -----
    With a 5x5 patch and degree 4 the polynomial interpolates the patch exactly, so the
    surface does not depend on how IDL's ``POLYWARP`` lays out its coefficients.
    """
    m_y, m_x = patch.shape
    yy, xx = np.mgrid[0:m_y, 0:m_x]
    pw = np.arange(degree + 1)
    design = (xx.ravel()[:, None, None] ** pw[None, :, None]) * (yy.ravel()[:, None, None] ** pw[None, None, :])
    design = design.reshape(xx.size, -1)
    coef, *_ = np.linalg.lstsq(design, patch.ravel().astype(np.float64), rcond=None)
    c = coef.reshape(degree + 1, degree + 1)

    def f(p):
        return -float(np.sum(c * (p[0] ** pw)[:, None] * (p[1] ** pw)[None, :]))

    def g(p):
        dxp = np.where(pw > 0, pw * p[0] ** np.clip(pw - 1, 0, None), 0.0)
        dyp = np.where(pw > 0, pw * p[1] ** np.clip(pw - 1, 0, None), 0.0)
        return -np.array([np.sum(c * dxp[:, None] * (p[1] ** pw)[None, :]),
                          np.sum(c * (p[0] ** pw)[:, None] * dyp[None, :])])

    fit = (design @ coef).reshape(m_y, m_x)
    j, i = np.unravel_index(np.argmax(fit), fit.shape)
    res = optimize.minimize(f, x0=np.array([float(i), float(j)]), jac=g, method="CG",
                            options={"gtol": 1e-10, "maxiter": 200})
    return float(res.x[0]), float(res.x[1])


def crosscorr(d1: np.ndarray, d2: np.ndarray, cut: bool = False, pmax: bool = False,
              rng: int = 2, degree: int = 4):
    """Cross-correlate two images as ``~/idl/crosscorr.pro`` does.

    Parameters
    ----------
    d1, d2 : ndarray, shape (ny, nx)
        Images of the same, even, size.
    cut : bool, default False
        Taper the outer tenth of each axis of both images with a sine (IDL ``/CUT``).
    pmax : bool, default False
        Also return the subpixel position of the correlation maximum.
    rng : int, default 2
        Half-size of the patch fitted around the maximum (IDL ``range``).
    degree : int, default 4
        Degree of the polynomial fitted to that patch.

    Returns
    -------
    out : ndarray, shape (ny, nx)
        Normalized cross-correlation, at most 1, with zero lag at (ny // 2, nx // 2). The mean
        and the highest frequencies are removed (the first and last rows and columns of both
        spectra are zeroed) before normalizing.
    peak : tuple of float
        ``(px, py)``, the maximum's position as IDL's ``pmax``: pixel index along x and y,
        made negative when above half the axis length [pmax=True].
    """
    a = np.array(d1, dtype=np.float64)
    b = np.array(d2, dtype=np.float64)
    ny, nx = a.shape
    if cut:
        tx = _taper(nx - 1)
        n = tx.size - 1
        a[:, :n + 1] *= tx[None, :]
        b[:, :n + 1] *= tx[None, :]
        a[:, nx - 1 - n:] *= tx[::-1][None, :]
        b[:, nx - 1 - n:] *= tx[::-1][None, :]
        ty = _taper(ny - 1)
        n = ty.size - 1
        a[:n + 1, :] *= ty[:, None]
        b[:n + 1, :] *= ty[:, None]
        a[ny - 1 - n:, :] *= ty[::-1][:, None]
        b[ny - 1 - n:, :] *= ty[::-1][:, None]
    fa = np.fft.fft2(a)
    fb = np.fft.fft2(b)
    for f in (fa, fb):
        f[:, 0] = 0
        f[0, :] = 0
        f[:, nx - 1] = 0
        f[ny - 1, :] = 0
    p = np.sqrt(np.sum(np.abs(fa) ** 2) * np.sum(np.abs(fb) ** 2))
    out = (a.size * np.fft.ifft2(fa * np.conj(fb) / p)).real
    out = np.roll(out, (ny // 2, nx // 2), axis=(0, 1))
    if not pmax:
        return out
    jy, ix = np.unravel_index(np.argmax(out), out.shape)
    rows = (jy - rng + np.arange(2 * rng + 1)) % ny
    cols = (ix - rng + np.arange(2 * rng + 1)) % nx
    px, py = polyfit2d_max(out[np.ix_(rows, cols)], degree)
    px += ix - rng
    py += jy - rng
    if py > ny // 2:
        py -= ny
    if px > nx // 2:
        px -= nx
    return out, (px, py)


def cntrd(img: np.ndarray, x: float, y: float, fwhm: float) -> Tuple[float, float]:
    """Centroid a star as the IDL Astronomy Library's ``CNTRD`` does.

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        Image.
    x, y : float
        Approximate position in pixels.
    fwhm : float
        Approximate FWHM in pixels; it sets the box half-width nhalf = max(int(0.637 fwhm), 2).

    Returns
    -------
    xcen, ycen : float
        The centroid, or (-1, -1) where ``CNTRD`` fails (too near an edge, or no decreasing
        derivative), as in IDL.
    """
    ny, nx = img.shape
    nhalf = max(int(0.637 * fwhm), 2)
    nbox = 2 * nhalf + 1
    ix, iy = idl_round(x), idl_round(y)
    if ix < nhalf or ix + nhalf > nx - 1 or iy < nhalf or iy + nhalf > ny - 1:
        return -1.0, -1.0
    big = img[iy - nhalf:iy + nhalf + 1, ix - nhalf:ix + nhalf + 1]
    mx = np.max(big)
    yy, xx = np.nonzero(big == mx)
    if xx.size > 1:
        idx, idy = idl_round(xx.mean()), idl_round(yy.mean())
    else:
        idx, idy = int(xx[0]), int(yy[0])
    xmax, ymax = ix - nhalf + idx, iy - nhalf + idy
    if xmax < nhalf or xmax + nhalf > nx - 1 or ymax < nhalf or ymax + nhalf > ny - 1:
        return -1.0, -1.0
    box = np.asarray(img[ymax - nhalf:ymax + nhalf + 1, xmax - nhalf:xmax + nhalf + 1], dtype=np.float64)
    ir = max(nhalf - 1, 1)
    dd = np.arange(nbox - 1) + 0.5 - nhalf
    w = 1.0 - 0.5 * (np.abs(dd) - 0.5) / (nhalf - 0.5)
    sumc = w.sum()
    deriv = np.roll(box, -1, axis=1) - box
    deriv = deriv[nhalf - ir:nhalf + ir + 1, 0:nbox - 1].sum(axis=0)
    sumd, sumxd, sumxsq = (w * deriv).sum(), (w * dd * deriv).sum(), (w * dd ** 2).sum()
    if sumxd >= 0:
        return -1.0, -1.0
    dx = sumxsq * sumd / (sumc * sumxd)
    if abs(dx) > nhalf:
        return -1.0, -1.0
    deriv = np.roll(box, -1, axis=0) - box
    deriv = deriv[0:nbox - 1, nhalf - ir:nhalf + ir + 1].sum(axis=1)
    sumd, sumxd = (w * deriv).sum(), (w * deriv * dd).sum()
    if sumxd >= 0:
        return -1.0, -1.0
    dy = sumxsq * sumd / (sumc * sumxd)
    if abs(dy) > nhalf:
        return -1.0, -1.0
    return float(xmax - dx), float(ymax - dy)


def psf_gaussian(npix: int, fwhm: float, centroid: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """Make a 2-d Gaussian of peak 1 as the IDL Astronomy Library's ``PSF_GAUSSIAN`` does.

    Parameters
    ----------
    npix : int
        Image size in pixels (square).
    fwhm : float
        FWHM in pixels.
    centroid : tuple of float, optional
        ``(x, y)`` center in pixels; default (npix - 1) / 2 in both.

    Returns
    -------
    psf : ndarray, shape (npix, npix)
        The Gaussian, evaluated at pixel centers, in single precision as in IDL.
    """
    cx, cy = centroid if centroid is not None else ((npix - 1) / 2.0, (npix - 1) / 2.0)
    sd = np.float32(fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0))))
    x = np.arange(npix, dtype=np.float32) - np.float32(cx)
    y = np.arange(npix, dtype=np.float32) - np.float32(cy)
    gx = np.exp(-0.5 * (x / sd) ** 2).astype(np.float32)
    gy = np.exp(-0.5 * (y / sd) ** 2).astype(np.float32)
    return (gy[:, None] * gx[None, :]).astype(np.float32)


class SurfaceFit:
    """Least-squares polynomial surface of one image size, as IDL ``SFIT(data, degree)``.

    IDL's ``SFIT`` without ``/MAX_DEGREE`` fits every term x^i y^j with i, j <= degree, so
    (degree + 1)^2 terms, x and y being the pixel indices. The pseudo-inverse is computed once,
    so fitting many frames of the same size costs two matrix products each.

    Parameters
    ----------
    shape : tuple of int
        ``(ny, nx)`` of the images to fit.
    degree : int, default 2
        Polynomial degree in each of x and y.
    """

    def __init__(self, shape: Tuple[int, int], degree: int = 2):
        ny, nx = shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        pw = np.arange(degree + 1)
        cols = [(xx.ravel() ** i) * (yy.ravel() ** j) for j in pw for i in pw]
        self.design = np.stack(cols, axis=1).astype(np.float64)
        self.pinv = np.linalg.pinv(self.design)
        self.shape = shape

    def surface(self, data: np.ndarray) -> np.ndarray:
        """Return the fitted surface of one image.

        Parameters
        ----------
        data : ndarray, shape (ny, nx)
            Image, all finite.

        Returns
        -------
        fit : ndarray, shape (ny, nx)
            The least-squares surface.
        """
        z = np.asarray(data, dtype=np.float64).ravel()
        return (self.design @ (self.pinv @ z)).reshape(self.shape)


@dataclass
class _FixGroup:
    """Bad pixels of one fixpix neighbor count.

    Parameters
    ----------
    bad : ndarray of int, shape (m,)
        Flat indices of the bad pixels, where m is their number.
    nbr : ndarray of int, shape (m, k)
        Flat indices of each one's k selected good neighbors.
    """

    bad: np.ndarray
    nbr: np.ndarray


class FixPix:
    """Bad-pixel replacement as ``~/idl/fixpix.pro`` (Mike Liu's ``FIXPIX``) does.

    For each bad pixel the routine grows a box (half-width 2, 4, 6, ..., clipped at the image
    edges) until it holds at least ``npix`` good pixels, keeps the good pixels at or within the
    ``npix``-th smallest distance, and replaces the bad pixel with their mean after dropping the
    lowest and the highest value. Values always come from the input image, never from pixels
    already replaced. Which neighbors are used depends only on the bad-pixel map, so they are
    found once here and every frame then needs only a gather, a sort and a mean.

    Parameters
    ----------
    badmap : ndarray, shape (ny, nx)
        1 (or True) for good pixels, 0 for bad ones.
    npix : int, default 8
        Number of nearest good pixels to use (IDL ``NPIX``).

    Notes
    -----
    At a clipped box, IDL looks up distances in the full-size ``dist_circle`` array with the
    clipped box's flat indices, so the distances there are those of other positions. This is
    reproduced, since it decides which neighbors are averaged near the image edges.
    """

    def __init__(self, badmap: np.ndarray, npix: int = 8):
        good = np.asarray(badmap) != 0
        ny, nx = good.shape
        self.shape = good.shape
        bad_y, bad_x = np.nonzero(~good)
        groups = {}
        for y, x in zip(bad_y, bad_x):
            dd = 2
            while True:
                x0, x1 = max(x - dd, 0), min(x + dd, nx - 1)
                y0, y1 = max(y - dd, 0), min(y + dd, ny - 1)
                cut = good[y0:y1 + 1, x0:x1 + 1]
                if cut.sum() >= npix or (x0 == 0 and y0 == 0 and x1 == nx - 1 and y1 == ny - 1):
                    break
                dd += 2
            wc = x1 - x0 + 1
            gy, gx = np.nonzero(cut)
            wgood = gy * wc + gx
            n = 2 * dd + 1
            gdist = np.hypot(wgood % n - dd, wgood // n - dd)
            k_th = np.sort(gdist)[min(npix, gdist.size) - 1]
            keep = gdist <= k_th
            idx = (gy[keep] + y0) * nx + (gx[keep] + x0)
            groups.setdefault(idx.size, ([], []))
            groups[idx.size][0].append(y * nx + x)
            groups[idx.size][1].append(idx)
        self.groups = [_FixGroup(np.asarray(b), np.asarray(nb)) for k, (b, nb) in sorted(groups.items())]
        self.nbad = int(bad_y.size)

    def apply(self, img: np.ndarray) -> np.ndarray:
        """Replace the bad pixels of one image.

        Parameters
        ----------
        img : ndarray, shape (ny, nx)
            Image.

        Returns
        -------
        out : ndarray, shape (ny, nx)
            Copy of ``img`` with every bad pixel replaced by the trimmed mean of its neighbors.
        """
        src = np.asarray(img, dtype=np.float64)
        out = src.copy()
        flat_src, flat_out = src.ravel(), out.ravel()
        for g in self.groups:
            k = g.nbr.shape[1]
            vals = np.sort(flat_src[g.nbr], axis=1)
            if k > 2:
                flat_out[g.bad] = vals[:, 1:k - 1].sum(axis=1) / (k - 2.0)
            else:
                flat_out[g.bad] = np.nan
        return out
