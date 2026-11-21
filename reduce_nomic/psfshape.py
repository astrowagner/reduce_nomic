"""PSF-shape test for a companion inside λ/D.

A companion at a separation s well inside λ/D with flux ratio f stays inside the star's PSF core:
it shifts the photocenter by about f s, which registration removes, and changes the shape of the
core, elongating it by an amount growing as f s² (spin 2) and making it lopsided as f s³ (spin 1
and spin 3). The companion is fixed on the sky, so in frames taken pupil-stabilized its shape
signal turns with the parallactic angle q, while the instrument's own elongation and asymmetry stay
fixed to the pupil. This module measures Gaussian-weighted moments of each registered frame,
fits each moment as a static term, a slow drift and a term turning with q, and calibrates the
turning term by injecting copies of the frames' own PSF (self-injection, as the IDL predecessor's
contrast curves do) at a set separation and position angle.

A moment of spin m is written as a complex number z (for spin 2, e1 + i e2 with
e1 = (Ixx - Iyy) / (Ixx + Iyy) and e2 = 2 Ixy / (Ixx + Iyy)); a companion at sky angle φ adds
A exp(i m (φ - σ q)) with A ∝ f, where σ = +1 or -1 is the sense in which the sky turns on the
detector. Both senses are fitted: the wrong one, which no sky-fixed source can follow, is a null
test with the same sensitivity to drifts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np


def weighted_moments(frames: np.ndarray, sigma_w: float, x0: float, y0: float, n_iter: int = 200,
                     bg_annulus: Optional[Sequence[float]] = None, tol: float = 1e-7) -> Dict[str, np.ndarray]:
    """Gaussian-weighted centroid and shape moments of each frame.

    Parameters
    ----------
    frames : ndarray, shape (n, ny, nx)
        Registered frames with the star near (x0, y0); n is the number of frames, ny and nx the
        frame size in pixels. NaN pixels count as zero.
    sigma_w : float
        Standard deviation of the circular Gaussian weight, pixels.
    x0, y0 : float
        Starting position of the weighted centroid (0-based pixels).
    n_iter : int, default 200
        Most iterations of the weighted centroid. Each moves the centroid by a fraction
        σ_psf² / (σ_psf² + σ_w²) of its remaining error, so a narrow weight converges fast and a
        wide one needs many.
    bg_annulus : sequence of float, optional
        Inner and outer radii (pixels, about (x0, y0)) of an annulus whose median is subtracted from
        each frame first. Default: no subtraction.
    tol : float, default 1e-7
        Stop when no centroid moves by more than this, pixels.

    Returns
    -------
    mom : dict of str to ndarray
        Per frame, shape (n,): ``xc``, ``yc`` (weighted centroid, pixels), ``flux`` (weighted sum),
        ``size2`` ((Ixx + Iyy) / 2, pixels²), ``spin2`` (complex e1 + i e2), ``spin1`` (complex
        Σ w I r² (x + i y) / (Σ w I size2^1.5)), ``spin3`` (complex Σ w I r³ exp(3iθ) /
        (Σ w I size2^1.5)), with x, y, r, θ measured from the weighted centroid.

    Notes
    -----
    The sums run over a box of half-width ceil(7 σ_w) + 2 pixels around the rounded start, where
    the weight has fallen below 10^-10; the background median uses the whole frame.
    """
    f = np.nan_to_num(np.asarray(frames, dtype=np.float64))
    n, ny, nx = f.shape
    if bg_annulus is not None:
        yy0, xx0 = np.mgrid[0:ny, 0:nx]
        r0 = np.hypot(xx0 - x0, yy0 - y0)
        ring = (r0 >= bg_annulus[0]) & (r0 < bg_annulus[1])
        f = f - np.median(f[:, ring], axis=1)[:, None, None]
    h = int(np.ceil(7 * sigma_w)) + 2
    ix0, iy0 = int(round(x0)), int(round(y0))
    xa, xb = max(ix0 - h, 0), min(ix0 + h + 1, nx)
    ya, yb = max(iy0 - h, 0), min(iy0 + h + 1, ny)
    f = f[:, ya:yb, xa:xb]
    yy, xx = np.mgrid[ya:yb, xa:xb].astype(np.float64)
    xc = np.full(n, float(x0))
    yc = np.full(n, float(y0))
    for _ in range(n_iter):
        w = np.exp(-((xx[None] - xc[:, None, None]) ** 2 + (yy[None] - yc[:, None, None]) ** 2) / (2 * sigma_w ** 2)) * f
        s = w.sum(axis=(1, 2))
        xn = (w * xx[None]).sum(axis=(1, 2)) / s
        yn = (w * yy[None]).sum(axis=(1, 2)) / s
        moved = max(float(np.max(np.abs(xn - xc))), float(np.max(np.abs(yn - yc))))
        xc, yc = xn, yn
        if moved < tol:
            break
    dx = xx[None] - xc[:, None, None]
    dy = yy[None] - yc[:, None, None]
    w = np.exp(-(dx ** 2 + dy ** 2) / (2 * sigma_w ** 2)) * f
    s = w.sum(axis=(1, 2))
    ixx = (w * dx ** 2).sum(axis=(1, 2)) / s
    iyy = (w * dy ** 2).sum(axis=(1, 2)) / s
    ixy = (w * dx * dy).sum(axis=(1, 2)) / s
    size2 = (ixx + iyy) / 2
    r2 = dx ** 2 + dy ** 2
    zc = dx + 1j * dy
    spin1 = (w * r2 * zc).sum(axis=(1, 2)) / s / size2 ** 1.5
    spin3 = (w * zc ** 3).sum(axis=(1, 2)) / s / size2 ** 1.5
    return {"xc": xc, "yc": yc, "flux": s, "size2": size2,
            "spin2": ((ixx - iyy) + 2j * ixy) / (ixx + iyy), "spin1": spin1, "spin3": spin3}


@dataclass
class RotatingFit:
    """Fit of one complex moment as static + drift + a term turning with the parallactic angle.

    Parameters
    ----------
    b : complex
        Amplitude of the turning term, exp(i m (φ - σ q)) × |b| at q = 0 (moment units).
    sigma_b : float
        Standard error of each of its real and imaginary parts, from the residual scatter.
    static : complex
        Static term (the moment at the middle of the sequence, without the turning term).
    resid_rms : float
        RMS of the complex residuals (moment units).
    n : int
        Number of frames fitted.
    rho : float
        Lag-1 autocorrelation of the residuals; ``sigma_b`` is multiplied by
        sqrt((1 + rho) / (1 - rho)) when it is positive, for frames that are not independent.
    """

    b: complex
    sigma_b: float
    static: complex
    resid_rms: float
    n: int
    rho: float = 0.0


def fit_rotating(z: np.ndarray, q_deg: np.ndarray, spin: int, sense: int = 1, drift_order: int = 1,
                 t: Optional[np.ndarray] = None) -> RotatingFit:
    """Least-squares fit of a complex moment series with a term turning with the parallactic angle.

    Parameters
    ----------
    z : ndarray, shape (n,)
        Complex moment of each frame; n is the number of frames.
    q_deg : ndarray, shape (n,)
        Parallactic angle of each frame, degrees.
    spin : int
        Spin m of the moment (1, 2 or 3).
    sense : int, default 1
        +1 or -1, the sense σ in which the sky turns on the detector.
    drift_order : int, default 1
        Order of the polynomial drift in ``t`` (0: static only).
    t : ndarray, shape (n,), optional
        Drift variable (e.g. time); default: frame index. Scaled to [-1, 1] internally.

    Returns
    -------
    fit : RotatingFit
        The turning amplitude, its standard error per component (scaled by the residual scatter
        and inflated for residuals correlated from frame to frame), the static term, the
        residual RMS and the residuals' lag-1 autocorrelation.
    """
    z = np.asarray(z, dtype=np.complex128)
    n = len(z)
    tt = np.arange(n, dtype=np.float64) if t is None else np.asarray(t, dtype=np.float64)
    tt = 2 * (tt - tt.min()) / max(tt.max() - tt.min(), 1e-12) - 1
    cols = [tt ** k for k in range(drift_order + 1)]
    cols.append(np.exp(-1j * spin * sense * np.deg2rad(q_deg)))
    a = np.column_stack([c.astype(np.complex128) for c in cols])
    coef, *_ = np.linalg.lstsq(a, z, rcond=None)
    res = z - a @ coef
    dof = max(n - a.shape[1], 1)
    rms = float(np.sqrt(np.sum(np.abs(res) ** 2) / dof))
    cov = np.linalg.inv(a.conj().T @ a)
    rho = float(np.real(np.vdot(res[:-1], res[1:])) / max(np.real(np.vdot(res, res)), 1e-300)) if n > 2 else 0.0
    inflate = float(np.sqrt((1 + rho) / (1 - rho))) if 0 < rho < 0.99 else (10.0 if rho >= 0.99 else 1.0)
    sigma_b = float(np.sqrt(np.abs(cov[-1, -1]) / 2) * rms * inflate)
    return RotatingFit(b=complex(coef[-1]), sigma_b=sigma_b, static=complex(coef[0]), resid_rms=rms, n=n,
                       rho=rho)


def shift_frames(frames: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Shift each frame by its own subpixel offset with a Fourier shift.

    Parameters
    ----------
    frames : ndarray, shape (n, ny, nx)
        Frames; NaN pixels count as zero.
    dx, dy : ndarray, shape (n,)
        Shift of each frame along x and y, pixels (positive moves the content to larger indices).

    Returns
    -------
    shifted : ndarray, shape (n, ny, nx)
        Shifted frames (periodic at the edges; keep shifts well below the frame half-size).
    """
    f = np.nan_to_num(np.asarray(frames, dtype=np.float64))
    n, ny, nx = f.shape
    ky = np.fft.fftfreq(ny)[None, :, None]
    kx = np.fft.fftfreq(nx)[None, None, :]
    ph = np.exp(-2j * np.pi * (kx * np.asarray(dx)[:, None, None] + ky * np.asarray(dy)[:, None, None]))
    return np.fft.ifft2(np.fft.fft2(f) * ph).real


def inject(frames: np.ndarray, contrast: float, sep_px: float, phi_deg: float, q_deg: np.ndarray,
           sense: int = 1) -> np.ndarray:
    """Add a companion made from each frame's own PSF at a fixed sky position.

    Parameters
    ----------
    frames : ndarray, shape (n, ny, nx)
        Registered frames.
    contrast : float
        Companion-to-star flux ratio.
    sep_px : float
        Separation, pixels.
    phi_deg : float
        Position angle on the sky in the fit's convention, degrees: in frame i the companion sits at
        detector angle phi - sense × q_i (counterclockwise from +x).
    q_deg : ndarray, shape (n,)
        Parallactic angle of each frame, degrees.
    sense : int, default 1
        +1 or -1, as in ``fit_rotating``.

    Returns
    -------
    out : ndarray, shape (n, ny, nx)
        frames + contrast × (frames shifted to the companion's position in each frame).
    """
    ang = np.deg2rad(phi_deg - sense * np.asarray(q_deg, dtype=np.float64))
    return np.nan_to_num(np.asarray(frames, dtype=np.float64)) + contrast * shift_frames(
        frames, sep_px * np.cos(ang), sep_px * np.sin(ang))


def response(frames: np.ndarray, q_deg: np.ndarray, sigma_w: float, x0: float, y0: float, spins: Sequence[int],
             sep_px: float, contrast: float, n_phi: int = 8, drift_order: int = 1, sense: int = 1,
             bg_annulus: Optional[Sequence[float]] = None, base: Optional[Dict[int, RotatingFit]] = None) -> Dict[int, Dict[str, float]]:
    """Calibrate the turning amplitude per unit flux ratio by injecting companions.

    Parameters
    ----------
    frames : ndarray, shape (n, ny, nx)
        Registered frames, star near (x0, y0).
    q_deg : ndarray, shape (n,)
        Parallactic angles, degrees.
    sigma_w : float
        Weight width for ``weighted_moments``, pixels.
    x0, y0 : float
        Star position, 0-based pixels.
    spins : sequence of int
        Moment spins to calibrate (1, 2, 3).
    sep_px : float
        Injected separation, pixels.
    contrast : float
        Injected flux ratio; small enough to stay in the linear regime (0.01-0.03).
    n_phi : int, default 8
        Number of sky angles, evenly spaced over 360 degrees.
    drift_order : int, default 1
        As in ``fit_rotating``.
    sense : int, default 1
        Sense of the injection and the fit.
    bg_annulus : sequence of float, optional
        As in ``weighted_moments``.
    base : dict of int to RotatingFit, optional
        Fits without injection by spin (computed when not given).

    Returns
    -------
    cal : dict of int to dict
        By spin: ``r`` (median over angles of |Δb| / contrast, moment units per unit flux ratio),
        ``r_spread`` (half the 16-84 % range of |Δb| / contrast over angles), and ``phase_sd``
        (circular standard deviation, degrees, of arg(Δb) - spin × φ over angles; small when the
        turning term follows the injected angle as it should).
    """
    if base is None:
        m0 = weighted_moments(frames, sigma_w, x0, y0, bg_annulus=bg_annulus)
        base = {m: fit_rotating(m0[f"spin{m}"], q_deg, m, sense, drift_order) for m in spins}
    db: Dict[int, list] = {m: [] for m in spins}
    phis = np.arange(n_phi) * 360.0 / n_phi
    for phi in phis:
        mi = weighted_moments(inject(frames, contrast, sep_px, phi, q_deg, sense), sigma_w, x0, y0, bg_annulus=bg_annulus)
        for m in spins:
            db[m].append(fit_rotating(mi[f"spin{m}"], q_deg, m, sense, drift_order).b - base[m].b)
    out = {}
    for m in spins:
        d = np.asarray(db[m])
        amp = np.abs(d) / contrast
        ph = np.angle(d) - np.deg2rad(m * phis)
        R = abs(np.mean(np.exp(1j * ph)))
        out[m] = {"r": float(np.median(amp)), "r_spread": float((np.percentile(amp, 84) - np.percentile(amp, 16)) / 2),
                  "phase_sd": float(np.rad2deg(np.sqrt(-2 * np.log(max(R, 1e-12)))))}
    return out


def combine(b: Sequence[complex], sigma_b: Sequence[float], r: Sequence[float]) -> Dict[str, float]:
    """Combine turning amplitudes from several frame sets into one flux-ratio estimate.

    Each set k gives b_k ≈ r_k f exp(i m φ) for a companion of flux ratio f at sky angle φ (the
    same in every set of one night), so b_k / r_k estimates f exp(i m φ); these are averaged with
    inverse-variance weights.

    Parameters
    ----------
    b : sequence of complex
        Turning amplitudes (moment units).
    sigma_b : sequence of float
        Their standard errors per component.
    r : sequence of float
        Responses (moment units per unit flux ratio) at the separation considered.

    Returns
    -------
    comb : dict of str to float
        ``f_re``, ``f_im`` (the weighted mean of b_k / r_k), ``f_abs`` (its modulus, the
        best-fit flux ratio), ``sigma_f`` (standard error per component of the mean), ``snr``
        (f_abs / sigma_f) and ``chi2`` (scatter of the sets about the mean, 2 (K - 1) degrees of
        freedom for K sets).
    """
    b = np.asarray(b, dtype=np.complex128)
    sb = np.asarray(sigma_b, dtype=np.float64)
    rr = np.asarray(r, dtype=np.float64)
    beta = b / rr
    sbeta = sb / rr
    w = 1.0 / sbeta ** 2
    mean = np.sum(w * beta) / np.sum(w)
    sig = float(1.0 / np.sqrt(np.sum(w)))
    chi2 = float(np.sum(np.abs(beta - mean) ** 2 / sbeta ** 2))
    return {"f_re": float(mean.real), "f_im": float(mean.imag), "f_abs": float(abs(mean)), "sigma_f": sig,
            "snr": float(abs(mean) / sig), "chi2": chi2}


def rayleigh_threshold(p: float = 2.866515719e-7, trials: int = 1) -> float:
    """Detection threshold for the modulus of a complex amplitude with Gaussian noise.

    Parameters
    ----------
    p : float, default 2.866515719e-7
        False-alarm probability (the one-sided tail of 5σ for a Gaussian).
    trials : int, default 1
        Independent tries (e.g. 2 for both rotation senses); the probability is split among them.

    Returns
    -------
    t : float
        Threshold in units of the per-component standard error: |b| / σ above t has a
        probability p of arising from noise (5.49 for one try).
    """
    return float(np.sqrt(2.0 * np.log(trials / p)))
