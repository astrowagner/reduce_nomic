"""Limits on stellar companions from contrast curves.

A contrast curve gives, at each angular separation, the flux ratio C that a point source must have
to be recovered at a set significance (5σ in the IDL predecessor's injection-recovery curves).
For a star of apparent magnitude m_star in the band and parallax ϖ, a companion at the limit has
the absolute magnitude

    M_lim = m_star - 2.5 log10(C) + 5 log10(ϖ / 100 mas),

and the main-sequence dwarf sequence turns M_lim into the earliest spectral type and the highest
mass a companion could have and still be missed. The N' band (9.8-12.4 µm) is matched to WISE W3
(12 µm) magnitudes, M_W3 = M_Ks - (Ks - W3). M_Ks and masses come from Mamajek's "Modern Mean
Dwarf Stellar Color and Effective Temperature Sequence" (version 2022.04.16; Pecaut & Mamajek
2013, ApJS 208, 9), ``data/EEM_dwarf_UBVIJHK_colors_Teff.txt``. Ks - W3 is that table's
(Ks - W1) + (W1 - W3) through K5V, and from K7V on the median color of single field dwarfs in the
Cifuentes et al. (2020, A&A 642, A115) catalog, ``data/dwarf_ks_w3_cifuentes2020.csv`` (made by
``tools/derive_ks_w3_colors.py``); subtypes with neither (K6V, K8V, K9V) are interpolated in
spectral type. For stars, K - W3 and K - N' differ by much less than the 0.2-0.5 mag between
neighboring subtypes, so W3 stands in for N' within the precision that matters here.

Spectral types are handled as numbers: O0 = 0, B0 = 10, A0 = 20, F0 = 30, G0 = 40, K0 = 50,
M0 = 60, L0 = 70, T0 = 80, Y0 = 90, so K5V is 55 and M3.5V is 63.5.
"""
from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

#: Folder of the tables shipped with the package.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
#: Mamajek's dwarf sequence, version 2022.04.16, unchanged.
MAMAJEK_FILE = os.path.join(DATA_DIR, "EEM_dwarf_UBVIJHK_colors_Teff.txt")
#: Ks - W3 colors of K5V-L dwarfs from the Cifuentes et al. (2020) catalog.
KSW3_FILE = os.path.join(DATA_DIR, "dwarf_ks_w3_cifuentes2020.csv")

_CLASSES = "OBAFGKMLTY"


def spt_to_num(spt: str) -> float:
    """Convert a spectral type to its number.

    Parameters
    ----------
    spt : str
        Spectral type such as ``K5V``, ``M3.5V``, ``L1`` or ``T0V`` (a luminosity class is ignored).

    Returns
    -------
    num : float
        10 × (index of the class in OBAFGKMLTY) + subtype, e.g. 55.0 for K5V and 63.5 for M3.5V.
    """
    m = re.match(r"^([OBAFGKMLTY])(\d+(?:\.\d+)?)", spt.strip())
    if not m:
        raise ValueError(f"not a spectral type: {spt!r}")
    return 10.0 * _CLASSES.index(m.group(1)) + float(m.group(2))


def num_to_spt(num: float, decimals: int = 1) -> str:
    """Convert a spectral-type number back to a dwarf spectral type.

    Parameters
    ----------
    num : float
        Spectral-type number (see the module docstring). NaN gives an empty string.
    decimals : int, default 1
        Decimals of the subtype; a whole subtype is written without them (``M3V``, ``M3.4V``).

    Returns
    -------
    spt : str
        Spectral type with luminosity class V, e.g. ``M3.4V``.
    """
    if not np.isfinite(num):
        return ""
    num = round(float(num), decimals)
    cls = int(num // 10)
    sub = round(num - 10 * cls, decimals)
    if sub >= 10:
        cls, sub = cls + 1, round(sub - 10, decimals)
    sub_s = f"{sub:.{decimals}f}".rstrip("0").rstrip(".")
    return f"{_CLASSES[cls]}{sub_s}V"


def _float(tok: str) -> float:
    """Read one table entry; Mamajek's placeholders (runs of dots) become NaN.

    Parameters
    ----------
    tok : str
        Table token.

    Returns
    -------
    value : float
        The number, or NaN for a placeholder.
    """
    try:
        return float(tok)
    except ValueError:
        return float("nan")


def read_mamajek(path: str = MAMAJEK_FILE) -> Dict[str, np.ndarray]:
    """Read the main table of Mamajek's dwarf sequence.

    Parameters
    ----------
    path : str, default MAMAJEK_FILE
        Path of ``EEM_dwarf_UBVIJHK_colors_Teff.txt``.

    Returns
    -------
    table : dict of str to ndarray, each shape (n,)
        ``SpT`` (str), ``sptnum`` and every numeric column under its header name (``M_Ks``,
        ``Ks-W1``, ``W1-W3``, ``Msun``, ...), NaN where the table has a placeholder; n is the
        number of rows of the main table (O3V to Y4V).
    """
    with open(path) as fh:
        lines = fh.read().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("SpT "))
    header = lines[start].split()
    rows = []
    for ln in lines[start + 1:]:
        if ln.startswith("#") or not ln.strip():
            break
        rows.append(ln.split())
    table: Dict[str, np.ndarray] = {"SpT": np.array([r[0] for r in rows])}
    table["sptnum"] = np.array([spt_to_num(s) for s in table["SpT"]])
    for j, name in enumerate(header[1:-1], start=1):          # the last column repeats SpT
        table[name] = np.array([_float(r[j]) for r in rows])
    return table


def read_ks_w3(path: str = KSW3_FILE) -> Dict[float, Tuple[float, int]]:
    """Read the Ks - W3 colors derived from the Cifuentes et al. (2020) catalog.

    Parameters
    ----------
    path : str, default KSW3_FILE
        Table written by ``tools/derive_ks_w3_colors.py``.

    Returns
    -------
    colors : dict of float to (float, int)
        Median Ks - W3 (mag) and number of stars, by spectral-type number.
    """
    with open(path) as fh:
        rows = list(csv.DictReader(ln for ln in fh if not ln.startswith("#")))
    out = {}
    for r in rows:
        name = r["spt"] if r["spt"].startswith("L") else r["spt"]
        out[spt_to_num(name)] = (float(r["ks_w3"]), int(r["n"]))
    return out


@dataclass
class DwarfSequence:
    """Main-sequence dwarfs in the N' (W3) band, ordered from early to late type.

    Parameters
    ----------
    spt : ndarray of str, shape (n,)
        Spectral types, e.g. ``K5V``; n is the number of subtypes kept.
    sptnum : ndarray, shape (n,)
        Spectral-type numbers (see the module docstring).
    m_ks : ndarray, shape (n,)
        Absolute Ks magnitude (Mamajek), mag.
    ks_w3 : ndarray, shape (n,)
        Ks - W3 color, mag.
    ks_w3_source : ndarray of str, shape (n,)
        Where each color comes from: ``mamajek``, ``cifuentes2020`` or ``interpolated``.
    m_w3 : ndarray, shape (n,)
        Absolute W3 magnitude, M_Ks - (Ks - W3), mag; strictly increasing.
    mass : ndarray, shape (n,)
        Mass (Mamajek), solar masses.
    """

    spt: np.ndarray
    sptnum: np.ndarray
    m_ks: np.ndarray
    ks_w3: np.ndarray
    ks_w3_source: np.ndarray
    m_w3: np.ndarray
    mass: np.ndarray

    def type_at(self, m_abs: np.ndarray) -> np.ndarray:
        """Spectral-type number of a dwarf with a given absolute N' magnitude.

        Parameters
        ----------
        m_abs : ndarray or float
            Absolute N' (W3) magnitude, mag.

        Returns
        -------
        sptnum : ndarray or float
            Spectral-type number, linear in magnitude between subtypes; NaN outside the sequence
            (brighter than its first or fainter than its last subtype).
        """
        return np.interp(m_abs, self.m_w3, self.sptnum, left=np.nan, right=np.nan)

    def mass_at(self, m_abs: np.ndarray) -> np.ndarray:
        """Mass of a dwarf with a given absolute N' magnitude.

        Parameters
        ----------
        m_abs : ndarray or float
            Absolute N' (W3) magnitude, mag.

        Returns
        -------
        mass : ndarray or float
            Mass in solar masses, linear in magnitude between subtypes; NaN outside the sequence.
        """
        return np.interp(m_abs, self.m_w3, self.mass, left=np.nan, right=np.nan)

    def m_w3_of(self, spt: str) -> float:
        """Absolute W3 magnitude of a subtype in the sequence.

        Parameters
        ----------
        spt : str
            Spectral type, e.g. ``K7V``.

        Returns
        -------
        m_w3 : float
            Its absolute W3 magnitude, mag.
        """
        i = np.flatnonzero(np.isclose(self.sptnum, spt_to_num(spt)))
        if not len(i):
            raise KeyError(f"{spt} is not in the sequence ({self.spt[0]} to {self.spt[-1]})")
        return float(self.m_w3[i[0]])


def dwarf_sequence(first: str = "F0V", last: str = "M7V", mamajek_path: str = MAMAJEK_FILE,
                   ksw3_path: str = KSW3_FILE, min_stars: int = 4) -> DwarfSequence:
    """Build the N' (W3) dwarf sequence from Mamajek's table and the Cifuentes et al. colors.

    Parameters
    ----------
    first : str, default "F0V"
        Earliest subtype kept. Earlier than F0V the absolute magnitudes in Mamajek's table are not
        monotonic in subtype (A7V/A8V), which the inversion needs.
    last : str, default "M7V"
        Latest subtype kept. Past M7V the catalog has fewer than four clean W3 detections per
        subtype and the colors scatter.
    mamajek_path : str, default MAMAJEK_FILE
        Mamajek's table.
    ksw3_path : str, default KSW3_FILE
        Ks - W3 colors from ``tools/derive_ks_w3_colors.py``.
    min_stars : int, default 4
        Fewest catalog stars for a Cifuentes color to be used; a subtype with fewer is
        interpolated.

    Returns
    -------
    seq : DwarfSequence
        The subtypes from ``first`` to ``last`` with M_Ks and mass given in Mamajek's table.
    """
    t = read_mamajek(mamajek_path)
    cif = {k: v for k, v in read_ks_w3(ksw3_path).items() if v[1] >= min_stars}
    lo, hi = spt_to_num(first), spt_to_num(last)
    keep = (t["sptnum"] >= lo) & (t["sptnum"] <= hi) & np.isfinite(t["M_Ks"]) & np.isfinite(t["Msun"])
    num = t["sptnum"][keep]
    mam = (t["Ks-W1"] + t["W1-W3"])[keep]
    color = np.full(len(num), np.nan)
    source = np.array([""] * len(num), dtype=object)
    for i, s in enumerate(num):
        if np.isfinite(mam[i]):
            color[i], source[i] = mam[i], "mamajek"
        elif s >= 57 and s in cif:                     # the catalog starts at K5V; K7V on is used
            color[i], source[i] = cif[s][0], "cifuentes2020"
    have = np.isfinite(color)
    if not have.any():
        raise ValueError("no Ks - W3 colors in range")
    for i in np.flatnonzero(~have):
        color[i] = np.interp(num[i], num[have], color[have])
        source[i] = "interpolated"
    m_w3 = t["M_Ks"][keep] - color
    if np.any(np.diff(m_w3) <= 0):
        bad = [t["SpT"][keep][i] for i in np.flatnonzero(np.diff(m_w3) <= 0)]
        raise ValueError(f"M_W3 is not increasing after {bad}; choose another range")
    return DwarfSequence(spt=t["SpT"][keep], sptnum=num, m_ks=t["M_Ks"][keep], ks_w3=color,
                         ks_w3_source=source.astype(str), m_w3=m_w3, mass=t["Msun"][keep])


def distance_modulus_term(parallax_mas: float) -> float:
    """Term that turns an apparent into an absolute magnitude.

    Parameters
    ----------
    parallax_mas : float
        Parallax, mas.

    Returns
    -------
    term : float
        5 log10(ϖ / 100 mas), mag, so that M = m + term (0.5736 for Vega's 130.23 mas).
    """
    return 5.0 * np.log10(parallax_mas / 100.0)


def contrast_to_absmag(contrast: np.ndarray, star_mag: float, parallax_mas: float) -> np.ndarray:
    """Absolute magnitude of a companion with a given flux ratio to its star.

    Parameters
    ----------
    contrast : ndarray or float
        Companion-to-star flux ratio in the band; values ≤ 0 or NaN give NaN.
    star_mag : float
        Apparent magnitude of the star in the band, mag.
    parallax_mas : float
        Parallax, mas.

    Returns
    -------
    m_abs : ndarray or float
        Absolute magnitude in the band, mag.
    """
    c = np.asarray(contrast, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        dm = np.where(c > 0, -2.5 * np.log10(np.where(c > 0, c, 1.0)), np.nan)
    return star_mag + dm + distance_modulus_term(parallax_mas)


def absmag_to_contrast(m_abs: np.ndarray, star_mag: float, parallax_mas: float) -> np.ndarray:
    """Flux ratio to its star of a companion with a given absolute magnitude.

    Parameters
    ----------
    m_abs : ndarray or float
        Absolute magnitude in the band, mag.
    star_mag : float
        Apparent magnitude of the star in the band, mag.
    parallax_mas : float
        Parallax, mas.

    Returns
    -------
    contrast : ndarray or float
        Companion-to-star flux ratio; the inverse of ``contrast_to_absmag``.
    """
    return 10.0 ** (-0.4 * (np.asarray(m_abs, float) - distance_modulus_term(parallax_mas) - star_mag))


@dataclass
class ContrastCurve:
    """Contrast limits at a set of separations and position angles.

    Parameters
    ----------
    separations : ndarray, shape (n_sep,)
        Angular separations, arcsec; n_sep is their number.
    contrast : ndarray, shape (n_pa, n_sep)
        Limit at each position angle and separation (flux ratio); NaN where the curve has no
        limit; n_pa is the number of position angles with at least one limit.
    pas : ndarray, shape (n_pa,)
        Position angles of the rows, degrees.
    sigma : float
        Significance of the limits (the IDL ``threshold``).
    source : str
        File the curve was read from.
    """

    separations: np.ndarray
    contrast: np.ndarray
    pas: np.ndarray
    sigma: float
    source: str

    def stats(self) -> Dict[str, np.ndarray]:
        """Statistics of the limits over position angle at each separation.

        Returns
        -------
        stats : dict of str to ndarray, each shape (n_sep,)
            ``n_ok`` (position angles with a limit), ``median``, ``mean``, ``worst`` (largest
            contrast) and ``best`` (smallest), over the position angles with a limit; NaN where
            none has one.
        """
        c = self.contrast
        ok = np.isfinite(c)
        n_ok = ok.sum(axis=0)
        out = {"n_ok": n_ok}
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                out["median"] = np.nanmedian(c, axis=0)
                out["mean"] = np.nanmean(c, axis=0)
                out["worst"] = np.nanmax(c, axis=0)
                out["best"] = np.nanmin(c, axis=0)
        return out


def read_idl_curve(path: str) -> ContrastCurve:
    """Read a contrast curve saved by the IDL predecessor's ``contrastcurve_nomic.pro``.

    The IDL routine injects a scaled copy of the star at each separation and position angle,
    reruns the reduction, and adjusts the injected flux ratio until the source is recovered at
    ``threshold`` σ (to within 10 %), with trials limited to its ``contrast_guess``. It appends one
    row per completed position angle and saves ``curves``, ``pas``, ``separations`` and
    ``threshold``. NaN marks a position the source was not recovered at within eight trials
    (blocked, or a limit above ``contrast_guess``); rows after an interrupted or failed run are
    NaN or zero throughout.

    Parameters
    ----------
    path : str
        The ``<night>_contrast_output.sav`` file.

    Returns
    -------
    curve : ContrastCurve
        The rows with at least one positive finite limit; zeros and negative values become NaN.
        ``pas`` are the angles of the kept rows (the IDL list may be longer than the rows saved).
    """
    from scipy.io import readsav

    s = readsav(path)
    seps = np.atleast_1d(np.asarray(s["separations"], float))
    c = np.atleast_2d(np.asarray(s["curves"], float))
    if c.shape[1] != len(seps) and c.shape[0] == len(seps):
        c = c.T
    c = np.where(np.isfinite(c) & (c > 0), c, np.nan)
    pas_all = np.atleast_1d(np.asarray(s["pas"], float)) if "pas" in s else np.full(len(c), np.nan)
    rows = np.flatnonzero(np.isfinite(c).any(axis=1))
    pas = np.array([pas_all[i] if i < len(pas_all) else np.nan for i in rows])
    sigma = float(s["threshold"]) if "threshold" in s else float("nan")
    return ContrastCurve(separations=seps, contrast=c[rows], pas=pas, sigma=sigma, source=path)


def klip_tpe_sections(path: str) -> List[Tuple[str, List[List[str]]]]:
    """Split a KLIP-TPE ``contrast_curve.txt`` into its sections.

    Parameters
    ----------
    path : str
        A run's or an annulus's ``contrast_curve.txt``.

    Returns
    -------
    sections : list of (str, list of list of str)
        For each block of data rows, in file order: the text of the ``#`` comment lines before
        it (joined with spaces, without the ``#``) and its rows split on whitespace.
    """
    sections: List[Tuple[str, List[List[str]]]] = []
    title: List[str] = []
    rows: List[List[str]] = []
    with open(path) as fh:
        for ln in fh:
            t = ln.strip()
            if not t:
                continue
            if t.startswith("#"):
                if rows:
                    sections.append((" ".join(title), rows))
                    title, rows = [], []
                title.append(t.lstrip("#").strip())
            else:
                rows.append(t.split())
    if rows:
        sections.append((" ".join(title), rows))
    return sections


def read_klip_tpe_curve(path: str, section: str = "injection") -> ContrastCurve:
    """Read a KLIP-TPE run's ``contrast_curve.txt`` (or an annulus's).

    KLIP-TPE writes several sections, each a block of ``#`` comment lines and then rows: the
    injection-recovery 5σ (SNR = 5) contrast curve, azimuthally combined (``separation_arcsec
    contrast``), the per-sample validation points (three columns) and a KLIP-FM cross-check
    (two columns). Their order differs between a run's file and an annulus's, so the section is
    found by its comment lines (``klip_tpe_sections``).

    Parameters
    ----------
    path : str
        The ``contrast_curve.txt`` file.
    section : {"injection", "klip_fm"}, default "injection"
        ``injection`` reads the first section whose comment lines mention an injection and are
        neither the per-sample points nor the KLIP-FM cross-check; ``klip_fm`` reads the KLIP-FM
        cross-check.

    Returns
    -------
    curve : ContrastCurve
        One row of limits (``pas`` = [NaN]: the curve is not per position angle), sigma 5.
    """
    if section not in ("injection", "klip_fm"):
        raise ValueError(f"section must be 'injection' or 'klip_fm', not {section!r}")
    for title, rows in klip_tpe_sections(path):
        low = title.lower()
        if section == "injection":
            ok = "injection" in low and "klip-fm" not in low and "per-sample" not in low
        else:
            ok = "klip-fm" in low
        pairs = [(float(r[0]), float(r[1])) for r in rows if len(r) == 2]
        if not ok or not pairs:
            continue
        seps = np.array([p[0] for p in pairs])
        vals = np.array([p[1] for p in pairs])
        o = np.argsort(seps, kind="stable")
        c = np.where(vals[o] > 0, vals[o], np.nan)[None, :]
        return ContrastCurve(separations=seps[o], contrast=c, pas=np.array([np.nan]), sigma=5.0, source=path)
    raise ValueError(f"no {section} curve section in {path}")


def companion_limits(contrast: np.ndarray, star_mag: float, parallax_mas: float,
                     seq: DwarfSequence) -> Dict[str, np.ndarray]:
    """Convert contrast limits to limits on main-sequence companions.

    Parameters
    ----------
    contrast : ndarray, shape (n,)
        Contrast limits (flux ratio); n is their number, NaN for none.
    star_mag : float
        Apparent magnitude of the star in the band, mag.
    parallax_mas : float
        Parallax, mas.
    seq : DwarfSequence
        Dwarf sequence in the band.

    Returns
    -------
    limits : dict of str to ndarray, each shape (n,)
        ``dmag`` (magnitude difference, mag), ``m_abs`` (limiting absolute magnitude, mag),
        ``sptnum`` (earliest spectral type that would be missed, as a number), ``spt`` (the same
        as text, ``""`` outside the sequence), ``mass`` (highest mass that would be missed, solar
        masses), and ``beyond`` (``"brighter"`` or ``"fainter"`` when the limit is outside the
        sequence, else ``""``).
    """
    c = np.asarray(contrast, float)
    m_abs = contrast_to_absmag(c, star_mag, parallax_mas)
    num = seq.type_at(m_abs)
    beyond = np.where(~np.isfinite(m_abs), "",
                      np.where(m_abs < seq.m_w3[0], "brighter", np.where(m_abs > seq.m_w3[-1], "fainter", "")))
    return {"dmag": m_abs - star_mag - distance_modulus_term(parallax_mas), "m_abs": m_abs,
            "sptnum": num, "spt": np.array([num_to_spt(x) for x in np.atleast_1d(num)]),
            "mass": seq.mass_at(m_abs), "beyond": beyond}


def lambda_over_d_arcsec(wavelength_um: float, aperture_m: float) -> float:
    """Diffraction scale λ/D.

    Parameters
    ----------
    wavelength_um : float
        Wavelength, µm.
    aperture_m : float
        Aperture diameter, m.

    Returns
    -------
    lod : float
        λ/D, arcsec (0.273 for 11.1 µm on 8.4 m).
    """
    return wavelength_um * 1e-6 / aperture_m * 180.0 / np.pi * 3600.0


def projected_au(sep_arcsec: np.ndarray, parallax_mas: float) -> np.ndarray:
    """Projected separation of an angular separation at a star's distance.

    Parameters
    ----------
    sep_arcsec : ndarray or float
        Angular separation, arcsec.
    parallax_mas : float
        Parallax, mas.

    Returns
    -------
    sep_au : ndarray or float
        Projected separation, au (angular separation in arcsec × distance in pc).
    """
    return np.asarray(sep_arcsec, float) * 1000.0 / parallax_mas


def arcsec_of_au(sep_au: np.ndarray, parallax_mas: float) -> np.ndarray:
    """Angular separation of a projected separation at a star's distance.

    Parameters
    ----------
    sep_au : ndarray or float
        Projected separation, au.
    parallax_mas : float
        Parallax, mas.

    Returns
    -------
    sep_arcsec : ndarray or float
        Angular separation, arcsec; the inverse of ``projected_au``.
    """
    return np.asarray(sep_au, float) * parallax_mas / 1000.0


def on_grid(seps: np.ndarray, lim: np.ndarray, grid: np.ndarray, interpolate: bool = True) -> np.ndarray:
    """Put one curve's limits on a grid of separations.

    Parameters
    ----------
    seps : ndarray, shape (n_sep,)
        Separations the curve was measured at, arcsec; n_sep is their number.
    lim : ndarray, shape (n_sep,)
        Its limits (flux ratio), NaN where it has none.
    grid : ndarray, shape (n_grid,)
        Separations to report at, arcsec.
    interpolate : bool, default True
        Interpolate log10(limit) linearly in separation between neighboring measured separations
        that both have a limit. False: only grid points equal to a measured separation get a value.

    Returns
    -------
    out : ndarray, shape (n_grid,)
        The limits on the grid; NaN outside the measured range, across a gap (a measured
        separation without a limit), and, without ``interpolate``, between measured separations.
    """
    seps = np.round(np.asarray(seps, float), 6)
    lim = np.asarray(lim, float)
    out = np.full(len(grid), np.nan)
    for j, g in enumerate(np.round(grid, 6)):
        hit = np.flatnonzero(np.isclose(seps, g))
        if len(hit):
            out[j] = lim[hit[0]]
            continue
        if not interpolate:
            continue
        lo = np.flatnonzero(seps < g)
        hi = np.flatnonzero(seps > g)
        if not len(lo) or not len(hi):
            continue
        a, b = lo[np.argmax(seps[lo])], hi[np.argmin(seps[hi])]
        if np.isfinite(lim[a]) and np.isfinite(lim[b]):
            t = (g - seps[a]) / (seps[b] - seps[a])
            out[j] = 10.0 ** ((1 - t) * np.log10(lim[a]) + t * np.log10(lim[b]))
    return out


def best_envelope(curves: Dict[str, Tuple[np.ndarray, np.ndarray]], grid: Optional[np.ndarray] = None,
                  interpolate: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lowest contrast limit at each separation among several curves.

    Parameters
    ----------
    curves : dict of str to (ndarray, ndarray)
        Separations (arcsec) and limits (flux ratio, NaN for none) of each curve, by name; the
        two arrays of a curve have the same shape (n_sep_i,).
    grid : ndarray, shape (n_grid,), optional
        Separations to report at, arcsec. Default: every separation of any curve, sorted.
    interpolate : bool, default True
        Fill a curve between its measured separations (see ``on_grid``), so that curves measured on
        different grids are compared at every grid point.

    Returns
    -------
    grid : ndarray, shape (n_grid,)
        Separations, arcsec.
    best : ndarray, shape (n_grid,)
        Lowest limit there, NaN where no curve has one.
    which : ndarray of str, shape (n_grid,)
        Name of the curve giving it, ``""`` where none.
    """
    if grid is None:
        grid = np.unique(np.concatenate([np.round(s, 6) for s, _ in curves.values()]))
    best = np.full(len(grid), np.nan)
    which = np.array([""] * len(grid), dtype=object)
    for name, (seps, lim) in curves.items():
        g = on_grid(seps, lim, grid, interpolate)
        better = np.isfinite(g) & ~(np.isfinite(best) & (best <= g))
        best[better], which[better] = g[better], name
    return grid, best, which.astype(str)
