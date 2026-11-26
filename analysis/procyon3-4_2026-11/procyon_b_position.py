"""Predict Procyon B's separation and position angle from Bond et al. (2015)'s relative orbit.

usage: python analysis/procyon3-4_2026-11/procyon_b_position.py EPOCH [EPOCH ...] [--check]
       [--out OUT]

The relative visual orbit of Procyon B about A is Bond et al. (2015, ApJ 813, 106), Table 8:
P = 40.840 yr, a = 4.3075", i = 31.408 deg, Omega = 100.683 deg, T0 = 1968.076,
e = 0.39785, omega = 89.23 deg. At an epoch t the mean anomaly is M = 2 pi (t - T0) / P,
Kepler's equation E - e sin E = M gives the eccentric anomaly E, the true anomaly is
nu = 2 atan(sqrt((1 + e) / (1 - e)) tan(E / 2)), the separation in the orbit r = a (1 - e cos E), and
on the sky

    rho cos(theta - Omega) = r cos(nu + omega)
    rho sin(theta - Omega) = r sin(nu + omega) cos i,

with theta the position angle east of north (J2000; precession to the epoch is below 0.01 deg here
and ignored).

With ``--check`` the prediction is compared with the paper's HST measurements (its Table 3, the
WFPC2 F218W and WFC3 F953N "average" rows the orbit was fitted to), as a test of this code.

Parameters
----------
EPOCH : str
    Each epoch, as a decimal year (2021.97) or a UT date (2021-12-20, 2021-12-20T06:00).
--check : flag
    Also compare with the HST measurements and print the residuals.
--out : str, optional
    Text file for the printed table; not overwritten if it exists.

Returns
-------
stdout : text
    Per epoch: the decimal year, the separation (arcsec) and the position angle (degrees); with
    ``--check``, the HST residuals in separation and angle and their largest absolute values.
OUT : file
    The same text [--out].
"""
import argparse
import datetime as dt
import math
import os
import sys
from typing import List, Optional, Sequence, Tuple

#: Bond et al. (2015) Table 8: the relative orbit of Procyon B about A
ORBIT = {"P": 40.840, "a": 4.3075, "i": 31.408, "Omega": 100.683, "T0": 1968.076, "e": 0.39785,
         "omega": 89.23}

#: Bond et al. (2015) Table 3, the rows used in the orbit fit: (decimal year, separation ["], PA [deg])
HST = [
    (1995.1745, 4.9389, 42.977), (1997.9072, 4.7058, 58.027), (1998.8257, 4.5973, 63.499),
    (1999.8342, 4.4583, 69.771), (2000.9093, 4.2809, 76.977), (2001.8839, 4.0859, 84.147),
    (2002.8537, 3.8584, 91.939), (2003.8220, 3.5988, 100.787), (2004.8629, 3.2840, 112.092),
    (2005.9040, 2.9293, 125.956), (2006.8046, 2.6266, 140.997), (2007.8011, 2.3452, 161.715),
    (2011.1040, 2.6406, 240.272), (2012.1877, 3.0137, 257.737), (2013.0947, 3.3160, 269.432),
    (2014.7038, 3.7966, 285.721),
]


def decimal_year(s: str) -> float:
    """Read an epoch as a decimal year.

    Parameters
    ----------
    s : str
        A decimal year (``2021.97``) or an ISO UT date or date-time (``2021-12-20``,
        ``2021-12-20T06:00``).

    Returns
    -------
    float
        The epoch in years, Julian (365.25 d) from J2000.0 (2000-01-01T12:00 UT), which matches the
        paper's Besselian dates to 0.001 yr over this range.
    """
    try:
        return float(s)
    except ValueError:
        pass
    t = dt.datetime.fromisoformat(s)
    j2000 = dt.datetime(2000, 1, 1, 12, 0, 0)
    return 2000.0 + (t - j2000).total_seconds() / (365.25 * 86400.0)


def position(t: float, orbit: Optional[dict] = None) -> Tuple[float, float]:
    """Separation and position angle of the companion at one epoch.

    Parameters
    ----------
    t : float
        Epoch, decimal year.
    orbit : dict or None, default None
        Elements ``P`` (yr), ``a`` (arcsec), ``i``, ``Omega``, ``omega`` (deg), ``T0`` (yr) and ``e``;
        None takes :data:`ORBIT`.

    Returns
    -------
    rho : float
        Separation, arcsec.
    theta : float
        Position angle east of north, degrees in [0, 360).
    """
    o = ORBIT if orbit is None else orbit
    m = 2.0 * math.pi * (t - o["T0"]) / o["P"]
    e = o["e"]
    E = m + e * math.sin(m)
    for _ in range(100):
        dE = (E - e * math.sin(E) - m) / (1.0 - e * math.cos(E))
        E -= dE
        if abs(dE) < 1e-14:
            break
    nu = 2.0 * math.atan2(math.sqrt(1.0 + e) * math.sin(E / 2.0), math.sqrt(1.0 - e) * math.cos(E / 2.0))
    r = o["a"] * (1.0 - e * math.cos(E))
    u = nu + math.radians(o["omega"])
    x = r * math.cos(u)
    y = r * math.sin(u) * math.cos(math.radians(o["i"]))
    rho = math.hypot(x, y)
    theta = (math.degrees(math.atan2(y, x)) + o["Omega"]) % 360.0
    return rho, theta


def check() -> Tuple[List[str], float, float]:
    """Compare the predictions with the HST measurements.

    Returns
    -------
    lines : list of str
        One line per measurement: epoch, measured and predicted separation and angle.
    max_drho : float
        Largest absolute separation residual, arcsec.
    max_dpa : float
        Largest absolute angle residual, degrees.
    """
    lines, mr, ma = [], 0.0, 0.0
    for t, rho, pa in HST:
        pr, pp = position(t)
        dpa = (pp - pa + 180.0) % 360.0 - 180.0
        mr, ma = max(mr, abs(pr - rho)), max(ma, abs(dpa))
        lines.append(f"  {t:9.4f}  rho {rho:6.4f} pred {pr:6.4f} ({pr - rho:+.4f})   "
                     f"PA {pa:8.3f} pred {pp:8.3f} ({dpa:+.3f})")
    return lines, mr, ma


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
    p.add_argument("epochs", nargs="+")
    p.add_argument("--check", action="store_true")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    if a.out and os.path.exists(a.out):
        raise SystemExit(f"refusing to overwrite {a.out}")
    out = [f"# Procyon B about A, Bond et al. (2015) Table 8: " +
           ", ".join(f"{k} {v}" for k, v in ORBIT.items())]
    for s in a.epochs:
        t = decimal_year(s)
        rho, pa = position(t)
        out.append(f"{s:>20s}  t = {t:9.4f}  rho = {rho:.4f}\"  PA = {pa:.3f} deg")
    if a.check:
        lines, mr, ma = check()
        out.append("# HST measurements (Table 3, rows fitted) against this code's prediction")
        out.extend(lines)
        out.append(f"# largest residuals: {mr:.4f}\" in separation, {ma:.3f} deg in angle")
    text = "\n".join(out) + "\n"
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
