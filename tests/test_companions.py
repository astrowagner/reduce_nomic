"""Tests of the companion-limit conversion (reduce_nomic.companions)."""
import numpy as np
import pytest

from reduce_nomic import companions as cp


def test_spectral_type_numbers_round_trip():
    """Spectral types map to numbers and back (K5V = 55, M3.5V = 63.5)."""
    assert cp.spt_to_num("K5V") == 55.0
    assert cp.spt_to_num("M3.5V") == 63.5
    assert cp.spt_to_num("T0V") == 80.0
    for s in ("F0V", "G2V", "K7V", "M0V", "M4.5V"):
        assert cp.num_to_spt(cp.spt_to_num(s)) == s
    assert cp.num_to_spt(63.44) == "M3.4V"
    assert cp.num_to_spt(np.nan) == ""


def test_distance_term_and_inverse():
    """M = m + 5 log10(parallax / 100 mas), and the contrast conversion inverts."""
    assert cp.distance_modulus_term(130.23) == pytest.approx(0.573555, abs=1e-6)
    assert cp.distance_modulus_term(100.0) == pytest.approx(0.0)
    c = np.array([1e-2, 3e-3, 5e-4])
    m = cp.contrast_to_absmag(c, 0.0, 130.23)
    assert m[0] == pytest.approx(5.0 + 0.573555, abs=1e-6)
    assert np.allclose(cp.absmag_to_contrast(m, 0.0, 130.23), c)
    assert np.isnan(cp.contrast_to_absmag(np.array([0.0, -1.0, np.nan]), 0.0, 130.23)).all()


def test_dwarf_sequence():
    """The N' sequence increases monotonically and splices the two color sources as documented."""
    seq = cp.dwarf_sequence("F0V", "M7V")
    assert np.all(np.diff(seq.m_w3) > 0)
    i = {s: k for k, s in enumerate(seq.spt)}
    assert seq.ks_w3_source[i["K5V"]] == "mamajek"
    assert seq.ks_w3[i["K5V"]] == pytest.approx(0.092 - 0.029, abs=1e-6)
    assert seq.ks_w3_source[i["K7V"]] == "cifuentes2020"
    assert seq.m_w3_of("K7V") == pytest.approx(4.81 - 0.132, abs=1e-6)
    assert seq.ks_w3_source[i["K6V"]] == "interpolated"
    assert seq.ks_w3[i["K5V"]] < seq.ks_w3[i["K6V"]] < seq.ks_w3[i["K7V"]]
    with pytest.raises(KeyError):
        seq.m_w3_of("A0V")


def test_companion_limits_at_a_known_type():
    """A limit equal to a subtype's contrast gives that subtype and its mass; outside is flagged."""
    seq = cp.dwarf_sequence()
    c_k7 = cp.absmag_to_contrast(seq.m_w3_of("K7V"), 0.0, 130.23)
    lim = cp.companion_limits(np.array([c_k7, 1.0, 1e-7, np.nan]), 0.0, 130.23, seq)
    assert lim["spt"][0] == "K7V"
    assert lim["mass"][0] == pytest.approx(0.64)
    assert list(lim["beyond"]) == ["", "brighter", "fainter", ""]
    assert c_k7 == pytest.approx(0.0228, abs=2e-4)


def test_on_grid_and_envelope():
    """Curves are interpolated in log contrast inside their measured range only, then minimized."""
    seps = np.array([0.3, 0.5, 0.7, 0.9])
    lim = np.array([1e-2, 1e-3, np.nan, 1e-3])
    g = cp.on_grid(seps, lim, np.array([0.2, 0.3, 0.4, 0.6, 0.8, 1.0]))
    assert np.isnan(g[0]) and np.isnan(g[-1])            # no extrapolation
    assert g[1] == pytest.approx(1e-2)
    assert g[2] == pytest.approx(10 ** -2.5)             # log-linear midpoint
    assert np.isnan(g[3]) and np.isnan(g[4])             # no bridging across a gap
    assert np.isnan(cp.on_grid(seps, lim, np.array([0.4]), interpolate=False))[0]
    grid, best, which = cp.best_envelope({"a": (seps, lim), "b": (np.array([0.5, 0.7]), np.array([2e-3, 5e-4]))})
    assert list(grid) == [0.3, 0.5, 0.7, 0.9]
    assert best[1] == pytest.approx(1e-3) and which[1] == "a"
    assert best[2] == pytest.approx(5e-4) and which[2] == "b"


def test_curve_stats_ignore_missing_angles():
    """Statistics over position angle use only the angles with a limit at each separation."""
    c = cp.ContrastCurve(separations=np.array([0.3, 0.5]), contrast=np.array([[1e-3, np.nan], [3e-3, 2e-3]]),
                         pas=np.array([0.0, 18.0]), sigma=5.0, source="test")
    s = c.stats()
    assert list(s["n_ok"]) == [2, 1]
    assert s["median"][0] == pytest.approx(2e-3)
    assert s["worst"][0] == pytest.approx(3e-3)
    assert s["worst"][1] == pytest.approx(2e-3)


def test_lambda_over_d_and_au():
    """λ/D at 11.1 µm on 8.4 m is 0.273 arcsec; 0.5 au at 7.68 pc is 65 mas."""
    assert cp.lambda_over_d_arcsec(11.1, 8.4) == pytest.approx(0.2726, abs=1e-4)
    assert cp.arcsec_of_au(0.5, 130.23) == pytest.approx(0.0651, abs=1e-4)
    assert cp.projected_au(cp.arcsec_of_au(3.0, 130.23), 130.23) == pytest.approx(3.0)


def test_read_klip_tpe_curve(tmp_path):
    """The first block of two-column rows of a KLIP-TPE contrast_curve.txt is the curve."""
    p = tmp_path / "contrast_curve.txt"
    p.write_text("# Injection-calibrated SNR=5 contrast curve (validated winners per annulus)\n"
                 "# separation_arcsec   contrast_snr5\n"
                 "     0.400     2.0000E-03\n     0.300     4.0000E-03\n     0.500     1.0000E-03\n"
                 "# per-sample 5*contrast/SNR points (validation injections):\n"
                 "#   sep(\")    5sig_contrast_sample   annulus\n"
                 "     0.310     3.9000E-03     1\n")
    c = cp.read_klip_tpe_curve(str(p))
    assert list(c.separations) == [0.3, 0.4, 0.5]
    assert np.allclose(c.contrast[0], [4e-3, 2e-3, 1e-3])
    assert c.sigma == 5.0 and c.contrast.shape == (1, 3)
