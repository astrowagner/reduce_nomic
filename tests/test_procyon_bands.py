"""Tests of the ±1σ bands and the clip of the Procyon figure: pa_clip.py, idl_pa_scatter.py and
plot_procyon_curves.py --bands --clip.

Synthetic final grids, IDL tables and contrast-curve PDF readings stand in for the archive's, so the
choice of position angles, the clip, the consistency checks and the band table are checked without
the data.
"""
import csv
import json
import os
import sys

import numpy as np
import pytest

pytest.importorskip("pymupdf")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis", "procyon3-4_2026-11"))
import idl_pa_scatter  # noqa: E402
import plot_procyon_curves as ppc  # noqa: E402
from pa_clip import sigma_clip_mask  # noqa: E402


def _fake_pdf(gray, black, title):
    """Return a stand-in for idl_curve_pdf.read_curves that gives fixed curves.

    Parameters
    ----------
    gray : list of list of (float, float)
        The gray curves, one per position angle in drawing order: (separation in arcsec, contrast).
    black : list of (float, float)
        The black line.
    title : str
        The plot title, with the test counter.

    Returns
    -------
    read_curves : callable
        Takes a path and returns ``{"pa": gray, "mean": black, "title": title, "info": {}}``.
    """
    return lambda path: {"pa": gray, "mean": black, "title": title, "info": {}}


def _write_run(run_dir, seps, values, statuses):
    """Write a run's final_grid/curve.txt and curve.json.

    Parameters
    ----------
    run_dir : pathlib.Path
        The run directory to create.
    seps : list of float
        Separations of the grid, in arcsec.
    values : list of list of float or None
        Per separation, the positions' contrasts (None for a blocked position).
    statuses : list of list of str
        Per separation, the positions' statuses.

    Returns
    -------
    medians : list of float
        The curve: the median of the settled (converged or accepted) contrasts at each separation.
    """
    grid = run_dir / "final_grid"
    grid.mkdir(parents=True)
    positions, medians = [], []
    for i, (vals, sts) in enumerate(zip(values, statuses)):
        for v, s in zip(vals, sts):
            positions.append({"index": len(positions), "sep_index": i, "status": s, "contrast": v})
        medians.append(float(np.median([v for v, s in zip(vals, sts) if s in ("converged", "accepted")])))
    (grid / "curve.json").write_text(json.dumps({"r_as": seps, "curve": medians, "positions": positions}))
    lines = ["# sep_arcsec   contrast_median"] + [f"    {s:.4f}     {m:.4e}" for s, m in zip(seps, medians)]
    (grid / "curve.txt").write_text("\n".join(lines) + "\n")
    return medians


def test_final_positions_use_the_settled_contrasts(tmp_path):
    """The run's band is the sample standard deviation of its converged and accepted contrasts."""
    seps = [0.35, 0.58]
    vals = [[1e-3, 2e-3, 4e-3, None], [1e-4, 1.5e-4, 2e-4]]
    sts = [["converged", "converged", "accepted", "blocked"], ["converged"] * 3]
    med = _write_run(tmp_path / "run", seps, vals, sts)
    curve, sd, n, nc = ppc.read_final_positions(str(tmp_path / "run"), np.array(seps), np.array(med))
    assert list(curve) == med and list(n) == [3, 3] and list(nc) == [0, 0]
    assert sd[0] == pytest.approx(np.std([1e-3, 2e-3, 4e-3], ddof=1))
    assert sd[1] == pytest.approx(np.std([1e-4, 1.5e-4, 2e-4], ddof=1))
    with pytest.raises(SystemExit, match="median of the settled"):
        ppc.read_final_positions(str(tmp_path / "run"), np.array(seps), np.array([2.1e-3, 1.5e-4]))
    with pytest.raises(SystemExit, match="separations do not match"):
        ppc.read_final_positions(str(tmp_path / "run"), np.array([0.35, 0.60]), np.array(med))


def test_clip_matches_astropy_and_drops_the_outlier(tmp_path):
    """The clip is astropy's iterative median-centered one; a run's clipped curve is the median kept."""
    astropy_stats = pytest.importorskip("astropy.stats")
    rng = np.random.default_rng(1)
    for _ in range(200):
        v = np.exp(rng.normal(-8.0, 0.5, size=rng.integers(5, 25)))
        v[rng.integers(v.size)] *= rng.choice([1.0, 30.0])
        for sig in (2.0, 2.5, 3.0):
            ref = ~astropy_stats.sigma_clip(v, sigma=sig, maxiters=None, cenfunc="median", stdfunc="std").mask
            assert np.array_equal(sigma_clip_mask(v, sig), ref)
    # the joint run's 0.35" separation: eleven settled contrasts, one of them 26 times the median
    vals = [4.43e-4, 6.59e-4, 1.32e-4, 4.31e-4, 8.17e-4, 1.70e-2, 8.17e-4, 8.17e-4, 2.44e-4, 9.70e-4, 6.11e-4]
    med = _write_run(tmp_path / "run", [0.35], [vals + [None]], [["converged"] * 5 + ["accepted"] + ["converged"] * 5
                                                                 + ["blocked"]])
    curve, sd, n, nc = ppc.read_final_positions(str(tmp_path / "run"), np.array([0.35]), np.array(med), clip=3.0)
    kept = [v for v in vals if v != 1.70e-2]
    assert n[0] == 10 and nc[0] == 1
    assert curve[0] == pytest.approx(np.median(kept)) and sd[0] == pytest.approx(np.std(kept, ddof=1))
    with pytest.raises(ValueError):
        sigma_clip_mask(np.array([1.0, np.nan]), 3.0)


def _scatter_csv(path, rows, clip=""):
    """Write a scatter table of versions a and b as idl_pa_scatter.py does.

    Parameters
    ----------
    path : pathlib.Path
        Table to write.
    rows : list of (str, tuple or None, tuple or None)
        Per separation: its value as text, and for a and b ``(mean_all, mean, sd, n, n_clipped)`` or
        None for no point.
    clip : str, default ""
        The ``clip_sigma`` cell.

    Returns
    -------
    path : pathlib.Path
        `path`.
    """
    fields = ("mean_all", "mean", "sd", "n", "n_clipped", "n_pa")
    lines = [",".join(["separation_arcsec"] + [f"{c}_{f}" for c in "ab" for f in fields] + ["clip_sigma"])]
    for sep, a, b in rows:
        cells = [sep]
        for v in (a, b):
            cells += ["", "", "", "", "", ""] if v is None else [f"{v[0]:.4e}", f"{v[1]:.4e}", f"{v[2]:.4e}",
                                                                 str(v[3]), str(v[4]), "20"]
        lines.append(",".join(cells + [clip]))
    path.write_text("\n".join(lines) + "\n")
    return path


def test_idl_scatter_takes_the_version_each_point_came_from(tmp_path):
    """Each IDL point gets the standard deviation of its own version, whose mean must match it."""
    path = _scatter_csv(tmp_path / "scatter.csv", [("0.40", (1e-3, 1e-3, 2e-4, 20, 0), (9e-4, 9e-4, 3e-4, 19, 0)),
                                                   ("0.50", (8e-4, 8e-4, 1e-4, 20, 0), None)])
    val, ver, sd, n, nc = ppc.read_idl_scatter(str(path), np.array([0.40, 0.50]), np.array([9.0e-4, 8.0e-4]),
                                               ["b", "a"], ["a", "b"])
    assert list(val) == pytest.approx([9e-4, 8e-4]) and ver == ["b", "a"]
    assert list(sd) == pytest.approx([3e-4, 1e-4]) and list(n) == [19, 20] and list(nc) == [0, 0]
    with pytest.raises(SystemExit, match="not the IDL table"):
        ppc.read_idl_scatter(str(path), np.array([0.40]), np.array([9.1e-4]), ["b"], ["a", "b"])
    with pytest.raises(SystemExit, match="no b at 0.50"):
        ppc.read_idl_scatter(str(path), np.array([0.40, 0.50]), np.array([9.0e-4, 8.0e-4]), ["b", "b"], ["b"])
    with pytest.raises(SystemExit, match="written with clip_sigma"):
        ppc.read_idl_scatter(str(path), np.array([0.40]), np.array([9.0e-4]), ["b"], ["a", "b"], clip=3.0)


def test_idl_clip_takes_the_deepest_clipped_mean(tmp_path):
    """With a clip, each point is the deepest of its versions' clipped means, which can change version."""
    path = _scatter_csv(tmp_path / "scatter.csv", [("0.40", (1e-3, 8.5e-4, 2e-4, 19, 1), (9e-4, 9e-4, 3e-4, 19, 0))],
                        clip="3")
    val, ver, sd, n, nc = ppc.read_idl_scatter(str(path), np.array([0.40]), np.array([9.0e-4]), ["b"], ["a", "b"],
                                               clip=3.0)
    assert val[0] == pytest.approx(8.5e-4) and ver == ["a"] and sd[0] == pytest.approx(2e-4)
    assert n[0] == 19 and nc[0] == 1
    with pytest.raises(SystemExit, match="written with clip_sigma"):
        ppc.read_idl_scatter(str(path), np.array([0.40]), np.array([9.0e-4]), ["b"], ["a", "b"])


def test_pdf_uses_only_the_complete_position_angles(monkeypatch):
    """A PDF drawn at test 7 of 8 (4 position angles x 2 separations) counts its first 3 position angles."""
    gray = [[(0.4, 1e-3), (0.5, 5e-4)], [(0.4, 2e-3), (0.5, 6e-4)], [(0.4, 3e-3), (0.5, 7e-4)],
            [(0.4, 9e-3), (0.5, 9e-3)]]
    monkeypatch.setattr(idl_pa_scatter, "read_curves",
                        _fake_pdf(gray, [(0.4, 2e-3), (0.5, 6e-4)], "Test 7 / 8 -- SNR = 5.10"))
    stats, n_pa, note, clipped = idl_pa_scatter.scatter_pdf("x.pdf")
    assert n_pa == 4 and "first 3 of 4" in note and clipped == []
    assert stats[0.4] == pytest.approx((2e-3, 2e-3, 1e-3, 3, 0))
    assert stats[0.5] == pytest.approx((6e-4, 6e-4, 1e-4, 3, 0))
    # a black line that is the mean of all four does not match the complete position angles
    monkeypatch.setattr(idl_pa_scatter, "read_curves",
                        _fake_pdf(gray, [(0.4, 3.75e-3), (0.5, 2.7e-3)], "Test 7 / 8 -- SNR = 5.10"))
    with pytest.raises(SystemExit, match="black line differs"):
        idl_pa_scatter.scatter_pdf("x.pdf")
    monkeypatch.setattr(idl_pa_scatter, "read_curves",
                        _fake_pdf(gray, [(0.4, 2e-3), (0.5, 6e-4)], "Test 7 / 9 -- SNR = 5.10"))
    with pytest.raises(SystemExit, match="does not match"):
        idl_pa_scatter.scatter_pdf("x.pdf")


def test_main_scales_to_the_common_snr_and_writes_the_table(tmp_path, monkeypatch, capsys):
    """A curve that targeted S/N 3 is scaled by 5/3, its standard deviation too."""
    gray = [[(0.4, 1e-3)], [(0.4, 2e-3)], [(0.4, 3e-3)]]
    monkeypatch.setattr(idl_pa_scatter, "read_curves", _fake_pdf(gray, [(0.4, 1.5e-3)], "Test 3 / 3 -- SNR = 3.0"))
    out = tmp_path / "t.csv"
    assert idl_pa_scatter.main([str(out), "5", "v:3:x.pdf"]) == 0
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    # test 3 of 3 under way: the first 2 position angles are complete
    assert rows[0]["separation_arcsec"] == "0.40" and rows[0]["v_n"] == "2" and rows[0]["v_n_pa"] == "3"
    assert rows[0]["v_n_clipped"] == "0" and rows[0]["clip_sigma"] == ""
    assert float(rows[0]["v_mean"]) == pytest.approx(1.5e-3 * 5 / 3, rel=1e-4)
    assert float(rows[0]["v_mean_all"]) == pytest.approx(1.5e-3 * 5 / 3, rel=1e-4)
    assert float(rows[0]["v_sd"]) == pytest.approx(np.std([1e-3, 2e-3], ddof=1) * 5 / 3, rel=1e-4)
    assert "scaled by 1.66667" in capsys.readouterr().out


def test_plot_bands_end_to_end(tmp_path, capsys):
    """--bands writes the band table, reports a band reaching below zero and needs every IDL scatter table."""
    med = _write_run(tmp_path / "run", [0.35, 0.58],
                     [[1e-4, 2e-4, 3e-3], [1e-4, 1.2e-4, 1.4e-4]], [["converged", "converged", "accepted"],
                                                                    ["converged"] * 3])
    table = tmp_path / "idl.csv"
    table.write_text("separation_arcsec,v1,deepest,deepest_curve\n0.40,1e-3,1e-3,v1\n0.50,8e-4,8e-4,v1\n")
    scatter = tmp_path / "scatter.csv"
    scatter.write_text("separation_arcsec,v1_mean_all,v1_mean,v1_sd,v1_n,v1_n_clipped,v1_n_pa,clip_sigma\n"
                       "0.40,1.0000e-03,1.0000e-03,2.0000e-04,20,0,20,\n"
                       "0.50,8.0000e-04,8.0000e-04,1.0000e-04,20,0,20,\n")
    common = ["--run", "Run", "#2a78d6", str(tmp_path / "run"), "--idl", "IDL", "#2a78d6", str(table), "v1"]
    out_csv = tmp_path / "f.csv"
    ppc.main([str(tmp_path / "f.png")] + common + ["--bands", "--idl-scatter", "IDL", str(scatter),
                                                   "--csv", str(out_csv)])
    assert (tmp_path / "f.png").stat().st_size > 0
    with open(out_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == ["curve", "separation_arcsec", "contrast", "sd", "n", "n_clipped", "band_low", "band_high",
                             "source"]
    sd0 = np.std([1e-4, 2e-4, 3e-3], ddof=1)
    assert float(rows[0]["sd"]) == pytest.approx(sd0, rel=1e-4)
    assert float(rows[0]["band_low"]) == pytest.approx(med[0] - sd0, rel=1e-3) and float(rows[0]["band_low"]) < 0
    assert rows[2]["curve"] == "IDL" and float(rows[2]["sd"]) == pytest.approx(2e-4) and rows[2]["n"] == "20"
    printed = capsys.readouterr().out
    assert "band below zero (lower edge off the plot): Run at 0.35\"" in printed
    # without --bands the table keeps its four columns
    ppc.main([str(tmp_path / "g.png")] + common + ["--csv", str(tmp_path / "g.csv")])
    with open(tmp_path / "g.csv", newline="") as f:
        assert next(csv.reader(f)) == ["curve", "separation_arcsec", "contrast", "source"]
    with pytest.raises(SystemExit, match="no --idl-scatter"):
        ppc.main([str(tmp_path / "h.png")] + common + ["--bands"])
    with pytest.raises(SystemExit, match="is for --bands or --clip"):
        ppc.main([str(tmp_path / "h.png")] + common + ["--idl-scatter", "IDL", str(scatter)])
    # a clip needs scatter tables written with the same clip
    with pytest.raises(SystemExit, match="written with clip_sigma"):
        ppc.main([str(tmp_path / "h.png")] + common + ["--bands", "--clip", "3", "--idl-scatter", "IDL", str(scatter)])
