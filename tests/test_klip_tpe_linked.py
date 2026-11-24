"""Tests of tools/klip_tpe_linked.py: the link step, and a linked klip-tpe run end to end on synthetic cubes.

They need klip_tpe importable (a KLIP-TPE clone on ``PYTHONPATH``) and are skipped otherwise.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
from astropy.io import fits

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WRAPPER = os.path.join(ROOT, "tools", "klip_tpe_linked.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))
klip_tpe = pytest.importorskip("klip_tpe")
import klip_tpe_linked as ktl  # noqa: E402
from klip_tpe.space import Param, SearchSpace  # noqa: E402

LINK = ["bin", "filter", "angsep", "anglemax"]


def two_chop_space():
    """Two partitions A and B, each with bin, filter, angsep, anglemax and k_klip, and two drop slots."""
    block = [Param("bin", 1, 10, "int", default=2), Param("filter", 0, 25, "int", default=5),
             Param("angsep", 0.0, 3.0, "float", default=1.0), Param("anglemax", 20, 82, "int", default=60),
             Param("k_klip", 1, 20, "int", default=5)]
    sp = SearchSpace()
    sp.replicate(block, ["A", "B"])
    sp.with_selection("two_slot", partitions=["A", "B"], max_drop=2)
    return sp


def vector(sp, **values):
    """The space's default vector with the named dimensions set."""
    x = sp.default_vector()
    for name, v in values.items():
        x[sp.index(name)] = v
    return x


def test_link_tie_takes_the_most_permissive_angles_and_the_first_partition_otherwise():
    sp = two_chop_space()
    x = vector(sp, bin_A=3, bin_B=5, filter_A=2, filter_B=24, angsep_A=0.8, angsep_B=0.3,
               anglemax_A=40, anglemax_B=55, k_klip_A=4, k_klip_B=9)
    y = ktl.LinkTie(LINK)(x, sp)
    got = {n: y[sp.index(n)] for n in sp.names}
    assert got["bin_A"] == got["bin_B"] == 3
    assert got["filter_A"] == got["filter_B"] == 2
    assert got["angsep_A"] == got["angsep_B"] == pytest.approx(0.3)
    assert got["anglemax_A"] == got["anglemax_B"] == 55
    assert (got["k_klip_A"], got["k_klip_B"]) == (4, 9)          # not linked
    assert np.array_equal(ktl.LinkTie(LINK)(y, sp), y)          # a linked vector stays as it is
    assert x[sp.index("filter_B")] == 24                         # the input is not changed


def test_link_tie_takes_the_angles_of_the_selected_partitions_only():
    sp = two_chop_space()
    # two-slot selection: a slot holding 2 drops the second partition, B
    x = vector(sp, angsep_A=0.8, angsep_B=0.3, anglemax_A=40, anglemax_B=55, drop1=2)
    assert sp.selected_partitions(x) == ["A"]
    y = ktl.LinkTie(["angsep", "anglemax"])(x, sp)
    assert y[sp.index("angsep_A")] == y[sp.index("angsep_B")] == pytest.approx(0.8)
    assert y[sp.index("anglemax_A")] == y[sp.index("anglemax_B")] == 40


def test_link_tie_leaves_a_global_base_and_records_itself():
    sp = SearchSpace()
    sp.add(Param("filter", 0, 25, "int", default=5))               # one global dimension, no replicas
    sp.partitions = ["A", "B"]
    x = vector(sp, filter=7)
    assert np.array_equal(ktl.LinkTie(["filter"])(x, sp), x)
    rec = ktl.LinkTie(["angsep", "angsep", "anglemax"]).describe()
    assert rec["name"] == "link_tie" and rec["bases"] == ["angsep", "anglemax"]


def test_split_argv_needs_the_separator_and_a_command():
    assert ktl.split_argv(["--link", "bin", "--", "generic", "--seed", "1"]) == (
        ["--link", "bin"], ["generic", "--seed", "1"])
    with pytest.raises(SystemExit):
        ktl.split_argv(["--link", "bin", "generic"])
    with pytest.raises(SystemExit):
        ktl.split_argv(["--link", "bin", "--"])


def write_synthetic(d):
    """Write two 40-frame 64x64 cubes with a pupil-fixed speckle field, their angles and a PSF.

    Returns the klip-tpe data arguments.
    """
    rng = np.random.default_rng(3)
    n, ny = 40, 64
    yy, xx = np.mgrid[:ny, :ny] - (ny - 1) / 2.0
    r = np.hypot(yy, xx)
    speckle = rng.normal(size=(ny, ny))
    speckle = np.real(np.fft.ifft2(np.fft.fft2(speckle) * np.exp(-0.5 * (np.hypot(*np.meshgrid(
        np.fft.fftfreq(ny), np.fft.fftfreq(ny))) / 0.08) ** 2)))
    speckle *= 30.0 / speckle.std() * np.exp(-r / 25.0)
    args = {"--cube": [], "--angles": [], "--psf": []}
    for name, offset in (("A", 0.0), ("B", 0.75)):                  # interleaved chops: offset angles
        cube = (speckle[None] * (1.0 + 0.05 * rng.normal(size=(n, 1, 1)))
                + rng.normal(size=(n, ny, ny))).astype(np.float32)
        angles = np.linspace(0.0, 60.0, n) + offset
        c, a = os.path.join(d, f"cube_{name}.fits"), os.path.join(d, f"angles_{name}.fits")
        fits.PrimaryHDU(cube).writeto(c)
        fits.PrimaryHDU(angles).writeto(a)
        args["--cube"].append(c)
        args["--angles"].append(a)
    py, px = np.mgrid[:21, :21] - 10.0
    psf = np.exp(-0.5 * (py ** 2 + px ** 2) / (3.5 / 2.3548) ** 2).astype(np.float32)
    p = os.path.join(d, "psf.fits")
    fits.PrimaryHDU(psf).writeto(p)
    args["--psf"] = [p, p]
    out = []
    for k, v in args.items():
        out += [k] + v
    # lambda/D = 1.6e-6 / 9.4 rad = 3.5 px at 0.01 arcsec per pixel
    return out + ["--names", "A", "B", "--pxscale", "0.01", "--lam", "1.6e-6", "--diam", "9.4"]


def run_wrapper(argv, tmp_path):
    """Run the wrapper in a fresh process, with klip_tpe importable and its registry in tmp_path."""
    env = dict(os.environ)
    kt_root = os.path.dirname(os.path.dirname(os.path.abspath(klip_tpe.__file__)))
    env["PYTHONPATH"] = kt_root + os.pathsep + env.get("PYTHONPATH", "")
    env["KLIP_TPE_DATA"] = str(tmp_path / "klip_tpe_data")
    env["MPLBACKEND"] = "Agg"
    return subprocess.run([sys.executable, WRAPPER] + list(argv), env=env, capture_output=True, text=True,
                          timeout=900)


def linked_values(run_dir):
    """Each evaluation's (bin, filter, angsep, anglemax) for A and for B, from results.jsonl."""
    rows = []
    with open(os.path.join(run_dir, "results.jsonl")) as fh:
        for line in fh:
            pp = json.loads(line)["config"]["per_partition"]
            rows.append(tuple(tuple(pp[p][b] for b in LINK) for p in ("A", "B")))
    return rows


def test_a_linked_run_links_every_evaluation_and_extends_only_with_the_same_link(tmp_path):
    data = write_synthetic(str(tmp_path))
    run = str(tmp_path / "run")
    quick = ["--contrast-grid", "off", "--final-grid", "off", "--no-param-verify", "--no-fm-curve",
             "--no-fm-preview"]
    r = run_wrapper(["--link"] + LINK + ["--", "generic"] + data
                    + ["--ann-edges", "6", "20", "--n-iter", "14", "--n-init", "8", "--n-top", "1",
                       "--n-valid", "1", "--n-remeasure", "1", "--contrast0", "1e-2", "--use-contrast", "5e-2",
                       "--workers", "2", "--seed", "1", "--no-display", "--run-dir", run] + quick, tmp_path)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    rows = linked_values(run)
    assert len(rows) >= 14
    assert all(a == b for a, b in rows), [x for x in rows if x[0] != x[1]][:3]
    with open(os.path.join(run, "run_setup.txt")) as fh:
        setup = fh.read()
    assert "link_params        : ['bin', 'filter', 'angsep', 'anglemax']" in setup
    with open(os.path.join(run, "run_setup.json")) as fh:
        proj = json.load(fh)["projection"]
    assert proj["name"] == "compose" and proj["steps"][-1]["name"] == "link_tie"
    with open(os.path.join(run, "results.txt")) as fh:
        header = fh.readline()
    assert "k_klip_A" in header and "k_klip_B" in header             # k stays per chop
    # a checkpoint that links these bases goes on only with the same --link
    bad = run_wrapper(["--link", "bin", "filter", "--", "extend", "--instrument", "generic"] + data
                      + ["--workers", "2", "--run-dir", run, "--n-iter", "18"], tmp_path)
    assert bad.returncode != 0 and "the checkpoint links" in (bad.stdout + bad.stderr)
    assert len(linked_values(run)) == len(rows)
    ok = run_wrapper(["--link"] + LINK + ["--", "extend", "--instrument", "generic"] + data
                     + ["--workers", "2", "--run-dir", run, "--n-iter", "18"], tmp_path)
    assert ok.returncode == 0, ok.stdout[-3000:] + ok.stderr[-3000:]
    more = linked_values(run)
    assert len(more) > len(rows)
    assert all(a == b for a, b in more)


def test_bin_unlinked_with_a_higher_top_across_partitions_of_unequal_length(tmp_path):
    """Without bin in --link each partition draws its own bin, up to --bin-max; the other links hold."""
    data = write_synthetic(str(tmp_path))
    # shorten partition B to 24 frames, as Procyon_3 (164) beside Procyon_4 (388)
    ib = data.index("--cube") + 2
    short = str(tmp_path / "cube_B_short.fits")
    fits.PrimaryHDU(fits.getdata(data[ib])[:24]).writeto(short)
    data[ib] = short
    ia = data.index("--angles") + 2
    short_a = str(tmp_path / "angles_B_short.fits")
    fits.PrimaryHDU(fits.getdata(data[ia])[:24]).writeto(short_a)
    data[ia] = short_a
    run = str(tmp_path / "run_unlinked")
    quick = ["--contrast-grid", "off", "--final-grid", "off", "--no-param-verify", "--no-fm-curve",
             "--no-fm-preview"]
    link = ["filter", "angsep", "anglemax"]
    r = run_wrapper(["--link"] + link + ["--", "generic"] + data + ["--bin-max", "5"]
                    + ["--ann-edges", "6", "20", "--n-iter", "24", "--n-init", "16", "--n-top", "1",
                       "--n-valid", "1", "--n-remeasure", "1", "--contrast0", "1e-2", "--use-contrast", "5e-2",
                       "--workers", "2", "--seed", "2", "--no-display", "--run-dir", run] + quick, tmp_path)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    with open(os.path.join(run, "run_setup.txt")) as fh:
        setup = fh.read()
    assert "link_params        : ['filter', 'angsep', 'anglemax']" in setup
    bins, tied = [], []
    with open(os.path.join(run, "results.jsonl")) as fh:
        for line in fh:
            pp = json.loads(line)["config"]["per_partition"]
            bins.append((pp["A"]["bin"], pp["B"]["bin"]))
            tied.append(tuple(pp["A"][b] == pp["B"][b] for b in link))
    assert len(bins) >= 24
    assert all(all(t) for t in tied)                                   # filter, angsep, anglemax still one value
    assert any(a != b for a, b in bins)                                # bin is each partition's own
    assert max(max(a, b) for a, b in bins) > 3                         # above 24 / 8, the default top
    assert max(max(a, b) for a, b in bins) <= 5                        # and no higher than --bin-max


def test_max_drop_and_weighting_reach_the_run(tmp_path):
    """A run started with --max-drop 1 and --weighting sqrt_texp records both and selects one partition at times."""
    data = write_synthetic(str(tmp_path))
    run = str(tmp_path / "run_md")
    quick = ["--contrast-grid", "off", "--final-grid", "off", "--no-param-verify", "--no-fm-curve",
             "--no-fm-preview"]
    r = run_wrapper(["--link", "filter", "angsep", "anglemax", "--", "generic"] + data
                    + ["--weighting", "sqrt_texp", "--max-drop", "1"]
                    + ["--ann-edges", "6", "20", "--n-iter", "24", "--n-init", "16", "--n-top", "1",
                       "--n-valid", "1", "--n-remeasure", "1", "--contrast0", "1e-2", "--use-contrast", "5e-2",
                       "--workers", "2", "--seed", "3", "--no-display", "--run-dir", run] + quick, tmp_path)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    with open(os.path.join(run, "run_setup.txt")) as fh:
        setup = fh.read()
    assert "partition_weighting: sqrt_texp" in setup
    sizes = set()
    with open(os.path.join(run, "results.jsonl")) as fh:
        for line in fh:
            cfg = json.loads(line)["config"]
            sizes.add(len(cfg.get("selected") or cfg["per_partition"]))
    assert sizes <= {1, 2} and 1 in sizes                               # one partition left out at most


def test_known_source_reaches_the_objective_the_sampler_and_the_verify(tmp_path):
    """A run with --known records it in the objective, keeps every injection 1.5 FWHM away and verifies it."""
    data = write_synthetic(str(tmp_path))
    run = str(tmp_path / "run_kn")
    quick = ["--contrast-grid", "off", "--final-grid", "off", "--no-param-verify", "--no-fm-curve",
             "--no-fm-preview"]
    r = run_wrapper(["--link", "filter", "angsep", "anglemax", "--", "generic"] + data
                    + ["--known", "0.13", "156"]
                    + ["--ann-edges", "6", "20", "--n-iter", "24", "--n-init", "16", "--n-top", "1",
                       "--n-valid", "1", "--n-remeasure", "1", "--contrast0", "1e-2", "--use-contrast", "5e-2",
                       "--workers", "2", "--seed", "3", "--no-display", "--verify", "--run-dir", run] + quick, tmp_path)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    with open(os.path.join(run, "run_setup.json")) as fh:
        setup = json.load(fh)
    assert setup["objective"]["metric"]["known"] == [[0.13, 156.0]]     # left out of the noise rings
    assert setup["sampler"]["known"] == [[0.13, 156.0]] and setup["sampler"]["excl_fwhm"] == 1.5
    keep = 1.5 * float(setup["fwhm_px"]) * 0.01                         # the sampler's keep-out, arcsec
    n = 0
    with open(os.path.join(run, "results.jsonl")) as fh:
        for line in fh:
            for s in json.loads(line).get("sources") or []:
                rho, theta = float(s[0]), float(s[1])
                d = np.sqrt(rho ** 2 + 0.13 ** 2 - 2 * rho * 0.13 * np.cos(np.radians(theta - 156.0)))
                assert d >= keep * (1 - 1e-9), (rho, theta, d, keep)
                n += 1
    assert n > 0
    with open(os.path.join(run, "annulus01", "verify_report.txt")) as fh:
        report = fh.read()
    head = '== candidate 1  rho=0.130"  PA=156.0 deg =='                    # measured as a candidate
    assert head in report
    assert report.split(head, 1)[1].lstrip().startswith("[A] combined S/N =")
