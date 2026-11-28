"""Tests of analysis/procyon3-4_2026-11/edge_fix_prototype.py on synthetic images and frames.

The pure helpers run everywhere. The tests of the patches need klip_tpe importable (a KLIP-TPE
clone on ``PYTHONPATH``) and are skipped otherwise; each restores the patched klip-tpe functions
when it ends, so later tests see klip-tpe as it is.
"""
import importlib.util
import json
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = importlib.util.spec_from_file_location(
    "edge_fix_prototype", os.path.join(HERE, "..", "analysis", "procyon3-4_2026-11", "edge_fix_prototype.py"))
efp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(efp)

EDGES = [139.66, 195.53, 251.4, 307.26]


# ----------------------------------------------------------------------------- pure helpers

def test_pad_edges_between_and_all():
    """Only edges another annulus shares are padded, unless edge_padding is "all"."""
    assert efp.pad_edges(139.66, 195.53, EDGES) == (False, True)
    assert efp.pad_edges(195.53, 251.4, EDGES) == (True, True)
    assert efp.pad_edges(251.4, 307.26, EDGES) == (True, False)
    assert efp.pad_edges(251.4, 307.26, EDGES, "all") == (True, True)


def test_parse_params_reads_json_values():
    """Values parse as JSON where they can and stay strings otherwise."""
    out = efp.parse_params(["post_hp=false", "comb_type=mean", "k_klip=20"])
    assert out == {"post_hp": False, "comb_type": "mean", "k_klip": 20}
    assert efp.parse_params(None) == {}


def test_patches_of_names_and_joins():
    """Named variants map to their patches; '+' joins patches; bad ones exit."""
    assert efp.patches_of("base") == ()
    assert efp.patches_of("hpmask_klip") == ("hpmask", "mklip")
    assert efp.patches_of("hpmask+mklip+nanvar") == ("hpmask", "mklip", "nanvar")
    for bad in ("nope", "hpmask+nope", "hpmask+hpmask2", "+"):
        with pytest.raises(SystemExit):
            efp.patches_of(bad)


def test_source_rows_rotate_and_extra(tmp_path):
    """Rotation adds to every position angle modulo 360, extra sources are appended."""
    rec = {"sources": [[3.8, 350.0, 1e-4], [4.0, 10.0, 1e-4]]}
    extra = tmp_path / "extra.txt"
    extra.write_text("# rho theta contrast\n4.1 100 2e-4\n")
    rows = efp.source_rows(rec, str(extra), rotate=20.0)
    assert [r[1] for r in rows] == pytest.approx([10.0, 30.0, 120.0])
    assert [r[0] for r in rows] == pytest.approx([3.8, 4.0, 4.1])
    assert rows[2][2] == pytest.approx(2e-4)


def test_expected_noise_formulas():
    """Per-pixel weights give 1/sum(n/s^2); equal weights give sum(s^2/n)/k^2 over partitions with data."""
    counts = np.zeros((2, 1, 3))
    counts[:, 0, 0] = [4, 18]         # both partitions
    counts[:, 0, 1] = [0, 9]          # the second only
    sig = [2.0, 3.0]                  # counts[:, 0, 2] = 0: no data
    pix = efp.expected_noise(counts, sig, True)[0]
    eq = efp.expected_noise(counts, sig, False)[0]
    assert pix[0] == pytest.approx(np.sqrt(1.0 / (4 / 4.0 + 18 / 9.0)))
    assert eq[0] == pytest.approx(np.sqrt((4.0 / 4 + 9.0 / 18) / 4.0))
    assert pix[1] == pytest.approx(1.0) and eq[1] == pytest.approx(1.0)
    assert np.isnan(pix[2]) and np.isnan(eq[2])


def test_expected_noise_matches_simulation():
    """The weighted mean of partitions' frame averages scatters as expected_noise says."""
    rng = np.random.default_rng(3)
    n = np.array([5, 20, 2])
    s = np.array([1.0, 2.0, 0.5])
    trials = 20000
    parts = np.stack([rng.normal(0.0, s[p], (trials, n[p])).mean(axis=1) for p in range(3)])
    w = n / s ** 2
    pix = (parts * w[:, None]).sum(axis=0) / w.sum()
    eq = parts.mean(axis=0)
    c = n.astype(float)[:, None, None]
    assert pix.std() == pytest.approx(efp.expected_noise(c, s, True)[0, 0], rel=0.03)
    assert eq.std() == pytest.approx(efp.expected_noise(c, s, False)[0, 0], rel=0.03)


def test_partition_weights_modes():
    """Inverse variance n/s^2, coverage n/N (1 at full coverage), equal 1; 0 without data; bad modes raise."""
    counts = np.zeros((2, 1, 3))
    counts[:, 0, 0] = [4, 18]
    counts[:, 0, 1] = [2, 9]
    sig = [2.0, 3.0]
    assert np.allclose(efp.partition_weights(counts, sig, "pixel")[:, 0, :2], [[1.0, 0.5], [2.0, 1.0]])
    assert np.allclose(efp.partition_weights(counts, sig, "coverage")[:, 0, :2], [[1.0, 0.5], [1.0, 0.5]])
    assert np.allclose(efp.partition_weights(counts, sig, "equal")[:, 0, :2], 1.0)
    assert np.all(efp.partition_weights(counts, sig, "coverage")[:, 0, 2] == 0)
    with pytest.raises(ValueError):
        efp.partition_weights(counts, sig, "nope")


def test_expected_noise_coverage_matches_simulation():
    """The coverage-weighted mean of partitions' frame averages scatters as expected_noise says."""
    rng = np.random.default_rng(10)
    n = np.array([5, 20, 2])
    nmax = np.array([10, 20, 8])
    s = np.array([1.0, 2.0, 0.5])
    trials = 20000
    parts = np.stack([rng.normal(0.0, s[p], (trials, n[p])).mean(axis=1) for p in range(3)])
    w = n / nmax
    est = (parts * w[:, None]).sum(axis=0) / w.sum()
    c = np.zeros((3, 1, 2))
    c[:, 0, 0] = n
    c[:, 0, 1] = nmax                      # sets each partition's largest count N_p
    assert est.std() == pytest.approx(efp.expected_noise(c, s, "coverage")[0, 0], rel=0.03)


def test_recombine_and_combine_mode():
    """Equal weights give the NaN-aware mean, inverse variance the weighted mean; the card names the mode."""
    rng = np.random.default_rng(11)
    st = rng.normal(size=(3, 4, 4))
    st[0, 0, :] = np.nan
    counts = np.full((3, 4, 4), 6.0)
    counts[0, 0, :] = 0
    sig = [1.0, 2.0, 0.5]
    assert np.allclose(efp.recombine(st, counts, sig, "equal"), np.nanmean(st, axis=0))
    w = 6.0 / np.array(sig) ** 2
    manual = (st[1:, 0, 0] * w[1:]).sum() / w[1:].sum()
    assert efp.recombine(st, counts, sig, "pixel")[0, 0] == pytest.approx(manual)
    assert efp.combine_mode({"COMBINE": "pixel inverse variance"}) == "pixel"
    assert efp.combine_mode({"COMBINE": "pixel coverage"}) == "coverage"
    assert efp.combine_mode({}) == "equal"


def test_kt_noise_map_without_counts_is_none():
    """A reduction without klip-tpe's counts (or older klip-tpe code) gives no noise map."""
    assert efp.kt_noise_map(object()) is None


def test_robust_sigma_and_minimum():
    """The MAD sigma of Gaussian values is near 1; too few values give NaN."""
    v = np.random.default_rng(4).normal(size=40000)
    v[:100] = 1e6
    assert efp.robust_sigma(v) == pytest.approx(1.0, rel=0.03)
    assert np.isnan(efp.robust_sigma([1.0, np.nan], min_n=2))


def test_source_xy_north_up_east_left():
    """Position angle 0 is up from the star, 90 is to the left."""
    x, y = efp.source_xy(1.0, 0.0, 0.1, (101, 101))
    assert (x, y) == pytest.approx((50.0, 60.0))
    x, y = efp.source_xy(1.0, 90.0, 0.1, (101, 101))
    assert (x, y) == pytest.approx((40.0, 50.0))


def test_matched_filter_support():
    """The support is the kernel's fraction on data, the map the zero-filled correlation."""
    img = np.ones((60, 60))
    img[:, 30:] = np.nan
    k = np.ones((5, 5)) / 25.0
    A, sup = efp.matched_filter(img, k)
    assert sup[30, 10] == pytest.approx(1.0) and sup[30, 50] == pytest.approx(0.0)
    assert sup[30, 29] == pytest.approx(3.0 / 5.0)
    assert A[30, 29] == pytest.approx(0.6)


def gaussian(shape, x0, y0, fwhm):
    """Return a unit-peak Gaussian image.

    Parameters
    ----------
    shape : tuple of 2 ints
        Image shape (ny, nx).
    x0, y0 : float
        Center, in pixels.
    fwhm : float
        FWHM, in pixels.

    Returns
    -------
    ndarray, shape (ny, nx)
        The Gaussian.
    """
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    s = fwhm / 2.3548
    return np.exp(-0.5 * ((xx - x0) ** 2 + (yy - y0) ** 2) / s ** 2)


def ring_setup(size=301, a_in=80.0, a_out=130.0, fwhm=6.0, seed=5):
    """Return a white-noise annulus, its coverage with a low-coverage west half, and the geometry.

    Parameters
    ----------
    size : int, default 301
        Image side, in pixels.
    a_in, a_out : float, default 80.0 and 130.0
        The annulus' radii, in pixels.
    fwhm : float, default 6.0
        FWHM of the sources and the matched filter, in pixels.
    seed : int, default 5
        Seed of the noise.

    Returns
    -------
    dict
        ``clean`` (white noise of sigma 1, twice that west of the star, NaN outside the annulus),
        ``noise`` (the true noise map), ``cov`` (0.3 west, 1 east), ``zone``, ``kernel`` (a
        unit-sum Gaussian), ``fwhm`` and ``pmask`` (nothing masked).
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    c = (size - 1) / 2.0
    r = np.hypot(xx - c, yy - c)
    zone = (r >= a_in) & (r < a_out)
    noise = np.where(xx > c, 1.0, 2.0)
    clean = np.where(zone, rng.normal(size=(size, size)) * noise, np.nan)
    k = gaussian((25, 25), 12, 12, fwhm)
    return {"clean": clean, "noise": noise, "cov": np.where(xx > c, 1.0, 0.3), "zone": zone,
            "kernel": k / k.sum(), "fwhm": fwhm, "pmask": np.zeros((size, size), bool)}


def test_measure_sources_scale_and_calibration():
    """S/N grows with the source's amplitude, c5 = 5 contrast / S/N, the true noise map calibrates to 1."""
    g = ring_setup()
    size = g["clean"].shape[0]
    pxs = 0.01
    # PA 270 lands right of the star (noise 1, coverage 1), PA 90 left (noise 2, coverage 0.3)
    srcs = [(1.05, 270.0, 1e-4), (1.05, 90.0, 1e-4)]
    inj = g["clean"].copy()
    for amp, (rho, th, _c) in zip((3.0, 6.0), srcs):
        x0, y0 = efp.source_xy(rho, th, pxs, (size, size))
        inj = inj + amp * gaussian((size, size), x0, y0, g["fwhm"])
    inj[~g["zone"]] = np.nan
    rows, calib = efp.measure_sources(g["clean"], inj, g["noise"], srcs, g["kernel"], g["fwhm"], pxs, g["cov"],
                                      g["pmask"], g["zone"], band=2.0)
    ks = [k for _c0, _c1, k, _n in calib if np.isfinite(k)]
    assert len(ks) == 2 and np.allclose(ks, 1.0, atol=0.1)
    right, left = rows
    assert right["cov"] == 1.0 and left["cov"] == 0.3
    # whitened, the left source of twice the amplitude in twice the noise has the right one's S/N
    assert left["snr"] == pytest.approx(right["snr"], rel=0.15)
    assert right["c5"] == pytest.approx(5e-4 / right["snr"])
    # with a flat noise map the left half's noise is underestimated, and the calibration says so
    rows2, calib2 = efp.measure_sources(g["clean"], inj, np.ones_like(g["noise"]), srcs, g["kernel"], g["fwhm"],
                                        pxs, g["cov"], g["pmask"], g["zone"], band=2.0)
    k_left = [k for c0, c1, k, _n in calib2 if c0 <= 0.3 < c1][0]
    k_right = [k for c0, c1, k, _n in calib2 if c0 <= 1.0 < c1][0]
    assert k_left / k_right == pytest.approx(2.0, rel=0.1)
    assert rows2[1]["snr"] == pytest.approx(left["snr"], rel=0.2)


def test_sources_file_names():
    """The pass 'inj' keeps the old name; other passes carry their tag."""
    assert efp.sources_file("o", "base", "inj") == os.path.join("o", "base_sources.json")
    assert efp.sources_file("o", "base", "inj_r20") == os.path.join("o", "base_inj_r20_sources.json")


def test_stage_profile_finds_a_noisy_rim():
    """A rim of triple noise within 4 px of a NaN edge shows as a sigma ratio of 3 there and 1 beyond."""
    from scipy import ndimage
    rng = np.random.default_rng(6)
    n, size = 12, 301
    yy, xx = np.mgrid[:size, :size]
    r = np.hypot(xx - 150.0, yy - 150.0)
    zone = (r >= 80.0) & (r <= 140.0)
    fr = rng.normal(size=(n, size, size))
    for i in range(n):
        fr[i, :, :60 + 2 * i] = np.nan
        d = ndimage.distance_transform_edt(np.isfinite(fr[i]) | ~zone)   # as stage_profile measures it
        fr[i][np.isfinite(fr[i]) & (d < 4)] *= 3.0                       # bins 1, 2 and 3
    prof = {k: (md, sg) for k, _n, md, sg in efp.stage_profile(fr, 80.0, 140.0, margin=5.0)}
    # 1500 to 2000 pixels a bin: the MAD sigma's error is about 0.08 at sigma 3 and 0.03 at 1, the
    # median's 0.09 and 0.03; the bounds are 5 of those or more
    assert all(abs(prof[k][1] - 3.0) < 0.45 for k in (1, 2, 3))
    assert all(abs(prof[k][1] - 1.0) < 0.15 for k in range(5, 20))
    assert all(abs(prof[k][0]) < 0.5 for k in (1, 2, 3))
    assert all(abs(prof[k][0]) < 0.2 for k in range(4, 20))


def test_write_image_refuses_overwrite(tmp_path):
    """A second write to the same path exits."""
    p = str(tmp_path / "a.fits")
    efp.write_image(p, np.zeros((3, 3)), {"LABEL": "x"})
    with pytest.raises(SystemExit):
        efp.write_image(p, np.zeros((3, 3)), {"LABEL": "x"})


def test_load_winner(tmp_path):
    """The winner is the done file's index in the annulus' results."""
    d = tmp_path / "annulus02"
    d.mkdir()
    (d / "annulus_done.json").write_text(json.dumps({"winner_index": 7}))
    recs = [{"annulus": 1, "index": 6}, {"annulus": 0, "index": 7}, {"annulus": 1, "index": 7, "ok": 1}]
    (tmp_path / "results.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\nnot json\n")
    assert efp.load_winner(str(tmp_path), 2) == recs[2]
    with pytest.raises(SystemExit):
        efp.load_winner(str(tmp_path), 3)


# ----------------------------------------------------------------------------- the patches (need klip_tpe)

@pytest.fixture
def kt(monkeypatch):
    """Import klip_tpe, or skip, and restore every function the patches replace when the test ends."""
    pytest.importorskip("klip_tpe")
    from klip_tpe import klip, reducer
    for mod, name in ((reducer, "_highpass_frames"), (reducer, "nw_ang_comb"), (reducer, "klip_annular"),
                      (klip, "klip_annular")):
        monkeypatch.setattr(mod, name, getattr(mod, name))
    monkeypatch.setattr(klip, "_MASKED_MIN_PX", 50, raising=False)
    return klip, reducer


def test_hpmask_blanks_the_one_sided_band(kt):
    """After the patch, pixels within w // 2 of NaN are NaN and the rest is the original high-pass."""
    klip, reducer = kt
    rng = np.random.default_rng(7)
    img = rng.normal(size=(2, 80, 80))
    img[:, :, :20] = np.nan
    w = 9
    ref = reducer._highpass_frames(img.copy(), w, True)
    efp.install_hpmask(1.0)
    out = reducer._highpass_frames(img.copy(), w, True)
    h = w // 2
    assert np.isnan(out[:, 10:70, 20:20 + h]).all()
    inner = (slice(None), slice(h, 80 - h), slice(20 + h, 80 - h))
    assert np.allclose(out[inner], ref[inner])
    assert np.isnan(out[:, :, :20]).all()


def rot90_like(img, k):
    """Return ``np.rot90(img, k)``, the exact 90 degree turn used to check the patched combination.

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        The image, square with an odd side.
    k : int
        Quarter turns of ``np.rot90``.

    Returns
    -------
    ndarray, shape (ny, nx)
        The turned image.
    """
    return np.rot90(img, k)


def test_nanvar_comb_weights_by_the_frames_with_data(kt):
    """With frames at 0 and 90 degrees, the patched weights equal an independent NaN-aware computation."""
    klip, reducer = kt
    rng = np.random.default_rng(8)
    n, size = 10, 41
    cube = rng.normal(size=(n, size, size))
    cube[:5, :, :12] = np.nan                       # frames at 0 deg lack the sky's west strip
    ang = np.array([0.0] * 5 + [90.0] * 5)
    t = rng.normal(size=(size, size))
    inner = (slice(2, -2), slice(2, -2))            # the 90 degree turn is exact off the border pixels
    k = next(k for k in (1, -1)
             if np.allclose(klip.rotate_ccw(t, 90.0, cval=0.0)[inner], rot90_like(t, k)[inner], atol=1e-6))
    efp.install_nanvar_comb(min_frames=3)
    out = reducer.nw_ang_comb(cube, ang)
    # independent: pupil-frame frames with NaN, variance over the frames with data, weights per frame
    pupil = np.stack([cube[i] if ang[i] == 0 else rot90_like(cube[i], -k) for i in range(n)])
    cnt = np.isfinite(pupil).sum(axis=0)
    var = np.nanvar(pupil, axis=0, ddof=1)
    med = np.median(var[cnt >= 3])
    var = np.where(cnt >= 3, var, med)
    wts = np.stack([1.0 / (var if ang[i] == 0 else rot90_like(var, k)) for i in range(n)])
    wts = np.where(np.isfinite(cube), wts, 0.0)
    expect = (np.where(np.isfinite(cube), cube, 0.0) * wts).sum(axis=0) / wts.sum(axis=0)
    assert np.allclose(out[inner], expect[inner], atol=1e-5)
    # the original, zeros in the variance, differs where only half the frames have data
    from klip_tpe.klip import nw_ang_comb as orig
    assert np.abs(orig(cube, ang)[inner] - expect[inner]).max() > 1e-2


def test_masked_klip_equals_original_without_nan_and_masks_with(kt):
    """With no NaN the masked fit is KLIP-TPE's; a frame's NaN strip blanks it in the targets it references."""
    klip, reducer = kt
    rng = np.random.default_rng(9)
    n, size = 20, 81
    ang = np.linspace(0.0, 100.0, n)
    speck = rng.normal(size=(size, size))
    cube = np.stack([speck + 0.3 * rng.normal(size=(size, size)) for _ in range(n)])
    kp = klip.KLIPParams(k_klip=5, inrad=10.0, outrad=38.0, n_ang=1, angsep=0.0, anglemax=50.0, n_min_ref=3)
    ref = klip.klip_annular(cube.copy(), ang, kp, 3.0)[0]
    efp.install_masked_klip(min_px=20)
    out = klip.klip_annular(cube.copy(), ang, kp, 3.0)[0]
    yy, xx = np.mgrid[:size, :size]
    r = np.hypot(xx - 40, yy - 40)
    zone = (r >= 12) & (r <= 36)
    assert np.allclose(out[:, zone], ref[:, zone], atol=1e-9, equal_nan=True)
    j0 = 3
    c2 = cube.copy()
    c2[j0, :, :25] = np.nan
    out2 = klip.klip_annular(c2, ang, kp, 3.0)[0]
    strip = zone & (xx < 25)
    for t in range(n):
        uses = t != j0 and abs(ang[t] - ang[j0]) <= 50.0
        if t == j0 or uses:
            assert np.isnan(out2[t][strip]).all(), t
        else:
            assert np.isfinite(out2[t][strip]).all(), t
    assert reducer.klip_annular is klip.klip_annular
