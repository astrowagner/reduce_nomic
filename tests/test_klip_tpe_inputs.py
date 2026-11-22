"""Test of tools/klip_tpe_inputs.py on a synthetic archive."""
import json
import os
import subprocess
import sys

import numpy as np
from astropy.io import fits

TOOL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "klip_tpe_inputs.py")


def _night(root, night, n, n_by_cube=None, angles=None, off_cube=None, nan_rows=None, off_frames=()):
    """Write one side's port products for a night: a centered star in 300 px frames.

    Parameters
    ----------
    root : str
        Archive root.
    night : str
        Night folder name.
    n : int
        Frames per cube.
    n_by_cube : dict, optional
        Frames per cube number, overriding ``n`` for the cubes it lists.
    angles : ndarray, optional
        Angles for every cube (degrees), sliced to each cube's frame count; default a linear
        ramp from -10 to 10.
    off_cube : int, optional
        Cube number whose frames have the star 30 px right of the center (a failed centering).
    nan_rows : sequence of int, optional
        For each frame k of every cube, the number of rows at the bottom set to NaN (the padding
        past the detector's edge), as ``nan_rows[k]``; default none.
    off_frames : sequence of int, default ()
        Frames of every cube whose star sits 30 px right of the center (a chunk whose centering
        failed).
    """
    red = os.path.join(root, night, "reduced_py")
    os.makedirs(red)
    yy, xx = np.mgrid[0:300, 0:300]
    star = np.exp(-((xx - 149.5) ** 2 + (yy - 149.5) ** 2) / 72.0).astype(np.float32)
    off = np.exp(-((xx - 179.5) ** 2 + (yy - 149.5) ** 2) / 72.0).astype(np.float32)
    h = fits.Header()
    h["XCEN"], h["YCEN"] = 149.5, 149.5
    for cn in (1, 2):
        pre = f"{night}_left"
        k = (n_by_cube or {}).get(cn, n)
        ang = np.linspace(-10, 10, k) if angles is None else np.asarray(angles[:k], dtype=float)
        for tag in ("", "_clean"):
            img = off if cn == off_cube else star
            frames = np.repeat(img[None], k, axis=0)
            for j in off_frames:
                if j < k:
                    frames[j] = off
            for j in range(k if nan_rows is not None else 0):
                frames[j, :nan_rows[j]] = np.nan
            fits.PrimaryHDU(frames, h).writeto(os.path.join(red, f"{pre}_cube{cn}_align_cen{tag}_bin.fits"))
            fits.PrimaryHDU(ang).writeto(os.path.join(red, f"{pre}_angles{cn}{tag}_bin.fits"))
        fits.PrimaryHDU(star, h).writeto(os.path.join(red, f"{pre}_cube{cn}_align_cen_clean_median_PSF.fits"))


def test_inputs_centered_with_references(tmp_path):
    """Cut cubes and PSFs keep the star at the array center; references stack; the command lists every partition."""
    root = str(tmp_path)
    _night(root, "Fake_20220101", 10)
    _night(root, "Fake_20220102", 7)
    r0 = subprocess.run([sys.executable, TOOL, root, "Fake_20220101", "--tag", "rdi", "--half", "40", "--psf-half", "10",
                         "--ref", "Fake_20220102"], capture_output=True, text=True)
    assert r0.returncode != 0 and "--ref-min-dpa" in r0.stderr              # the exclusion must be chosen
    r = subprocess.run([sys.executable, TOOL, root, "Fake_20220101", "--tag", "rdi", "--half", "40", "--psf-half", "10",
                        "--ref", "Fake_20220102", "Missing_1", "--ref-min-dpa", "0", "--min-frames", "5"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    d = os.path.join(root, "Fake_20220101", "klip_tpe", "inputs_rdi")
    cube = fits.getdata(os.path.join(d, "Fake_20220101_left_cube1_all_c80.fits"))
    ref = fits.getdata(os.path.join(d, "Fake_20220101_left_cube1_ref_c80.fits"))
    psf = fits.getdata(os.path.join(d, "Fake_20220101_left_cube1_psf20.fits"))
    assert cube.shape == (10, 80, 80) and ref.shape == (7, 80, 80) and psf.shape == (20, 20)
    yy, xx = np.mgrid[0:80, 0:80]
    img = cube[0]
    assert abs((img * xx).sum() / img.sum() - 39.5) < 1e-4 and abs((img * yy).sum() / img.sum() - 39.5) < 1e-4
    m = json.load(open(os.path.join(d, "manifest.json")))
    assert [p["name"] for p in m["partitions"]] == ["left_A", "left_B"]
    assert m["partitions"][0]["ref_nights_used"] == ["Fake_20220102"]
    sh = open(os.path.join(root, "Fake_20220101", "klip_tpe", "run_rdi.sh")).read()
    assert "set -euo pipefail" in sh and "--ref-cube" in sh and "--names left_A left_B" in sh
    assert '--workers "${WORKERS:-auto}"' in sh
    smoke = open(os.path.join(root, "Fake_20220101", "klip_tpe", "run_rdi_smoke.sh")).read()
    assert "--n-iter 10" in smoke and "smoke_rdi" in smoke and "--ann-edges 11 34 " in smoke
    again = subprocess.run([sys.executable, TOOL, root, "Fake_20220101", "--tag", "rdi"], capture_output=True, text=True)
    assert again.returncode != 0                         # an existing tag is refused
    r2 = subprocess.run([sys.executable, TOOL, root, "Fake_20220101", "--tag", "adi", "--half", "40", "--psf-half", "10",
                         "--command-archive", "/mnt/RAID64TB/NOMIC"], capture_output=True, text=True)
    assert r2.returncode == 0, r2.stderr
    sh2 = open(os.path.join(root, "Fake_20220101", "klip_tpe", "run_adi.sh")).read()
    assert "/mnt/RAID64TB/NOMIC/Fake_20220101/klip_tpe/inputs_adi/Fake_20220101_left_cube1_all_c80.fits" in sh2
    assert root not in sh2
    assert "--show" in sh2 and "--no-display" not in sh2                 # the live window by default
    r3 = subprocess.run([sys.executable, TOOL, root, "Fake_20220101", "--tag", "quiet", "--half", "40", "--psf-half", "10",
                         "--display", "none"], capture_output=True, text=True)
    assert r3.returncode == 0, r3.stderr
    sh3 = open(os.path.join(root, "Fake_20220101", "klip_tpe", "run_quiet.sh")).read()
    smoke3 = open(os.path.join(root, "Fake_20220101", "klip_tpe", "run_quiet_smoke.sh")).read()
    assert "--no-display" in sh3 and "--show" not in sh3 and "--no-display" in smoke3


def test_fixed_injection_contrast_written_and_checked(tmp_path):
    """--use-contrast reaches both scripts and the manifest; without it neither script has it; bad values stop the tool."""
    root = str(tmp_path)
    _night(root, "Fake_20220101", 10)
    base = [sys.executable, TOOL, root, "Fake_20220101", "--half", "40", "--psf-half", "10"]
    r = subprocess.run(base + ["--tag", "fix", "--ann-edges", "0", "20", "35", "--n-iter", "20", "20",
                               "--n-init", "5", "5", "--use-contrast", "2e-4"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    kt = os.path.join(root, "Fake_20220101", "klip_tpe")
    sh = open(os.path.join(kt, "run_fix.sh")).read()
    smoke = open(os.path.join(kt, "run_fix_smoke.sh")).read()
    assert "--contrast0 1e-3 --use-contrast 2e-4 \\" in sh and "--use-contrast 2e-4" in smoke
    assert json.load(open(os.path.join(kt, "inputs_fix", "manifest.json")))["use_contrast"] == [2e-4]
    r2 = subprocess.run(base + ["--tag", "cal"], capture_output=True, text=True)
    assert r2.returncode == 0, r2.stderr
    assert "--use-contrast" not in open(os.path.join(kt, "run_cal.sh")).read()
    assert json.load(open(os.path.join(kt, "inputs_cal", "manifest.json")))["use_contrast"] is None
    for bad in (["-1e-4"], ["abc"], ["nan"]):
        r3 = subprocess.run(base + ["--tag", "bad", "--use-contrast"] + bad, capture_output=True, text=True)
        assert r3.returncode != 0 and "--use-contrast" in r3.stderr
        assert not os.path.exists(os.path.join(kt, "inputs_bad"))


def test_transit_angles_unwrapped_and_short_cube_skipped(tmp_path):
    """Angles crossing ±180° are written without the jump, a bin averaged across the wrap is left
    out, and a cube below --min-frames is left out."""
    root = str(tmp_path)
    wrong = 61.0                                         # a bin averaged across the wrap
    full = np.array([170.0, 175.0, 179.5, wrong, -178.0, -172.0, -165.0, -160.0, -150.0, -140.0, -130.0, -120.0, -110.0])
    ang = np.delete(full, 3)
    _night(root, "Fake_20220521", 13, n_by_cube={2: 1}, angles=full)
    r = subprocess.run([sys.executable, TOOL, root, "Fake_20220521", "--tag", "adi", "--half", "40", "--psf-half", "10"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    d = os.path.join(root, "Fake_20220521", "klip_tpe", "inputs_adi")
    out = fits.getdata(os.path.join(d, "Fake_20220521_left_angles1_all.fits"))
    assert np.allclose(out, np.where(ang < 0, ang + 360.0, ang))        # 170 ... 179.5, 182 ... 250
    assert np.all(np.diff(out) > 0)
    m = json.load(open(os.path.join(d, "manifest.json")))
    assert [p["name"] for p in m["partitions"]] == ["left_A"]
    p = m["partitions"][0]
    assert p["unwrapped"] and abs(p["rotation_deg"] - 80.0) < 1e-9 and p["angle_first"] == 170.0
    assert p["frames"] == 12 and p["dropped_wrap_averaged"] == [{"index": 3, "angle_deg": wrong}]
    assert fits.getdata(os.path.join(d, "Fake_20220521_left_cube1_all_c80.fits")).shape[0] == 12
    assert m["skipped"] == [{"name": "left_B", "frames": 1, "reason": "fewer than 10 frames"}]
    assert not os.path.exists(os.path.join(d, "Fake_20220521_left_cube2_all_c80.fits"))
    sh = open(os.path.join(root, "Fake_20220521", "klip_tpe", "run_adi.sh")).read()
    assert "--names left_A \\" in sh and "cube2" not in sh


def test_reference_frames_near_the_science_angles_left_out(tmp_path):
    """Reference frames within --ref-min-dpa of any science angle are left out of the library."""
    root = str(tmp_path)
    _night(root, "Fake_20220101", 10)                                    # science angles -10 .. 10
    _night(root, "Fake_20220102", 7, angles=np.array([-50.0, -40.0, -30.0, -12.0, 0.0, 12.0, 30.0]))
    r = subprocess.run([sys.executable, TOOL, root, "Fake_20220101", "--tag", "rdi", "--half", "40", "--psf-half", "10",
                        "--ref", "Fake_20220102", "--ref-min-dpa", "15", "--min-frames", "5"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    d = os.path.join(root, "Fake_20220101", "klip_tpe", "inputs_rdi")
    assert fits.getdata(os.path.join(d, "Fake_20220101_left_cube1_ref_c80.fits")).shape[0] == 4   # -50, -40, -30, 30
    m = json.load(open(os.path.join(d, "manifest.json")))
    det = m["partitions"][0]["ref_detail"][0]
    assert (det["frames"], det["within_min_dpa"], det["used"], det["wrap_averaged"]) == (7, 3, 4, 0)
    assert m["partitions"][0]["ref_min_dpa"] == 15.0
    _night(root, "Fake_20220103", 1, angles=np.array([-60.0]))           # a failed reduction: one frame
    r1 = subprocess.run([sys.executable, TOOL, root, "Fake_20220101", "--tag", "rdi1", "--half", "40", "--psf-half", "10",
                         "--ref", "Fake_20220102", "Fake_20220103", "--ref-min-dpa", "15", "--min-frames", "5"], capture_output=True, text=True)
    assert r1.returncode == 0, r1.stderr
    m1 = json.load(open(os.path.join(root, "Fake_20220101", "klip_tpe", "inputs_rdi1", "manifest.json")))
    assert m1["partitions"][0]["ref_frames"] == 4 and m1["partitions"][0]["ref_detail"][1]["used"] == 0
    r2 = subprocess.run([sys.executable, TOOL, root, "Fake_20220101", "--tag", "rdi2", "--half", "40", "--psf-half", "10",
                         "--ref", "Fake_20220102", "--ref-min-dpa", "60", "--min-frames", "5"], capture_output=True, text=True)
    assert r2.returncode == 1 and "skipped" in r2.stdout                # no reference frame is 60 deg away
    assert not os.path.exists(os.path.join(root, "Fake_20220101", "klip_tpe", "inputs_rdi2"))   # nothing left behind


def test_star_off_center_left_out(tmp_path):
    """A science cube or a reference cube whose star is not at the center is left out."""
    root = str(tmp_path)
    _night(root, "Fake_20220520", 12, off_cube=2)                      # chop B centered on the wrong place
    _night(root, "Fake_20220601", 12, angles=np.linspace(170, 150, 12), off_cube=1)
    r = subprocess.run([sys.executable, TOOL, root, "Fake_20220520", "--tag", "adi", "--half", "40", "--psf-half", "10"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    m = json.load(open(os.path.join(root, "Fake_20220520", "klip_tpe", "inputs_adi", "manifest.json")))
    assert [p["name"] for p in m["partitions"]] == ["left_A"]
    assert m["skipped"][0]["name"] == "left_B" and "not at the center" in m["skipped"][0]["reason"]
    r2 = subprocess.run([sys.executable, TOOL, root, "Fake_20220520", "--tag", "rdi", "--half", "40", "--psf-half", "10",
                         "--ref", "Fake_20220601", "--ref-min-dpa", "30"], capture_output=True, text=True)
    assert r2.returncode == 1 and "no reference frames" in r2.stdout      # the one reference cube is off center
    m2 = os.path.join(root, "Fake_20220520", "klip_tpe", "inputs_rdi", "manifest.json")
    assert not os.path.exists(m2)


def test_star_offset_ignores_far_artifacts():
    """A bright artifact far from the center does not hide a centered star; no star near the center fails."""
    sys.path.insert(0, os.path.dirname(TOOL))
    import klip_tpe_inputs as k
    yy, xx = np.mgrid[0:240, 0:240]
    star = np.exp(-((xx - 119.5) ** 2 + (yy - 119.5) ** 2) / 72.0)
    img = 0.94 * star
    img[238, 43] = 1.0                                      # an edge artifact brighter than the star
    off, pk, ratio = k.star_offset(img)
    assert off < 1.0 and ratio > 0.9
    empty = np.zeros((240, 240))
    empty[14, 1] = 0.8
    empty[120, 113] = 0.2                                   # noise near the center, the bright thing at the edge
    off2, pk2, ratio2 = k.star_offset(empty)
    assert ratio2 < 0.5


def test_frames_with_nan_in_the_cut_left_out(tmp_path):
    """Science and reference frames with NaN padding inside the cut are left out and listed; a
    narrower cut that stays finite keeps them; too few finite frames skips the partition."""
    root = str(tmp_path)
    rows = [0] * 12
    rows[3], rows[5], rows[7] = 125, 111, 125           # NaN up to row 124 (110) reaches 15 px (1 px) into a 40 px half cut
    _night(root, "Fake_20220522", 12, nan_rows=rows)
    ref_rows = [125, 125, 0, 0, 0, 0, 0]
    _night(root, "Fake_20220523", 7, angles=np.array([-50.0, -40.0, -30.0, 30.0, 40.0, 50.0, 60.0]), nan_rows=ref_rows)
    r = subprocess.run([sys.executable, TOOL, root, "Fake_20220522", "--tag", "rdi", "--half", "40", "--psf-half", "10",
                        "--ref", "Fake_20220523", "--ref-min-dpa", "15", "--min-frames", "5"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "left out 3 of 12 frame(s) with NaN" in r.stdout
    d = os.path.join(root, "Fake_20220522", "klip_tpe", "inputs_rdi")
    cube = fits.getdata(os.path.join(d, "Fake_20220522_left_cube1_all_c80.fits"))
    ang = fits.getdata(os.path.join(d, "Fake_20220522_left_angles1_all.fits"))
    ref = fits.getdata(os.path.join(d, "Fake_20220522_left_cube1_ref_c80.fits"))
    assert cube.shape == (9, 80, 80) and np.all(np.isfinite(cube))
    assert np.allclose(ang, np.delete(np.linspace(-10, 10, 12), [3, 5, 7]))
    assert ref.shape == (5, 80, 80) and np.all(np.isfinite(ref))
    m = json.load(open(os.path.join(d, "manifest.json")))
    p = m["partitions"][0]
    assert p["frames"] == 9 and p["dropped_nan"] == [3, 5, 7]
    assert p["finite_half_px"] == {"min": 25, "p05": 25, "median": 40}  # 150 - 125 = 25 px; 150 - 111 = 39 px
    det = p["ref_detail"][0]
    assert (det["frames"], det["nan_within_cut"], det["used"]) == (7, 2, 5)
    r2 = subprocess.run([sys.executable, TOOL, root, "Fake_20220522", "--tag", "adi20", "--half", "20", "--psf-half", "10",
                         "--min-frames", "5"], capture_output=True, text=True)
    assert r2.returncode == 0, r2.stderr
    m2 = json.load(open(os.path.join(root, "Fake_20220522", "klip_tpe", "inputs_adi20", "manifest.json")))
    assert m2["partitions"][0]["frames"] == 12 and m2["partitions"][0]["dropped_nan"] == []
    assert m2["partitions"][0]["finite_half_px"]["min"] == 20                 # capped at the cut's own half-width
    many = [125] * 5 + [0] * 7                                                # 7 finite frames, fewer than 10
    _night(root, "Fake_20220524", 12, nan_rows=many)
    r3 = subprocess.run([sys.executable, TOOL, root, "Fake_20220524", "--tag", "adi", "--half", "40", "--psf-half", "10"],
                        capture_output=True, text=True)
    assert r3.returncode == 1 and "fewer than --min-frames" in r3.stdout
    assert not os.path.exists(os.path.join(root, "Fake_20220524", "klip_tpe", "inputs_adi"))


def test_frames_with_nan_kept_with_keep_nan(tmp_path):
    """--keep-nan keeps science and reference frames with NaN in the cut, NaN and all, and the
    manifest counts them; --min-coverage reaches both scripts; a bad fraction stops the tool."""
    root = str(tmp_path)
    rows = [0] * 12
    rows[3], rows[5], rows[7] = 125, 111, 125           # as in the test above: 3 frames with NaN in a 40 px half cut
    _night(root, "Fake_20220522", 12, nan_rows=rows)
    ref_rows = [125, 125, 0, 0, 0, 0, 0]
    _night(root, "Fake_20220523", 7, angles=np.array([-50.0, -40.0, -30.0, 30.0, 40.0, 50.0, 60.0]), nan_rows=ref_rows)
    r = subprocess.run([sys.executable, TOOL, root, "Fake_20220522", "--tag", "pad", "--half", "40", "--psf-half", "10",
                        "--ref", "Fake_20220523", "--ref-min-dpa", "15", "--min-frames", "5", "--keep-nan",
                        "--min-coverage", "0"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "kept 3 of 12 frame(s) with NaN" in r.stdout
    d = os.path.join(root, "Fake_20220522", "klip_tpe", "inputs_pad")
    cube = fits.getdata(os.path.join(d, "Fake_20220522_left_cube1_all_c80.fits"))
    ref = fits.getdata(os.path.join(d, "Fake_20220522_left_cube1_ref_c80.fits"))
    assert cube.shape == (12, 80, 80) and ref.shape == (7, 80, 80)
    bad = [k for k in range(12) if not np.all(np.isfinite(cube[k]))]
    assert bad == [3, 5, 7]                                                  # the padding is kept as NaN
    assert np.all(np.isnan(cube[3, :15])) and np.all(np.isfinite(cube[3, 15:]))
    m = json.load(open(os.path.join(d, "manifest.json")))
    assert m["keep_nan"] is True and m["min_coverage"] == 0.0
    p = m["partitions"][0]
    assert p["frames"] == 12 and p["dropped_nan"] == [] and p["frames_with_nan"] == 3
    assert abs(p["nan_fraction"] - (15 + 1 + 15) * 80 / (12 * 80 * 80)) < 1e-9
    det = p["ref_detail"][0]
    assert (det["nan_within_cut"], det["nan_kept"], det["used"]) == (2, True, 7)
    kt = os.path.join(root, "Fake_20220522", "klip_tpe")
    assert "--min-coverage 0 \\" in open(os.path.join(kt, "run_pad.sh")).read()
    assert "--min-coverage 0" in open(os.path.join(kt, "run_pad_smoke.sh")).read()
    r2 = subprocess.run([sys.executable, TOOL, root, "Fake_20220522", "--tag", "bad", "--half", "40", "--psf-half", "10",
                         "--min-coverage", "1.5"], capture_output=True, text=True)
    assert r2.returncode != 0 and "--min-coverage" in r2.stderr


def test_finite_half_counts_from_the_center():
    """The largest finite centered square: no NaN gives the cut's half-width, a NaN row or column
    limits it to its distance from the center."""
    sys.path.insert(0, os.path.dirname(TOOL))
    import klip_tpe_inputs as k
    c = np.ones((4, 80, 80), dtype=np.float32)
    c[1, 0] = np.nan                     # bottom row: 39.5 px from the center, half-width 39
    c[2, :, 70] = np.nan                 # a column 30.5 px right of the center: 30
    c[3, 41, 41] = np.inf                # 1.5 px from the center: 1
    assert k.finite_half(c).tolist() == [40, 39, 30, 1]


def test_frames_with_the_star_off_center_left_out(tmp_path):
    """Single science or reference frames whose star is far from the center (a chunk whose
    centering failed) are left out and listed; the median-image check alone would pass them."""
    root = str(tmp_path)
    _night(root, "Fake_20220521", 12, off_frames=(4, 9))
    _night(root, "Fake_20220522", 7, angles=np.array([-50.0, -40.0, -30.0, 30.0, 40.0, 50.0, 60.0]), off_frames=(0,))
    r = subprocess.run([sys.executable, TOOL, root, "Fake_20220521", "--tag", "rdi", "--half", "40", "--psf-half", "10",
                        "--ref", "Fake_20220522", "--ref-min-dpa", "15", "--min-frames", "5"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "left out 2 frame(s) whose star is more than 5.0 px from the center" in r.stdout
    d = os.path.join(root, "Fake_20220521", "klip_tpe", "inputs_rdi")
    m = json.load(open(os.path.join(d, "manifest.json")))
    p = m["partitions"][0]
    assert p["frames"] == 10 and [x["index"] for x in p["dropped_off_center"]] == [4, 9]
    assert abs(p["dropped_off_center"][0]["offset_px"] - 29.5) < 0.1     # the brightest pixel of a star 30 px off
    ang = fits.getdata(os.path.join(d, "Fake_20220521_left_angles1_all.fits"))
    assert np.allclose(ang, np.delete(np.linspace(-10, 10, 12), [4, 9]))
    det = p["ref_detail"][0]
    assert (det["off_center"], det["used"]) == (1, 6)                     # frame 0 (-50 deg) left out
    sys.path.insert(0, os.path.dirname(TOOL))
    import klip_tpe_inputs as k
    cube = fits.getdata(os.path.join(d, "Fake_20220521_left_cube1_all_c80.fits"))
    assert np.all(k.frame_offsets(cube) < 1.0)                            # 0.71 px: the half-pixel center


def _fake_venv(d):
    """A folder ``bin`` with ``klip-tpe`` and ``python`` that print their name and arguments, one per line."""
    b = os.path.join(d, "venv", "bin")
    os.makedirs(b)
    for name in ("klip-tpe", "python"):
        p = os.path.join(b, name)
        with open(p, "w") as fh:
            fh.write('#!/usr/bin/env bash\nprintf "%s\\n" "$(basename "$0")" "$@"\n')
        os.chmod(p, 0o755)
    return os.path.join(b, "klip-tpe")


def test_resume_script_repeats_the_data_arguments_and_link_runs_the_wrapper(tmp_path):
    """resume_<TAG>.sh gives klip-tpe resume the run's data arguments and nothing the checkpoint holds;
    --link runs all three scripts through the copied klip_tpe_linked.py with the venv's Python."""
    root = str(tmp_path)
    _night(root, "Fake_20220101", 10)
    base = [sys.executable, TOOL, root, "Fake_20220101", "--half", "40", "--psf-half", "10",
            "--ann-edges", "0", "20", "35", "--n-iter", "20", "20", "--n-init", "5", "5"]
    kt = os.path.join(root, "Fake_20220101", "klip_tpe")
    r = subprocess.run(base + ["--tag", "plain"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    run = open(os.path.join(kt, "run_plain.sh")).read().splitlines()
    res = open(os.path.join(kt, "resume_plain.sh")).read().splitlines()
    data = [ln for ln in run if ln.startswith(("  --cube", "  --angles", "  --psf", "  --names", "  --pxscale"))]
    assert len(data) == 5 and [ln for ln in res if ln in data] == data
    assert '"${KT:?}" resume --instrument generic \\' in res
    assert '  --workers "${WORKERS:-auto}" --show \\' in res and res[-1] == '  --run-dir "${RUN:?}" "$@"'
    assert not any(opt in ln for ln in res for opt in ("--n-iter", "--ann-edges", "--use-contrast", "--seed"))
    assert json.load(open(os.path.join(kt, "inputs_plain", "manifest.json")))["link"] is None
    r = subprocess.run(base + ["--tag", "lnk", "--use-contrast", "5e-4", "--link", "bin", "filter", "angsep",
                               "anglemax", "angsep"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    wrapper = os.path.join(kt, "inputs_lnk", "klip_tpe_linked.py")
    src = os.path.join(os.path.dirname(TOOL), "klip_tpe_linked.py")
    assert open(wrapper, "rb").read() == open(src, "rb").read()
    m = json.load(open(os.path.join(kt, "inputs_lnk", "manifest.json")))
    assert m["link"]["bases"] == ["bin", "filter", "angsep", "anglemax"] and m["link"]["wrapper"] == wrapper
    sys.path.insert(0, os.path.dirname(TOOL))
    import klip_tpe_inputs as k
    assert m["link"]["wrapper_sha256"] == k.sha256_of(wrapper)
    for name in ("run_lnk.sh", "run_lnk_smoke.sh", "resume_lnk.sh", "run_plain.sh", "resume_plain.sh"):
        assert subprocess.run(["bash", "-n", os.path.join(kt, name)]).returncode == 0, name
    # what each script hands over, with stand-ins for klip-tpe and the venv's Python
    env = dict(os.environ, KT=_fake_venv(root))
    out = subprocess.run(["bash", os.path.join(kt, "run_lnk.sh")], env=env, capture_output=True, text=True)
    args = out.stdout.splitlines()
    assert out.returncode == 0 and args[:8] == ["python", wrapper, "--link", "bin", "filter", "angsep",
                                                 "anglemax", "--"]
    assert args[8] == "generic" and "--use-contrast" in args and args[-1] == os.path.join(kt, "run_lnk")
    out = subprocess.run(["bash", os.path.join(kt, "resume_lnk.sh")], env=env, capture_output=True, text=True)
    assert out.returncode != 0 and "has no checkpoint.json" in out.stdout
    os.makedirs(os.path.join(kt, "run_lnk"))
    open(os.path.join(kt, "run_lnk", "checkpoint.json"), "w").close()
    out = subprocess.run(["bash", os.path.join(kt, "resume_lnk.sh"), "--gpu-share", "auto", "--gpu-device", "GPU-x"],
                         env=env, capture_output=True, text=True)
    args = out.stdout.splitlines()
    assert out.returncode == 0 and args[8:11] == ["resume", "--instrument", "generic"]
    assert args[-6:] == ["--run-dir", os.path.join(kt, "run_lnk"), "--gpu-share", "auto", "--gpu-device", "GPU-x"]
    out = subprocess.run(["bash", os.path.join(kt, "resume_lnk.sh")], env=env, capture_output=True, text=True)
    args = out.stdout.splitlines()
    assert "--use-contrast" not in args and args[-1] == os.path.join(kt, "run_lnk")
    open(os.path.join(kt, "run_lnk", "final_results.json"), "w").close()
    out = subprocess.run(["bash", os.path.join(kt, "resume_lnk.sh")], env=env, capture_output=True, text=True)
    assert out.returncode != 0 and "is already finished" in out.stdout
    os.makedirs(os.path.join(kt, "run_plain"))
    open(os.path.join(kt, "run_plain", "checkpoint.json"), "w").close()
    out = subprocess.run(["bash", os.path.join(kt, "resume_plain.sh")], env=env, capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.splitlines()[:4] == ["klip-tpe", "resume", "--instrument", "generic"]
    r = subprocess.run(base + ["--tag", "bad", "--link", "bin;rm"], capture_output=True, text=True)
    assert r.returncode != 0 and "--link takes parameter names" in r.stderr
    assert not os.path.exists(os.path.join(kt, "inputs_bad"))


def test_batch_written_into_both_scripts_and_checked(tmp_path):
    """--batch N reaches the run and smoke scripts and the manifest; without it neither script has it."""
    root = str(tmp_path)
    _night(root, "Fake_20220101", 10)
    base = [sys.executable, TOOL, root, "Fake_20220101", "--half", "40", "--psf-half", "10"]
    kt = os.path.join(root, "Fake_20220101", "klip_tpe")
    r = subprocess.run(base + ["--tag", "b16", "--use-contrast", "5e-4", "--batch", "16"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    sh = open(os.path.join(kt, "run_b16.sh")).read()
    assert "--contrast0 1e-3 --use-contrast 5e-4 --batch 16 \\" in sh
    assert "--batch 16" in open(os.path.join(kt, "run_b16_smoke.sh")).read()
    assert "--batch" not in open(os.path.join(kt, "resume_b16.sh")).read()       # the checkpoint holds it
    assert json.load(open(os.path.join(kt, "inputs_b16", "manifest.json")))["batch"] == 16
    r = subprocess.run(base + ["--tag", "b1"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "--batch" not in open(os.path.join(kt, "run_b1.sh")).read()
    assert json.load(open(os.path.join(kt, "inputs_b1", "manifest.json")))["batch"] == 1
    for bad in ("0", "-2"):
        r = subprocess.run(base + ["--tag", "bad", "--batch", bad], capture_output=True, text=True)
        assert r.returncode != 0 and "--batch" in r.stderr
        assert not os.path.exists(os.path.join(kt, "inputs_bad"))


def test_several_nights_joined_into_one_run(tmp_path):
    """Two nights become four labeled partitions of one run in --out-folder; the checks refuse bad setups."""
    root = str(tmp_path)
    _night(root, "Fake_20220101", 10)
    _night(root, "Fake_20220102", 12, angles=np.linspace(20, 40, 12))
    joint = "Fake_20220101-20220102"
    os.makedirs(os.path.join(root, joint))
    base = [sys.executable, TOOL, root, "Fake_20220101", "Fake_20220102", "--half", "40", "--psf-half", "10"]
    r = subprocess.run(base + ["--tag", "nofolder"], capture_output=True, text=True)
    assert r.returncode != 0 and "--out-folder" in r.stderr                       # where would it go?
    r = subprocess.run(base + ["--tag", "j", "--out-folder", joint, "--labels", "P1"], capture_output=True, text=True)
    assert r.returncode != 0 and "one label per night" in r.stderr
    r = subprocess.run(base + ["--tag", "j", "--out-folder", joint, "--labels", "P1", "P_2"], capture_output=True, text=True)
    assert r.returncode != 0 and "letters and digits" in r.stderr
    r = subprocess.run(base + ["--tag", "j", "--out-folder", joint, "--ref", "Fake_20220102", "--ref-min-dpa", "0"],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "science nights" in r.stderr
    r = subprocess.run(base + ["--tag", "j", "--out-folder", joint, "--labels", "P1", "P2", "--link", "bin", "filter"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    kt = os.path.join(root, joint, "klip_tpe")
    man = json.load(open(os.path.join(kt, "inputs_j", "manifest.json")))
    assert man["night"] == joint and man["nights"] == ["Fake_20220101", "Fake_20220102"] and man["labels"] == ["P1", "P2"]
    assert [p["name"] for p in man["partitions"]] == ["P1_left_A", "P1_left_B", "P2_left_A", "P2_left_B"]
    assert [p["night"] for p in man["partitions"]] == ["Fake_20220101"] * 2 + ["Fake_20220102"] * 2
    assert [p["frames"] for p in man["partitions"]] == [10, 10, 12, 12]
    sh = open(os.path.join(kt, "run_j.sh")).read()
    assert "--names P1_left_A P1_left_B P2_left_A P2_left_B \\" in sh
    assert sh.count("_cube1_all_c80.fits") == 2 and sh.count("_cube2_all_c80.fits") == 2
    assert os.path.join(root, joint, "klip_tpe", "run_j") in sh
    assert "for Fake_20220101-20220102 (Fake_20220101 + Fake_20220102)" in sh
    for name in ("run_j_smoke.sh", "resume_j.sh"):
        assert "--names P1_left_A P1_left_B P2_left_A P2_left_B \\" in open(os.path.join(kt, name)).read()
    ang = fits.getdata(os.path.join(kt, "inputs_j", "Fake_20220102_left_angles1_all.fits"))
    assert np.allclose(ang, np.linspace(20, 40, 12))
    # without labels, each night's UT date labels its partitions
    r = subprocess.run(base + ["--tag", "jd", "--out-folder", joint], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    man = json.load(open(os.path.join(kt, "inputs_jd", "manifest.json")))
    assert [p["name"] for p in man["partitions"]] == ["20220101_left_A", "20220101_left_B",
                                                      "20220102_left_A", "20220102_left_B"]
    # one night keeps its unlabeled names and its own folder
    r = subprocess.run([sys.executable, TOOL, root, "Fake_20220101", "--half", "40", "--psf-half", "10", "--tag", "one"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    man = json.load(open(os.path.join(root, "Fake_20220101", "klip_tpe", "inputs_one", "manifest.json")))
    assert man["night"] == "Fake_20220101" and man["labels"] is None
    assert [p["name"] for p in man["partitions"]] == ["left_A", "left_B"]
