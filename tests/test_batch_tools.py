"""Tests of the helpers shared by the run and the batch tools, the pre-flight and the batch driver."""
import csv
import json
import os
import sys

import pytest

from reduce_nomic.pipeline import planned_outputs, sky_folders

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import batch_reduce  # noqa: E402
import preflight_nights  # noqa: E402
from test_names import make_archive  # noqa: E402
from test_pipeline_synthetic import make_night  # noqa: E402


def test_planned_outputs():
    """A run plans 24 products, all under its prefix."""
    out = planned_outputs("Procyon_20230205_left")
    assert len(out) == 24 and len(set(out)) == 24
    assert all(f.startswith("Procyon_20230205_left_") for f in out)
    assert "Procyon_20230205_left_cube2_align_cen_bin.fits" in out


def test_sky_folders(tmp_path):
    """Round k reads skys<min(k, 3)>; explicit folders are renamed and cut to n_sky."""
    make_archive(str(tmp_path))
    a = str(tmp_path)
    assert sky_folders({"n_sky": 0, "archive": a}) == []
    assert sky_folders({"n_sky": 5, "archive": a}) == ["skys1", "skys2", "skys3", "skys3", "skys3"]
    assert sky_folders({"n_sky": 1, "archive": a, "sky_dirs": ["../Alcor/skys1", "x"]}) == ["../Alcor_20230407/skys1"]


def test_preflight_flags_missing_skies_and_existing_outputs(tmp_path, capsys):
    """The pre-flight counts frames and flags a missing sky folder and an existing product."""
    nights = make_archive(str(tmp_path))
    a = str(tmp_path)
    blocks = json.load(open(nights))
    blocks["Procyon_13"]["settings"].update({"side": "both", "n_sky": 2})
    json.dump(blocks, open(nights, "w"))
    sky = os.path.join(a, "Procyon_20230205", "skys1")
    os.makedirs(sky)
    open(os.path.join(sky, "n_230205_000100.fits"), "w").close()
    out = os.path.join(a, "Procyon_20230205", "reduced_py")
    os.makedirs(out)
    open(os.path.join(out, "Procyon_20230205_right_summary.json"), "w").close()
    table = os.path.join(a, "pre.csv")
    assert preflight_nights.main([a, table, "--nights", "Procyon_13", "--nights-json", nights]) == 1
    rows = {r["side"]: r for r in csv.DictReader(open(table))}
    assert rows["left"]["n_science"] == "4" and rows["left"]["sky_folders"] == "skys1:1;skys2:MISSING"
    assert rows["left"]["problems"] == "missing sky folders skys2"
    assert rows["right"]["existing_outputs"] == "1" and "1 outputs exist" in rows["right"]["problems"]
    assert rows["left"]["night_old"] == "Procyon_13" and rows["left"]["settings"] == "nights.json:Procyon_13"
    with pytest.raises(SystemExit):
        preflight_nights.main([a, table, "--nights", "Procyon_13", "--nights-json", nights])


SYNTH = {"obj": "fakestar", "prebin": 5, "background_limit": 50000.0,
         "pre_corr_thresh_1": 0.5, "pre_corr_thresh_2": 0.5, "corr_thresh_1": 0.5, "corr_thresh_2": 0.5,
         "pre_noise_thresh_1": 3, "pre_noise_thresh_2": 3, "noise_thresh_1": 3, "noise_thresh_2": 3,
         "pre_peak_thresh_1": 0.5, "pre_peak_thresh_2": 0.5, "peak_thresh_1": 0.5, "peak_thresh_2": 0.5}


def write_plan(path: str, rows: list) -> None:
    """Write a batch plan.

    Parameters
    ----------
    path : str
        Output CSV.
    rows : list of dict
        Rows with ``id``, ``night``, ``side``, ``out``, ``set`` (dict) and ``note``.

    Returns
    -------
    None
        Writes ``path``.
    """
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "night", "side", "out", "set", "note"])
        w.writeheader()
        for r in rows:
            w.writerow(dict(r, set=json.dumps(r["set"])))


def test_run_state(tmp_path):
    """A run is done with its summary, partial with only some products, and to do with none."""
    out = str(tmp_path)
    assert batch_reduce.run_state(out, "N_1", "left") == "todo"
    open(os.path.join(out, "N_1_left_badmap.fits"), "w").close()
    assert batch_reduce.run_state(out, "N_1", "left") == "partial"
    open(os.path.join(out, "N_1_left_summary.json"), "w").close()
    assert batch_reduce.run_state(out, "N_1", "left") == "done"
    assert batch_reduce.run_state(out, "N_1", "right") == "todo"


def test_command_passes_settings_as_json(tmp_path):
    """Settings reach the command line as NAME=JSON, with the output and work folders."""
    row = {"night": "Procyon_20230407", "side": "left", "out": "reduced_py/skytest_nosky",
           "set": json.dumps({"n_sky": 0, "sky_dirs": ["../Alcor_20230407/skys1"], "center_method": "idl"})}
    argv = batch_reduce.command(row, "/arc", "/arc/_batch/work", 64, None, False)
    assert argv[1:4] == ["-m", "reduce_nomic.cli", "Procyon_20230407"]
    assert argv[argv.index("--out") + 1] == "/arc/Procyon_20230407/reduced_py/skytest_nosky"
    assert argv[argv.index("--work") + 1] == "/arc/_batch/work"
    sets = [argv[i + 1] for i, x in enumerate(argv) if x == "--set"]
    assert sets == ["n_sky=0", 'sky_dirs=["../Alcor_20230407/skys1"]', 'center_method="idl"']
    assert "--display" not in argv and "--nights-json" not in argv


def test_batch_runs_skips_and_reports(tmp_path, capsys):
    """A batch reduces a synthetic night, skips it when run again, and reports a partial run."""
    root = str(tmp_path / "archive")
    make_night(root)
    nights = str(tmp_path / "none.json")
    plan = str(tmp_path / "plan.csv")
    write_plan(plan, [{"id": "a", "night": "Fake_1", "side": "left", "out": "", "set": SYNTH, "note": ""},
                      {"id": "b", "night": "Fake_1", "side": "left", "out": "reduced_py/x", "set": SYNTH, "note": ""}])
    os.makedirs(os.path.join(root, "Fake_1", "reduced_py", "x"))
    open(os.path.join(root, "Fake_1", "reduced_py", "x", "Fake_1_left_badmap.fits"), "w").close()
    work = str(tmp_path / "work")
    args = [plan, root, "--work", work, "--workers", "2", "--nights-json", nights, "--min-free-gb", "0"]
    assert batch_reduce.main(args + ["--dry-run"]) == 1
    assert "1 partial, 1 to do" in capsys.readouterr().out
    assert batch_reduce.main(args) == 1
    out = capsys.readouterr().out
    assert "a (Fake_1 left, reduced_py): done in" in out and "b (Fake_1 left, reduced_py/x): partial, skipped" in out
    assert os.path.exists(os.path.join(root, "Fake_1", "reduced_py", "Fake_1_left_cube1_align_cen_bin.fits"))
    log = list(csv.DictReader(open(str(tmp_path / "plan_log.csv"))))
    assert len(log) == 1 and log[0]["id"] == "a" and log[0]["exit"] == "0" and os.path.exists(log[0]["log"])
    assert os.path.exists(os.path.join(work, "left_round0_chopsub.f32"))
    assert batch_reduce.main(args + ["--only", "a"]) == 0
    assert "a (Fake_1 left, reduced_py): done, skipped" in capsys.readouterr().out
