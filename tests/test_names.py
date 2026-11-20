"""Tests of the folder-name map: resolving either name, settings, frame selection and the CLI."""
import csv
import json
import os

import numpy as np
import pytest
from astropy.io import fits

from reduce_nomic import cli, names
from reduce_nomic.config import load_settings
from reduce_nomic.frames import select_science_frames

MAP = [("Procyon_13", "Procyon_20230205"), ("Alcor", "Alcor_20230407"), ("darks", "darks")]


def make_archive(root: str, with_map: bool = True) -> str:
    """Build a tiny archive whose ``Procyon_13`` is now ``Procyon_20230205``.

    The renamed folder holds frames 3 to 6 of ``n_230205``; the sorting's move log, which uses
    the old names, moved frames 1 and 2 out of ``Procyon_13/raw``, so the folder held frames 1 to
    6 before the sorting.

    Parameters
    ----------
    root : str
        Empty folder to fill.
    with_map : bool, default True
        Write ``folder_name_map.csv`` (``MAP``).

    Returns
    -------
    nights_json : str
        Path of an object-block file with one block, ``Procyon_13`` (``radius`` 123).
    """
    raw = os.path.join(root, "Procyon_20230205", "raw")
    os.makedirs(raw)
    for k in range(3, 7):
        fits.PrimaryHDU(np.zeros((4, 4), np.int16)).writeto(os.path.join(raw, f"n_230205_{k:06d}.fits"))
    os.makedirs(os.path.join(root, "_raw_inventory", "sorting"))
    with open(os.path.join(root, "_raw_inventory", "sorting", "moves_log.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["when", "id", "src", "dst"])
        for k in (1, 2):
            w.writerow(["2026-11-17T00:00:00Z", k, f"Procyon_13/raw/n_230205_{k:06d}.fits",
                        f"_other/230205_x/n_230205_{k:06d}.fits"])
    if with_map:
        with open(os.path.join(root, names.MAP_FILE), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["old_name", "new_name", "kind", "ut_date", "idl_name"])
            for old, new in MAP:
                w.writerow([old, new, "night", "", old])
    nights = os.path.join(root, "nights.json")
    json.dump({"Procyon_13": {"settings": {"radius": 123}, "unevaluated": {}}}, open(nights, "w"))
    return nights


def test_resolve_either_name(tmp_path):
    """The current and the old name resolve to the same pair; unlisted names stand for themselves."""
    make_archive(str(tmp_path))
    a = str(tmp_path)
    assert names.resolve(a, "Procyon_20230205") == names.NightName("Procyon_20230205", "Procyon_13")
    assert names.resolve(a, "Procyon_13") == names.NightName("Procyon_20230205", "Procyon_13")
    assert names.resolve(a, "darks") == names.NightName("darks", "darks")
    assert names.resolve(a, "New_20270101") == names.NightName("New_20270101", "New_20270101")


def test_resolve_without_map(tmp_path):
    """Without a map, a name is both the current and the old name."""
    make_archive(str(tmp_path), with_map=False)
    assert names.load_map(str(tmp_path)) == {}
    assert names.resolve(str(tmp_path), "Procyon_13") == names.NightName("Procyon_13", "Procyon_13")


def test_current_rel(tmp_path):
    """An old folder name after ``../`` is renamed; other paths are unchanged."""
    make_archive(str(tmp_path))
    a = str(tmp_path)
    assert names.current_rel(a, "../Alcor/skys1") == "../Alcor_20230407/skys1"
    assert names.current_rel(a, "../Alcor_20230407/skys1") == "../Alcor_20230407/skys1"
    assert names.current_rel(a, "skys1") == "skys1"
    assert names.current_rel(a, "../Unknown/skys1") == "../Unknown/skys1"


@pytest.mark.parametrize("given", ["Procyon_20230205", "Procyon_13"])
def test_settings_use_the_old_name_block(tmp_path, given):
    """Either name gets the current folder name and the object block of the old name."""
    nights = make_archive(str(tmp_path))
    st = load_settings(given, nights, {"archive": str(tmp_path)})
    assert st.night == "Procyon_20230205" and st.night_old == "Procyon_13"
    assert st.radius == 123 and st.source == "nights.json:Procyon_13"


def test_frame_selection_reads_the_move_log_under_the_old_name(tmp_path):
    """The position cut counts the frames moved out under the old name."""
    make_archive(str(tmp_path))
    a = str(tmp_path)
    # position 2 of the pre-sorting folder (frames 1-6) is frame 3: frames 3-6 are selected
    fr = select_science_frames(a, "Procyon_20230205", start_pos=2, old_night="Procyon_13")
    assert [f.num for f in fr] == [3, 4, 5, 6]
    # under the new name the move log has no rows, the folder looks like frames 3-6, and
    # position 2 would wrongly be frame 5
    fr = select_science_frames(a, "Procyon_20230205", start_pos=2)
    assert [f.num for f in fr] == [5, 6]


def test_cli_dry_run_by_old_name(tmp_path, capsys):
    """The command line takes the old name, reports both and selects frames in the renamed folder."""
    nights = make_archive(str(tmp_path))
    assert cli.main(["Procyon_13", "--archive", str(tmp_path), "--nights-json", nights, "--dry-run",
                     "--set", "start_frame_left=2"]) == 0
    out = capsys.readouterr().out
    assert "night Procyon_20230205 (named Procyon_13 before the 2026-11-20 renaming)" in out
    assert "left: 4 science frames, n_230205_3 .. n_230205_6" in out


def test_cli_refuses_a_missing_folder(tmp_path):
    """A name with no folder stops the command line before any work."""
    nights = make_archive(str(tmp_path))
    with pytest.raises(SystemExit):
        cli.main(["Nope_1", "--archive", str(tmp_path), "--nights-json", nights, "--dry-run"])
