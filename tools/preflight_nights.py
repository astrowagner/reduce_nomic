#!/usr/bin/env python3
"""Check every night before a batch reduction: settings, frames, skies, outputs, scratch and time.

For each night folder of the archive (``folder_name_map.csv`` kind ``night``) that has an IDL
object block, or every night folder with ``--all-nights``, and each side its settings reduce,
reports without reading any frame:

* where the settings come from (the object block under the old name, or the defaults);
* the science frames selected (``frames.select_science_frames``): count, first and last;
* the sky folders the settings imply (``pipeline.sky_folders``) and their raw-frame counts,
  with any missing folder flagged, since a run stops at a missing sky folder;
* the products that already exist in ``<night>/reduced_py`` under the run's prefix, since a run
  refuses to start when any exists;
* the scratch the run needs at most: two 512 x 512 float32 memmaps per round, one frame each
  per segment, so at most 2 MiB per raw frame of the science round and the sky rounds;
* the run time, scaled from Procyon_13 on patrick with 64 workers (run6 and run7: 0.013 to 0.020
  s per raw frame per side, including its sky rounds).

Usage: preflight_nights.py ARCHIVE OUT_CSV [--all-nights] [--nights NAME ...] [--nights-json PATH]

Parameters
----------
ARCHIVE : str
    The archive root, with ``folder_name_map.csv``.
OUT_CSV : str
    Output table; it must not exist.
--all-nights : flag
    Include the night folders without an IDL object block, with the default settings.
--nights : list of str, optional
    Night folders to check, by current or old name; default all (see above).
--nights-json : str, optional
    Object blocks to use instead of the package's ``config/nights_idl.json``.

Outputs
-------
OUT_CSV
    One row per night and side: ``folder``, ``night_old``, ``side``, ``settings``,
    ``n_science``, ``first``, ``last``, ``sky_folders`` (``folder:frames`` or
    ``folder:MISSING``, ``;``-separated), ``n_sky_frames``, ``missing_skies``,
    ``existing_outputs``, ``scratch_gb`` (GB, 1e9 bytes), ``minutes_low``, ``minutes_high``,
    ``problems``.
Standard output
    One line per night and side, the problems, and the totals.
"""
import argparse
import csv
import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reduce_nomic import names  # noqa: E402
from reduce_nomic.config import chop_frames_per_side, load_settings  # noqa: E402
from reduce_nomic.frames import list_raw, select_science_frames  # noqa: E402
from reduce_nomic.pipeline import planned_outputs, sky_folders  # noqa: E402

#: Scratch per raw frame and round at most: two memmaps of one 512 x 512 float32 frame, bytes.
SCRATCH_PER_FRAME = 2 * 512 * 512 * 4
#: Run time per raw frame per side on patrick with 64 workers (Procyon_13 run6 and run7), seconds.
SEC_PER_FRAME = (0.013, 0.020)


def check_side(archive: str, st, side: str) -> dict:
    """Check one night and side.

    Parameters
    ----------
    archive : str
        Archive root.
    st : Settings
        The night's settings (``config.load_settings``).
    side : str
        ``left`` or ``right``.

    Returns
    -------
    row : dict
        The OUT_CSV fields for this night and side.
    """
    v = dict(st.values)
    start = v["start_frame_left"] if side == "left" else v["start_frame_right"]
    end = v["end_frame_left"] if side == "left" else v["end_frame_right"]
    sci = select_science_frames(archive, v["night"], start, end, frame_range=v.get("frames"),
                                old_night=v["night_old"])
    skies, missing, n_sky = [], [], 0
    for d in sky_folders(v):
        path = os.path.join(archive, v["night"], d)
        if os.path.isdir(path):
            k = len(list_raw(path))
            n_sky += k
            skies.append(f"{d}:{k}")
        else:
            missing.append(d)
            skies.append(f"{d}:MISSING")
    out_dir = os.path.join(archive, v["night"], v["out_subdir"])
    existing = [f for f in planned_outputs(f"{v['night']}_{side}") if os.path.exists(os.path.join(out_dir, f))]
    problems = []
    if not sci:
        problems.append("no science frames")
    if missing:
        problems.append("missing sky folders " + ",".join(missing))
    if existing:
        problems.append(f"{len(existing)} outputs exist")
    if st.source == "defaults":
        problems.append("no IDL block, defaults")
    n = len(sci) + n_sky
    return {"folder": v["night"], "night_old": v["night_old"], "side": side, "settings": st.source,
            "n_science": len(sci), "first": f"{sci[0].stem}_{sci[0].num}" if sci else "",
            "last": f"{sci[-1].stem}_{sci[-1].num}" if sci else "", "sky_folders": ";".join(skies),
            "n_sky_frames": n_sky, "missing_skies": ";".join(missing), "existing_outputs": len(existing),
            "scratch_gb": round(n * SCRATCH_PER_FRAME / 1e9, 1),
            "minutes_low": round(n * SEC_PER_FRAME[0] / 60, 1), "minutes_high": round(n * SEC_PER_FRAME[1] / 60, 1),
            "problems": "; ".join(problems)}


def main(argv: Optional[List[str]] = None) -> int:
    """Run the pre-flight; see the module docstring.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments; default ``sys.argv[1:]``.

    Returns
    -------
    status : int
        0 when no night has a problem, 1 otherwise.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("archive")
    ap.add_argument("out_csv")
    ap.add_argument("--all-nights", action="store_true")
    ap.add_argument("--nights", nargs="+")
    ap.add_argument("--nights-json")
    a = ap.parse_args(argv)
    if os.path.lexists(a.out_csv):
        sys.exit(f"{a.out_csv} exists")
    rows_map = list(csv.DictReader(open(os.path.join(a.archive, names.MAP_FILE))))
    if a.nights:
        folders = [names.resolve(a.archive, n).folder for n in a.nights]
    else:
        folders = sorted((r["new_name"] for r in rows_map if r["kind"] == "night" and (r["idl_name"] or a.all_nights)),
                         key=lambda f: (f.rsplit("_", 1)[-1], f))
    out = []
    for folder in folders:
        st = load_settings(folder, a.nights_json, {"archive": a.archive})
        for side in chop_frames_per_side(st):
            r = check_side(a.archive, st, side)
            out.append(r)
            print(f"{r['folder']:26s} {r['night_old']:16s} {side:5s} {r['n_science']:7d} frames, skies "
                  f"{r['sky_folders'] or '-'}, scratch {r['scratch_gb']:.0f} GB, "
                  f"{r['minutes_low']:.0f}-{r['minutes_high']:.0f} min{'  PROBLEM: ' + r['problems'] if r['problems'] else ''}",
                  flush=True)
    with open(a.out_csv, "x", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    bad = [r for r in out if r["problems"]]
    lo, hi = sum(r["minutes_low"] for r in out) / 60, sum(r["minutes_high"] for r in out) / 60
    print(f"{len(out)} night-sides in {len(folders)} nights: {sum(r['n_science'] for r in out):,} science and "
          f"{sum(r['n_sky_frames'] for r in out):,} sky frame-sides; largest scratch "
          f"{max(r['scratch_gb'] for r in out):.0f} GB; about {lo:.0f}-{hi:.0f} h in all; "
          f"{len(bad)} with problems")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
