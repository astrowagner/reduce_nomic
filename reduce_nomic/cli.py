"""Command line: reduce one NOMIC night with the Python port of reduce_nomic.pro.

Usage
-----
    python3 -m reduce_nomic.cli NIGHT [--side left|right|both] [--workers N] [--archive PATH]
                                [--out DIR] [--work DIR] [--frames FIRST LAST]
                                [--set NAME=VALUE ...] [--nights-json PATH] [--display] [--dry-run]

NIGHT is the night folder's name in the archive, current (``Procyon_20230406``) or before the
2026-11-20 renaming (``Procyon_17``); ``<archive>/folder_name_map.csv`` relates the two
(``reduce_nomic.names``). Settings come from the IDL predecessor's defaults and the night's object
block (``config/nights_idl.json``, under the old name); ``--set`` overrides any setting by name,
with the value parsed as JSON when possible (``--set radius=400 --set
sky_dirs='["../Alcor_20230407/skys1"]'``; an old folder name after ``../`` is renamed too). ``--display`` opens a live view of the stages (chop
identification, chop subtraction, frame selection, progress) in a window of the KLIP-TPE live
window's size, 1850 x 990 pixels, shrunk on a smaller screen; it is redrawn without taking the
focus, and the final view is written to ``<out>/<night>_<side>_display.png``. With no X display
the view goes to that PNG only. It is drawn by its own process, so closing the window or losing
the X connection does not stop the run. It needs matplotlib, and Tk for the window.
``--dry-run`` prints the settings and the selected frames and stops. Progress lines with
estimated remaining time are printed for every stage.

Outputs
-------
<out>/<night>_<side>_*.fits, *.csv, *_summary.json
    The products listed in ``reduce_nomic.pipeline.reduce``, named with the folder's current
    name; ``<out>`` defaults to ``<archive>/<night>/reduced_py``.
<work>/
    Memory-mapped intermediates, default ``<out>/work_<side>``; they can be deleted after the run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

from .config import chop_frames_per_side, load_settings
from .frames import select_science_frames


def parse_value(text: str):
    """Parse a ``--set`` value as JSON, falling back to the plain string.

    Parameters
    ----------
    text : str
        The value as given.

    Returns
    -------
    value : any
        Number, list, bool or null from JSON, otherwise the string.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def main(argv: List[str]) -> int:
    """Run the command line.

    Parameters
    ----------
    argv : list of str
        Arguments without the program name.

    Returns
    -------
    status : int
        0 on success.
    """
    ap = argparse.ArgumentParser(prog="reduce_nomic", description=__doc__.split("\n\n")[0])
    ap.add_argument("night")
    ap.add_argument("--side", choices=["left", "right", "both"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--archive")
    ap.add_argument("--out")
    ap.add_argument("--work")
    ap.add_argument("--frames", type=int, nargs=2, metavar=("FIRST", "LAST"))
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument("--nights-json")
    ap.add_argument("--display", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    over = {}
    for item in a.set:
        k, v = item.split("=", 1)
        over[k.strip()] = parse_value(v)
    if a.archive:
        over["archive"] = a.archive
    if a.side:
        over["side"] = a.side
    if a.frames:
        over["frames"] = list(a.frames)
    over["workers"] = a.workers
    if a.display:
        over["display"] = True
    st = load_settings(a.night, a.nights_json, over)
    if not os.path.isdir(os.path.join(st.archive, st.night)):
        ap.error(f"no night folder {st.night} in {st.archive}")
    print(f"night {st.night}" + (f" (named {st.night_old} before the 2026-11-20 renaming)" if st.night_old != st.night
                                 else "") + f"; settings from {st.source}", flush=True)
    sides = chop_frames_per_side(st)
    if a.dry_run:
        print(json.dumps({"source": st.source, "unevaluated": st.unevaluated, "sides": sides,
                          "settings": st.values}, indent=1, default=str))
        for side in sides:
            start = st.start_frame_left if side == "left" else st.start_frame_right
            end = st.end_frame_left if side == "left" else st.end_frame_right
            fr = select_science_frames(st.archive, st.night, start, end, frame_range=st.values.get("frames"),
                                       old_night=st.night_old)
            print(f"{side}: {len(fr)} science frames" + (f", {fr[0].stem}_{fr[0].num} .. {fr[-1].stem}_{fr[-1].num}" if fr else ""))
        return 0
    from .pipeline import reduce
    for side in sides:
        reduce(st, side, workers=a.workers, out_dir=a.out, work_dir=a.work)
    return 0


def main_entry() -> None:
    """Entry point of the ``reduce-nomic`` console script.

    Returns
    -------
    None
        Exits with ``main``'s status.
    """
    sys.exit(main(sys.argv[1:]))


if __name__ == "__main__":
    main_entry()
