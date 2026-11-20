#!/usr/bin/env python3
"""Check that every night's science-frame selection is the same after the folder renaming.

For each night of ``config/nights_idl.json`` whose folder the archive's
``folder_name_map.csv`` lists, and each side, selects the science frames three ways with the
night's ``start_frame_<side>`` and ``end_frame_<side>``:

* **after:** the renamed folder, with the move log read under the old name, as the port now does
  (``select_science_frames(ARCHIVE, new, old_night=old)``);
* **before:** the layout before the renaming, rebuilt in SCRATCH as symbolic links from each old
  name to its renamed folder plus one to ``_raw_inventory``, selected by the old name as the port
  did before (``select_science_frames(SCRATCH, old)``);
* **untranslated:** the renamed folder with the move log read under the new name, to show which
  nights need the translation.

Prints one line per night and side: the frame counts and first and last frames, ``same`` when
"after" equals "before" frame for frame, and whether "untranslated" differs. Nights the map does
not list, or whose folder has no ``raw/``, are skipped with a line saying so. Changes nothing in
the archive.

Usage: check_selection_after_rename.py ARCHIVE SCRATCH [--nights NAME ...]

Parameters
----------
ARCHIVE : str
    The renamed archive (with ``folder_name_map.csv``).
SCRATCH : str
    Empty or absent folder for the symbolic links of the old layout.
--nights : list of str, optional
    Old night names to check; default every block of ``config/nights_idl.json``.

Outputs
-------
Standard output
    One line per night and side, then a summary. Exit status 1 when any "after" differs from
    "before".
SCRATCH/
    The symbolic links.
"""
import argparse
import json
import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reduce_nomic import names  # noqa: E402
from reduce_nomic.config import load_settings  # noqa: E402
from reduce_nomic.frames import select_science_frames  # noqa: E402


def describe(frames: list) -> str:
    """Summarize a frame selection.

    Parameters
    ----------
    frames : list of RawFrame
        Selected frames.

    Returns
    -------
    text : str
        Count and first and last frame, e.g. ``1200 n_230205_100..n_230205_1299``.
    """
    if not frames:
        return "0"
    return f"{len(frames)} {frames[0].stem}_{frames[0].num}..{frames[-1].stem}_{frames[-1].num}"


def main(argv: Optional[List[str]] = None) -> int:
    """Run the check; see the module docstring.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments; default ``sys.argv[1:]``.

    Returns
    -------
    status : int
        0 when every night selects the same frames before and after the renaming.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("archive")
    ap.add_argument("scratch")
    ap.add_argument("--nights", nargs="+")
    a = ap.parse_args(argv)
    new_of = names.load_map(a.archive)
    if not new_of:
        sys.exit(f"no {names.MAP_FILE} in {a.archive}")
    blocks = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                         "config", "nights_idl.json")))
    olds = a.nights or sorted(blocks)
    os.makedirs(a.scratch, exist_ok=True)
    for name in ["_raw_inventory"] + [o for o in olds if o in new_of]:
        link = os.path.join(a.scratch, name)
        target = os.path.join(os.path.abspath(a.archive), new_of.get(name, name))
        if not os.path.lexists(link):
            os.symlink(target, link)
        elif os.readlink(link) != target:
            sys.exit(f"{link} points elsewhere")
    differ, needed, n = [], [], 0
    for old in olds:
        if old not in new_of:
            print(f"{old}: not in the map, skipped")
            continue
        new = new_of[old]
        if not os.path.isdir(os.path.join(a.archive, new, "raw")):
            print(f"{old} -> {new}: no raw/ folder, skipped")
            continue
        st = load_settings(new, None, {"archive": a.archive})
        sides = ["left", "right"] if st.side == "both" else [st.side]
        for side in sides:
            start = st.start_frame_left if side == "left" else st.start_frame_right
            end = st.end_frame_left if side == "left" else st.end_frame_right
            after = select_science_frames(a.archive, new, start, end, old_night=old)
            before = select_science_frames(a.scratch, old, start, end)
            plain = select_science_frames(a.archive, new, start, end)
            same = [(f.stem, f.num) for f in after] == [(f.stem, f.num) for f in before]
            untrans = [(f.stem, f.num) for f in plain] != [(f.stem, f.num) for f in after]
            n += 1
            if not same:
                differ.append(f"{old} {side}")
            if untrans:
                needed.append(f"{old} {side}")
            print(f"{old} -> {new} {side} (start {start}, end {end:g}): after {describe(after)}; "
                  f"before {describe(before)}; {'same' if same else 'DIFFERENT'}"
                  f"{'; untranslated would differ: ' + describe(plain) if untrans else ''}", flush=True)
    print(f"{n} night-sides: {n - len(differ)} same, {len(differ)} different {differ}; "
          f"the translation matters for {len(needed)}: {needed}")
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main())
