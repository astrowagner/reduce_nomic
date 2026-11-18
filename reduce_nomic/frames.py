"""Raw-frame selection and header reading for one night.

The IDL predecessor read every file of a night's ``raw/`` folder and cut the list by position
(``start_frame_left``, ``end_frame_left``, ...). The archive has since been sorted
(``_raw_inventory/sorting/moves_log.csv``), so the port selects frames by frame number instead:
a position cut is translated to the frame number it pointed at in the folder as it was before
the sorting, and every frame still in the folder from that number on is used.
"""
from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from astropy.io import fits

RAW = re.compile(r"^([A-Za-z0-9]+_\d{6})_(\d{5,6})\.fits(\.gz|\.Z)?$")

#: Header keywords read from each raw frame.
KEYS = ("LBT_PARA", "TIME-END", "TIME-OBS", "DATE-OBS", "EXPTIME", "CHOP_POS", "NOD_POS",
        "NCOADDS", "OBJNAME", "NOMICFW1", "NOMICFW2", "LBT_AIRM")


@dataclass
class RawFrame:
    """One raw frame of a night.

    Parameters
    ----------
    path : str
        Absolute path of the FITS file.
    stem : str
        Name stem ``<prefix>_<yymmdd>``.
    num : int
        Frame number from the file name.
    inum : str
        The IDL predecessor's frame label: the 6 characters before ``.fits`` in the name.
    """

    path: str
    stem: str
    num: int
    inum: str


def list_raw(folder: str) -> List[RawFrame]:
    """List the raw frames of one folder, sorted by name stem and frame number.

    Parameters
    ----------
    folder : str
        Folder holding ``<stem>_<n>.fits`` files.

    Returns
    -------
    frames : list of RawFrame
        Every raw-named file, sorted as the Mac's volume listed them for zero-padded names.
    """
    out = []
    for name in os.listdir(folder):
        m = RAW.match(name)
        if m:
            base = name.split(".fits")[0]
            out.append(RawFrame(os.path.join(folder, name), m.group(1), int(m.group(2)), base[-6:]))
    out.sort(key=lambda f: (f.stem, f.num))
    return out


def presort_names(archive: str, rel_dir: str, moves_log: str) -> List[str]:
    """Rebuild a folder's file names as they were before the archive was sorted.

    Parameters
    ----------
    archive : str
        Archive root.
    rel_dir : str
        Folder relative to the archive, e.g. ``EpsEri/raw``.
    moves_log : str
        The sorting's move log (``when, id, src, dst``); rows undone later are ignored.

    Returns
    -------
    names : list of str
        Raw file names of the folder before the sorting (those there now, plus those moved out,
        minus those moved in), sorted by stem and frame number.
    """
    now = {f.path.rsplit("/", 1)[1] for f in list_raw(os.path.join(archive, rel_dir))}
    if os.path.exists(moves_log):
        rows = list(csv.DictReader(open(moves_log)))
        undone = {(r["src"][7:], r["dst"][7:]) for r in rows if r["src"].startswith("undone:")}
        for r in rows:
            if r["id"] == "park" or r["src"].startswith("undone:") or (r["src"], r["dst"]) in undone:
                continue
            if os.path.dirname(r["src"]) == rel_dir:
                now.add(os.path.basename(r["src"]))
            if os.path.dirname(r["dst"]) == rel_dir:
                now.discard(os.path.basename(r["dst"]))

    def key(n: str) -> Tuple[str, int]:
        m = RAW.match(n)
        return (m.group(1), int(m.group(2))) if m else (n, 0)

    return sorted(now, key=key)


def select_science_frames(archive: str, night: str, start_pos: int = 0, end_pos: float = 99e9,
                          moves_log: Optional[str] = None,
                          frame_range: Optional[Tuple[int, int]] = None) -> List[RawFrame]:
    """Select a night's science frames as the IDL predecessor's position cut did.

    Parameters
    ----------
    archive : str
        Archive root.
    night : str
        Night folder name.
    start_pos : int, default 0
        IDL ``start_frame_<side>``: list position of the first frame in the pre-sorting folder.
    end_pos : float, default 99e9
        IDL ``end_frame_<side>``: list position one past the last frame.
    moves_log : str, optional
        The sorting's move log; default ``<archive>/_raw_inventory/sorting/moves_log.csv``.
    frame_range : tuple of int, optional
        ``(first, last)`` frame numbers to use instead of the position cut.

    Returns
    -------
    frames : list of RawFrame
        The frames now in ``<night>/raw`` whose (stem, number) lies in the selected range,
        sorted by stem and frame number.
    """
    rel = f"{night}/raw"
    current = list_raw(os.path.join(archive, rel))
    if frame_range is not None:
        lo, hi = frame_range
        return [f for f in current if lo <= f.num <= hi]
    if moves_log is None:
        moves_log = os.path.join(archive, "_raw_inventory", "sorting", "moves_log.csv")
    old = presort_names(archive, rel, moves_log)
    if not old:
        return []
    start_pos = int(start_pos)
    end = int(min(end_pos, len(old)))
    if start_pos >= end:
        return []

    def key(n: str) -> Tuple[str, int]:
        m = RAW.match(n)
        return m.group(1), int(m.group(2))

    lo_key, hi_key = key(old[start_pos]), key(old[end - 1])
    return [f for f in current if lo_key <= (f.stem, f.num) <= hi_key]


def read_frame(path: str) -> Tuple[np.ndarray, Dict[str, object]]:
    """Read one raw frame and the header values the reduction uses.

    Parameters
    ----------
    path : str
        FITS file.

    Returns
    -------
    data : ndarray, shape (ny, nx)
        The frame as float64, indexed ``[y, x]``; ny = nx = 512 for NOMIC.
    head : dict
        The values of ``KEYS``, None where absent.
    """
    with fits.open(path, memmap=False) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float64)
        h = hdul[0].header
        head = {k: h.get(k) for k in KEYS}
    while data.ndim > 2:
        data = data[0]
    return data, head
