"""Archive folder names: each night folder's current name and its name before the renaming.

On 2026-11-20 the night folders of the NOMIC archive were renamed from the IDL predecessor's
names (``Procyon_13``) to ``<target>_<UT date>`` (``Procyon_20230205``), as listed in
``<archive>/folder_name_map.csv``; ``FOLDER_NAMES.md`` beside it gives the convention. The port
reads and writes in the folders under their current names, and looks a night up under its old
name in the IDL object blocks (``config/nights_idl.json``) and in the sorting's move log
(``_raw_inventory/sorting/moves_log.csv``), which predate the renaming. A night can be given by
either name.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Dict

#: File name of the folder-name map in the archive root.
MAP_FILE = "folder_name_map.csv"


@dataclass(frozen=True)
class NightName:
    """A night folder's current name and its name before the renaming.

    Parameters
    ----------
    folder : str
        Current folder name in the archive, e.g. ``Procyon_20230205``.
    old : str
        Its name before the renaming, e.g. ``Procyon_13``; equal to ``folder`` for a folder the
        renaming kept or the map does not list, and in an archive without a map.
    """

    folder: str
    old: str


def load_map(archive: str) -> Dict[str, str]:
    """Read the archive's folder-name map.

    Parameters
    ----------
    archive : str
        Archive root.

    Returns
    -------
    new_of : dict of str to str
        Current name by old name for every top-level folder in the map (a kept folder maps to
        itself); empty when the archive has no ``folder_name_map.csv``.
    """
    path = os.path.join(archive, MAP_FILE)
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as fh:
        return {r["old_name"]: r["new_name"] for r in csv.DictReader(fh)}


def resolve(archive: str, name: str) -> NightName:
    """Resolve a night folder given by its current or its old name.

    Parameters
    ----------
    archive : str
        Archive root.
    name : str
        Folder name, current (``Procyon_20230205``) or old (``Procyon_13``).

    Returns
    -------
    night : NightName
        Both names. A name the map does not list, or any name in an archive without a map, is
        taken as both.
    """
    new_of = load_map(archive)
    old_of = {new: old for old, new in new_of.items()}
    if name in old_of:
        return NightName(name, old_of[name])
    if name in new_of:
        return NightName(new_of[name], name)
    return NightName(name, name)


def current_rel(archive: str, rel: str) -> str:
    """Rename an old folder name in a path relative to a night folder.

    Parameters
    ----------
    archive : str
        Archive root.
    rel : str
        Path relative to a night folder, e.g. the sky folder of another night ``../Alcor/skys1``.

    Returns
    -------
    rel : str
        The path with the folder after a leading ``../`` under its current name
        (``../Alcor_20230407/skys1``); any other path unchanged.
    """
    parts = rel.split("/")
    if len(parts) > 1 and parts[0] == "..":
        parts[1] = load_map(archive).get(parts[1], parts[1])
    return "/".join(parts)
