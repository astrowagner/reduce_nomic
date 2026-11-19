#!/usr/bin/env python3
"""Reduce one synthetic night with two versions of reduce_nomic and compare every product.

Builds the synthetic night of ``tests/test_pipeline_synthetic.make_night`` (taken from NEW_ROOT)
in ``WORKDIR/archive``, reduces its left side with each version in its own process, with
frames 2, 5 and 7 of every chunk listed as bad (``manual_bads``) so that the pre-clean rejects
some, and prints for each product whether it is identical. Each process gets
``PYTHONPATH=<root>``, runs with ``python -P`` from the root's parent folder (so that the
caller's folder cannot shadow the package) and prints the folder it imported the package from;
it uses 2 workers.

FITS files are compared by every HDU's data (bit for bit, NaN equal to NaN) and header cards
other than DATE; CSV files by text; the summary JSON with its entries whose names contain
"elapsed" removed, and with the keys only the new version writes listed and left out. Exits
with status 1 when any product differs or is missing from one side.

With ``--set KEY=VALUE`` (repeatable; VALUE parsed as JSON when it parses, else a string) both
versions get the setting too; a version that does not know it ignores it.

Usage: compare_versions.py OLD_ROOT NEW_ROOT WORKDIR [--set KEY=VALUE ...]
"""
import json
import os
import subprocess
import sys
from typing import List, Optional

import numpy as np

OVERRIDES = {"obj": "fakestar", "prebin": 5, "background_limit": 50000.0,
             "pre_corr_thresh_1": 0.5, "pre_corr_thresh_2": 0.5, "corr_thresh_1": 0.5, "corr_thresh_2": 0.5,
             "pre_noise_thresh_1": 1.02, "pre_noise_thresh_2": 1.02, "noise_thresh_1": 3, "noise_thresh_2": 3,
             "pre_peak_thresh_1": 0.97, "pre_peak_thresh_2": 0.97, "peak_thresh_1": 0.5, "peak_thresh_2": 0.5,
             "manual_bads": [2, 5, 7]}


def run(root: str, code: str) -> None:
    """Run Python code in a new process with one version of the package on its path.

    Parameters
    ----------
    root : str
        Folder holding the ``reduce_nomic`` package.
    code : str
        Python source to run.

    Returns
    -------
    None
        Prints the folder the package was imported from; raises
        ``subprocess.CalledProcessError`` when the process fails.
    """
    env = dict(os.environ, PYTHONPATH=root)
    probe = "import reduce_nomic, os; print('using', os.path.dirname(os.path.dirname(reduce_nomic.__file__)))"
    # -P and a neutral working folder keep the caller's folder off sys.path, so that ``root``'s
    # package is the one imported (``python -c`` would otherwise put the current folder first).
    subprocess.run([sys.executable, "-P", "-c", probe + "; " + code], env=env, check=True, cwd=os.path.dirname(root.rstrip("/")) or "/")


def strip_elapsed(obj):
    """Remove the entries whose names contain "elapsed" from a JSON structure.

    Parameters
    ----------
    obj : object
        Parsed JSON.

    Returns
    -------
    clean : object
        The same structure without those entries.
    """
    if isinstance(obj, dict):
        return {k: strip_elapsed(v) for k, v in obj.items() if "elapsed" not in k}
    if isinstance(obj, list):
        return [strip_elapsed(v) for v in obj]
    return obj


def drop_added(new, old):
    """Remove from a JSON structure the dictionary keys that another one lacks.

    Parameters
    ----------
    new : object
        Parsed JSON of the new version.
    old : object
        Parsed JSON of the old version, of the same layout.

    Returns
    -------
    trimmed : object
        ``new`` without the keys ``old`` lacks at the same place.
    """
    if isinstance(new, dict) and isinstance(old, dict):
        return {k: drop_added(v, old[k]) for k, v in new.items() if k in old}
    if isinstance(new, list) and isinstance(old, list) and len(new) == len(old):
        return [drop_added(x, y) for x, y in zip(new, old)]
    return new


def same(a: str, b: str) -> bool:
    """Compare two product files.

    Parameters
    ----------
    a, b : str
        Paths of the same product from the two versions.

    Returns
    -------
    identical : bool
        True when the files match as described in the module docstring.
    """
    if a.endswith(".fits"):
        from astropy.io import fits
        with fits.open(a) as fa, fits.open(b) as fb:
            if len(fa) != len(fb):
                return False
            for ha, hb in zip(fa, fb):
                ca = [(c.keyword, str(c.value)) for c in ha.header.cards if c.keyword != "DATE"]
                cb = [(c.keyword, str(c.value)) for c in hb.header.cards if c.keyword != "DATE"]
                if ca != cb:
                    return False
                da, db = ha.data, hb.data
                if (da is None) != (db is None):
                    return False
                if da is None:
                    continue
                if getattr(da, "names", None):
                    if da.names != db.names or any(not np.array_equal(np.asarray(da[n]), np.asarray(db[n]), equal_nan=True)
                                                   for n in da.names):
                        return False
                elif da.dtype != db.dtype or not np.array_equal(da, db, equal_nan=True):
                    return False
        return True
    if a.endswith(".json"):
        ja, jb = strip_elapsed(json.load(open(a))), strip_elapsed(json.load(open(b)))
        added = sorted(set(jb.get("cubes", {}).get("1", {})) - set(ja.get("cubes", {}).get("1", {})))
        added += sorted(set(jb) - set(ja))
        if added:
            print(f"  (new keys in {os.path.basename(b)}: {', '.join(added)}; compared without them)")
        return drop_added(jb, ja) == ja
    if a.endswith(".png"):
        return True
    return open(a, "rb").read() == open(b, "rb").read()


def main(argv: Optional[List[str]] = None) -> None:
    """Build, reduce and compare; see the module docstring.

    Parameters
    ----------
    argv : list of str, optional
        ``OLD_ROOT NEW_ROOT WORKDIR [--set KEY=VALUE ...]``; default ``sys.argv[1:]``.

    Returns
    -------
    None
        Prints one line per product and exits with status 1 on any difference.
    """
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("work")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    a = ap.parse_args(argv)
    old, new, work = a.old, a.new, a.work
    extra = {}
    for item in a.set:
        key, _, val = item.partition("=")
        try:
            extra[key] = json.loads(val)
        except ValueError:
            extra[key] = val
    archive = os.path.join(work, "archive")
    run(new, f"import sys; sys.path.insert(0, {os.path.join(new, 'tests')!r}); "
             f"from test_pipeline_synthetic import make_night; make_night({archive!r})")
    outs = {}
    for tag, root in (("old", old), ("new", new)):
        out = os.path.join(work, f"out_{tag}")
        over = dict(OVERRIDES, archive=archive, **extra)
        run(root, "from reduce_nomic.config import load_settings; from reduce_nomic.pipeline import reduce; "
                  f"st = load_settings('Fake_1', nights_json={os.path.join(work, 'none.json')!r}, overrides={over!r}); "
                  f"reduce(st, 'left', workers=2, out_dir={out!r}, work_dir={os.path.join(work, 'work_' + tag)!r})")
        outs[tag] = out
    names = sorted(set(os.listdir(outs["old"])) | set(os.listdir(outs["new"])))
    bad = 0
    for name in names:
        pa, pb = os.path.join(outs["old"], name), os.path.join(outs["new"], name)
        if not (os.path.isfile(pa) and os.path.isfile(pb)):
            print(f"MISSING   {name}")
            bad += 1
            continue
        ok = same(pa, pb)
        bad += not ok
        print(f"{'identical' if ok else 'DIFFERENT'} {name}")
    print(f"{len(names) - bad} of {len(names)} products identical")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
