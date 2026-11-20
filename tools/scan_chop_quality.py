"""Scan a night's chop identification: star position and peak S/N of sampled frames, per variant.

Usage
-----
    python3 tools/scan_chop_quality.py ARCHIVE NIGHT SIDE OUT_CSV [--step K] [--first I] [--count C]
                                       [--variant NAME:KEY=VALUE[,KEY=VALUE...] ...]

Takes every K-th frame of the night's sorted raw frames (from position ``nbg + 1``), computes
its rolling background as ``combine.frame_states`` does, and for each variant of the settings
computes the crude star image (``combine.crude_star_image``), its maximum and the maximum's S/N
(``combine.chopid_snr``). Appends one row per frame and variant to ``OUT_CSV``.
``--first`` and ``--count`` select which sampled frames to run, so a scan can be split over
several calls.

Parameters
----------
ARCHIVE : str
    Archive root (holding the night folders and ``calib/``).
NIGHT : str
    Night folder name, current (``Procyon_20230205``) or before the 2026-11-20 renaming
    (``Procyon_13``).
SIDE : str
    ``left`` or ``right``.
OUT_CSV : str
    CSV to append to (created with a header when absent). Columns: ``variant``, ``pos``
    (position in the sorted raw list), ``num`` (frame number), ``median`` (frame median, ADU),
    ``star_x``, ``star_y`` (pixels; -1 over the background limit), ``snr``.
--step : int, default 100
    Sampling step in frames.
--first, --count : int, default 0 and all
    Sampled frames to run, by index.
--variant : str
    ``NAME:KEY=VALUE,...`` settings applied on top of the night's (JSON values); default the
    two variants ``destripe:chopid_destripe=1`` and ``median5:chopid_destripe=1,chopid_median=5``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def main(argv=None) -> int:
    """Run the scan.

    Parameters
    ----------
    argv : list of str, optional
        Arguments without the program name; default ``sys.argv[1:]``.

    Returns
    -------
    status : int
        0 on success. Appends to ``OUT_CSV`` and prints a line every 100 sampled frames.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("archive")
    ap.add_argument("night")
    ap.add_argument("side", choices=["left", "right"])
    ap.add_argument("out_csv")
    ap.add_argument("--step", type=int, default=100)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--count", type=int)
    ap.add_argument("--variant", action="append", default=[])
    a = ap.parse_args(argv)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo)
    import time
    import numpy as np
    from astropy.io import fits
    from reduce_nomic import combine, idl
    from reduce_nomic.config import load_settings
    from reduce_nomic.frames import list_raw, read_frame

    variants = {}
    for spec in a.variant or ["destripe:chopid_destripe=1", "median5:chopid_destripe=1,chopid_median=5"]:
        name, _, kvs = spec.partition(":")
        over = {}
        for kv in filter(None, kvs.split(",")):
            k, v = kv.split("=", 1)
            try:
                over[k.strip()] = json.loads(v)
            except ValueError:
                over[k.strip()] = v
        variants[name] = over
    st0 = dict(load_settings(a.night, None, {"archive": a.archive}).values, side=a.side)
    nbg = int(st0["nbg"])
    files = list_raw(os.path.join(a.archive, st0["night"], "raw"))
    n = len(files)
    positions = list(range(nbg + 1, n - 1 - nbg, a.step))
    last = len(positions) if a.count is None else min(len(positions), a.first + a.count)
    init = combine.initial_background(files)
    master = np.asarray(fits.getdata(os.path.join(a.archive, st0["badmap_master"])), dtype=np.float64)
    badmap = combine.make_badmap(master, init["badtest"], st0["badsigthresh"], int(st0["badsz"]),
                                 int(st0["badbuff"]), bool(st0["fresh_badmap"]))
    new = not os.path.exists(a.out_csv)
    t0 = time.time()
    with open(a.out_csv, "a") as out:
        if new:
            out.write("variant,pos,num,median,star_x,star_y,snr\n")
        for k in range(a.first, last):
            i = positions[k]
            hp = {}
            for d in range(-nbg, nbg + 1):
                r = read_frame(files[i + d].path)[0]
                hp[d] = (r, r - idl.smooth(r, 20))
            data = hp[0][0]
            med = float(idl.median(data))
            bg = np.min([hp[d][1] for d in hp if d != 0], axis=0)
            for name, over in variants.items():
                if med >= st0["background_limit"]:
                    out.write(f"{name},{i},{files[i].num},{med:.1f},-1,-1,nan\n")
                    continue
                img = combine.crude_star_image(data, bg, badmap, dict(st0, **over))
                j = int(np.argmax(img))
                y, x = divmod(j, img.shape[1])
                out.write(f"{name},{i},{files[i].num},{med:.1f},{x},{y},{combine.chopid_snr(img, j):.2f}\n")
            if (k - a.first) % 100 == 99:
                out.flush()
                print(f"{k + 1}/{len(positions)} sampled frames, {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
