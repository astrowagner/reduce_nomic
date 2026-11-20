"""Sample the chop identification across a night: crude star positions of blocks of frames.

Usage
-----
    python3 tools/sample_chop_positions.py ARCHIVE NIGHT OUT_CSV [--samples N] [--block M]
                                           [--first I] [--count C] [--side left|right ...]
                                           [--set NAME=VALUE ...] [--png PREFIX]

Spreads ``N`` blocks of ``M`` consecutive frames evenly over the night's sorted raw frames,
runs ``combine.frame_states`` on each block for each side, exactly as the combine stage does
(same initial background, bad-pixel map and settings), and appends one row per frame to
``OUT_CSV``. ``--first`` and ``--count`` select which of the ``N`` blocks to run, so a long
sampling can be split over several calls.

Parameters
----------
ARCHIVE : str
    Archive root (the folder holding the night folders and ``calib/``).
NIGHT : str
    Night folder name, current (``Procyon_20230205``) or before the 2026-11-20 renaming
    (``Procyon_13``).
OUT_CSV : str
    CSV file to append to (created with a header when absent). Columns: ``side``, ``block``,
    ``pos`` (position in the sorted raw list), ``num`` (frame number), ``median`` (frame median,
    ADU), ``star_x``, ``star_y`` (crude star position, detector pixels, -1 over the background
    limit), ``stripe`` (mean of rows 126-130, columns 50-100 of the crude image).
--samples : int, default 40
    Number of blocks spread over the night.
--block : int, default 20
    Frames per block.
--first, --count : int, default 0 and all
    Blocks to run, by index among the ``N``.
--side : str, default both sides
    ``left`` and/or ``right``.
--set : str
    Setting overrides, ``NAME=VALUE`` (JSON values), e.g. ``--set chopid_destripe=0``.
--png : str, optional
    Also feed the sampled blocks to the live display's view (``display.LiveView``, PNG only)
    through the driver's own callback (``pipeline._chop_feed``), apply the no-star cut to all
    sampled frames together (``combine.reject_no_star`` with ``chopid_nostar_frac``), and write
    the final view to ``PREFIX_<side>.png`` (must not exist). Needs all blocks in one call.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def main(argv=None) -> int:
    """Run the sampled blocks and append their frame states.

    Parameters
    ----------
    argv : list of str, optional
        Arguments without the program name; default ``sys.argv[1:]``.

    Returns
    -------
    status : int
        0 on success. Appends to ``OUT_CSV``, writes ``PREFIX_<side>.png`` with ``--png``, and
        prints one line per block and side.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("archive")
    ap.add_argument("night")
    ap.add_argument("out_csv")
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--block", type=int, default=20)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--count", type=int)
    ap.add_argument("--side", action="append", choices=["left", "right"])
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument("--png")
    a = ap.parse_args(argv)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo)
    import time
    import numpy as np
    from astropy.io import fits
    from reduce_nomic import combine
    from reduce_nomic.config import load_settings
    from reduce_nomic.frames import list_raw

    over = {"archive": a.archive}
    for item in a.set:
        k, v = item.split("=", 1)
        try:
            over[k.strip()] = json.loads(v)
        except ValueError:
            over[k.strip()] = v
    st = dict(load_settings(a.night, None, over).values)
    files = list_raw(os.path.join(a.archive, st["night"], "raw"))
    n = len(files)
    nbg = int(st["nbg"])
    starts = np.linspace(nbg + 1, n - 2 - nbg - a.block, a.samples).astype(int)
    init = combine.initial_background(files)
    master = np.asarray(fits.getdata(os.path.join(a.archive, st["badmap_master"])), dtype=np.float64)
    badmap = combine.make_badmap(master, init["badtest"], st["badsigthresh"], int(st["badsz"]),
                                 int(st["badbuff"]), bool(st["fresh_badmap"]))
    new = not os.path.exists(a.out_csv)
    last = a.samples if a.count is None else min(a.samples, a.first + a.count)
    sides = a.side or ["left", "right"]
    views, feeds, kept = {}, {}, {side: [] for side in sides}
    if a.png:
        os.environ.pop("DISPLAY", None)
        from reduce_nomic.display import LiveView
        from reduce_nomic.pipeline import _chop_feed
        for side in sides:
            png = f"{a.png}_{side}.png"
            if os.path.exists(png):
                print(f"{png} exists", file=sys.stderr)
                return 2
            views[side] = LiveView(f"{a.night} {side} (sampled blocks)", png, log=print, backend="Agg")
            feeds[side] = _chop_feed(views[side], dict(st, side=side), False, f"{side}_round0 (sampled)")
    with open(a.out_csv, "a") as out:
        if new:
            out.write("side,block,pos,num,median,star_x,star_y,stripe\n")
        for b in range(a.first, last):
            lo = int(starts[b])
            for side in sides:
                t0 = time.time()
                images = {} if a.png else None
                sts = combine.frame_states(files, lo, lo + a.block, init["bg"], badmap, dict(st, side=side),
                                           images=images)
                if a.png:
                    thumb = None
                    if images:
                        pos, img, bsub = images["last"]
                        q = sts[pos - lo]
                        thumb = (img.astype(np.float32), bsub.astype(np.float32), (float(q.star_x), float(q.star_y)),
                                 f"{q.num}")
                    feeds[side]((sts, thumb))
                    kept[side].extend(sts)
                for q in sts:
                    out.write(f"{side},{b},{q.pos},{q.num},{q.median:.1f},{q.star_x},{q.star_y},{q.stripe:.3f}\n")
                out.flush()
                xs = sorted({q.star_x for q in sts})
                print(f"block {b} ({lo}..{lo + a.block - 1}) {side}: x in {xs[:6]}{'...' if len(xs) > 6 else ''}, "
                      f"{time.time() - t0:.1f} s", flush=True)
    for side, view in views.items():
        frac = float(st.get("chopid_nostar_frac") or 0.0)
        snr = np.array([q.snr for q in kept[side] if q.star_x >= 0], dtype=float)
        if frac > 0 and np.isfinite(snr).any():
            cut = frac * float(np.nanmedian(snr))
            nrej = combine.reject_no_star(kept[side], frac)
            view.mark_no_star([q.num for q in kept[side] if q.nostar], cut)
            print(f"{side}: {nrej} of {snr.size} sampled frames without a star (S/N below {cut:.0f})", flush=True)
        view.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
