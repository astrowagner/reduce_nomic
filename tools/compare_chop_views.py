"""Compare candidate images for the display's chop-identification panel on real raw frames.

Usage
-----
    python3 tools/compare_chop_views.py RAW_DIR BADMAP OUT_PNG [--night NAME] [--frame K]

Reads the raw frames in ``RAW_DIR`` (consecutive ``n_*.fits`` frames of one night, at least
``2 nbg + 1``), takes frame ``K`` of them, and draws side by side:

1. the current panel: the frame minus its 20-px boxcar, 2x2 binned;
2. the same minus the rolling background (per-pixel minimum of the same quantity over the
   ``nbg`` frames on each side), as ``combine.frame_states`` computes it;
3. the crude star image ``combine.crude_star_image`` takes the maximum of, for the left (SX)
   aperture;
4. the same for the right (DX) aperture;
5. the frame minus the mean of its two neighbors (the nearest-neighbor chop subtraction of a
   chop that flips every frame), 20-px high-passed;
6. the mean of the rolling-background-subtracted frames of each chop state around frame ``K``,
   chop A minus chop B.

Each panel marks its maximum. Settings are the night's (``config.load_settings``).

Parameters
----------
RAW_DIR : str
    Folder of consecutive raw frames.
BADMAP : str
    Master bad-pixel map (``calib/master_badmap_2022.fits``).
OUT_PNG : str
    PNG to write (must not exist).
--night : str, default Procyon_13
    Night whose settings to use.
--frame : int, default middle
    Index of the frame to show among the sorted frames.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys


def main(argv=None) -> int:
    """Draw the comparison.

    Parameters
    ----------
    argv : list of str, optional
        Arguments without the program name; default ``sys.argv[1:]``.

    Returns
    -------
    status : int
        0 on success. Writes ``OUT_PNG`` and prints, for each panel, the maximum's position and
        its contrast (maximum over the robust standard deviation of the panel).
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("raw_dir")
    ap.add_argument("badmap")
    ap.add_argument("out_png")
    ap.add_argument("--night", default="Procyon_13")
    ap.add_argument("--frame", type=int)
    a = ap.parse_args(argv)
    if os.path.exists(a.out_png):
        print(f"{a.out_png} exists", file=sys.stderr)
        return 2
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo)
    import numpy as np
    from astropy.io import fits
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from reduce_nomic import combine, idl
    from reduce_nomic.config import load_settings
    from reduce_nomic.frames import read_frame

    st = dict(load_settings(a.night, None, {}).values)
    nbg = int(st["nbg"])
    paths = sorted(glob.glob(os.path.join(a.raw_dir, "n_*.fits")))
    raws = [read_frame(p)[0] for p in paths]
    k = len(raws) // 2 if a.frame is None else a.frame
    if not nbg <= k < len(raws) - nbg:
        print(f"frame {k} needs {nbg} frames on each side", file=sys.stderr)
        return 2
    hp = [r - idl.smooth(r, 20) for r in raws]
    master = np.asarray(fits.getdata(a.badmap), dtype=np.float64)
    badtest = np.min(raws[:11], axis=0)
    badmap = combine.make_badmap(master, badtest, st["badsigthresh"], int(st["badsz"]), int(st["badbuff"]),
                                 bool(st["fresh_badmap"]))

    def rolling(i):
        """Return the rolling background of frame i (minimum over nbg frames on each side)."""
        return np.min([hp[i + d] for d in range(-nbg, nbg + 1) if d != 0], axis=0)

    bg = rolling(k)
    views = {}
    views["1. Current: Frame - 20-px Boxcar"] = hp[k]
    views["2. Minus Rolling Background"] = hp[k] - bg
    for side in ("left", "right"):
        s2 = dict(st, side=side)
        views[f"{3 if side == 'left' else 4}. Crude Star Image, {side.upper()}"] = combine.crude_star_image(raws[k], bg, badmap, s2)
    views["5. Frame - Mean of Neighbors (High-Passed)"] = idl.smooth(raws[k] - 0.5 * (raws[k - 1] + raws[k + 1]), 3) - \
        idl.smooth(raws[k] - 0.5 * (raws[k - 1] + raws[k + 1]), 20)
    lo, hi = max(nbg, k - 6), min(len(raws) - nbg, k + 7)
    sub = {i: hp[i] - rolling(i) for i in range(lo, hi)}
    labs = {}
    for i, img in sub.items():
        cimg = combine.crude_star_image(raws[i], rolling(i), badmap, dict(st, side="left"))
        y, x = divmod(int(np.argmax(cimg)), 512)
        labs[i] = "A" if x < 256 else "B"
    a_imgs = [sub[i] for i in sub if labs[i] == "A"]
    b_imgs = [sub[i] for i in sub if labs[i] == "B"]
    if a_imgs and b_imgs:
        views[f"6. Mean A - Mean B ({len(a_imgs)} + {len(b_imgs)} Frames)"] = np.mean(a_imgs, axis=0) - np.mean(b_imgs, axis=0)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), layout="constrained")
    for ax, (title, img) in zip(axes.ravel(), views.items()):
        v = img[np.isfinite(img)]
        med = float(np.median(v))
        rsd = float(1.4826 * np.median(np.abs(v - med))) or 1.0
        y, x = divmod(int(np.nanargmax(img)), img.shape[1])
        contrast = (img[y, x] - med) / rsd
        print(f"{title}: max at x={x}, y={y}, contrast {contrast:.1f} robust sigma")
        ax.imshow(img, origin="lower", cmap="gray", vmin=med - 3 * rsd, vmax=med + 12 * rsd, interpolation="nearest")
        ax.plot([x], [y], "o", mfc="none", mec="red", ms=16, mew=1.5)
        ax.axhline(int(st["sx_top"]), color="C0", lw=0.8, ls="--")
        ax.axhline(int(st["dx_bottom"]), color="C1", lw=0.8, ls="--")
        ax.set_title(f"{title}\nMax at ({x}, {y}), {contrast:.0f} Sigma")
        ax.set_xlabel("X (Pixels)")
        ax.set_ylabel("Y (Pixels)")
    fig.suptitle(f"{a.night}: Frame {os.path.basename(paths[k])}, Stretch -3 to +12 Robust Sigma; "
                 f"Dashed: sx_top (Blue) and dx_bottom (Orange)")
    fig.savefig(a.out_png, dpi=80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
