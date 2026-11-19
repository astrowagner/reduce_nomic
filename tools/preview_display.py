"""Render the live display of a synthetic night, to check the panel layout.

Usage
-----
    python3 tools/preview_display.py OUT_DIR [--frames N] [--window]

Writes a synthetic NOMIC night (the test suite's ``make_night``) under ``OUT_DIR/archive``,
reduces its left aperture with ``display`` on, and prints the path of the display PNG,
``OUT_DIR/out/Fake_1_left_display.png`` (1850 x 990 pixels). Without ``--window`` the X display
is ignored, so the PNG-only view is drawn; with it, the live window opens when an X display and
Tk are available. ``OUT_DIR`` must not exist yet.

Parameters
----------
OUT_DIR : str
    New folder for the synthetic archive, the outputs and the work files.
--frames : int, default 240
    Number of synthetic raw frames.
--window : flag
    Keep ``DISPLAY`` so that the live window opens.
"""
from __future__ import annotations

import argparse
import os
import sys


def main(argv=None) -> int:
    """Build the synthetic night, reduce it with the display on, and print the PNG path.

    Parameters
    ----------
    argv : list of str, optional
        Arguments without the program name; default ``sys.argv[1:]``.

    Returns
    -------
    status : int
        0 on success. Writes ``OUT_DIR/archive`` (synthetic raw frames and master bad-pixel
        map), ``OUT_DIR/out`` (the reduction's products and the display PNG) and
        ``OUT_DIR/out/work_left`` (memory-mapped intermediates).
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out_dir")
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--window", action="store_true")
    a = ap.parse_args(argv)
    if os.path.exists(a.out_dir):
        print(f"{a.out_dir} exists; give a new folder", file=sys.stderr)
        return 2
    if not a.window:
        os.environ.pop("DISPLAY", None)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path[:0] = [repo, os.path.join(repo, "tests")]
    from test_pipeline_synthetic import make_night
    from reduce_nomic.config import load_settings
    from reduce_nomic.pipeline import reduce
    root = os.path.join(a.out_dir, "archive")
    make_night(root, n=a.frames)
    over = {"archive": root, "obj": "fakestar", "prebin": 5, "background_limit": 50000.0, "display": True}
    for cn in (1, 2):
        for pre in ("pre_", ""):
            over.update({f"{pre}corr_thresh_{cn}": 0.5, f"{pre}noise_thresh_{cn}": 3, f"{pre}peak_thresh_{cn}": 0.5})
    st = load_settings("Fake_1", nights_json=os.path.join(a.out_dir, "none.json"), overrides=over)
    out = os.path.join(a.out_dir, "out")
    reduce(st, "left", workers=2, out_dir=out)
    print(os.path.join(out, "Fake_1_left_display.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
