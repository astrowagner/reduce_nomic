#!/usr/bin/env python3
"""Time one chunk task of the split-to-pre-bin stage and measure its peak memory.

Builds a synthetic chunk of chop-subtracted frames in a temporary work file: 2 M rows of
512 x 512 float32, alternating chop A and chop B (a LEFT-RIGHT chop with the star at (150, 100)
and (400, 100) in detector pixels, the other chop's image negative), unit Gaussian background
noise, NaN in the outer 3 pixels and above row 214 (an SX frame of a double-sided night).
Every 7th chop-A frame has 3 times the noise and every 11th a star at half the peak, so that
the pre-clean rejects some frames. It then runs ``pipeline._chunk_task`` for chop A with the
settings of NIGHT (default Procyon_13) and the positions ``geometry.find_positions`` gives, as
the pipeline would, and prints one JSON line: the frames processed and rejected, the wall time,
the time per frame in seconds, and the process's peak resident memory (``ru_maxrss``) in GiB.

With ``--save PATH`` the task's binned frames, angles, exposure times, rejected frames and
shifts go to an .npz file; with ``--compare PATH`` they are compared with such a file and the
JSON line gains ``identical`` (true when every array is bit-identical).

Usage: bench_chunk.py M [--night NIGHT] [--radius R] [--save PATH | --compare PATH]
Run with the reduce_nomic package on PYTHONPATH, one configuration per process, so that the
peak memory belongs to that configuration.
"""
import argparse
import json
import os
import resource
import sys
import tempfile
import time
from typing import List, Optional

import numpy as np

from reduce_nomic import config, geometry, pipeline


def synthetic_chunk(path: str, rows: int, seed: int = 0) -> List[str]:
    """Write a synthetic chunk of chop-subtracted frames.

    Parameters
    ----------
    path : str
        Work file to create, ``rows`` x 512 x 512 float32.
    rows : int
        Number of frames, alternating chop A and chop B, starting with A.
    seed : int, default 0
        Seed of the noise generator.

    Returns
    -------
    chops : list of str
        ``CHOP_A`` or ``CHOP_B`` for each frame.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:512, 0:512]
    star_a = 1000.0 * np.exp(-((xx - 150.3) ** 2 + (yy - 100.6) ** 2) / (2 * 4.0 ** 2))
    star_b = 1000.0 * np.exp(-((xx - 400.2) ** 2 + (yy - 101.1) ** 2) / (2 * 4.0 ** 2))
    mm = np.memmap(path, dtype=np.float32, mode="w+", shape=(rows, 512, 512))
    chops = []
    for i in range(rows):
        a = i % 2 == 0
        k = i // 2
        noise = 3.0 if a and k % 7 == 3 else 1.0
        scale = 0.5 if a and k % 11 == 5 else 1.0
        f = (scale * star_a - star_b if a else star_b - star_a) + rng.normal(0.0, noise, (512, 512))
        f[:3], f[-3:], f[:, :3], f[:, -3:] = np.nan, np.nan, np.nan, np.nan
        f[215:] = np.nan
        mm[i] = f
        chops.append("CHOP_A" if a else "CHOP_B")
    mm.flush()
    return chops


def main(argv: Optional[List[str]] = None) -> None:
    """Run the benchmark; see the module docstring.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments; default ``sys.argv[1:]``.

    Returns
    -------
    None
        Prints one JSON line.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("m", type=int, help="frames per chop in the chunk")
    ap.add_argument("--night", default="Procyon_13", help="night whose settings are used")
    ap.add_argument("--radius", type=int, default=None, help="override the night's radius, pixels")
    ap.add_argument("--save", help="write the task's outputs to this .npz file")
    ap.add_argument("--compare", help="compare the task's outputs with this .npz file")
    a = ap.parse_args(argv)
    over = {"side": "left", "display": False}
    if a.radius:
        over["radius"] = a.radius
    st = dict(config.load_settings(a.night, overrides=over).values)
    rows = 2 * a.m
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "chunk.f32")
        chops = synthetic_chunk(path, rows)
        mm = np.memmap(path, dtype=np.float32, mode="r", shape=(rows, 512, 512))
        fw = int(st["framew"])
        first = geometry.padded(np.asarray(mm[0], np.float64), fw)
        second = geometry.padded(np.asarray(mm[1], np.float64), fw)
        p1, p2 = geometry.find_positions(first, second, first, st)
        q1 = geometry.edge_checked(p1, None, int(st["radius"]))
        del mm
        zeros = [0.0] * rows
        task = (0, 0, rows, 1, (p1, p2), q1, path, rows, chops, zeros, zeros, [1.0] * rows, st)
        t0 = time.time()
        res = pipeline._chunk_task(task)
        wall = time.time() - t0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 2 ** 30
    arrays = {"binned": np.asarray(res["binned"]), "angles": np.asarray(res["angles"]),
              "exptimes": np.asarray(res["exptimes"]), "bad": np.asarray(res["bad"]),
              "shifts": np.asarray(res["shifts"]), "corrs": np.asarray(res["corrs"]),
              "noises": np.asarray(res["noises"]), "peaks": np.asarray(res["peaks"])}
    line = {"frames": res["n_in"], "bad": len(res["bad"]), "radius": int(st["radius"]),
            "wall_s": round(wall, 2), "s_per_frame": round(wall / max(res["n_in"], 1), 4),
            "peak_gib": round(peak, 3)}
    if a.save:
        np.savez(a.save, **arrays)
    if a.compare:
        ref = np.load(a.compare)
        line["identical"] = bool(set(ref.files) == set(arrays) and all(
            ref[k].shape == arrays[k].shape and ref[k].dtype == arrays[k].dtype
            and np.array_equal(ref[k], arrays[k], equal_nan=True) for k in arrays))
    print(json.dumps(line))


if __name__ == "__main__":
    main(sys.argv[1:])
