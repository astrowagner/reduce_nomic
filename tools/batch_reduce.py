#!/usr/bin/env python3
"""Run a plan of reductions one after another, each in its own process; resumable.

Usage: batch_reduce.py PLAN_CSV ARCHIVE --work DIR [--workers N] [--log LOG_CSV]
                       [--min-free-gb G] [--only ID ...] [--nights-json PATH] [--display] [--dry-run]

PLAN_CSV holds one row per run, in the order to run them:

* ``id``: unique label of the run;
* ``night``: night folder, current or old name;
* ``side``: ``left`` or ``right``;
* ``out``: output folder relative to the night folder, default ``reduced_py`` (sky tests use
  subfolders such as ``reduced_py/skytest_nosky``);
* ``set``: JSON object of settings for this run, passed as ``--set NAME=VALUE`` (may be empty);
* ``note``: free text.

Each run is in one of three states, from its output folder alone:

* **done**: ``<night>_<side>_summary.json`` exists, the run's last product; skipped;
* **partial**: some of the run's products (``pipeline.planned_outputs``) exist but not the
  summary, as after a crash; skipped and reported, never deleted, since the run would refuse to
  start over them;
* **to do**: none exists.

A run to do starts only when the archive's file system has at least ``--min-free-gb`` free. It
runs ``python -m reduce_nomic.cli NIGHT --side SIDE --workers N --archive ARCHIVE --out OUT --work
DIR [--set ...]`` with this interpreter, its output in ``<out>/<night>_<side>_run_<UTC>.log``. All
runs share the work folder ``DIR``, one after another: each round recreates its memmaps there
(``mode="w+"``), so the scratch stays at the size of the largest run instead of growing with
every night. A failed run is recorded and the batch goes on. Ctrl-C stops the batch; the running
reduction gets the same interrupt, and the run is recorded as interrupted.

Parameters
----------
PLAN_CSV : str
    The plan.
ARCHIVE : str
    Archive root.
--work : str
    Shared work folder for the memmaps; created if missing.
--workers : int, default 64
    Worker processes per run.
--log : str, optional
    Batch log to append to; default ``<PLAN_CSV without .csv>_log.csv``.
--min-free-gb : float, default 1500
    Free space, in GB (1e9 bytes), below which no further run starts.
--only : list of str, optional
    Run only these ids.
--nights-json : str, optional
    Object blocks for every run, instead of the package's ``config/nights_idl.json``.
--display : flag
    Pass ``--display`` to every run.
--dry-run : flag
    Print each run's state and command and stop.

Outputs
-------
<night>/<out>/
    Each run's products and its log ``<night>_<side>_run_<UTC>.log``.
LOG_CSV
    One row per run started: ``id``, ``night``, ``side``, ``out``, ``start``, ``end`` (UTC),
    ``minutes``, ``exit`` (the process's exit status, ``interrupted``, or ``missing folder``),
    ``log``.
Standard output
    One line per run.
"""
import argparse
import csv
import datetime
import json
import os
import subprocess
import sys
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reduce_nomic import names  # noqa: E402
from reduce_nomic.pipeline import planned_outputs  # noqa: E402

LOG_FIELDS = ["id", "night", "side", "out", "start", "end", "minutes", "exit", "log"]


def utc_now() -> datetime.datetime:
    """Return the current UTC time.

    Returns
    -------
    now : datetime
        Time zone aware, UTC.
    """
    return datetime.datetime.now(datetime.timezone.utc)


def run_state(out_dir: str, night: str, side: str) -> str:
    """Tell from its output folder whether a run is done, partial or still to do.

    Parameters
    ----------
    out_dir : str
        The run's output folder.
    night : str
        Current night folder name, the products' prefix.
    side : str
        ``left`` or ``right``.

    Returns
    -------
    state : str
        ``done``, ``partial`` or ``todo`` (see the module docstring).
    """
    pre = f"{night}_{side}"
    if os.path.exists(os.path.join(out_dir, f"{pre}_summary.json")):
        return "done"
    if any(os.path.exists(os.path.join(out_dir, f)) for f in planned_outputs(pre)):
        return "partial"
    return "todo"


def command(row: dict, archive: str, work: str, workers: int, nights_json: Optional[str], display: bool) -> List[str]:
    """Build the command line of one run.

    Parameters
    ----------
    row : dict
        Plan row, with ``night`` resolved to the current folder name.
    archive : str
        Archive root.
    work : str
        Shared work folder.
    workers : int
        Worker processes.
    nights_json : str, optional
        Object blocks to use instead of the package's.
    display : bool
        Add ``--display``.

    Returns
    -------
    argv : list of str
        The command, starting with this interpreter.
    """
    out_dir = os.path.join(archive, row["night"], row.get("out") or "reduced_py")
    argv = [sys.executable, "-m", "reduce_nomic.cli", row["night"], "--side", row["side"], "--workers", str(workers),
            "--archive", archive, "--out", out_dir, "--work", work]
    for k, v in json.loads(row.get("set") or "{}").items():
        argv += ["--set", f"{k}={json.dumps(v)}"]
    if nights_json:
        argv += ["--nights-json", nights_json]
    if display:
        argv.append("--display")
    return argv


def free_gb(path: str) -> float:
    """Return the free space of the file system holding a path.

    Parameters
    ----------
    path : str
        Any path on the file system.

    Returns
    -------
    free : float
        Space available to this user, in GB (1e9 bytes).
    """
    s = os.statvfs(path)
    return s.f_bavail * s.f_frsize / 1e9


def append_log(path: str, row: dict) -> None:
    """Append one row to the batch log, creating it with a header when absent.

    Parameters
    ----------
    path : str
        Batch log.
    row : dict
        Values of ``LOG_FIELDS``.

    Returns
    -------
    None
        Appends to ``path`` and flushes it to disk.
    """
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
        fh.flush()
        os.fsync(fh.fileno())


def main(argv: Optional[List[str]] = None) -> int:
    """Run the plan; see the module docstring.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments; default ``sys.argv[1:]``.

    Returns
    -------
    status : int
        0 when every run of the plan is done, 1 when any failed, is partial or was not started.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("plan_csv")
    ap.add_argument("archive")
    ap.add_argument("--work", required=True)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--log")
    ap.add_argument("--min-free-gb", type=float, default=1500.0)
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--nights-json")
    ap.add_argument("--display", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    log_csv = a.log or (a.plan_csv[:-4] if a.plan_csv.endswith(".csv") else a.plan_csv) + "_log.csv"
    rows = list(csv.DictReader(open(a.plan_csv)))
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        sys.exit("plan ids are not unique")
    if a.only:
        unknown = set(a.only) - set(ids)
        if unknown:
            sys.exit(f"unknown ids: {sorted(unknown)}")
        rows = [r for r in rows if r["id"] in a.only]
    for r in rows:
        r["night"] = names.resolve(a.archive, r["night"]).folder
        if r["side"] not in ("left", "right"):
            sys.exit(f"{r['id']}: side must be left or right")
        json.loads(r.get("set") or "{}")
    if not a.dry_run:
        os.makedirs(a.work, exist_ok=True)
    states = {r["id"]: run_state(os.path.join(a.archive, r["night"], r.get("out") or "reduced_py"), r["night"], r["side"])
              for r in rows}
    print(f"{len(rows)} runs: {sum(s == 'done' for s in states.values())} done, "
          f"{sum(s == 'partial' for s in states.values())} partial, {sum(s == 'todo' for s in states.values())} to do; "
          f"log {log_csv}", flush=True)
    failed = []
    for k, r in enumerate(rows, 1):
        tag = f"[{k}/{len(rows)}] {r['id']} ({r['night']} {r['side']}, {r.get('out') or 'reduced_py'})"
        state = states[r["id"]]
        if state != "todo":
            print(f"{tag}: {state}, skipped", flush=True)
            if state == "partial":
                failed.append(r["id"])
            continue
        argv_run = command(r, a.archive, a.work, a.workers, a.nights_json, a.display)
        if a.dry_run:
            print(f"{tag}: to do\n  {' '.join(argv_run)}", flush=True)
            continue
        if not os.path.isdir(os.path.join(a.archive, r["night"])):
            print(f"{tag}: no such night folder", flush=True)
            append_log(log_csv, {"id": r["id"], "night": r["night"], "side": r["side"], "out": r.get("out") or "reduced_py",
                                 "start": "", "end": "", "minutes": "", "exit": "missing folder", "log": ""})
            failed.append(r["id"])
            continue
        free = free_gb(a.archive)
        if free < a.min_free_gb:
            print(f"{tag}: only {free:.0f} GB free (< {a.min_free_gb:.0f}); stopping", flush=True)
            return 1
        out_dir = os.path.join(a.archive, r["night"], r.get("out") or "reduced_py")
        os.makedirs(out_dir, exist_ok=True)
        t0 = utc_now()
        log_path = os.path.join(out_dir, f"{r['night']}_{r['side']}_run_{t0.strftime('%Y%m%dT%H%M%SZ')}.log")
        print(f"{tag}: started {t0.strftime('%Y-%m-%d %H:%M:%S')} UTC, log {log_path}", flush=True)
        interrupted = False
        with open(log_path, "x") as fh:
            fh.write(" ".join(argv_run) + "\n")
            fh.flush()
            proc = subprocess.Popen(argv_run, stdout=fh, stderr=subprocess.STDOUT)
            try:
                code = proc.wait()
            except KeyboardInterrupt:
                interrupted = True
                code = proc.wait()
        t1 = utc_now()
        minutes = (t1 - t0).total_seconds() / 60
        result = "interrupted" if interrupted else str(code)
        append_log(log_csv, {"id": r["id"], "night": r["night"], "side": r["side"], "out": r.get("out") or "reduced_py",
                             "start": t0.strftime("%Y-%m-%dT%H:%M:%SZ"), "end": t1.strftime("%Y-%m-%dT%H:%M:%SZ"),
                             "minutes": f"{minutes:.1f}", "exit": result, "log": log_path})
        print(f"{tag}: {'done' if code == 0 and not interrupted else 'FAILED (' + result + ')'} in {minutes:.1f} min",
              flush=True)
        if interrupted:
            print("interrupted; stopping", flush=True)
            return 1
        if code != 0:
            failed.append(r["id"])
    if failed:
        print(f"not done: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
