#!/usr/bin/env python3
"""Report how fast a KLIP-TPE run goes and what limits it: the CPU workers, the card or neither.

Reads a run folder and, when given, the file its output went to, and prints a text report:

- from ``results.jsonl`` (one record per evaluation), per annulus: the evaluations, the
  median wall time of one evaluation, the median and best-decile seconds per evaluation
  over consecutive windows of W evaluations, and the mean seconds per evaluation within
  segments of finish order (warm-up, early search, late search);
- from ``card_shares.jsonl`` (a line at each re-balance of ``--gpu-share auto`` that
  changes the share), per window between two consecutive re-balances of one annulus: the
  seconds per trial, the fraction of the window the CPU workers and the card workers were
  busy, their mean seconds per job, the card's share of the injected draws and the jobs
  per trial;
- from the log: the parallel and card set-up lines, the "worker memory" lines and the
  main-process accounting KLIP-TPE prints at each annulus end, and the number of listed
  card jobs that did not fit a card worker's memory and went to the CPU.

An evaluation finishes at its dispatch time plus its wall time (``meta.t_dispatch`` plus
``wall_s``). A gap of ``max_gap`` seconds or more between two finishes (a stop and a
resume, or the end-of-annulus validation) is left out of the rates. The run may still be
writing; lines that do not parse are skipped and counted.

The file name leaves out "klip_tpe" on purpose. KLIP-TPE's liveness scan (``klip-tpe
runs`` and the check a resume makes) counts as one of a run's processes any process whose
command line holds "klip_tpe" or "klip-tpe" together with the run's folder name, so a
tool named klip_tpe_* and given the run folder would block a resume while it ran. For the
same reason, give the run folder as a relative path from its parent.

Usage
-----
python tools/tpe_throughput.py RUN_DIR [--log LOG] [--window W] [--segments I ...]
    [--every K] [--max-gap S] [--max-window T] [--out FILE]

RUN_DIR is the run folder; LOG is the file the run's output went to; W is the window, in
evaluations, of the per-annulus median (default 100); I ... are the finish-order indices
that start the segments of each annulus (default 0 500 1500 4000 7000 9000); K prints
every K-th re-balance window of each annulus, and always its last (default 1); S is the
shortest gap, in seconds, left out of the rates (default 600); T is the longest
re-balance window kept, in seconds (default 1800); FILE also writes the report there
(refused if it exists).
"""

import argparse
import json
import os
import re
import statistics
from datetime import datetime

#: Regular expressions that select a line of the run's log: the parallel and card set-up
#: lines, the per-annulus "worker memory" lines and the main-process accounting at each
#: annulus end.
LOG_PATTERNS = (r"parallel:", r"card: \d+ workers on ", r"card: share auto", r"worker memory:",
                r"CPU workers were busy", r"the main process was busy", r"the main process took",
                r"of that, ")

#: Text of the log line KLIP-TPE writes when a job does not fit its card worker's share of
#: the card's memory and goes to the CPU workers instead. It lists the first few and
#: counts the rest; the report counts the listed lines.
CARD_FALLBACK = "needed more than its worker's share of the card's memory"

#: Finish-order indices that start the default segments of an annulus: the 500-evaluation
#: warm-up, the early search and three stretches of the late search.
DEFAULT_SEGMENTS = (0, 500, 1500, 4000, 7000, 9000)


def read_jsonl(path):
    """Read a JSON-lines file, skipping lines that do not parse.

    Parameters
    ----------
    path : str
        Path of the file.

    Returns
    -------
    rows : list of dict
        The parsed lines, in file order.
    n_bad : int
        Number of non-empty lines that did not parse (a partial last line while the run
        writes).
    """
    rows, n_bad = [], 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                n_bad += 1
    return rows, n_bad


def parse_time(text):
    """Parse a KLIP-TPE time stamp such as ``2026-11-25T23:24:31-0700``.

    Parameters
    ----------
    text : str
        ISO 8601 date and time with a UTC offset, with or without a colon in the offset.

    Returns
    -------
    datetime
        The time, offset-aware.
    """
    return datetime.strptime(re.sub(r"([+-]\d\d):(\d\d)$", r"\1\2", text.strip()),
                             "%Y-%m-%dT%H:%M:%S%z")


def finish_times(records):
    """Group the evaluations' finish times and wall times by annulus.

    Parameters
    ----------
    records : list of dict
        Records of ``results.jsonl``. Each has ``annulus`` (counted from 0), ``wall_s``
        and ``meta.t_dispatch`` (Unix time, in seconds); a record missing one is left out.

    Returns
    -------
    dict of int to list of tuple
        For each annulus, counted from 1, the (finish time, wall time) pairs, in seconds,
        sorted by finish time.
    """
    out = {}
    for r in records:
        t0 = (r.get("meta") or {}).get("t_dispatch")
        wall, ann = r.get("wall_s"), r.get("annulus")
        if t0 is None or wall is None or ann is None:
            continue
        out.setdefault(int(ann) + 1, []).append((float(t0) + float(wall), float(wall)))
    for rows in out.values():
        rows.sort()
    return out


def kept_gaps(finish, max_gap):
    """Return the gaps between consecutive finishes shorter than ``max_gap``.

    Parameters
    ----------
    finish : sequence of float
        Finish times, in seconds, sorted.
    max_gap : float
        Shortest gap left out, in seconds.

    Returns
    -------
    list of float
        The kept gaps, in seconds.
    """
    return [b - a for a, b in zip(finish, finish[1:]) if b - a < max_gap]


def window_rates(finish, window, max_gap):
    """Return the seconds per evaluation over consecutive windows of ``window`` evaluations.

    Parameters
    ----------
    finish : sequence of float
        Finish times, in seconds, sorted.
    window : int
        Evaluations per window.
    max_gap : float
        Shortest gap, in seconds, that leaves the window holding it out.

    Returns
    -------
    list of float
        The seconds per evaluation of each kept window, in finish order.
    """
    out = []
    for i in range(window, len(finish), window):
        seg = finish[i - window:i + 1]
        if all(b - a < max_gap for a, b in zip(seg, seg[1:])):
            out.append((seg[-1] - seg[0]) / window)
    return out


def segment_rates(finish, starts, max_gap):
    """Return the mean seconds per evaluation within segments of finish order.

    Parameters
    ----------
    finish : sequence of float
        Finish times, in seconds, sorted; n is their number.
    starts : sequence of int
        Finish-order indices that start the segments. Indices outside 0 to n - 1 are
        dropped, 0 is added, and the last segment runs to n.
    max_gap : float
        Shortest gap, in seconds, left out of the means.

    Returns
    -------
    list of tuple
        (first index, end index, mean seconds per evaluation) for each segment, the mean
        None when the segment keeps no gap.
    """
    n = len(finish)
    if n == 0:
        return []
    cuts = sorted({0} | {int(s) for s in starts if 0 <= int(s) < n})
    out = []
    for lo, hi in zip(cuts, cuts[1:] + [n]):
        g = kept_gaps(finish[lo:hi], max_gap)
        out.append((lo, hi, sum(g) / len(g) if g else None))
    return out


def share_windows(entries, max_window):
    """Measure each window between two consecutive re-balances of one annulus.

    Parameters
    ----------
    entries : list of dict
        Lines of ``card_shares.jsonl``, in file order. Each has ``annulus`` (counted from
        1), ``after_trials``, ``trials`` (trials since the previous re-balance), ``shares``
        (percent of the injected draws, one per card), ``time``, ``cpu`` and ``cards``.
        ``cpu`` and each card are dicts of ``jobs``, ``busy_s`` (the seconds the workers
        spent on jobs in the window) and ``workers``.
    max_window : float
        Longest window kept, in seconds; a longer one spans a stop.

    Returns
    -------
    list of dict
        One per kept window: ``annulus``, ``after_trials``, ``end`` (datetime of the
        closing re-balance), ``s_trial``, ``cpu_busy`` and ``card_busy`` (fractions of the
        window, the cards summed), ``cpu_s_job`` and ``card_s_job`` (mean seconds per job,
        None without jobs), ``share`` (the first card's percent), ``jobs_trial`` and
        ``cpu_workers``.

    Notes
    -----
    KLIP-TPE writes a line only when the share changes, and each line measures only the
    last ``trials`` trials. A window is kept only when the two lines are that many trials
    apart, so that the measurement covers the whole window.
    """
    out = []
    for a, b in zip(entries, entries[1:]):
        if a.get("annulus") != b.get("annulus"):
            continue
        if None not in (a.get("after_trials"), b.get("after_trials"), b.get("trials")) \
                and b["after_trials"] - a["after_trials"] != b["trials"]:
            continue
        end = parse_time(b["time"])
        dt = (end - parse_time(a["time"])).total_seconds()
        if dt <= 0 or dt > max_window:
            continue
        cpu, cards = b["cpu"], b.get("cards") or []
        card_jobs = sum(c["jobs"] for c in cards)
        card_busy_s = sum(c["busy_s"] for c in cards)
        card_workers = sum(c["workers"] for c in cards)
        trials = b.get("trials") or 1
        out.append({
            "annulus": b["annulus"], "after_trials": b.get("after_trials"), "end": end,
            "s_trial": dt / trials,
            "cpu_busy": cpu["busy_s"] / (cpu["workers"] * dt) if cpu["workers"] else 0.0,
            "card_busy": card_busy_s / (card_workers * dt) if card_workers else 0.0,
            "cpu_s_job": cpu["busy_s"] / cpu["jobs"] if cpu["jobs"] else None,
            "card_s_job": card_busy_s / card_jobs if card_jobs else None,
            "share": (b.get("shares") or [None])[0],
            "jobs_trial": (cpu["jobs"] + card_jobs) / trials,
            "cpu_workers": cpu["workers"],
        })
    return out


def log_lines(path, patterns=LOG_PATTERNS, count=CARD_FALLBACK):
    """Pick the parallel, card, memory and main-process lines out of a run's log.

    Parameters
    ----------
    path : str
        The file the run's output went to.
    patterns : sequence of str, default LOG_PATTERNS
        Regular expressions; a line matching one is selected.
    count : str, default CARD_FALLBACK
        Text whose lines are counted rather than selected.

    Returns
    -------
    lines : list of tuple
        (line number, line without its newline) of each selected line; a line repeated
        verbatim is kept once.
    n_count : int
        Number of lines holding ``count``.
    """
    regs = [re.compile(p) for p in patterns]
    lines, seen, n_count = [], set(), 0
    with open(path, errors="replace") as f:
        for i, line in enumerate(f, 1):
            s = line.rstrip("\n")
            if count and count in s:
                n_count += 1
            elif s not in seen and any(r.search(s) for r in regs):
                seen.add(s)
                lines.append((i, s))
    return lines, n_count


def _fmt(x, spec):
    """Format a number, or a dash for None.

    Parameters
    ----------
    x : float or None
        The number.
    spec : str
        Format specification, such as ``"5.2f"``.

    Returns
    -------
    str
        The formatted number, or a dash right-aligned to the specification's width.
    """
    if x is None:
        width = re.match(r"\d*", spec).group(0)
        return "-".rjust(int(width) if width else 1)
    return format(x, spec)


def report(run_dir, log=None, window=100, starts=DEFAULT_SEGMENTS, every=1, max_gap=600.0,
           max_window=1800.0):
    """Build the throughput report of one run.

    Parameters
    ----------
    run_dir : str
        The run folder, holding ``results.jsonl`` and, for a run with ``--gpu-share``,
        ``card_shares.jsonl``.
    log : str or None, default None
        The file the run's output went to; None leaves the log section out.
    window : int, default 100
        Evaluations per window of the per-annulus median.
    starts : sequence of int, default DEFAULT_SEGMENTS
        Finish-order indices that start the segments of each annulus.
    every : int, default 1
        Print every ``every``-th re-balance window of each annulus, and always its last.
    max_gap : float, default 600.0
        Shortest gap between two finishes, in seconds, left out of the rates.
    max_window : float, default 1800.0
        Longest re-balance window kept, in seconds.

    Returns
    -------
    str
        The report, lines joined by newlines.
    """
    name = os.path.basename(os.path.normpath(run_dir))
    lines = [f"KLIP-TPE throughput: {name}"]
    records, n_bad = read_jsonl(os.path.join(run_dir, "results.jsonl"))
    lines.append(f"results.jsonl: {len(records)} records ({n_bad} unparsable lines)")
    for ann, rows in sorted(finish_times(records).items()):
        finish = [r[0] for r in rows]
        lines.append("")
        lines.append(f"Annulus {ann}: {len(rows)} evaluations, median wall "
                     f"{statistics.median(r[1] for r in rows):.1f} s")
        rates = sorted(window_rates(finish, window, max_gap))
        if rates:
            lines.append(f"  s/evaluation over {window}-evaluation windows: median "
                         f"{statistics.median(rates):.2f}, best decile {rates[len(rates) // 10]:.2f} "
                         f"({len(rates)} windows)")
        lines.append(f"  mean s/evaluation by finish order (gaps of {max_gap:g} s or more left out):")
        for lo, hi, mean in segment_rates(finish, starts, max_gap):
            lines.append(f"    {lo:5d}-{hi:<5d} {_fmt(mean, '6.2f')}")
    shares_path = os.path.join(run_dir, "card_shares.jsonl")
    if os.path.exists(shares_path):
        entries, n_bad = read_jsonl(shares_path)
        wins = share_windows(entries, max_window)
        lines.append("")
        lines.append(f"Card re-balances: {len(entries)} lines ({n_bad} unparsable), {len(wins)} windows "
                     f"between two of one annulus (busy = share of the window the workers spent on jobs)")
        lines.append("  ann  after   end       s/trial  CPU busy (procs, s/job)  card busy (s/job)  "
                     "share %  jobs/trial")
        by_ann = {}
        for w in wins:
            by_ann.setdefault(w["annulus"], []).append(w)
        for ann in sorted(by_ann):
            ws = by_ann[ann]
            for i, w in enumerate(ws):
                if i % max(every, 1) and i != len(ws) - 1:
                    continue
                lines.append(
                    f"  {ann:3d}  {w['after_trials']:5d}   {w['end']:%H:%M:%S}  {w['s_trial']:7.2f}"
                    f"  {100 * w['cpu_busy']:7.0f}% ({w['cpu_workers']:3d}, {_fmt(w['cpu_s_job'], '5.1f')})"
                    f"     {100 * w['card_busy']:5.0f}% ({_fmt(w['card_s_job'], '4.2f')})"
                    f"     {_fmt(w['share'], '5.1f')}  {w['jobs_trial']:10.1f}")
    if log is not None:
        lines.append("")
        picked, n_fallback = log_lines(log)
        lines.append(f"Log ({os.path.basename(log)}), selected lines:")
        for i, s in picked:
            lines.append(f"  {i:7d}: {s.strip()}")
        lines.append(f"  card jobs sent to the CPU for want of card memory: {n_fallback} listed in the log "
                     f"(KLIP-TPE lists the first few and counts the rest)")
    return "\n".join(lines)


def main(argv=None):
    """Parse the command line, print the report and, with ``--out``, write it.

    Parameters
    ----------
    argv : list of str or None, default None
        Command-line arguments; None reads ``sys.argv[1:]``.

    Returns
    -------
    None
        The report goes to standard output, and to the ``--out`` file when given.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir", help="the run folder (relative to its parent, see the module doc)")
    ap.add_argument("--log", default=None, help="the file the run's output went to")
    ap.add_argument("--window", type=int, default=100,
                    help="evaluations per window of the per-annulus median (default 100)")
    ap.add_argument("--segments", type=int, nargs="+", default=list(DEFAULT_SEGMENTS),
                    help="finish-order indices that start the segments (default 0 500 1500 4000 7000 9000)")
    ap.add_argument("--every", type=int, default=1,
                    help="print every K-th re-balance window, and always the last (default 1)")
    ap.add_argument("--max-gap", type=float, default=600.0,
                    help="shortest gap between finishes, in seconds, left out of the rates (default 600)")
    ap.add_argument("--max-window", type=float, default=1800.0,
                    help="longest re-balance window kept, in seconds (default 1800)")
    ap.add_argument("--out", default=None, help="also write the report here (refused if it exists)")
    args = ap.parse_args(argv)
    text = report(args.run_dir, args.log, args.window, args.segments, args.every, args.max_gap,
                  args.max_window)
    print(text)
    if args.out:
        with open(args.out, "x") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
