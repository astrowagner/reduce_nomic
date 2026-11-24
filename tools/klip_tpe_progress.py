#!/usr/bin/env python3
"""Summarize the search in one annulus of a KLIP-TPE run from its results.jsonl.

KLIP-TPE appends one JSON record per evaluation to ``results.jsonl`` in the run
folder. This script reads the records of one annulus and prints:

- the number of evaluations, failures and evaluations per search phase;
- the injected contrast of each run of consecutive evaluations (the
  re-calibration revisit can change it early in the search);
- the best scores among the evaluations at the latest contrast, with their
  partitions and parameters (scores are comparable only at one contrast);
- the partition selections over the whole annulus and over the last M
  evaluations, and the bin and filter values over the last M;
- the evaluation rate from the dispatch times, over the whole annulus and
  over the last M evaluations.

The run may still be writing the file; a partial last line is skipped.

Usage
-----
python tools/klip_tpe_progress.py RESULTS_JSONL [--annulus N] [--last M] [--top T]

RESULTS_JSONL is the run's ``results.jsonl``; N is the annulus, counted from 1
(default: the annulus of the last record); M is the number of latest
evaluations for the late-search statistics (default 1000); T is the number of
best evaluations to list (default 5).
"""

import argparse
import collections
import json
import sys


def read_records(path, annulus=None):
    """Read the evaluation records of one annulus from a results.jsonl file.

    Parameters
    ----------
    path : str
        Path of the run's ``results.jsonl``.
    annulus : int or None, default None
        Annulus number, counted from 1. None takes the annulus of the last
        complete record.

    Returns
    -------
    annulus : int
        The annulus read, counted from 1.
    records : list of dict
        The records of that annulus in file order, each reduced to the fields
        this script uses: ``index``, ``phase``, ``score``, ``raw_score``,
        ``contrast``, ``failed``, ``t_dispatch``, ``selected`` (tuple of
        partition names), ``params`` (the linked and mean parameters) and
        ``per_partition`` (dict of each selected partition's parameters).
    n_partial : int
        Number of lines that could not be parsed (a partial last line while
        the run is writing).
    """
    rows = []
    n_partial = 0
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                n_partial += 1
                continue
            rows.append(rec)
    if not rows:
        raise SystemExit(f"no complete records in {path}")
    ann0 = rows[-1]["annulus"] if annulus is None else annulus - 1
    records = []
    for rec in rows:
        if rec.get("annulus") != ann0:
            continue
        cfg = rec.get("config") or {}
        meta = rec.get("meta") or {}
        selected = tuple(sorted(cfg.get("selected") or meta.get("selected") or ()))
        per_part = cfg.get("per_partition") or {}
        records.append({
            "index": rec.get("index"),
            "phase": rec.get("phase"),
            "score": rec.get("score"),
            "raw_score": rec.get("raw_score"),
            "contrast": rec.get("contrast"),
            "failed": bool(meta.get("failed", False)),
            "t_dispatch": meta.get("t_dispatch"),
            "selected": selected,
            "params": cfg.get("params") or {},
            "per_partition": {p: per_part.get(p, {}) for p in selected},
        })
    return ann0 + 1, records, n_partial


def contrast_runs(records):
    """Group consecutive records by injected contrast.

    Parameters
    ----------
    records : list of dict
        Records from `read_records`, in file order.

    Returns
    -------
    runs : list of tuple
        One ``(contrast, n, first_index, last_index)`` per run of consecutive
        records at one contrast, where n is the number of records in the run.
    """
    runs = []
    for rec in records:
        c = rec["contrast"]
        if runs and runs[-1][0] == c:
            contrast, n, first, _ = runs[-1]
            runs[-1] = (contrast, n + 1, first, rec["index"])
        else:
            runs.append((c, 1, rec["index"], rec["index"]))
    return runs


def rate(records):
    """Evaluation rate from the dispatch times.

    Parameters
    ----------
    records : list of dict
        Records from `read_records`.

    Returns
    -------
    per_s : float or None
        Evaluations per second between the first and last dispatch time, or
        None with fewer than two dispatch times.
    span_s : float or None
        Seconds between the first and last dispatch time, or None.
    """
    times = sorted(r["t_dispatch"] for r in records if r["t_dispatch"] is not None)
    if len(times) < 2 or times[-1] <= times[0]:
        return None, None
    span = times[-1] - times[0]
    return (len(times) - 1) / span, span


def describe(rec):
    """Format one record's partitions and parameters on one line.

    Parameters
    ----------
    rec : dict
        A record from `read_records`.

    Returns
    -------
    text : str
        Selected partitions, the linked parameters (bin, filter, angsep,
        anglemax) and each partition's n_ang and k_klip.
    """
    p = rec["params"]
    parts = rec["per_partition"]
    n_ang = " / ".join(str(parts[q].get("n_ang")) for q in rec["selected"])
    k = " / ".join(str(parts[q].get("k_klip")) for q in rec["selected"])
    return (f"{'+'.join(rec['selected'])}  bin {p.get('bin')}  filter {p.get('filter')}  "
            f"angsep {p.get('angsep')}  anglemax {p.get('anglemax')}  "
            f"n_ang {n_ang}  k {k}")


def summarize(annulus, records, n_partial, last, top, out=sys.stdout):
    """Print the summary of one annulus.

    Parameters
    ----------
    annulus : int
        Annulus number, counted from 1.
    records : list of dict
        Records of that annulus from `read_records`.
    n_partial : int
        Number of unparsed lines, from `read_records`.
    last : int
        Number of latest records for the late-search statistics.
    top : int
        Number of best records to list at the latest contrast.
    out : file, default sys.stdout
        Where to print.

    Returns
    -------
    None
        The summary goes to `out`.
    """
    pr = lambda *a: print(*a, file=out)
    n = len(records)
    pr(f"annulus {annulus}: {n} evaluations (index {records[0]['index']}..{records[-1]['index']}), "
       f"{sum(r['failed'] for r in records)} failed, {n_partial} unparsed line(s)")
    phases = collections.Counter(r["phase"] for r in records)
    pr("phases: " + ", ".join(f"{k} {v}" for k, v in sorted(phases.items(), key=lambda kv: -kv[1])))
    pr("contrast runs (contrast, n, first index, last index):")
    for c, m, first, lst in contrast_runs(records):
        pr(f"  {c:.3e}  {m:6d}  {first}..{lst}")
    c_last = records[-1]["contrast"]
    at_c = [r for r in records if r["contrast"] == c_last and not r["failed"]
            and r["score"] is not None]
    best = sorted(at_c, key=lambda r: r["score"], reverse=True)[:top]
    pr(f"best {len(best)} of {len(at_c)} at contrast {c_last:.3e} (score, raw score, index, phase):")
    for r in best:
        pr(f"  {r['score']:7.3f}  {r['raw_score']:7.3f}  {r['index']:6d}  {r['phase']:8s}  {describe(r)}")
    tail = records[-last:]
    sel_all = collections.Counter(r["selected"] for r in records)
    sel_tail = collections.Counter(r["selected"] for r in tail)
    pr(f"selections (all {n} / last {len(tail)}):")
    for s, m in sel_all.most_common():
        pr(f"  {'+'.join(s) or '(none)'}: {m} / {sel_tail.get(s, 0)}")
    bins = collections.Counter(r["params"].get("bin") for r in tail)
    filters = collections.Counter(r["params"].get("filter") for r in tail)
    pr(f"bin, last {len(tail)}: " + ", ".join(f"{k}: {v}" for k, v in sorted(bins.items(), key=lambda kv: (kv[0] is None, kv[0]))))
    pr(f"filter, last {len(tail)} (most common 5): " + ", ".join(f"{k}: {v}" for k, v in filters.most_common(5)))
    r_all, span_all = rate(records)
    r_tail, span_tail = rate(tail)
    if r_all is not None:
        pr(f"rate: {r_all:.3f} per s over {span_all:.0f} s (all); "
           f"{r_tail:.3f} per s over {span_tail:.0f} s (last {len(tail)})")


def main(argv=None):
    """Parse the command line and print the summary.

    Parameters
    ----------
    argv : list of str or None, default None
        Command-line arguments; None reads ``sys.argv[1:]``.

    Returns
    -------
    None
        The summary goes to standard output.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("results", help="the run's results.jsonl")
    ap.add_argument("--annulus", type=int, default=None,
                    help="annulus number counted from 1; default the last record's")
    ap.add_argument("--last", type=int, default=1000,
                    help="latest evaluations for the late-search statistics (default 1000)")
    ap.add_argument("--top", type=int, default=5,
                    help="best evaluations to list at the latest contrast (default 5)")
    args = ap.parse_args(argv)
    annulus, records, n_partial = read_records(args.results, args.annulus)
    if not records:
        raise SystemExit(f"no records for annulus {args.annulus}")
    summarize(annulus, records, n_partial, args.last, args.top)


if __name__ == "__main__":
    main()
