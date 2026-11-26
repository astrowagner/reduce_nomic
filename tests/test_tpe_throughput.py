"""Tests of tools/tpe_throughput.py on synthetic run folders.

The folders hold a ``results.jsonl`` and a ``card_shares.jsonl`` in KLIP-TPE's formats:
annuli counted from 0 in the results and from 1 in the re-balances, time stamps with a
-0700 offset.
"""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import tpe_throughput as tt  # noqa: E402


def write_jsonl(path, rows, tail=None):
    """Write rows as JSON lines, with an optional unparsable last line.

    Parameters
    ----------
    path : str
        File to write.
    rows : list of dict
        The records.
    tail : str or None, default None
        Text appended after the records without a newline, as a run caught mid-write leaves.

    Returns
    -------
    None
        The file is written.
    """
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        if tail:
            f.write(tail)


def evals(annulus, t0, step, n, wall=10.0, start_index=0):
    """Return n evaluation records finishing every ``step`` seconds from ``t0 + wall``.

    Parameters
    ----------
    annulus : int
        Annulus, counted from 0.
    t0 : float
        Dispatch time of the first record, Unix seconds.
    step : float
        Seconds between consecutive dispatches (and finishes).
    n : int
        Number of records.
    wall : float, default 10.0
        Wall time of every record, in seconds.
    start_index : int, default 0
        Index of the first record.

    Returns
    -------
    list of dict
        Records with ``annulus``, ``index``, ``wall_s`` and ``meta.t_dispatch``.
    """
    return [{"annulus": annulus, "index": start_index + i, "wall_s": wall,
             "meta": {"t_dispatch": t0 + i * step}} for i in range(n)]


def share(annulus, after, time, cpu_jobs, cpu_busy, card_jobs, card_busy, share_pct=50.0):
    """Return one re-balance line of ``card_shares.jsonl``.

    Parameters
    ----------
    annulus : int
        Annulus, counted from 1.
    after : int
        Trials of the annulus done at the re-balance.
    time : str
        Time stamp, as KLIP-TPE writes it.
    cpu_jobs, card_jobs : int
        Jobs the CPU workers and the card finished in the window.
    cpu_busy, card_busy : float
        Seconds the 24 CPU workers and the 6 card workers spent on jobs in the window.
    share_pct : float, default 50.0
        The card's share of the injected draws, in percent.

    Returns
    -------
    dict
        The line's fields.
    """
    return {"annulus": annulus, "after_trials": after, "shares": [share_pct], "trials": 25,
            "cpu": {"jobs": cpu_jobs, "busy_s": cpu_busy, "workers": 24},
            "cards": [{"jobs": card_jobs, "busy_s": card_busy, "workers": 6}],
            "eligible": card_jobs, "balanced": [90.0], "time": time}


def test_parse_time_accepts_both_offset_forms():
    """A -0700 offset and a -07:00 offset give the same instant, 7 h behind UTC."""
    a = tt.parse_time("2026-11-25T23:24:31-0700")
    b = tt.parse_time("2026-11-25T23:24:31-07:00")
    assert a == b
    assert a.utcoffset().total_seconds() == -7 * 3600


def test_read_jsonl_skips_a_partial_line(tmp_path):
    """Two complete records and a half-written one: two rows, one unparsable line."""
    p = tmp_path / "results.jsonl"
    write_jsonl(p, evals(0, 1000.0, 2.0, 2), tail='{"annulus": 0, "wall')
    rows, n_bad = tt.read_jsonl(str(p))
    assert len(rows) == 2 and n_bad == 1


def test_finish_times_groups_by_annulus_from_one_and_sorts():
    """Annulus 0 becomes 1, finish = dispatch + wall, and the pairs come out in finish order."""
    recs = evals(1, 100.0, 5.0, 3, wall=1.0)[::-1] + evals(0, 0.0, 1.0, 2, wall=3.0)
    recs.append({"annulus": 0, "wall_s": 1.0, "meta": {}})        # no dispatch time: left out
    ft = tt.finish_times(recs)
    assert sorted(ft) == [1, 2]
    assert ft[1] == [(3.0, 3.0), (4.0, 3.0)]
    assert [f for f, _ in ft[2]] == [101.0, 106.0, 111.0]


def test_window_rates_leave_out_a_window_holding_a_stop():
    """150 finishes 2 s apart, a 1 h stop, then 150 finishes 1 s apart, in windows of 50."""
    finish = [2.0 * i for i in range(150)]
    finish += [finish[-1] + 3600.0 + i for i in range(150)]
    rates = tt.window_rates(finish, 50, 600.0)
    assert rates == pytest.approx([2.0, 2.0, 1.0, 1.0])      # windows end at 50, 100, 200, 250


def test_segment_rates_by_finish_order():
    """Segments start at 0, 100 and 250; the stop's gap is left out of its segment's mean."""
    finish = [2.0 * i for i in range(150)]
    finish += [finish[-1] + 3600.0 + i for i in range(150)]
    segs = tt.segment_rates(finish, [100, 250, 999], 600.0)
    assert [(lo, hi) for lo, hi, _ in segs] == [(0, 100), (100, 250), (250, 300)]
    assert segs[0][2] == pytest.approx(2.0)
    assert segs[1][2] == pytest.approx((49 * 2.0 + 99 * 1.0) / 148)
    assert segs[2][2] == pytest.approx(1.0)
    assert tt.segment_rates([], [0], 600.0) == []


def test_share_windows_busy_fractions_and_per_job_times():
    """A 50 s window of 25 trials: 0.5 CPU busy, 0.2 card busy, 6 s and 0.4 s a job."""
    entries = [share(1, 25, "2026-11-25T23:00:00-0700", 1, 1.0, 1, 1.0),
               share(1, 50, "2026-11-25T23:00:50-0700", 100, 600.0, 150, 60.0, share_pct=88.5),
               share(2, 25, "2026-11-25T23:01:00-0700", 10, 10.0, 0, 0.0),      # new annulus
               share(2, 50, "2026-11-26T00:01:00-0700", 10, 10.0, 0, 0.0),      # spans a stop
               share(2, 150, "2026-11-26T00:02:00-0700", 10, 10.0, 0, 0.0)]     # 100 trials apart
    wins = tt.share_windows(entries, 1800.0)
    assert len(wins) == 1
    w = wins[0]
    assert w["annulus"] == 1 and w["after_trials"] == 50
    assert w["s_trial"] == pytest.approx(2.0)
    assert w["cpu_busy"] == pytest.approx(0.5)
    assert w["card_busy"] == pytest.approx(0.2)
    assert w["cpu_s_job"] == pytest.approx(6.0)
    assert w["card_s_job"] == pytest.approx(0.4)
    assert w["jobs_trial"] == pytest.approx(10.0)
    assert w["share"] == 88.5 and w["cpu_workers"] == 24


def test_log_lines_select_keep_repeats_once_and_count_fallbacks(tmp_path):
    """The parallel line, a worker-memory line printed twice, a re-balance line (not picked),
    the card set-up line, two card-memory fallbacks (counted) and an accounting line."""
    p = tmp_path / "run.log"
    p.write_text("\n".join([
        "  parallel: 48 workers on 96 cores (192 logical CPUs): 24 worker processes x 2 thread(s) per job",
        "  something else",
        "  worker memory: 24 processes peaked at 4.93 GB each",
        "  worker memory: 24 processes peaked at 4.93 GB each",
        "  card: share 12 % -> 17 % over the last 25 trials, jobs a minute at full load card 1413",
        "  card: 6 workers on NVIDIA RTX PRO 6000 (driver 595.99.02), 12 % of the search's injected draws",
        "  card: a1_e76_d2_inj_P4_left_A needed more than its worker's share of the card's memory (OOM)",
        "  card: a1_e77_d1_inj_P4_left_B needed more than its worker's share of the card's memory (OOM)",
        "  annulus 1: the main process took 0.232 s a trial, 0.010 s scoring",
    ]) + "\n")
    got, n_fallback = tt.log_lines(str(p))
    assert [i for i, _ in got] == [1, 3, 6, 9]
    assert n_fallback == 2


def test_main_writes_the_report_and_refuses_to_overwrite(tmp_path, capsys):
    """A two-annulus run with re-balances: the report names both annuli and the windows."""
    run = tmp_path / "run_x"
    run.mkdir()
    write_jsonl(run / "results.jsonl", evals(0, 0.0, 2.0, 120) + evals(1, 1000.0, 3.0, 60, start_index=120))
    write_jsonl(run / "card_shares.jsonl",
                [share(1, 25, "2026-11-25T23:00:00-0700", 1, 1.0, 1, 1.0),
                 share(1, 50, "2026-11-25T23:00:50-0700", 100, 600.0, 150, 60.0)])
    log = tmp_path / "run_x.log"
    log.write_text("  parallel: 192 workers on 96 cores (192 logical CPUs): 96 worker processes\n")
    out = tmp_path / "report.txt"
    tt.main([str(run), "--log", str(log), "--window", "50", "--out", str(out)])
    text = out.read_text()
    assert "Annulus 1: 120 evaluations" in text and "Annulus 2: 60 evaluations" in text
    assert "median 2.00" in text and "23:00:50" in text and "96 worker processes" in text
    assert capsys.readouterr().out.strip() == text.strip()
    with pytest.raises(FileExistsError):
        tt.main([str(run), "--out", str(out)])
