"""Tests of tools/memlog.sh: one snapshot at a time (INTERVAL_S 0), its header and its fields.

They need Linux's /proc and are skipped elsewhere.
"""
import os
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "tools", "memlog.sh")
pytestmark = pytest.mark.skipif(not os.path.exists("/proc/meminfo"), reason="needs Linux /proc")

HEADER = "time,mem_total_gb,mem_available_gb,cached_gb,swap_used_gb,n_procs,pss_sum_gb,top5"


def run(*args):
    """Run memlog.sh with the given arguments.

    Parameters
    ----------
    *args : str
        Its command-line arguments.

    Returns
    -------
    subprocess.CompletedProcess
        The finished process, with its standard output and error as text.
    """
    return subprocess.run(["bash", SCRIPT, *args], capture_output=True, text=True, timeout=60)


def test_two_snapshots_append_under_one_header(tmp_path):
    """Two one-shot calls: the header once, then two lines of eight fields with sane values."""
    out = tmp_path / "mem.csv"
    for _ in range(2):
        r = run(str(out), "0")
        assert r.returncode == 0, r.stderr
    lines = out.read_text().splitlines()
    assert lines[0] == HEADER and len(lines) == 3
    for line in lines[1:]:
        f = line.split(",")
        assert len(f) == 8
        total, avail, cached, swap = map(float, f[1:5])
        assert total > 0 and 0 <= avail <= total and cached >= 0 and swap >= 0
        assert int(f[5]) >= 1 and float(f[6]) > 0
        top = f[7].split(" ")
        assert 1 <= len(top) <= 5 and all(len(t.split(":")) == 3 for t in top)


def test_rejects_a_short_or_bad_interval(tmp_path):
    """An interval of 3 s or a non-integer is refused with exit status 2 and nothing written."""
    out = tmp_path / "mem.csv"
    for bad in ("3", "1.5", "x"):
        r = run(str(out), bad)
        assert r.returncode == 2 and "INTERVAL_S" in r.stderr
    assert not out.exists()
