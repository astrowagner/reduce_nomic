"""Tests of analysis/procyon3-4_2026-11/run_known_sequence_kn.sh with stand-in run scripts.

The stand-ins record the options and the worker budget each run got and write the run folder (with
``final_results.json`` unless told to fail), so the sequence's order, skips, stops and options are
checked without klip-tpe.
"""
import os
import shutil
import stat
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEQ = os.path.join(ROOT, "analysis", "procyon3-4_2026-11", "run_known_sequence_kn.sh")
RUNS = [("Procyon_20211219", "adi3ann5c1p5kn"), ("Procyon_20211220", "adi3ann5kn"),
        ("Procyon_20211219-20211220", "adi3ann5kn")]
ALICE = "GPU-eb5590c8-23aa-43e8-d3ca-1225db70ca0e"


def _setup(tmp_path, fail_tag=None):
    """Write a stand-in archive, a stand-in klip-tpe and a copy of the sequence script.

    Parameters
    ----------
    tmp_path : pathlib.Path
        pytest's temporary folder.
    fail_tag : tuple of 2 str or None, default None
        ``(folder, tag)`` of the run whose stand-in exits with status 3 and no final_results.json.

    Returns
    -------
    seq : str
        Path of the sequence script's copy (its folder takes STOP_AFTER_CURRENT).
    env : dict
        Environment with ARCHIVE and KT set.
    """
    archive = tmp_path / "NOMIC"
    for folder, tag in RUNS:
        kt = archive / folder / "klip_tpe"
        kt.mkdir(parents=True)
        run = kt / f"run_{tag}"
        fail = "exit 3" if fail_tag == (folder, tag) else ""
        sh = kt / f"run_{tag}.sh"
        sh.write_text("#!/usr/bin/env bash\nset -euo pipefail\n"
                      f"RUN={run}\n"
                      '[ ! -e "${RUN}" ] || { echo "${RUN} exists"; exit 1; }\n'
                      'mkdir "${RUN}"\necho "$*" > "${RUN}/args.txt"\necho "${WORKERS}" > "${RUN}/workers.txt"\n'
                      f'echo "{folder} {tag} running"\n{fail}\n'
                      'echo "{}" > "${RUN}/final_results.json"\n')
    kt_exe = tmp_path / "venv" / "bin" / "klip-tpe"
    kt_exe.parent.mkdir(parents=True)
    kt_exe.write_text("#!/usr/bin/env bash\nexit 0\n")
    kt_exe.chmod(kt_exe.stat().st_mode | stat.S_IXUSR)
    seq_dir = tmp_path / "seq"
    seq_dir.mkdir()
    seq = str(seq_dir / os.path.basename(SEQ))
    shutil.copyfile(SEQ, seq)
    env = dict(os.environ, ARCHIVE=str(archive), KT=str(kt_exe))
    env.pop("WORKERS", None)
    return seq, env


def _run_dir(tmp_path, folder, tag):
    """Path of a stand-in run's folder."""
    return tmp_path / "NOMIC" / folder / "klip_tpe" / f"run_{tag}"


def test_three_runs_in_order_with_max_workers_and_alice(tmp_path):
    """The three run in order, each with WORKERS=max and Alice on auto share, logged beside its script."""
    seq, env = _setup(tmp_path)
    r = subprocess.run(["bash", seq], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    starts = [ln for ln in r.stdout.splitlines() if " starting " in ln]
    assert [ln.split(" starting ")[1].split(":")[0] for ln in starts] == [f"{f} {t}" for f, t in RUNS]
    for folder, tag in RUNS:
        d = _run_dir(tmp_path, folder, tag)
        assert (d / "args.txt").read_text().split() == ["--gpu-share", "auto", "--gpu-device", ALICE,
                                                        "--gpu-workers", "6"]
        assert (d / "workers.txt").read_text().strip() == "max"
        log = tmp_path / "NOMIC" / folder / "klip_tpe" / f"run_{tag}.log"
        assert f"{folder} {tag} running" in log.read_text()
    assert "all three runs done" in r.stdout
    again = subprocess.run(["bash", seq], env=env, capture_output=True, text=True)
    assert again.returncode == 0 and again.stdout.count("has finished: skipped") == 3


def test_stops_at_a_started_run_and_at_the_stop_file(tmp_path):
    """A run folder without final_results.json stops the sequence; STOP_AFTER_CURRENT stops it cleanly."""
    seq, env = _setup(tmp_path)
    _run_dir(tmp_path, *RUNS[1]).mkdir()                    # Procyon_4 started, not finished
    r = subprocess.run(["bash", seq], env=env, capture_output=True, text=True)
    assert r.returncode == 1
    assert "resume it with resume_adi3ann5kn.sh" in r.stdout
    assert (_run_dir(tmp_path, *RUNS[0]) / "final_results.json").exists()
    assert not _run_dir(tmp_path, *RUNS[2]).exists()
    seq2, env2 = _setup(tmp_path / "b")
    open(os.path.join(os.path.dirname(seq2), "STOP_AFTER_CURRENT"), "w").close()
    r = subprocess.run(["bash", seq2], env=env2, capture_output=True, text=True)
    assert r.returncode == 0 and "is there: stopping before Procyon_20211219 adi3ann5c1p5kn" in r.stdout
    assert not _run_dir(tmp_path / "b", *RUNS[0]).exists()


def test_a_failed_run_stops_the_sequence(tmp_path):
    """A run that fails stops the sequence before the next one starts."""
    seq, env = _setup(tmp_path, fail_tag=RUNS[1])
    r = subprocess.run(["bash", seq], env=env, capture_output=True, text=True)
    assert r.returncode != 0
    assert (_run_dir(tmp_path, *RUNS[0]) / "final_results.json").exists()
    assert not (_run_dir(tmp_path, *RUNS[1]) / "final_results.json").exists()
    assert not _run_dir(tmp_path, *RUNS[2]).exists()
