"""Tests of the live display: window size, PNG view, and a real Tk window under Xvfb."""
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pytest

from reduce_nomic import display


def test_fit_scale():
    """Check the window scale on full-HD, 4K, small and high-DPI screens."""
    assert display.fit_scale(1920, 1080) == 1.0
    assert display.fit_scale(3840, 2160) == 1.0
    s = display.fit_scale(1280, 800)
    assert np.isclose(s, min((1280 - 16) / 1850, (800 - 80) / 990))
    assert 1850 * s + 16 <= 1280 and 990 * s + 80 <= 800
    assert display.fit_scale(100, 100) == 0.1
    r = 1.25
    s = display.fit_scale(1920, 1080, r)
    assert np.isclose(s, min((1920 - 16 * r) / (1850 * r), (1080 - 80 * r) / (990 * r)))
    assert 1850 * s * r + 16 * r <= 1920 + 1e-9 and 990 * s * r + 80 * r <= 1080 + 1e-9
    assert display.fit_scale(1920, 1080, 0.85) == 1.0


def _feed(d):
    """Give a view one round of chop identification, a chop subtraction and a chunk.

    Parameters
    ----------
    d : reduce_nomic.display.LiveView or reduce_nomic.display.Display
        View to feed, directly or through the display process.

    Returns
    -------
    None
        Every panel has data.
    """
    rng = np.random.default_rng(1)
    d.start_round("left_round0", "Y of Detected Star (Pixels)", 256.0, "UP-DOWN", desc="rolling background, destriped")
    for b in range(5):
        nums = np.arange(200 * b, 200 * b + 200)
        chop_a = (nums // 6) % 2 == 0
        ys = np.where(chop_a, 192, 320)
        d.add_states(nums, np.full(200, 150), ys, np.where(chop_a, "CHOP_A", "CHOP_B"), ys.astype(float), skipped=3)
        img = np.zeros((512, 512))
        img[21:190, 21:245] = rng.normal(size=(169, 224))
        img[140:150, 135:145] += 50.0
        d.show_chop_image(img, (140.0, 145.0), "CHOP_A", str(nums[-1]), via_bg=(b == 4))
        d.show_progress("left_round0 chop states", b + 1, 5, 10.0 * (b + 1), 10.0 * (4 - b))
    d.show_segments(np.full(166, 6))
    d.show_chopsub(rng.normal(size=(512, 512)), rng.normal(size=(512, 512)), "segment 165")
    d.show_chunk(rng.normal(size=(200, 200)), {"chunk": 0, "cn": 1, "corrs": [0.9] * 20, "noises": [1.0] * 20,
                                                 "peaks": [1.0] * 20, "bad": [3]},
                 {"corr": 0.5, "noise": 3.0, "peak": 0.5})


def test_png_view(tmp_path, monkeypatch):
    """Check the PNG-only view: its size, and the frames and skips it keeps."""
    monkeypatch.delenv("DISPLAY", raising=False)
    png = str(tmp_path / "view.png")
    v = display.LiveView("Fake_1 left", png, log=lambda s: None)
    assert v.backend == "Agg" and "PNG" in v.note
    _feed(v)
    v.close()
    from PIL import Image
    assert Image.open(png).size == display.PANEL_PX
    assert v._skipped == 15 and sum(a.size for a in v._frames["num"]) == 1000
    import matplotlib.pyplot as plt
    plt.close("all")


def test_display_process_png(tmp_path, monkeypatch):
    """Check the display process: it draws what the driver sends, writes the PNG, and is reused."""
    monkeypatch.delenv("DISPLAY", raising=False)
    lines = []
    d = display.Display("Fake_1 left", str(tmp_path / "left.png"), log=lines.append)
    assert d.backend == "Agg" and not d.dead, d.note
    pid = display._SERVER["proc"].pid
    _feed(d)
    d.pump()
    d.close()
    from PIL import Image
    assert Image.open(tmp_path / "left.png").size == display.PANEL_PX
    d2 = display.Display("Fake_1 right", str(tmp_path / "right.png"), log=lines.append)
    assert display._SERVER["proc"].pid == pid and "reused" in d2.note
    _feed(d2)
    d2.close()
    assert Image.open(tmp_path / "right.png").size == display.PANEL_PX
    assert not lines, lines


def test_display_process_death_is_harmless(tmp_path, monkeypatch):
    """Check that the driver goes on when the display process dies (as on a lost X connection)."""
    monkeypatch.delenv("DISPLAY", raising=False)
    lines = []
    d = display.Display("Fake_1 left", str(tmp_path / "left.png"), log=lines.append)
    proc = display._SERVER["proc"]
    proc.kill()
    proc.join(10)
    _feed(d)
    d.pump()
    d.close()
    assert d.dead and any("display process ended" in s for s in lines), lines
    d2 = display.Display("Fake_1 right", str(tmp_path / "right.png"), log=lines.append)
    assert not d2.dead and display._SERVER["proc"].pid != proc.pid
    d2.close()
    assert os.path.exists(tmp_path / "right.png")


_ORPHAN_SCRIPT = r"""
import sys, time
from reduce_nomic import display
d = display.Display("Fake_1 left", sys.argv[1], log=print)
print("CHILD", display._SERVER["proc"].pid, flush=True)
time.sleep(120)
"""


def test_display_process_ends_with_driver(tmp_path, monkeypatch):
    """Check that the display process ends by itself when the driver is killed."""
    monkeypatch.delenv("DISPLAY", raising=False)
    import signal
    env = dict(os.environ)
    env.pop("DISPLAY", None)
    p = subprocess.Popen([sys.executable, "-c", _ORPHAN_SCRIPT, str(tmp_path / "v.png")], env=env,
                         stdout=subprocess.PIPE, text=True)
    line = p.stdout.readline()
    while line and not line.startswith("CHILD"):
        line = p.stdout.readline()
    child = int(line.split()[1])
    os.kill(p.pid, signal.SIGKILL)
    p.wait(10)
    for _ in range(100):
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        os.kill(child, signal.SIGKILL)
        raise AssertionError("the display process outlived its driver")


_UNGUARDED_SCRIPT = """
import sys
print("TOP-LEVEL RUN", flush=True)
from reduce_nomic import display
d = display.Display("Fake_1 left", sys.argv[1], log=print)
print("BACKEND", d.backend, flush=True)
d.close()
"""


def test_display_process_skips_unguarded_main(tmp_path):
    """Check that a driver script without a main guard is not run again in the display process."""
    script = tmp_path / "unguarded.py"
    script.write_text(_UNGUARDED_SCRIPT)
    env = dict(os.environ)
    env.pop("DISPLAY", None)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = os.pathsep.join([repo] + [x for x in [env.get("PYTHONPATH")] if x])
    r = subprocess.run([sys.executable, str(script), str(tmp_path / "v.png")], env=env, capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.count("TOP-LEVEL RUN") == 1 and "BACKEND Agg" in r.stdout, (r.stdout, r.stderr[-2000:])
    assert os.path.exists(tmp_path / "v.png")


#: run in a fresh process with DISPLAY on an Xvfb screen; prints a JSON report
_TK_SCRIPT = r"""
import json, sys, time
import numpy as np
sys.path.insert(0, sys.argv[1])
from test_display import _feed
from reduce_nomic import display
import matplotlib
calls = []
d = display.LiveView("Fake_1 left", sys.argv[2], log=calls.append)
import matplotlib.pyplot as plt
win = d.fig.canvas.manager.window
canvas = d.fig.canvas.get_tk_widget()
win.update()
rep = {"backend": d.backend, "note": d.note, "resizable": [bool(v) for v in win.resizable()],
       "canvas": [canvas.winfo_width(), canvas.winfo_height()],
       "screen": [win.winfo_screenwidth(), win.winfo_screenheight()]}
spy = []
for name in ("attributes", "wm_attributes", "lift", "tkraise", "focus_force", "deiconify"):
    orig = getattr(win, name)
    setattr(win, name, (lambda n, o: (lambda *a, **k: (spy.append([n, [str(x) for x in a]]), o(*a, **k))[1]))(name, orig))
plt.pause = lambda *a, **k: spy.append(["plt.pause", []])
d.interval = 0.0
_feed(d)
ran = []
win.after(0, lambda: ran.append(1))
d.pump()
rep["pump_ran_events"] = bool(ran)
fig0 = d.fig
d.reset("Fake_1 right", sys.argv[2])
_feed(d)
win.update()
rep["reused"] = d.fig is fig0 and [canvas.winfo_width(), canvas.winfo_height()] == rep["canvas"]
rep["spy"] = spy
plt.close(d.fig)
d.refresh(force=True)
d.pump()
rep["closed"] = d.closed
d.close()
rep["log"] = calls
print("REPORT " + json.dumps(rep))
"""


def _xvfb_report(tmp_path, screen, dpi=96):
    """Run ``_TK_SCRIPT`` against a new Xvfb screen and return its report.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Folder for the PNG.
    screen : str
        Xvfb screen geometry, e.g. ``1920x1080x24``.
    dpi : int, default 96
        Dots per inch the screen reports; matplotlib's Tk canvas scales by dpi / 96 on Linux.

    Returns
    -------
    rep : dict
        The script's report; the test is skipped without Xvfb or Tk.
    """
    pytest.importorskip("tkinter")
    xvfb = shutil.which("Xvfb")
    if xvfb is None:
        pytest.skip("no Xvfb")
    num = next(n for n in range(97, 200) if not os.path.exists(f"/tmp/.X11-unix/X{n}") and not os.path.exists(f"/tmp/.X{n}-lock"))
    proc = subprocess.Popen([xvfb, f":{num}", "-screen", "0", screen, "-dpi", str(dpi), "-nolisten", "tcp"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            if os.path.exists(f"/tmp/.X11-unix/X{num}"):
                break
            time.sleep(0.05)
        env = dict(os.environ, DISPLAY=f":{num}")
        env.pop("MPLBACKEND", None)
        here = os.path.dirname(os.path.abspath(__file__))
        r = subprocess.run([sys.executable, "-c", _TK_SCRIPT, here, str(tmp_path / "view.png")], env=env,
                           capture_output=True, text=True, timeout=300)
        line = [x for x in r.stdout.splitlines() if x.startswith("REPORT ")]
        assert r.returncode == 0 and line, r.stderr[-3000:]
        return json.loads(line[0][7:])
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_tk_window_full_hd(tmp_path):
    """Check the Tk window on a 1920 x 1080, 96-dpi screen: size, fixed, never raised, reused, closable."""
    rep = _xvfb_report(tmp_path, "1920x1080x24")
    assert rep["backend"] == "TkAgg"
    assert rep["canvas"] == list(display.PANEL_PX), rep
    assert rep["resizable"] == [False, False]
    assert rep["pump_ran_events"] and rep["reused"]
    assert not any(c[0] in ("lift", "tkraise", "focus_force", "plt.pause", "deiconify") for c in rep["spy"]), rep["spy"]
    assert not any("-topmost" in c[1] for c in rep["spy"]), rep["spy"]
    assert rep["closed"] and any("window closed" in s for s in rep["log"])
    assert os.path.getsize(tmp_path / "view.png") > 10000


def test_tk_window_small_screen(tmp_path):
    """Check that the Tk window is shrunk to fit a 1280 x 800 screen."""
    rep = _xvfb_report(tmp_path, "1280x800x24")
    s = display.fit_scale(1280, 800)
    assert abs(rep["canvas"][0] - 1850 * s) <= 2 and abs(rep["canvas"][1] - 990 * s) <= 2, rep
    assert "largest that fits" in rep["note"]


def test_tk_window_scaled_screen(tmp_path):
    """Check that the Tk window fits a 1920 x 1080 screen at 120 dpi, where matplotlib scales it by 1.25."""
    rep = _xvfb_report(tmp_path, "1920x1080x24", dpi=120)
    r = 120 / 96
    s = display.fit_scale(1920, 1080, r)
    assert abs(rep["canvas"][0] - 1850 * s * r) <= 3 and abs(rep["canvas"][1] - 990 * s * r) <= 3, rep
    assert rep["canvas"][0] + 16 * r <= 1920 and rep["canvas"][1] + 80 * r <= 1080 + 1, rep
    assert "1.25" in rep["note"]
