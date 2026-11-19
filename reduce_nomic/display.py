"""Live display of a reduction, in the spirit of the IDL predecessor's ``cgimage`` windows.

With ``--display`` the driver shows, as the stages run: the latest raw frame's crude star
image with the detected star and chop state, the latest chop-subtracted frame and its 35-px
high-pass, each chunk's centered median, the frame-selection measures with their thresholds,
and a progress panel. The window opens when an X display and Tk are available; otherwise the
same panels are written to a PNG file, rewritten at each refresh.
"""
from __future__ import annotations

import os
import time
from typing import Dict, Optional, Sequence

import numpy as np


def _backend() -> str:
    """Choose the matplotlib backend for the display.

    Returns
    -------
    name : str
        ``TkAgg`` when ``DISPLAY`` is set and tkinter imports, otherwise ``Agg``.
    """
    if os.environ.get("DISPLAY"):
        try:
            import tkinter  # noqa: F401
            return "TkAgg"
        except ImportError:
            pass
    return "Agg"


def _cmap(name: str):
    """Return a copy of a matplotlib colormap that draws NaN pixels in navy.

    Parameters
    ----------
    name : str
        Colormap name.

    Returns
    -------
    cmap : matplotlib.colors.Colormap
        The colormap, NaN shown in navy so masked pixels stand out.
    """
    import matplotlib
    return matplotlib.colormaps[name].with_extremes(bad="navy")


def _stretch(img: np.ndarray, lo: float = 1.0, hi: float = 99.5):
    """Return display limits from percentiles of the finite pixels.

    Parameters
    ----------
    img : ndarray
        Image.
    lo, hi : float, default 1.0 and 99.5
        Percentiles for the lower and upper limits.

    Returns
    -------
    vmin, vmax : float
        Limits, or (0, 1) when no pixel is finite.
    """
    v = img[np.isfinite(img)]
    if v.size == 0:
        return 0.0, 1.0
    a, b = np.percentile(v, [lo, hi])
    return float(a), float(b if b > a else a + 1.0)


class Display:
    """A six-panel live view of one night and aperture.

    Parameters
    ----------
    title : str
        Window title, e.g. ``Procyon_13 left``.
    png : str
        File written at each refresh when no window can open.
    min_interval : float, default 1.0
        Minimum time between two refreshes, in seconds.
    """

    def __init__(self, title: str, png: str, min_interval: float = 1.0):
        import matplotlib
        self.backend = _backend()
        matplotlib.use(self.backend, force=True)
        import matplotlib.pyplot as plt
        self.plt = plt
        self.png = png
        self.min_interval = min_interval
        self.last = 0.0
        self.title = title
        if self.backend != "Agg":
            plt.ion()
        self.fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
        self.ax = axes.ravel()
        self.fig.suptitle(title)
        for a, t in zip(self.ax, ("Raw Frame (Crude Star Image)", "Chop-Subtracted Frame",
                                  "Chop-Subtracted, 35-px High-Pass", "Centered Median (Central 200 px)",
                                  "Frame Selection", "Progress")):
            a.set_title(t)
        self.ax[5].axis("off")
        self.progress = {}
        self._text = None
        self.refresh(force=True)

    def show_raw(self, img: np.ndarray, star: Sequence[int], chop: str, label: str) -> None:
        """Show a raw frame's crude star image with the detected star.

        Parameters
        ----------
        img : ndarray, shape (ny, nx)
            Crude star image (downsampled by ``scale``-free factor; positions in its pixels).
        star : sequence of int
            ``(x, y)`` of the detected star in ``img`` pixels.
        chop : str
            The frame's chop state.
        label : str
            Frame label shown in the panel title.

        Returns
        -------
        None
            The panel is redrawn at the next refresh.
        """
        a = self.ax[0]
        a.clear()
        vmin, vmax = _stretch(img)
        a.imshow(img, origin="lower", cmap=_cmap("gray"), vmin=vmin, vmax=vmax)
        a.plot([star[0]], [star[1]], "o", mfc="none", mec="red", ms=14)
        a.set_title(f"Raw Frame {label}: {chop}")
        self.refresh()

    def show_chopsub(self, img: np.ndarray, hp: np.ndarray, label: str) -> None:
        """Show a chop-subtracted frame and its high-pass.

        Parameters
        ----------
        img : ndarray, shape (ny, nx)
            Chop-subtracted, corrected frame.
        hp : ndarray, shape (ny, nx)
            The same minus its 35-px boxcar.
        label : str
            Frame label shown in the panel titles.

        Returns
        -------
        None
            The panels are redrawn at the next refresh.
        """
        for a, im, t in ((self.ax[1], img, "Chop-Subtracted"), (self.ax[2], hp, "35-px High-Pass")):
            a.clear()
            vmin, vmax = _stretch(im)
            a.imshow(im, origin="lower", cmap=_cmap("gray"), vmin=vmin, vmax=vmax)
            a.set_title(f"{t} {label}")
        self.refresh()

    def show_chunk(self, med: np.ndarray, res: Dict, thresholds: Dict) -> None:
        """Show one chunk's centered median and frame-selection measures.

        Parameters
        ----------
        med : ndarray, shape (200, 200)
            Median of the chunk's centered frames, central 200x200 pixels.
        res : dict
            The chunk result: ``chunk``, ``cn``, ``corrs``, ``noises``, ``peaks``, ``bad``.
        thresholds : dict
            ``corr``, ``noise`` and ``peak`` thresholds of the pre-clean for this chop.

        Returns
        -------
        None
            The panels are redrawn at the next refresh.
        """
        a = self.ax[3]
        a.clear()
        vmin, vmax = _stretch(med, 0.5, 99.9)
        a.imshow(med, origin="lower", cmap=_cmap("inferno"), vmin=vmin, vmax=vmax)
        a.axhline(99.5, color="cyan", lw=0.5)
        a.axvline(99.5, color="cyan", lw=0.5)
        a.set_title(f"Centered Median: Chunk {res['chunk']}, Cube {res['cn']}")
        b = self.ax[4]
        b.clear()
        c = np.asarray(res["corrs"])
        n = np.asarray(res["noises"])
        p = np.asarray(res["peaks"])
        x = np.arange(c.size)
        if c.size:
            b.plot(x, c, ".", ms=3, label="Correlation")
            b.plot(x, n / np.nanmean(n), ".", ms=3, label="Noise / Mean")
            b.plot(x, p / np.nanmean(p), ".", ms=3, label="Peak / Mean")
            b.axhline(thresholds["corr"], color="C0", ls="--", lw=0.8)
            b.axhline(thresholds["noise"], color="C1", ls="--", lw=0.8)
            b.axhline(thresholds["peak"], color="C2", ls="--", lw=0.8)
            bad = np.asarray(res["bad"], dtype=int)
            bad = bad[bad < c.size]
            if bad.size:
                b.plot(bad, c[bad], "x", color="red", ms=5, label="Rejected")
        b.set_xlabel("Frame In Chunk")
        b.set_ylabel("Value")
        b.set_title(f"Frame Selection: {len(res['bad'])} of {c.size} Rejected")
        b.legend(fontsize=7, loc="lower left")
        self.refresh()

    def show_progress(self, stage: str, done: int, total: int, elapsed: float, eta: Optional[float]) -> None:
        """Update the progress panel.

        Parameters
        ----------
        stage : str
            Stage label.
        done, total : int
            Tasks done and in all.
        elapsed : float
            Seconds since the stage started.
        eta : float or None
            Estimated seconds left.

        Returns
        -------
        None
            The panel is redrawn at the next refresh.
        """
        self.progress[stage] = (done, total, elapsed, eta)
        lines = [self.title, ""]
        for s, (d, t, e, r) in self.progress.items():
            pct = 100.0 * d / t if t else 100.0
            left = f", {r / 60:.1f} min left" if r is not None and d < t else ""
            lines.append(f"{s}: {d}/{t} ({pct:.0f} %), {e / 60:.1f} min{left}")
        a = self.ax[5]
        if self._text is not None:
            self._text.remove()
        self._text = a.text(0.0, 1.0, "\n".join(lines[-22:]), va="top", family="monospace", fontsize=8, transform=a.transAxes)
        self.refresh()

    def refresh(self, force: bool = False) -> None:
        """Redraw the window or rewrite the PNG, at most once per ``min_interval``.

        Parameters
        ----------
        force : bool, default False
            Redraw regardless of the interval.

        Returns
        -------
        None
            The window is updated, or the PNG is written.
        """
        now = time.time()
        if not force and now - self.last < self.min_interval:
            return
        self.last = now
        if self.backend == "Agg":
            self.fig.savefig(self.png, dpi=80)
        else:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            self.plt.pause(0.001)

    def close(self) -> None:
        """Write the final view and leave the window open until the user closes it.

        Returns
        -------
        None
            The PNG is written in every case.
        """
        self.refresh(force=True)
        self.fig.savefig(self.png, dpi=80)
