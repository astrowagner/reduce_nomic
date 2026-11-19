"""Live display of a reduction, in the spirit of the IDL predecessor's ``cgimage`` windows.

With ``--display`` the driver shows, as the stages run:

* top row, the chop identification of the current round: the image the latest frame's chop
  state was decided from (the crude star image), with its detected star; the detected star position of every frame so far, colored by
  the chop state the pipeline gives it; the quantity that decides the chop state over the most
  recent frames, which alternates around its threshold while the chop is being identified (the
  star's x or y, or the change of the stripe mean for sky rounds and ``set_chop_via_bg``);
  and, once the round is segmented, the histogram of frames per segment;
* bottom row, chop subtraction and frame selection: the latest chop-subtracted frame and its
  35-px high-pass, each chunk's centered median, and its frame-selection measures with their
  thresholds;
* a strip with the progress of every stage, the latest one also in the title.

The window has the size of the KLIP-TPE live window, ``PANEL_PX`` = 1850 x 990 pixels at
``DPI`` = 100, which fits a 1920 x 1080 screen with room for a title bar and a desktop top bar;
on a smaller screen it is shrunk to fit. Its resizing is off. It opens once per run and is then
redrawn in place, at most once per ``min_interval``, without being raised or given focus.
Without an X display or Tk, the same view is written to a PNG file instead, at most every
``PNG_INTERVAL`` seconds and at the end. Window and PNG use the KLIP-TPE live panel's dark theme
(``DARK_RC``), with colorblind-safe colors that read on black.

The view (``LiveView``) runs in its own process, which ``Display``, the object the driver
uses, starts and feeds: the window answers the desktop at all times, and a display that fails
or loses its X connection cannot stop the reduction.
"""
from __future__ import annotations

import multiprocessing
import multiprocessing.queues
import os
import time
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

#: window size in pixels at ``DPI``: the KLIP-TPE live window, which fits a 1920 x 1080 screen
PANEL_PX = (1850, 990)
#: figure resolution, in dots per inch
DPI = 100
#: pixels kept free when the window is fitted to the screen (KLIP-TPE's values): a title bar,
#: the window border on each side, and a desktop top bar that Tk cannot see
TITLE_PX, BORDER_PX, TOPBAR_PX = 48, 8, 32
#: minimum time between two PNG writes when no window can open, in seconds
PNG_INTERVAL = 10.0
#: chop-identification frames shown in the chop-decision panel
RECENT_FRAMES = 600
#: most frames drawn in the star-position scatter (every k-th frame beyond this)
MAX_SCATTER = 20000
#: dark theme of the view, the KLIP-TPE live panel's (its ``_RC_DARK`` colors and IDL-window
#: look: black background, white text and axes, boxed axes with inward ticks, no legend frames)
DARK_RC = {"figure.facecolor": "black", "axes.facecolor": "black", "savefig.facecolor": "black",
           "text.color": "white", "axes.labelcolor": "white", "axes.edgecolor": "white",
           "xtick.color": "white", "ytick.color": "white", "grid.color": "#888888",
           "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
           "legend.frameon": False}
#: colors on the dark background (Okabe-Ito, colorblind-safe): chop A, chop B, a third series
CHOP_COLORS = {1: "#56B4E9", 2: "#E69F00"}
THIRD_COLOR = "#009E73"
#: reference lines (thresholds, splits), connecting lines, gray text, rejected frames and the
#: detected-star circle
LINE_COLOR, DIM_COLOR, NOTE_COLOR, MARK_COLOR = "#cccccc", "#5a5a5a", "#888888", "#FF4D4D"
#: masked (NaN) pixels in images: a dark blue that stands apart from the black of low values
#: without dominating the dark view
MASK_COLOR = "#26345c"


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


def fit_scale(screen_w: int, screen_h: int, ratio: float = 1.0) -> float:
    """Return the scale at which the ``PANEL_PX`` window fits a screen.

    Parameters
    ----------
    screen_w, screen_h : int
        Screen width and height in screen pixels.
    ratio : float, default 1.0
        Screen pixels per figure pixel: matplotlib's device pixel ratio, which for Tk on Linux
        is the screen's dots per inch over 96.

    Returns
    -------
    scale : float
        The largest scale, at most 1 and at least 0.1, at which the window, ``PANEL_PX`` x
        ``ratio`` screen pixels at scale 1, fits the screen with its border (``BORDER_PX`` each
        side), a title bar (``TITLE_PX``) and a desktop top bar (``TOPBAR_PX``), these margins
        times ``ratio`` when it exceeds 1. It is 1 on a 1920 x 1080 screen at ratio 1.
    """
    m = max(1.0, float(ratio))
    room_w = screen_w - 2 * BORDER_PX * m
    room_h = screen_h - (TOPBAR_PX + TITLE_PX) * m
    return max(0.1, min(1.0, room_w / (PANEL_PX[0] * ratio), room_h / (PANEL_PX[1] * ratio)))


def _pixel_ratio(fig) -> float:
    """Return the screen pixels per figure pixel that the Tk canvas will use once mapped.

    matplotlib's Tk canvas scales itself to the screen when its window is first mapped (on
    Linux by the screen's dots per inch over 96, on Windows by Tk's scaling), so a window sized
    before it is shown must allow for that.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure of a TkAgg window, not yet shown.

    Returns
    -------
    ratio : float
        The ratio matplotlib will apply, or 1 when its Tk canvas does not scale itself.
    """
    import sys
    if not hasattr(fig.canvas, "_update_device_pixel_ratio"):
        return 1.0
    w = fig.canvas.get_tk_widget()
    try:
        if sys.platform == "linux":
            return float(w.winfo_fpixels("1i")) / 96.0
        if sys.platform == "win32":
            return round(float(w.tk.call("tk", "scaling")) / (96 / 72), 2)
    except Exception:
        pass
    return 1.0


def _open_window(title: str) -> Tuple[object, str]:
    """Open the figure window hidden, fit it to the screen, switch its resizing off, then show it.

    A window shown first and fitted afterwards can open at the wrong size: GNOME maximizes a
    resizable window that covers most of the screen the moment it appears. Once shown, the
    window's real size is checked, and it is shrunk again if the canvas scaled itself more than
    expected.

    Parameters
    ----------
    title : str
        Window title.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure in its window: ``PANEL_PX`` at ``DPI``, times the scale from ``fit_scale``
        when the screen is smaller.
    note : str
        Line for the log: the backend, the window size in screen pixels, its scale when shrunk,
        and the pixel ratio when it is not 1.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams["toolbar"] = "None"      # no navigation bar: the window is the canvas
    was = plt.isinteractive()
    plt.ioff()                       # in interactive mode pyplot would show the window at once
    try:
        fig = plt.figure(figsize=(PANEL_PX[0] / DPI, PANEL_PX[1] / DPI), dpi=DPI)
    finally:
        if was:
            plt.ion()
    mgr = fig.canvas.manager
    mgr.set_window_title(title)
    win = getattr(mgr, "window", None)
    if win is None or not hasattr(win, "winfo_screenwidth"):
        mgr.show()
        return fig, f"window ({matplotlib.get_backend()}), {PANEL_PX[0]} x {PANEL_PX[1]} px"
    sw, sh = int(win.winfo_screenwidth()), int(win.winfo_screenheight())
    ratio = _pixel_ratio(fig)
    scale = fit_scale(sw, sh, ratio)
    if scale < 1.0:
        fig.set_size_inches(PANEL_PX[0] * scale / DPI, PANEL_PX[1] * scale / DPI, forward=True)
    win.resizable(False, False)
    mgr.show()
    canvas = fig.canvas.get_tk_widget()
    t0 = time.time()
    while not canvas.winfo_ismapped() and time.time() - t0 < 2.0:
        win.update()
        time.sleep(0.02)
    win.update()
    real = canvas.winfo_width() / (PANEL_PX[0] * scale)
    if canvas.winfo_ismapped() and real > ratio * 1.01:
        again = fit_scale(sw, sh, real)
        if again < scale - 1e-3:
            scale, ratio = again, real
            fig.set_size_inches(PANEL_PX[0] * scale / DPI, PANEL_PX[1] * scale / DPI, forward=True)
            win.update()
    note = f"window ({matplotlib.get_backend()}), {canvas.winfo_width()} x {canvas.winfo_height()} px, fixed size"
    if scale < 1.0:
        note += f", the {PANEL_PX[0]} x {PANEL_PX[1]} view at {scale:.2f}, the largest that fits the {sw} x {sh} screen"
    if abs(ratio - 1.0) > 0.005:
        note += f" (screen pixels per figure pixel {ratio:.2f})"
    return fig, note


def _is_open(fig) -> bool:
    """Tell whether a pyplot figure still exists (its window was not closed).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure made through pyplot.

    Returns
    -------
    open : bool
        True while pyplot still manages the figure.
    """
    import matplotlib.pyplot as plt
    return plt.fignum_exists(fig.number)


def _cmap(name: str):
    """Return a copy of a matplotlib colormap that draws NaN pixels in ``MASK_COLOR``.

    Parameters
    ----------
    name : str
        Colormap name.

    Returns
    -------
    cmap : matplotlib.colors.Colormap
        The colormap, NaN shown in ``MASK_COLOR`` so masked pixels stand out.
    """
    import matplotlib
    return matplotlib.colormaps[name].with_extremes(bad=MASK_COLOR)


def _opaque_legend(leg) -> None:
    """Draw a legend's markers fully opaque, whatever the alpha of the plotted points.

    Parameters
    ----------
    leg : matplotlib.legend.Legend or None
        Legend to change.

    Returns
    -------
    None
        The legend's handles get alpha 1.
    """
    if leg is None:
        return
    for h in getattr(leg, "legend_handles", None) or getattr(leg, "legendHandles", []):
        h.set_alpha(1.0)


def _stretch(img: np.ndarray, lo: float = 1.0, hi: float = 99.5) -> Tuple[float, float]:
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


class LiveView:
    """The live view itself: one figure with the panels, in a window or written to a PNG file.

    It runs in the display process (``Display`` starts it there); it can also be used directly.

    Parameters
    ----------
    title : str
        Title, e.g. ``Procyon_13 left``.
    png : str
        File the view is written to at ``close``, and when no window can open also at most
        every ``PNG_INTERVAL`` seconds.
    min_interval : float, default 1.0
        Minimum time between two window redraws, in seconds.
    log : callable, optional
        Called with one line of text for the run's log (the window closed, no window could
        open, a PNG not written); default ``print``.
    backend : str, optional
        ``TkAgg`` or ``Agg``; default from ``_backend``. When the window cannot open (no X
        server behind ``DISPLAY``), the view falls back to ``Agg``.

    Attributes
    ----------
    backend : str
        ``TkAgg`` (window) or ``Agg`` (PNG only).
    note : str
        Line for the log: the window and its size, or the PNG file.
    closed : bool
        True once the user closed the window; the view is still written to the PNG at ``close``.

    Notes
    -----
    Data passed to the ``show_*`` and ``add_states`` methods is kept, and the panels are drawn
    from it only at a redraw, so that frequent updates cost little. ``reset`` starts the panels
    afresh in the same window, for the next aperture.
    """

    def __init__(self, title: str, png: str, min_interval: float = 1.0,
                 log: Optional[Callable[[str], None]] = None, backend: Optional[str] = None):
        import matplotlib
        self.backend = backend or _backend()
        matplotlib.use(self.backend, force=True)
        matplotlib.rcParams.update({"figure.raise_window": False, "toolbar": "None", "font.size": 9,
                                    "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8,
                                    "ytick.labelsize": 8, "legend.fontsize": 8, **DARK_RC})
        import matplotlib.pyplot as plt
        self.plt = plt
        self.png = png
        self.title = title
        self._log = log or print
        self.closed = False
        self.last = 0.0
        self.fig = None
        if self.backend != "Agg":
            try:
                self.fig, self.note = _open_window(f"reduce-nomic live -- {title}")
            except Exception as exc:          # e.g. DISPLAY set but no X server behind it
                self._log(f"display: no window ({type(exc).__name__}: {exc}); writing {png} instead")
                self.backend = "Agg"
                matplotlib.use("Agg", force=True)
        if self.fig is None:
            self.fig = plt.figure(figsize=(PANEL_PX[0] / DPI, PANEL_PX[1] / DPI), dpi=DPI)
            self.note = f"PNG {png} ({PANEL_PX[0]} x {PANEL_PX[1]} px), rewritten at most every {PNG_INTERVAL:.0f} s"
        self.interval = float(min_interval) if self.backend != "Agg" else max(float(min_interval), PNG_INTERVAL)
        self._draw = {"raw": self._draw_raw, "states": self._draw_states, "seg": self._draw_seg,
                      "sub": self._draw_sub, "chunk": self._draw_chunk, "prog": self._draw_prog}
        self._build()
        self.refresh(force=True)

    def reset(self, title: str, png: str) -> None:
        """Start the panels afresh in the same window, for another night or aperture.

        Parameters
        ----------
        title : str
            New title.
        png : str
            New PNG file.

        Returns
        -------
        None
            The window keeps its size and place; its title and panels are new.
        """
        self.title, self.png = title, png
        if self._window_open():
            self.fig.canvas.manager.set_window_title(f"reduce-nomic live -- {title}")
        self._build()
        self.refresh(force=True)

    # -- layout and state ------------------------------------------------------------------
    def _build(self) -> None:
        """Lay out the panels on the (new or reused) figure and clear the kept data.

        Returns
        -------
        None
            ``self.ax`` holds the axes by name: ``raw``, ``pos``, ``dec``, ``seg`` (top row),
            ``sub``, ``hp``, ``med``, ``sel`` (bottom row) and ``prog`` (progress strip).
        """
        fig = self.fig
        fig.clear()
        try:
            fig.set_layout_engine("constrained")
        except AttributeError:
            fig.set_constrained_layout(True)
        gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 0.2])
        names = ("raw", "pos", "dec", "seg", "sub", "hp", "med", "sel")
        self.ax = {name: fig.add_subplot(gs[i // 4, i % 4]) for i, name in enumerate(names)}
        self.ax["prog"] = fig.add_subplot(gs[2, :])
        self._suptitle = fig.suptitle(self.title, fontsize=11, fontweight="bold")
        self._round = {"tag": "", "value_name": "", "threshold": 256.0, "direction": "LEFT-RIGHT",
                       "via_bg": False, "desc": ""}
        self._frames = {"num": [], "x": [], "y": [], "code": [], "value": []}
        self._skipped = 0
        self._raw = None
        self._seg = None
        self._sub = None
        self._chunk = None
        self.progress: Dict[str, Tuple[int, int, float, Optional[float]]] = {}
        self._dirty = set(self._draw)

    def _mark(self, key: str) -> None:
        """Mark a group of panels for redrawing and redraw if the interval has passed.

        Parameters
        ----------
        key : str
            Panel group: ``raw``, ``states``, ``seg``, ``sub``, ``chunk`` or ``prog``.

        Returns
        -------
        None
            The group is redrawn at the next redraw.
        """
        self._dirty.add(key)
        self.refresh()

    @staticmethod
    def _empty(ax, title: str, text: str) -> None:
        """Show a panel that has no data yet.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Panel.
        title : str
            Panel title.
        text : str
            Gray text in the middle of the panel.

        Returns
        -------
        None
            The panel shows its title and the text, without ticks.
        """
        ax.set_title(title)
        ax.text(0.5, 0.5, text, ha="center", va="center", color=NOTE_COLOR, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])

    # -- data from the driver --------------------------------------------------------------
    def start_round(self, tag: str, value_name: str, threshold: float, direction: str,
                    via_bg: bool = False, desc: str = "") -> None:
        """Start the chop-identification panels afresh for a new round.

        Parameters
        ----------
        tag : str
            Round label, e.g. ``left_round0``.
        value_name : str
            Axis title of the quantity that decides the chop state, with units.
        threshold : float
            Value at which the decision flips: 256 pixels for the star position, 0 for the
            stripe-mean change.
        direction : str
            Chop direction, ``LEFT-RIGHT`` (the star-position split is at x = 256) or another
            (split at y = 256).
        via_bg : bool, default False
            True when the stripe-mean change decides (``set_chop_via_bg``, and every sky round).
        desc : str, default ""
            How the chop-ID image was made, written in a corner of its panel.

        Returns
        -------
        None
            The top-row panels are cleared and redrawn at the next redraw.
        """
        self._round = {"tag": tag, "value_name": value_name, "threshold": float(threshold),
                       "direction": direction, "via_bg": bool(via_bg), "desc": str(desc)}
        self._frames = {"num": [], "x": [], "y": [], "code": [], "value": []}
        self._skipped = 0
        self._raw = None
        self._seg = None
        self._dirty.update(("raw", "states", "seg"))
        self.refresh(force=True)

    def add_states(self, nums: Sequence[int], xs: Sequence[int], ys: Sequence[int], chops: Sequence[str],
                   values: Sequence[float], skipped: int = 0) -> None:
        """Add a block of identified frames to the chop-identification panels.

        Parameters
        ----------
        nums : sequence of int
            Frame numbers of the block's frames under the background limit, in order.
        xs, ys : sequence of int
            Their detected star positions in detector pixels.
        chops : sequence of str
            Their chop states, ``CHOP_A`` or ``CHOP_B``, as the pipeline assigns them.
        values : sequence of float
            The quantity that decided each chop state (see ``start_round``).
        skipped : int, default 0
            Frames of the block over the background limit, which get no chop state.

        Returns
        -------
        None
            The star-position and chop-decision panels are redrawn at the next redraw.
        """
        f = self._frames
        f["num"].append(np.asarray(nums, dtype=np.int64))
        f["x"].append(np.asarray(xs, dtype=np.float32))
        f["y"].append(np.asarray(ys, dtype=np.float32))
        f["code"].append(np.asarray([1 if c == "CHOP_A" else 2 for c in chops], dtype=np.int8))
        f["value"].append(np.asarray(values, dtype=np.float64))
        self._skipped += int(skipped)
        self._mark("states")

    def show_chop_image(self, img: np.ndarray, star: Sequence[float], chop: str, label: str,
                        via_bg: bool = False) -> None:
        """Show the image a frame's chop state was decided from, with its detected star.

        Parameters
        ----------
        img : ndarray, shape (ny, nx)
            The crude star image of ``combine.crude_star_image`` (zeros where masked), or with
            ``via_bg`` the rolling-background-subtracted frame; (ny, nx) = (512, 512).
        star : sequence of float
            ``(x, y)`` of the detected star in pixels.
        chop : str
            The frame's chop state, ``CHOP_A`` or ``CHOP_B``.
        label : str
            Frame number, shown in the panel title.
        via_bg : bool, default False
            True when the stripe-mean change decided the chop state; the stripe region (rows
            126-130, columns 50-100) is outlined.

        Returns
        -------
        None
            The panel is redrawn at the next redraw.
        """
        self._raw = (np.asarray(img), tuple(star), str(chop), str(label), bool(via_bg))
        self._mark("raw")

    def show_segments(self, lengths: Sequence[int]) -> None:
        """Show the histogram of frames per segment of the current round.

        Parameters
        ----------
        lengths : sequence of int
            Number of frames in each segment.

        Returns
        -------
        None
            The panel is redrawn at once.
        """
        self._seg = np.asarray(lengths, dtype=int)
        self._dirty.add("seg")
        self.refresh(force=True)

    def show_chopsub(self, img: np.ndarray, hp: np.ndarray, label: str) -> None:
        """Show a chop-subtracted frame and its high-pass.

        Parameters
        ----------
        img : ndarray, shape (ny, nx)
            Chop-subtracted, corrected frame, where (ny, nx) = (512, 512).
        hp : ndarray, shape (ny, nx)
            The same minus its 35-px boxcar.
        label : str
            Frame label shown in the panel titles.

        Returns
        -------
        None
            The panels are redrawn at the next redraw.
        """
        self._sub = (np.asarray(img), np.asarray(hp), str(label))
        self._mark("sub")

    def show_chunk(self, med: np.ndarray, res: Dict, thresholds: Dict) -> None:
        """Show one chunk's centered median and frame-selection measures.

        Parameters
        ----------
        med : ndarray, shape (200, 200)
            Median of the chunk's centered frames, central 200 x 200 pixels.
        res : dict
            The chunk result: ``chunk``, ``cn``, ``corrs``, ``noises``, ``peaks``, ``bad``.
        thresholds : dict
            ``corr``, ``noise`` and ``peak`` thresholds of the pre-clean for this chop.

        Returns
        -------
        None
            The panels are redrawn at the next redraw.
        """
        self._chunk = (np.asarray(med), res, thresholds)
        self._mark("chunk")

    def show_progress(self, stage: str, done: int, total: int, elapsed: float, eta: Optional[float]) -> None:
        """Update the progress strip.

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
            The strip and the title are redrawn at the next redraw.
        """
        self.progress[stage] = (int(done), int(total), float(elapsed), eta)
        self._mark("prog")

    # -- drawing ---------------------------------------------------------------------------
    def _arrays(self) -> Tuple[np.ndarray, ...]:
        """Return the identified frames of the round so far, sorted by frame number.

        Returns
        -------
        num, x, y, code, value : ndarray, shape (m,)
            Frame number, star x and y in pixels, chop code (1 = A, 2 = B) and decision value of
            the m frames under the background limit so far.
        """
        f = self._frames
        if not f["num"]:
            e = np.zeros(0)
            return e.astype(np.int64), e, e, e.astype(np.int8), e
        out = [np.concatenate(f[k]) for k in ("num", "x", "y", "code", "value")]
        order = np.argsort(out[0], kind="stable")
        return tuple(a[order] for a in out)

    def _draw_raw(self) -> None:
        """Draw the chop-ID image panel.

        Returns
        -------
        None
            The panel is drawn from the last ``show_chop_image`` data: masked pixels in ``MASK_COLOR``, a
            square-root stretch from the 1st percentile to the star's peak, and the detected
            star circled; with ``via_bg`` a percentile stretch and the stripe region outlined.
        """
        from matplotlib.colors import PowerNorm
        from matplotlib.patches import Rectangle
        a = self.ax["raw"]
        a.clear()
        if self._raw is None:
            self._empty(a, "Chop-ID Image", "Waiting for the first frames")
            return
        img, star, chop, label, via_bg = self._raw
        if via_bg:
            vmin, vmax = _stretch(img)
            a.imshow(img, origin="lower", cmap=_cmap("gray"), vmin=vmin, vmax=vmax, interpolation="nearest")
            a.add_patch(Rectangle((49.5, 125.5), 51, 5, fill=False, ec="cyan", lw=1.0))
            a.set_title(f"Chop-ID Frame {label}, Chop {chop[-1]} (Stripe Mean)")
        else:
            # masked pixels are zero, or within rounding of zero after the boxcar's running sums
            scale = float(np.nanmax(np.abs(img))) if np.any(np.isfinite(img)) else 0.0
            shown = np.where(np.abs(img) <= 1e-9 * scale, np.nan, img)
            v = shown[np.isfinite(shown)]
            if v.size:
                vmin = float(np.percentile(v, 1))
                vmax = float(np.nanmax(shown))
                vmax = vmax if vmax > vmin else vmin + 1.0
                a.imshow(shown, origin="lower", cmap=_cmap("gray"), interpolation="nearest",
                         norm=PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax))
            a.plot([star[0]], [star[1]], "o", mfc="none", mec=MARK_COLOR, ms=14, mew=1.5)
            a.set_title(f"Chop-ID Image {label}, Chop {chop[-1]}")
        if self._round.get("desc"):
            a.text(0.01, 0.01, self._round["desc"], transform=a.transAxes, fontsize=7, color="white",
                   va="bottom", ha="left", bbox={"facecolor": "black", "alpha": 0.5, "pad": 1.5, "lw": 0})
        a.set_xlabel("X (Pixels)")
        a.set_ylabel("Y (Pixels)")

    def _draw_states(self) -> None:
        """Draw the star-position and chop-decision panels.

        Returns
        -------
        None
            The panels are drawn from the frames added since ``start_round``.
        """
        num, x, y, code, value = self._arrays()
        r = self._round
        a = self.ax["pos"]
        a.clear()
        b = self.ax["dec"]
        b.clear()
        if num.size == 0:
            msg = "Waiting for the chop identification"
            if self._skipped:
                msg = f"{self._skipped:,} frames so far, all over the background limit"
            self._empty(a, "Detected Star Positions", msg)
            self._empty(b, "Chop Decision", msg)
            return
        thin = max(1, num.size // MAX_SCATTER)
        for c, col in CHOP_COLORS.items():
            m = code == c
            a.plot(x[m][::thin], y[m][::thin], ".", ms=2, alpha=0.4, color=col,
                   label=f"Chop {'AB'[c - 1]}: {int(m.sum()):,}")
        a.plot([], [], " ", label=f"Over Bkg. Limit: {self._skipped:,}")
        if not r["via_bg"]:
            if r["direction"] == "LEFT-RIGHT":
                a.axvline(256, color=LINE_COLOR, lw=0.8, ls="--")
            else:
                a.axhline(256, color=LINE_COLOR, lw=0.8, ls="--")
        a.set_xlim(0, 512)
        a.set_ylim(0, 512)
        a.set_aspect("equal")
        a.set_xlabel("X (Pixels)")
        a.set_ylabel("Y (Pixels)")
        _opaque_legend(a.legend(loc="upper right", markerscale=5))
        a.set_title(f"Detected Star Positions, {num.size:,} Frames")
        k = slice(max(0, num.size - RECENT_FRAMES), num.size)
        n_, v_, c_ = num[k], value[k], code[k]
        b.plot(n_, v_, "-", color=DIM_COLOR, lw=0.6, zorder=1)
        for c, col in CHOP_COLORS.items():
            m = c_ == c
            b.plot(n_[m], v_[m], ".", ms=3.5, color=col, label=f"Chop {'AB'[c - 1]}", zorder=2)
        b.axhline(r["threshold"], color=LINE_COLOR, lw=0.8, ls="--")
        flips = int(np.count_nonzero(np.diff(c_))) if c_.size > 1 else 0
        b.set_title(f"Chop Decision, Last {n_.size:,} Frames: {flips:,} Flips")
        b.set_xlabel("Frame Number")
        b.set_ylabel(r["value_name"])
        b.ticklabel_format(axis="x", style="plain", useOffset=False)
        _opaque_legend(b.legend(loc="center right", markerscale=2))

    def _draw_seg(self) -> None:
        """Draw the frames-per-segment panel.

        Returns
        -------
        None
            The panel is drawn from the last ``show_segments`` data.
        """
        a = self.ax["seg"]
        a.clear()
        if self._seg is None:
            self._empty(a, "Frames per Segment", "Shown once the round is segmented")
            return
        ln = self._seg
        if ln.size:
            hi = int(ln.max())
            a.hist(ln, bins=np.arange(0.5, hi + 1.5, 1.0) if hi <= 200 else 60, color=THIRD_COLOR)
        med = float(np.median(ln)) if ln.size else 0.0
        a.set_xlabel("Frames per Segment")
        a.set_ylabel("Segments")
        a.set_title(f"Frames per Segment: {ln.size:,} Segments, Median {med:.0f}")

    def _draw_sub(self) -> None:
        """Draw the chop-subtracted frame and its high-pass.

        Returns
        -------
        None
            The panels are drawn from the last ``show_chopsub`` data.
        """
        if self._sub is None:
            for name, t in (("sub", "Chop-Subtracted Frame"), ("hp", "35-px High-Pass")):
                self.ax[name].clear()
                self._empty(self.ax[name], t, "Shown during the chop subtraction")
            return
        img, hp, label = self._sub
        for name, im, t in (("sub", img, "Chop-Subtracted"), ("hp", hp, "35-px High-Pass")):
            a = self.ax[name]
            a.clear()
            vmin, vmax = _stretch(im)
            a.imshow(im, origin="lower", cmap=_cmap("gray"), vmin=vmin, vmax=vmax, interpolation="nearest")
            a.set_title(f"{t}, {label.capitalize()}")
            a.set_xlabel("X (Pixels)")
            a.set_ylabel("Y (Pixels)")

    def _draw_chunk(self) -> None:
        """Draw the centered-median and frame-selection panels.

        Returns
        -------
        None
            The panels are drawn from the last ``show_chunk`` data.
        """
        a = self.ax["med"]
        b = self.ax["sel"]
        a.clear()
        b.clear()
        if self._chunk is None:
            self._empty(a, "Centered Median", "Shown during the frame selection")
            self._empty(b, "Frame Selection", "Shown during the frame selection")
            return
        med, res, thr = self._chunk
        vmin, vmax = _stretch(med, 0.5, 99.9)
        a.imshow(med, origin="lower", cmap=_cmap("inferno"), vmin=vmin, vmax=vmax, interpolation="nearest")
        a.axhline(99.5, color="cyan", lw=0.5)
        a.axvline(99.5, color="cyan", lw=0.5)
        a.set_title(f"Centered Median: Chunk {res['chunk']}, Chop {'AB'[int(res['cn']) - 1]}")
        a.set_xlabel("X (Pixels)")
        a.set_ylabel("Y (Pixels)")
        c = np.asarray(res["corrs"], dtype=float)
        n = np.asarray(res["noises"], dtype=float)
        p = np.asarray(res["peaks"], dtype=float)
        x = np.arange(c.size)
        if c.size:
            with np.errstate(invalid="ignore", divide="ignore"):
                b.plot(x, c, ".", ms=3, color=CHOP_COLORS[1], label="Correlation")
                b.plot(x, n / np.nanmean(n), ".", ms=3, color=CHOP_COLORS[2], label="Noise / Mean")
                b.plot(x, p / np.nanmean(p), ".", ms=3, color=THIRD_COLOR, label="Peak / Mean")
            b.axhline(thr["corr"], color=CHOP_COLORS[1], ls="--", lw=0.8)
            b.axhline(thr["noise"], color=CHOP_COLORS[2], ls="--", lw=0.8)
            b.axhline(thr["peak"], color=THIRD_COLOR, ls="--", lw=0.8)
            bad = np.asarray(res["bad"], dtype=int)
            bad = bad[bad < c.size]
            if bad.size:
                b.plot(bad, c[bad], "x", color=MARK_COLOR, ms=5, label="Rejected")
        b.set_xlabel("Frame in Chunk")
        b.set_ylabel("Value")
        b.set_title(f"Frame Selection: {len(res['bad']):,} of {c.size:,} Rejected")
        _opaque_legend(b.legend(loc="best"))

    @staticmethod
    def _progress_line(stage: str, item: Tuple[int, int, float, Optional[float]]) -> str:
        """Format one stage's progress.

        Parameters
        ----------
        stage : str
            Stage label.
        item : tuple
            ``(done, total, elapsed, eta)`` as given to ``show_progress``.

        Returns
        -------
        line : str
            Stage, tasks done of all, percent, elapsed minutes and, while running, minutes left.
        """
        d, t, e, r = item
        pct = 100.0 * d / t if t else 100.0
        left = f", {r / 60:.1f} min left" if r is not None and d < t else ""
        return f"{stage}: {d:,}/{t:,} ({pct:.0f} %), {e / 60:.1f} min{left}"

    def _draw_prog(self) -> None:
        """Draw the progress strip and the title.

        Returns
        -------
        None
            Up to 24 stages, the most recent last, in four columns; the latest stage also in
            the figure title.
        """
        a = self.ax["prog"]
        a.clear()
        a.axis("off")
        items = list(self.progress.items())[-24:]
        for j in range(4):
            col = items[6 * j:6 * j + 6]
            if col:
                a.text(j / 4.0, 1.0, "\n".join(self._progress_line(s, v) for s, v in col), va="top",
                       ha="left", family="monospace", fontsize=8, transform=a.transAxes)
        head = self.title
        if items:
            head += "  |  " + self._progress_line(*items[-1])
        self._suptitle.set_text(head)

    # -- window ----------------------------------------------------------------------------
    def _window_open(self) -> bool:
        """Tell whether the window can still be drawn, noting once when the user closed it.

        Returns
        -------
        open : bool
            False for the PNG-only backend and once the window is closed.
        """
        if self.backend == "Agg" or self.closed:
            return False
        if not _is_open(self.fig):
            self.closed = True
            self._log(f"display: window closed; the run goes on and the view is written to {self.png} at the end")
            return False
        return True

    def _save(self) -> None:
        """Write the view to the PNG file.

        Returns
        -------
        None
            ``png`` is written at ``DPI`` (``PANEL_PX`` pixels at full scale); a failure is logged.
        """
        try:
            self.fig.savefig(self.png, dpi=DPI)
        except Exception as exc:          # a display problem must not stop the reduction
            self._log(f"display: could not write {self.png}: {exc}")

    def refresh(self, force: bool = False) -> None:
        """Redraw the panels that changed, at most once per interval.

        The window is redrawn in place, without being raised or given focus (no ``plt.pause``,
        and ``figure.raise_window`` off), so it does not take over the desktop at each update.

        Parameters
        ----------
        force : bool, default False
            Redraw regardless of the interval.

        Returns
        -------
        None
            The window is updated, or (PNG-only backend) the PNG is written.
        """
        now = time.time()
        if not force and now - self.last < self.interval:
            return
        self.last = now
        for key in [k for k in self._draw if k in self._dirty]:
            self._draw[key]()
        self._dirty.clear()
        if self.backend == "Agg":
            self._save()
        elif self._window_open():
            try:
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()
            except Exception as exc:      # e.g. the window was destroyed mid-draw
                self.closed = True
                self._log(f"display: window lost ({exc}); the run goes on")

    def pump(self) -> None:
        """Process the window's events, redrawing first if changes wait and the interval passed.

        Called while the driver waits for its workers, so the window answers the desktop and can
        be moved even between updates.

        Returns
        -------
        None
            Pending window events are handled.
        """
        if self._dirty and time.time() - self.last >= self.interval:
            self.refresh()
            return
        if self._window_open():
            try:
                self.fig.canvas.flush_events()
            except Exception as exc:
                self.closed = True
                self._log(f"display: window lost ({exc}); the run goes on")

    def close(self) -> None:
        """Draw the final view and write it to the PNG file, leaving the window open.

        Returns
        -------
        None
            The PNG is written in every case; the window stays until the run ends or the user
            closes it.
        """
        self.refresh(force=True)
        if self.backend != "Agg":
            self._save()


# -- the display process ---------------------------------------------------------------------
class _Queue(multiprocessing.queues.Queue):
    """A multiprocessing queue whose feeder thread stays quiet once the display process is gone.

    Parameters
    ----------
    ctx : multiprocessing context
        The ``spawn`` context the display process uses.
    """

    @staticmethod
    def _on_queue_feeder_error(e, obj) -> None:
        """Ignore a message that could not be sent (the display process ended).

        Parameters
        ----------
        e : Exception
            The error, usually a broken pipe.
        obj : object
            The message.

        Returns
        -------
        None
            Nothing is printed; ``Display`` notices the ended process and stops sending.
        """
        return None


def _serve(q, ack, title: str, png: str, min_interval: float, backend: str) -> None:
    """Run the live view in the display process until told to stop.

    Parameters
    ----------
    q : _Queue
        Messages from the driver, ``(method name, arguments)`` for ``LiveView``, or
        ``("save", (png,))`` to draw the final view and write it, or ``("stop", ())``.
    ack : _Queue
        Messages to the driver: ``("ready", backend, note)`` once the view exists,
        ``("log", line)`` for the run's log, and ``("saved", png)`` after each save.
    title, png, min_interval, backend
        As for ``LiveView``.

    Returns
    -------
    None
        Between messages, and at most every 0.05 s while they stream in, the window's events
        are processed, so that it answers the desktop at all times. Ctrl-C in the terminal is
        left to the driver, and the process ends when the driver is gone.
    """
    import queue as std_queue
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    ppid = os.getppid()
    view = LiveView(title, png, min_interval, log=lambda line: ack.put(("log", line)), backend=backend)
    ack.put(("ready", view.backend, view.note))
    last_pump = time.time()
    while True:
        try:
            name, args = q.get(timeout=0.05)
        except std_queue.Empty:
            if os.getppid() != ppid:          # the driver ended without stopping us
                break
            view.pump()
            last_pump = time.time()
            continue
        if name == "stop":
            break
        if name == "save":
            view.close()
            ack.put(("saved", args[0]))
        else:
            getattr(view, name)(*args)
        if time.time() - last_pump > 0.05:
            view.pump()
            last_pump = time.time()


def _start_without_main(proc) -> None:
    """Start a ``spawn`` process without importing the driver's main script in it.

    The spawn method imports the parent's ``__main__`` script in the new process, which runs
    again any code the script has outside an ``if __name__ == "__main__":`` guard. The display
    process needs nothing from it, so ``__main__`` is replaced by an empty module while the
    process starts.

    Parameters
    ----------
    proc : multiprocessing.context.SpawnProcess
        Process to start.

    Returns
    -------
    None
        The process is started; ``sys.modules["__main__"]`` is restored.
    """
    import sys
    import types
    main = sys.modules.get("__main__")
    sys.modules["__main__"] = types.ModuleType("__main__")
    try:
        proc.start()
    finally:
        if main is not None:
            sys.modules["__main__"] = main


#: the display process of this run and its queues, shared by the Display objects (one per aperture)
_SERVER: Dict[str, object] = {"proc": None, "q": None, "ack": None, "backend": "", "note": ""}


class Display:
    """Live view of one night and aperture, drawn by a separate display process.

    The driver calls the same methods as on ``LiveView``; each sends its data to the display
    process, which owns the window. The driver never touches the window, so a display that fails
    or loses its X connection (which ends a Tk program) cannot stop the reduction, and the
    window stays responsive while the driver computes.

    Parameters
    ----------
    title : str
        Title, e.g. ``Procyon_13 left``.
    png : str
        File the final view is written to at ``close`` (see ``LiveView``).
    min_interval : float, default 1.0
        Minimum time between two window redraws, in seconds.
    log : callable, optional
        Called with one line of text for the run's log; default ``print``.
    start_timeout : float, default 60.0
        Seconds to wait for the display process to open its view.

    Attributes
    ----------
    backend : str
        ``TkAgg`` (window), ``Agg`` (PNG only), or ``none`` when the display process did not
        start.
    note : str
        Line for the log: the window and its size, or the PNG file.
    dead : bool
        True once the display process has ended; the methods then do nothing.

    Notes
    -----
    The display process is started with the ``spawn`` method, so the driver's forked workers
    inherit no window state. It is shared by the Display objects of one run: the next aperture
    resets the panels in the same window. It ends with the run (it is a daemon process).
    """

    def __init__(self, title: str, png: str, min_interval: float = 1.0,
                 log: Optional[Callable[[str], None]] = None, start_timeout: float = 60.0):
        self._log = log or print
        self.png = png
        self.dead = False
        srv = _SERVER
        proc = srv["proc"]
        if proc is not None and proc.is_alive():
            srv["q"].put(("reset", (title, png)))
            self.backend, self.note = str(srv["backend"]), f"{srv['note']} (the run's window, reused)"
        else:
            ctx = multiprocessing.get_context("spawn")
            q, ack = _Queue(ctx=ctx), _Queue(ctx=ctx)
            proc = ctx.Process(target=_serve, args=(q, ack, title, png, float(min_interval), _backend()),
                               name="reduce-nomic-display", daemon=True)
            _start_without_main(proc)
            srv.update(proc=proc, q=q, ack=ack, backend="none", note="")
            ready = self._wait("ready", start_timeout)
            if ready is None:
                self.backend, self.note = "none", "none (the display process did not start)"
                self.dead = True
                if proc.is_alive():
                    proc.terminate()
            else:
                self.backend, self.note = str(ready[1]), str(ready[2])
            srv.update(backend=self.backend, note=self.note)

    def _wait(self, kind: str, timeout: float):
        """Wait for a message of one kind from the display process, logging its log lines.

        Parameters
        ----------
        kind : str
            ``ready`` or ``saved``.
        timeout : float
            Seconds to wait at most.

        Returns
        -------
        msg : tuple or None
            The message, or None when the process ended or the time ran out.
        """
        import queue as std_queue
        ack, proc = _SERVER["ack"], _SERVER["proc"]
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                msg = ack.get(timeout=0.2)
            except std_queue.Empty:
                if not proc.is_alive():
                    return None
                continue
            if msg[0] == "log":
                self._log(msg[1])
            elif msg[0] == kind:
                return msg
        return None

    def _alive(self) -> bool:
        """Tell whether the display process still runs, noting once when it has ended.

        Returns
        -------
        alive : bool
            False once the process has ended.
        """
        if self.dead:
            return False
        if not _SERVER["proc"].is_alive():
            self.dead = True
            self._log("display: the display process ended (X connection lost or killed); the run goes on")
            return False
        return True

    def _send(self, name: str, *args) -> None:
        """Send one ``LiveView`` call to the display process.

        Parameters
        ----------
        name : str
            Method name.
        *args
            Its arguments, picklable.

        Returns
        -------
        None
            Nothing is sent once the process has ended.
        """
        if self._alive():
            _SERVER["q"].put((name, args))

    def start_round(self, tag: str, value_name: str, threshold: float, direction: str,
                    via_bg: bool = False, desc: str = "") -> None:
        """Start the chop-identification panels afresh for a new round (``LiveView.start_round``).

        Parameters
        ----------
        tag, value_name, threshold, direction, via_bg, desc
            As for ``LiveView.start_round``.

        Returns
        -------
        None
            The call is sent to the display process.
        """
        self._send("start_round", tag, value_name, float(threshold), direction, bool(via_bg), str(desc))

    def add_states(self, nums: Sequence[int], xs: Sequence[int], ys: Sequence[int], chops: Sequence[str],
                   values: Sequence[float], skipped: int = 0) -> None:
        """Add a block of identified frames (``LiveView.add_states``).

        Parameters
        ----------
        nums, xs, ys, chops, values, skipped
            As for ``LiveView.add_states``.

        Returns
        -------
        None
            The block is sent to the display process as arrays.
        """
        self._send("add_states", np.asarray(nums, np.int64), np.asarray(xs, np.float32),
                   np.asarray(ys, np.float32), [str(c) for c in chops], np.asarray(values, np.float64), int(skipped))

    def show_chop_image(self, img: np.ndarray, star: Sequence[float], chop: str, label: str,
                        via_bg: bool = False) -> None:
        """Show the image a frame's chop state was decided from (``LiveView.show_chop_image``).

        Parameters
        ----------
        img, star, chop, label, via_bg
            As for ``LiveView.show_chop_image``.

        Returns
        -------
        None
            The call is sent to the display process.
        """
        self._send("show_chop_image", np.asarray(img, np.float32), tuple(float(v) for v in star), str(chop),
                   str(label), bool(via_bg))

    def show_segments(self, lengths: Sequence[int]) -> None:
        """Show the histogram of frames per segment (``LiveView.show_segments``).

        Parameters
        ----------
        lengths
            As for ``LiveView.show_segments``.

        Returns
        -------
        None
            The call is sent to the display process.
        """
        self._send("show_segments", np.asarray(lengths, int))

    def show_chopsub(self, img: np.ndarray, hp: np.ndarray, label: str) -> None:
        """Show a chop-subtracted frame and its high-pass (``LiveView.show_chopsub``).

        Parameters
        ----------
        img, hp, label
            As for ``LiveView.show_chopsub``.

        Returns
        -------
        None
            The call is sent to the display process.
        """
        self._send("show_chopsub", np.asarray(img, np.float32), np.asarray(hp, np.float32), str(label))

    def show_chunk(self, med: np.ndarray, res: Dict, thresholds: Dict) -> None:
        """Show one chunk's centered median and frame selection (``LiveView.show_chunk``).

        Parameters
        ----------
        med, res, thresholds
            As for ``LiveView.show_chunk``; only the keys the panels use are sent from ``res``
            (not its binned frames).

        Returns
        -------
        None
            The call is sent to the display process.
        """
        keep = {k: res[k] for k in ("chunk", "cn", "corrs", "noises", "peaks", "bad")}
        self._send("show_chunk", np.asarray(med, np.float32), keep, {k: float(v) for k, v in thresholds.items()})

    def show_progress(self, stage: str, done: int, total: int, elapsed: float, eta: Optional[float]) -> None:
        """Update the progress strip (``LiveView.show_progress``).

        Parameters
        ----------
        stage, done, total, elapsed, eta
            As for ``LiveView.show_progress``.

        Returns
        -------
        None
            The call is sent to the display process.
        """
        self._send("show_progress", str(stage), int(done), int(total), float(elapsed),
                   None if eta is None else float(eta))

    def pump(self) -> None:
        """Pass on the display process's log lines and notice when it has ended.

        Returns
        -------
        None
            Called while the driver waits for its workers; the window itself is served by the
            display process.
        """
        import queue as std_queue
        if not self._alive():
            return
        ack = _SERVER["ack"]
        while True:
            try:
                msg = ack.get_nowait()
            except std_queue.Empty:
                return
            if msg[0] == "log":
                self._log(msg[1])

    def close(self, timeout: float = 120.0) -> None:
        """Have the display process draw the final view and write the PNG, and wait for it.

        Parameters
        ----------
        timeout : float, default 120.0
            Seconds to wait for the PNG at most.

        Returns
        -------
        None
            ``png`` is written by the display process; the window stays open until the run
            ends. A display process that ended or did not answer is logged.
        """
        self._send("save", self.png)
        if self.dead:
            self._log(f"display: no final view written ({self.png})")
            return
        if self._wait("saved", timeout) is None:
            self._log(f"display: the final view was not confirmed within {timeout:.0f} s ({self.png})")
