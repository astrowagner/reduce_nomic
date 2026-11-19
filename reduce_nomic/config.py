"""Reduction settings: the IDL predecessor's defaults and each night's overrides.

``DEFAULTS`` holds the settings of reduce_nomic.pro (lines 17-326 of the 2026-09-14 version)
that the reduction stages use, under their IDL names and with their IDL values, so that a night's
settings can be compared line by line with its IDL object block. ``config/nights_idl.json``,
written by ``tools/extract_idl_settings.py``, holds the 78 object blocks' overrides. Settings
that exist only in the port are in ``PORT_DEFAULTS``.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

#: Settings of the IDL predecessor used by the reduction stages, with their default values.
DEFAULTS: Dict[str, Any] = {
    # combine (coadd frames within each chop)
    "initial_combination": "mean", "start_frame_left": 0, "start_frame_right": 0,
    "end_frame_left": 99e9, "end_frame_right": 99e9, "min_n": 1, "skip_first": 0, "skip_last": 0,
    "block_right": 0, "block_stripe": 1, "block_bad_channel": 1, "reset_flags": 1,
    "make_badmap": 1, "badsigthresh": 2.0, "badsz": 10, "badbuff": 5, "fresh_badmap": 0,
    "bg_sub": 1, "rolling_background": 1, "nbg": 5, "set_chop_via_bg": 0,
    "background_limit": 28000.0, "fix_pa": 0, "lat": 32.7014, "lng": -109.889, "tz": -7.0,
    "double_sided": 0, "block_extra": 0, "reorder": 0, "skip_target_check": 0, "smooth_bg": 5,
    "n_sky": 0, "limit_sky_to_one": 1, "chop_direction": "UP-DOWN", "obj": "", "ra": None, "dec": None,
    # chop subtraction and cosmetics
    "nearest_neighbor_chop_sub": 1, "pca_background": 0, "skysub": 0, "flip_sky_chop": 0,
    "checksky": 1, "priorpoly_bgsub": 0, "midpoly_bgsub": 0, "normalize_middle": 1, "xww": 8,
    "yww": 10, "postpoly_bgsub": 1, "polydegree": 2, "fix_bad_channel": 0, "polyfit_edges": 0,
    "subtract_crosstalk": 0, "flatten_channels": 0, "flatten_horizontal": 0,
    "flatten_vertical": 0, "replace_bads": 1, "remove_oddeven_noise": 1, "zero_edges": 0,
    "nan_edges": 1, "badw": 2, "nan_bad_region": 0, "nan_exterior": 1, "nan_exterior_extra": 0,
    "mask_stripes": 0, "pad_frame": 1, "framew": 1400, "frame_limit": 1000,
    # split
    "destripe_at_split": 0, "destripe_bottom_at_split": 0, "auto_positions": 1,
    "auto_positions_edgecheck": 1, "filter_auto_pos": 0, "blockrad": 30.0, "radius": 500,
    "test_pos_negative": 1, "pos1": None, "pos2": None,
    # align and center
    "align_filter": 1, "align_filter_width": 20.0, "align_smooth": 1, "align_smooth_width": 5.0,
    "align_convolve": 0, "align_median_sub": 0, "align_radius_thresh": 50, "rot_center": 0,
    "censteps": 21, "censquare": 4.0,
    # aperture
    "side": "left", "sx_top": 215, "dx_bottom": 305,
    # pre-clean
    "pre_corr_thresh_1": 0.99, "pre_corr_thresh_2": 0.99, "pre_noise_clean": 1,
    "pre_noise_thresh_1": 1.05, "pre_noise_thresh_2": 1.05, "pre_coronoise_clean": 0,
    "pre_coronoise_thresh_1": 1.25, "pre_coronoise_thresh_2": 1.25, "pre_peak_clean": 1,
    "pre_peak_thresh_1": 0.95, "pre_peak_thresh_2": 0.95, "pre_peak_scale": 1,
    "pre_clean_angs": 0, "badang": -22.3, "badangrange": 1.0, "badangtwo": 68.8,
    # pre-binning and replacement
    "prebin": 100, "prebin_type": "median", "skip_replace": 0,
    # clean
    "corr_thresh_1": 0.99, "corr_thresh_2": 0.99, "annular_clean": 0, "manual_bads": [],
    "noise_clean": 1, "noise_thresh_1": 1.05, "noise_thresh_2": 1.05, "coronoise_clean": 0,
    "coronoise_thresh": 2.0, "peak_clean": 1, "peak_thresh_1": 0.95, "peak_thresh_2": 0.95,
    "peak_scale": 1, "clean_angs": 0,
    # geometry
    "pxscale": 0.0179,
}

#: Settings that exist only in the port, with their defaults.
PORT_DEFAULTS: Dict[str, Any] = {
    "night": "",            # night folder name in the archive
    "archive": "/mnt/RAID64TB/NOMIC",
    "manifest": "_raw_inventory/tables/raw_frame_manifest_sorted.csv",
    "badmap_master": "calib/master_badmap_2022.fits",
    "frames": None,         # explicit [first, last] frame numbers of the science run; None = manifest
    "sky_dirs": [],         # explicit sky folders, relative to the night folder, in round order
    "out_subdir": "reduced_py",
    "idl_oddeven": True,    # reproduce the 2x2 smoothing of remove_oddeven_noise (IDL precedence bug)
    "idl_smooth_bg": True,  # reproduce smooth_bg having no effect (IDL misspelling)
    "chopid_destripe": True,  # subtract each row's median, per channel half, from the chop-ID image
                              # (not in the IDL predecessor; False reproduces it)
    "seed": 0,              # seed for the random frame draws of the center stage
    "workers": 1,
    "display": False,       # live display (window, or a PNG when no X display is available)
}


@dataclass
class Settings:
    """Settings for reducing one night, one aperture.

    Parameters
    ----------
    values : dict
        Every setting by its IDL name (``DEFAULTS``) or port name (``PORT_DEFAULTS``).
    unevaluated : dict
        Object-block lines the extractor could not evaluate, by setting name, kept for review.
    source : str
        Where the overrides came from, e.g. ``config/nights_idl.json:Procyon_17``.
    """

    values: Dict[str, Any] = field(default_factory=dict)
    unevaluated: Dict[str, str] = field(default_factory=dict)
    source: str = "defaults"

    def __getattr__(self, name: str) -> Any:
        try:
            return self.values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def side_value(self, base: str, chop: int) -> Any:
        """Return a per-chop setting such as ``pre_corr_thresh_1``.

        Parameters
        ----------
        base : str
            Setting name without the suffix, e.g. ``pre_corr_thresh``.
        chop : int
            1 for chop A (cube1), 2 for chop B (cube2).

        Returns
        -------
        value : any
            The value of ``<base>_<chop>``.
        """
        return self.values[f"{base}_{chop}"]


#: Settings of the object blocks the port ignores: step switches from partial IDL reruns, paths
#: of the Mac, and settings of the KLIP and ADI stages, which KLIP-TPE replaces.
IGNORED = {"root_dir", "do_combine", "do_chop_sub", "do_split", "do_align", "do_center",
           "do_pre_clean", "do_pre_binning", "do_replace", "do_clean", "delete_chop",
           "delete_subcubes", "delete_splitcubes", "verify_clean", "verify_pre_clean",
           "verify_total_pre_clean", "bin", "k_klip", "angsep", "anglemax", "n_ang", "nrings",
           "wr", "adi_filter", "klip_filter", "second_klip_filter", "do_destripe", "radfilt",
           "annmode_inout", "subrefsky", "gpsf", "refname", "nref", "ref_file_l1", "ref_file_l2",
           "ref_file_r1", "ref_file_r2", "ivert_pos"}


def load_settings(night: str, nights_json: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Settings:
    """Build the settings of one night from the defaults and its IDL object block.

    Parameters
    ----------
    night : str
        Night folder name, e.g. ``Procyon_17``.
    nights_json : str, optional
        Path of the extracted object blocks; default ``config/nights_idl.json`` beside the package.
    overrides : dict, optional
        Final overrides by setting name, applied last (e.g. port settings for this run).

    Returns
    -------
    settings : Settings
        The merged settings. ``postpoly_bgsub`` follows ``normalize_middle`` unless the block
        sets it, as in IDL.
    """
    if nights_json is None:
        nights_json = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "nights_idl.json")
    values = dict(DEFAULTS)
    values.update(PORT_DEFAULTS)
    values["night"] = night
    unevaluated: Dict[str, str] = {}
    source = "defaults"
    if os.path.exists(nights_json):
        blocks = json.load(open(nights_json))
        if night in blocks:
            block = blocks[night]
            for k, v in block["settings"].items():
                if k not in IGNORED:
                    values[k] = v
            if "normalize_middle" in block["settings"] and "postpoly_bgsub" not in block["settings"]:
                values["postpoly_bgsub"] = block["settings"]["normalize_middle"]
            unevaluated = {k: v for k, v in block.get("unevaluated", {}).items() if k not in IGNORED}
            source = f"{os.path.basename(nights_json)}:{night}"
    if overrides:
        values.update(overrides)
    return Settings(values=values, unevaluated=unevaluated, source=source)


def chop_frames_per_side(settings: Settings) -> List[str]:
    """List the apertures to reduce for a night.

    Parameters
    ----------
    settings : Settings
        The night's settings; ``side`` is ``left``, ``right`` or ``both``.

    Returns
    -------
    sides : list of str
        ``["left"]``, ``["right"]`` or ``["left", "right"]``.
    """
    return ["left", "right"] if settings.side == "both" else [settings.side]
