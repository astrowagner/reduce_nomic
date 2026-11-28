#!/usr/bin/env python3
"""Prototype fixes for KLIP-TPE's edge-of-frame artifacts on one annulus' winner, before and after.

KLIP-TPE ``bf99474`` reduces frames that hold NaN past the detector's edge with steps that leave
lines along each frame's edge in the final images: the NaN-aware boxcar high-pass averages one side
only within half its width of a NaN edge, KLIP fills NaN with zeros before it builds the basis and
projects, the noise-weighted combination measures each pixel's variance with NaN as zeros, the
partitions are combined with one weight each, so a partition's few frames at its coverage edge
enter at full weight, and the S/N map's noise is per radius only. This script reduces one finished
annulus' winning configuration with KLIP-TPE's own reducer, as the run did, and again with
prototype fixes patched into the loaded ``klip_tpe`` modules (the klip-tpe code on disk is not
changed). A variant is a name in `VARIANTS` or patches of `PATCHES` joined with "+":

- ``base``: KLIP-TPE as it is; it must reproduce the run's ``best_clean.fits``;
- ``hpmask``: after the high-pass of the input frames, NaN every pixel whose boxcar is not
  entirely on data;
- ``hpmask2``: as ``hpmask``, with the blanked band twice as wide (the boxcar's full width);
- ``mklip``: a masked KLIP fit, each target's basis and projection on the pixels that hold data
  in the target and in all of its references, NaN elsewhere, in place of the zero fill
  (per-target path only, the run's ``klip_gram="target"``);
- ``nanvar``: the noise-weighted combination's variance over the frames with data at each pixel;
- ``hpmask_klip``: ``hpmask`` and ``mklip``.

``reduce --pixcomb`` also replaces the partitions' one-weight combination with per-pixel
inverse-variance weights from each partition's frame counts (`pixel_combine`), and the
``recombine`` step combines the partition images it saved with other weights (`partition_weights`):
the fraction of each partition's frames with data, which leaves data without NaN as they are.

The ``compare`` step reads the images and makes, for every variant, KLIP-TPE's S/N map
(``klip_tpe.products.snr_map``) and a noise-map one: the same map of the image divided by its
expected noise from the partitions' frame counts (`expected_noise`), so the noise model follows
the coverage. It writes a table, a figure and the injected sources' S/N. The ``sensitivity``
step measures each injected source's signal in the injected image minus the clean one and the
noise of the clean image at the source's radius, both after the noise map, and gives the 5 sigma
contrast there. The ``figure`` step draws one variant before and after another.

Usage
-----
python edge_fix_prototype.py reduce --kt-src DIR --run-dir RUN --inputs DIR --annulus N --variant V
    --out DIR [--inject] [--inj-tag TAG] [--rotate-sources DEG] [--skip-clean] [--pixcomb]
    [--workers 2] [--extra-sources FILE] [--param KEY=VALUE]... [--label L]
python edge_fix_prototype.py stages ... --partition P
python edge_fix_prototype.py compare --kt-src DIR --run-dir RUN --annulus N --out DIR
    [--only L...] [--counts LABEL:SOURCE]...
python edge_fix_prototype.py sensitivity --kt-src DIR --run-dir RUN --annulus N --out DIR
    --labels L... --tags TAG... [--counts LABEL:SOURCE]... [--ref L] [--prefix NAME]
python edge_fix_prototype.py recombine --out DIR --source L --label NEW --mode {pixel,coverage,equal}
    [--tags TAG...]
python edge_fix_prototype.py figure --kt-src DIR --run-dir RUN --annulus N --out DIR
    --before L --after L [--counts LABEL:SOURCE]...

``--kt-src`` is a directory holding the ``klip_tpe`` package (a checkout of ``bf99474`` or a commit
with the same code), ``--run-dir`` the finished run, ``--inputs`` the folder of its input cubes,
angles and PSFs (the run's ``inputs_*`` folder, with ``manifest.json``), N the annulus counted from
1 and V a variant above.
"""

import argparse
import inspect
import json
import os
import sys
import time

import numpy as np
from astropy.io import fits

#: The patches a variant can join with "+" (see `install`).
PATCHES = ("hpmask", "hpmask2", "mklip", "nanvar")

#: Named variants and their patches, in the order the comparison lists them.
VARIANTS = {"base": (), "hpmask": ("hpmask",), "hpmask2": ("hpmask2",), "hpmask_klip": ("hpmask", "mklip"),
            "mklip": ("mklip",), "nanvar": ("nanvar",), "hpmask_klip_nanvar": ("hpmask", "mklip", "nanvar")}


# ----------------------------------------------------------------------------- the run's records

def load_winner(run_dir, annulus):
    """Return the record of an annulus' winning evaluation from the run's results.

    Parameters
    ----------
    run_dir : str
        The run folder, with ``annulusNN/annulus_done.json`` and ``results.jsonl``.
    annulus : int
        Annulus, counted from 1.

    Returns
    -------
    dict
        The winner's ``results.jsonl`` record (``config``, ``sources``, ``contrast``, ``meta``).

    Raises
    ------
    SystemExit
        If the annulus is not done or its winner is not in ``results.jsonl``.
    """
    done = os.path.join(run_dir, f"annulus{annulus:02d}", "annulus_done.json")
    if not os.path.exists(done):
        raise SystemExit(f"{done} is missing: annulus {annulus} has not finished")
    idx = int(json.load(open(done))["winner_index"])
    with open(os.path.join(run_dir, "results.jsonl")) as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("annulus") == annulus - 1 and r.get("index") == idx:
                return r
    raise SystemExit(f"winner {idx} of annulus {annulus} is not in results.jsonl")


def run_setup(run_dir):
    """Return the run's ``run_setup.json``.

    Parameters
    ----------
    run_dir : str
        The run folder.

    Returns
    -------
    dict
        The set-up record.
    """
    return json.load(open(os.path.join(run_dir, "run_setup.json")))


def pad_edges(a_in, a_out, edges, edge_padding="between"):
    """Return whether a zone's inner and outer edges are padded, as ``Runner._pad_edges`` decides.

    Parameters
    ----------
    a_in, a_out : float
        The zone's radii, in pixels.
    edges : sequence of float
        The run's annulus edges, in pixels.
    edge_padding : {"between", "all"}, default "between"
        ``RunConfig.edge_padding``: pad only the edges another annulus shares, or every edge.

    Returns
    -------
    tuple of bool
        ``(pad_in, pad_out)``.
    """
    if str(edge_padding).lower() == "all":
        return True, True
    e = [float(v) for v in edges]
    lo, hi = (e[0], e[-1]) if len(e) >= 2 else (0.0, e[0] if e else 70.0)
    return a_in > lo + 1e-6, not (a_out >= hi - 1e-6)


# ----------------------------------------------------------------------------- the patches

def install_hpmask(factor=1.0):
    """Patch klip-tpe's high-pass of the input frames to blank the pixels whose boxcar is not on data.

    After ``_highpass_frames`` with ``nan_aware=True``, every pixel within ``factor * (w // 2)`` of
    a NaN pixel (a square neighborhood, w the odd boxcar width) or of the array's border is set to
    NaN. ``factor`` 1 blanks exactly the band where the boxcar averages one side only. Only
    ``klip_tpe.reducer._highpass_frames`` is patched: the reducer's fused preparation calls it by
    that name on the whole input frames, whose NaN is the detector's edge. The residuals' second
    high-pass (``post_hp``, through ``klip_tpe.klip.highpass_stack``) runs on frames that are NaN
    outside the KLIP zone, which this rule would eat into, so it keeps klip-tpe's behavior.

    Parameters
    ----------
    factor : float, default 1.0
        Width of the blanked band in units of the boxcar's half-width.

    Returns
    -------
    None
        The modules are patched in place.
    """
    from scipy import ndimage
    from klip_tpe import reducer
    orig = reducer._highpass_frames

    def patched(img, width, nan_aware):
        """High-pass as ``_highpass_frames`` does, then NaN the pixels whose boxcar is not all on data."""
        out = orig(img, width, nan_aware)
        w = int(width)
        if not nan_aware or w <= 1:
            return out
        if w % 2 == 0:
            w += 1
        h = int(round(factor * (w // 2)))
        if h < 1:
            return out
        fin = np.isfinite(np.asarray(img))
        size = (1,) * (fin.ndim - 2) + (2 * h + 1, 2 * h + 1)
        full = ndimage.uniform_filter(fin.astype(np.float32), size, mode="constant") > 1.0 - 1e-4
        out[~full] = np.nan
        return out

    reducer._highpass_frames = patched


#: The per-target block of ``klip_annular`` (bf99474) that the masked fit replaces.
_ZERO_FILL_BLOCK = """            R = _zero_nan(BD[refs])
            Z, lam, V = _basis(R)
            T = _zero_nan(D[t:t + 1].copy())
            Star = T.copy()               # raw target (the FM uses the un-adjusted stellar frame, L493)
            if p.spat_mean:
                T -= T.mean()
            if p.temp_mean:
                T -= R.mean(axis=0, keepdims=True)
            kk_max = min(Z.shape[0], k)
            if p.k_scan:
                S = T @ Z.T
                for kk in range(k_req):
                    m = min(kk + 1, kk_max)                 # past the cap: repeat the cap
                    out[kk, t].reshape(-1)[idx] = (T - S[:, :m] @ Z[:m] if m else T)[0]
            else:
                out[t].reshape(-1)[idx] = _project(T, Z, kk_max)[0]
"""

_MASKED_BLOCK = """            Rr = BD[refs]
            Dt = D[t:t + 1]
            P = np.isfinite(Dt[0]) & np.isfinite(Rr).all(axis=0)   # data in the target and every reference
            o_t = out[t].reshape(-1)
            o_t[idx] = np.nan
            if int(P.sum()) < _MASKED_MIN_PX or p.k_scan or do_fm:
                if p.k_scan or do_fm:
                    raise NotImplementedError("masked KLIP prototype: no k_scan, no FM")
                continue
            R = Rr[:, P]
            Z, lam, V = _basis(R)
            T = Dt[:, P].copy()
            Star = T.copy()
            if p.spat_mean:
                T -= T.mean()
            if p.temp_mean:
                T -= R.mean(axis=0, keepdims=True)
            kk_max = min(Z.shape[0], k)
            o_t[idx[P]] = _project(T, Z, kk_max)[0]
"""


def install_masked_klip(min_px=50):
    """Patch ``klip_annular``'s per-target path to fit on the pixels with data, in place of the zero fill.

    For each target frame and zone, the basis is built from the reference frames on the zone's
    pixels that hold data in the target and in every reference, and the target is projected on the
    same pixels; the zone's other pixels come out NaN. The patched function is compiled from
    ``klip_annular``'s own source with the one block replaced, and both ``klip_tpe.klip`` and
    ``klip_tpe.reducer`` get it.

    Parameters
    ----------
    min_px : int, default 50
        Fewest such pixels for a zone of a target to be reduced; below, it stays NaN.

    Returns
    -------
    None
        The modules are patched in place.

    Raises
    ------
    RuntimeError
        If the block to replace is not found exactly once (another klip-tpe version).
    """
    from klip_tpe import klip, reducer
    src = inspect.getsource(klip.klip_annular)
    if src.count(_ZERO_FILL_BLOCK) != 1:
        raise RuntimeError("klip_annular's per-target block is not the bf99474 one")
    src = src.replace(_ZERO_FILL_BLOCK, _MASKED_BLOCK)
    klip.__dict__["_MASKED_MIN_PX"] = int(min_px)
    exec(compile(src, klip.__file__, "exec"), klip.__dict__)
    reducer.klip_annular = klip.klip_annular


def install_nanvar_comb(min_frames=3):
    """Patch the noise-weighted combination to measure each pixel's variance over its frames with data.

    KLIP-TPE's ``nw_ang_comb`` turns the derotated frames back to the pupil frame with NaN as zeros
    and takes each pixel's variance over all the frames, so where only a fraction f of the frames
    has data the variance comes out about f times too small and those frames' pixels, at their
    detector edge, are weighted up. Here each frame's data mask is turned back with it, and the
    variance is taken over the frames whose pixel is entirely on data; a pixel with fewer than
    ``min_frames`` such frames takes the median variance of the others. The rest is
    ``nw_ang_comb``'s: the same floor, the same frames turned back and forth, the same weights.
    A call with ``ref_cube`` (KLIP-FM frames) or one frame goes to the original.

    Parameters
    ----------
    min_frames : int, default 3
        Fewest frames with data for a pixel's own variance.

    Returns
    -------
    None
        ``klip_tpe.reducer.nw_ang_comb`` is replaced; ``klip_tpe.klip`` keeps the original.
    """
    from klip_tpe import klip, reducer
    orig = klip.nw_ang_comb
    rot = klip.rotate_ccw

    def nw_ang_comb_nanvar(derot_cube, angles, ref_cube=None, threads=1):
        """Combine as ``nw_ang_comb`` does, each pixel's variance over its frames with data."""
        cube = np.asarray(derot_cube)
        n = cube.shape[0]
        if n == 1 or ref_cube is not None:
            return orig(derot_cube, angles, ref_cube=ref_cube, threads=threads)
        fin = np.isfinite(cube)
        A = np.empty(cube.shape, np.float64)
        on = np.empty(cube.shape, bool)
        for i in range(n):
            A[i] = rot(np.where(fin[i], cube[i], 0.0), -float(angles[i]), cval=0.0)
            on[i] = rot(fin[i].astype(np.float64), -float(angles[i]), cval=0.0) >= 1.0 - 1e-3
        A[~np.isfinite(A)] = 0.0
        cnt = on.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.where(on, A, 0.0).sum(axis=0) / np.maximum(cnt, 1)
            var = np.where(on, (A - m) ** 2, 0.0).sum(axis=0) / np.maximum(cnt - 1, 1)
        ok = (cnt >= int(min_frames)) & (var > 0)
        med = float(np.median(var[ok])) if ok.any() else 1.0
        var = np.where(ok, var, med)
        vfloor = med * 1e-6
        var = np.maximum(var, vfloor)
        num = np.zeros(cube.shape[1:])
        den = np.zeros(cube.shape[1:])
        for i in range(n):
            Ai = rot(A[i], float(angles[i]), cval=0.0)
            Vi = np.maximum(rot(var, float(angles[i]), cval=0.0), vfloor)
            Ai[~np.isfinite(Ai)] = 0.0
            Vi[~np.isfinite(Vi)] = vfloor
            inv = fin[i] / Vi
            num += Ai * inv
            den += inv
        out = num / np.maximum(den, 1e-30)
        out[den <= 0] = np.nan
        return out

    reducer.nw_ang_comb = nw_ang_comb_nanvar


def patches_of(variant):
    """Return the patches of a variant: a name in `VARIANTS` or patches of `PATCHES` joined with "+".

    Parameters
    ----------
    variant : str
        The variant.

    Returns
    -------
    tuple of str
        Its patches, empty for ``base``.

    Raises
    ------
    SystemExit
        If the variant is neither.
    """
    if variant in VARIANTS:
        return tuple(VARIANTS[variant])
    parts = tuple(p for p in variant.split("+") if p)
    if not parts or any(p not in PATCHES for p in parts) or ("hpmask" in parts and "hpmask2" in parts):
        raise SystemExit(f"--variant must be one of {tuple(VARIANTS)} or patches of {PATCHES} joined with '+'")
    return parts


def install(variant):
    """Apply a variant's patches to the loaded klip_tpe modules.

    Parameters
    ----------
    variant : str
        A name in `VARIANTS` or patches of `PATCHES` joined with "+" (`patches_of`).

    Returns
    -------
    None
        The modules are patched in place (``base`` patches nothing).
    """
    parts = patches_of(variant)
    if "hpmask" in parts:
        install_hpmask(1.0)
    if "hpmask2" in parts:
        install_hpmask(2.0)
    if "mklip" in parts:
        install_masked_klip()
    if "nanvar" in parts:
        install_nanvar_comb()


# ----------------------------------------------------------------------------- the reduction

def build_reducer(inputs, setup, selected, workers):
    """Build KLIP-TPE's generic reducer of the selected partitions as the run built it.

    Parameters
    ----------
    inputs : str
        Folder of the run's input cubes, angles and PSFs, with ``manifest.json``.
    setup : dict
        The run's ``run_setup.json`` (``pxscale``, ``rotation``, each partition's ``lam_m`` and ``diam_m``).
    selected : sequence of str
        Partition names to load (the winner's).
    workers : int
        Worker threads.

    Returns
    -------
    reducer : klip_tpe.reducer.PartitionedReducer
        The reducer.
    """
    from klip_tpe import cli, klip
    klip.set_rotation_backend(str(setup.get("rotation") or "scipy").split()[0])
    man = json.load(open(os.path.join(inputs, "manifest.json")))
    parts = {p["name"]: p for p in man["partitions"]}
    rp = setup["reducer"]["partitions"][selected[0]]
    args = ["generic"]
    for opt, key in (("--cube", "cube"), ("--angles", "angles"), ("--psf", "psf")):
        args.append(opt)
        args += [os.path.join(inputs, os.path.basename(parts[n][key])) for n in selected]
    args += ["--names", *selected, "--pxscale", str(setup["pxscale"]), "--lam", repr(float(rp["lam_m"])),
             "--diam", repr(float(rp["diam_m"])), "--workers", str(int(workers)), "--pool", "threads",
             "--run-dir", os.path.join(inputs, "_unused_run_dir")]
    a = cli._build_parser().parse_args(args)
    red, _obj, _smp, _guard, _thr = cli._generic_objects(a, log=lambda s: None)
    return red


def winner_config(rec, setup, selected, extra=None):
    """Return the winner's Config with the annulus' zone and padding, as ``Runner._reduce`` adds them.

    Parameters
    ----------
    rec : dict
        The winner's record (`load_winner`).
    setup : dict
        The run's ``run_setup.json`` (``config.ann_edges``, ``config.edge_padding``).
    selected : sequence of str
        The partitions loaded.
    extra : dict or None, default None
        Reducer parameters set on top of the winner's, such as ``post_hp`` or ``comb_type``.

    Returns
    -------
    klip_tpe.space.Config
        The configuration.
    """
    from klip_tpe.space import Config
    a_in, a_out = (float(v) for v in rec["meta"]["zone"])
    rc = setup["config"]
    pad_in, pad_out = pad_edges(a_in, a_out, rc["ann_edges"], rc.get("edge_padding") or "between")
    over = {"inrad": a_in, "outrad": a_out, "pad_in": pad_in, "pad_out": pad_out}
    over.update(extra or {})
    c = rec["config"]
    params = dict(c["params"], **over)
    per = {n: dict(c["per_partition"][n], **over) for n in selected}
    return Config(params, per, list(selected), np.asarray(c["x"], float))


def source_rows(rec, extra=None, rotate=0.0):
    """Return the winner's injected sources, any extra ones, turned by an angle, as rows.

    Parameters
    ----------
    rec : dict
        The winner's record; ``sources`` holds ``[rho_arcsec, theta_deg, contrast]`` rows.
    extra : str or None, default None
        A text file with one ``rho_arcsec theta_deg contrast`` row per extra source.
    rotate : float, default 0.0
        Degrees added to every source's position angle, so one run's pattern samples other
        position angles at the same separations and contrast.

    Returns
    -------
    list of tuple of 3 floats
        ``(rho_arcsec, theta_deg, contrast)``, theta in [0, 360).
    """
    rows = [tuple(map(float, s[:3])) for s in rec["sources"]]
    if extra:
        rows += [tuple(map(float, ln.split()[:3])) for ln in open(extra) if ln.strip() and not ln.startswith("#")]
    return [(r, (t + float(rotate)) % 360.0, c) for r, t, c in rows]


def sources_of(rec, extra=None, rotate=0.0):
    """Return the injected sources of `source_rows` as klip-tpe Sources.

    Parameters
    ----------
    rec : dict
        The winner's record.
    extra : str or None, default None
        A file of extra sources (`source_rows`).
    rotate : float, default 0.0
        Degrees added to every position angle.

    Returns
    -------
    list of klip_tpe.metrics.Source
        The sources.
    """
    from klip_tpe.metrics import Source
    return [Source(r, t, c) for r, t, c in source_rows(rec, extra, rotate)]


def write_image(path, img, header):
    """Write an image or a stack to a new FITS file, refusing to overwrite.

    Parameters
    ----------
    path : str
        Output path.
    img : ndarray
        The image, shape (ny, nx), or a stack (npart, ny, nx).
    header : dict
        Header cards.

    Returns
    -------
    None
        The file is written.
    """
    if os.path.exists(path):
        raise SystemExit(f"{path} exists; not overwriting")
    h = fits.Header()
    for k, v in header.items():
        h[k] = v
    fits.writeto(path, np.asarray(img, np.float32), h)


def pixel_combine(red, cfg, sources, selected):
    """Reduce each partition alone and combine them with per-pixel inverse-variance weights.

    KLIP-TPE combines the partitions with one weight each (``weighting="equal"``), so where one
    partition holds few frames (its coverage edge) its noise enters the combination at full weight.
    Here each partition p is reduced alone (KLIP-TPE's own reduction, its frames combined by
    ``nw_ang_comb``), the number of its derotated frames with data at each pixel, n_p, is counted,
    and the per-frame noise s_p is the robust sigma of its derotated frames where every frame has
    data. The images are combined with weights n_p / s_p**2, the inverse of each partition's
    variance s_p**2 / n_p at the pixel.

    Parameters
    ----------
    red : klip_tpe.reducer.PartitionedReducer
        The reducer.
    cfg : klip_tpe.space.Config
        The configuration, with every selected partition's parameters.
    sources : sequence of klip_tpe.metrics.Source or None
        Sources to inject, None for the clean reduction.
    selected : sequence of str
        The partitions to combine; npart is their number.

    Returns
    -------
    image : ndarray, shape (ny, nx)
        The combined image, NaN where no partition has data.
    stack : ndarray, shape (npart, ny, nx)
        Each partition's image.
    counts : ndarray, shape (npart, ny, nx)
        Each partition's number of derotated frames with data at each pixel.
    sig : list of float
        Each partition's per-frame noise s_p.
    """
    from klip_tpe import reducer as kr
    from klip_tpe.space import Config
    d0 = kr.derotate
    grab = {}

    def d_hook(cube, *args, **kw):
        """Derotate as klip-tpe does and keep the first derotated cube of the reduction."""
        out = d0(cube, *args, **kw)
        grab.setdefault("dr", np.array(out, np.float32))
        return out

    kr.derotate = d_hook
    imgs, counts, sig = [], [], []
    try:
        for pid in selected:
            grab.clear()
            one = Config(dict(cfg.params), {pid: dict(cfg.per_partition[pid])}, [pid], cfg.x)
            ev = red.reduce_config(one, sources)
            dr = grab["dr"]
            n = np.isfinite(dr).sum(axis=0).astype(np.float32)
            full = n >= dr.shape[0]
            v = dr[:, full]
            v = v[np.isfinite(v)]
            s = float(1.4826 * np.median(np.abs(v - np.median(v)))) if v.size else float("nan")
            imgs.append(np.asarray(ev.image, float))
            counts.append(n)
            sig.append(s)
    finally:
        kr.derotate = d0
    stack = np.stack(imgs)
    cnt = np.stack(counts)
    w = np.stack([np.where(np.isfinite(im), c / s ** 2, 0.0) for im, c, s in zip(imgs, cnt, sig)])
    num = np.nansum(np.where(w > 0, stack, 0.0) * w, axis=0)
    den = w.sum(axis=0)
    img = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    return img, stack, cnt, sig


def parse_params(items):
    """Parse ``KEY=VALUE`` reducer parameters, the values read as JSON where they parse.

    Parameters
    ----------
    items : sequence of str or None
        The ``--param`` values.

    Returns
    -------
    dict
        ``{key: value}``; ``post_hp=false`` gives ``{"post_hp": False}``, ``comb_type=mean`` gives
        ``{"comb_type": "mean"}``.
    """
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        try:
            out[k] = json.loads(v)
        except ValueError:
            out[k] = v
    return out


def cmd_reduce(a):
    """Reduce the winner with one variant, clean and optionally injected, and write the images.

    Parameters
    ----------
    a : argparse.Namespace
        The parsed command line of the ``reduce`` step.

    Returns
    -------
    None
        Writes ``LABEL_clean.fits`` and ``LABEL_clean_stack.fits`` (not with ``--skip-clean``) and,
        with ``--inject``, ``LABEL_TAG.fits``, ``LABEL_TAG_stack.fits`` and the sources'
        ``[rho_arcsec, theta_deg, contrast]`` rows in ``LABEL_sources.json`` (TAG ``inj``) or
        ``LABEL_TAG_sources.json`` into ``--out``; LABEL is ``--label`` or the variant and TAG
        ``--inj-tag``. With ``--pixcomb`` each pass also writes ``*_counts.fits``.
    """
    sys.path.insert(0, a.kt_src)
    install(a.variant)
    setup = run_setup(a.run_dir)
    rec = load_winner(a.run_dir, a.annulus)
    selected = list(rec["config"]["selected"])
    os.makedirs(a.out, exist_ok=True)
    label = a.label or a.variant
    tag_inj = a.inj_tag
    if a.skip_clean and not a.inject:
        raise SystemExit("--skip-clean leaves nothing to do without --inject")
    if a.inject and (tag_inj == "clean" or not tag_inj.replace("_", "").isalnum()):
        raise SystemExit("--inj-tag must be letters, digits and '_', and not 'clean'")
    src_json = os.path.join(a.out, f"{label}_sources.json" if tag_inj == "inj" else f"{label}_{tag_inj}_sources.json")
    if a.inject and os.path.exists(src_json):
        raise SystemExit(f"{src_json} exists; not overwriting")
    t0 = time.time()
    red = build_reducer(a.inputs, setup, selected, a.workers)
    extra = parse_params(a.param)
    cfg = winner_config(rec, setup, selected, extra)
    hdr = {"VARIANT": a.variant, "LABEL": label, "ANNULUS": a.annulus, "WINNER": rec["index"],
           "PARTS": ",".join(selected), "INRAD": cfg.params["inrad"], "OUTRAD": cfg.params["outrad"],
           "PARAMS": json.dumps(extra)[:60]}
    passes = [] if a.skip_clean else [("clean", None)]
    if a.inject:
        passes.append((tag_inj, sources_of(rec, a.extra_sources, a.rotate_sources)))
        hdr["SRCROT"] = float(a.rotate_sources)
    for tag, srcs in passes:
        t1 = time.time()
        if a.pixcomb:
            img, stack, counts, sig = pixel_combine(red, cfg, srcs, selected)
            write_image(os.path.join(a.out, f"{label}_{tag}.fits"), img, dict(hdr, COMBINE="pixel inverse variance"))
            write_image(os.path.join(a.out, f"{label}_{tag}_stack.fits"), stack, dict(hdr, STACKOF=",".join(selected)))
            write_image(os.path.join(a.out, f"{label}_{tag}_counts.fits"), counts,
                        dict(hdr, STACKOF=",".join(selected), FRAMESIG=",".join(f"{v:.4g}" for v in sig)))
            print(f"{label} {tag}: {time.time() - t1:.0f} s, finite {np.isfinite(img).mean():.3f}")
            continue
        ev = red.reduce_config(cfg, srcs)
        write_image(os.path.join(a.out, f"{label}_{tag}.fits"), ev.image, hdr)
        if ev.stack is not None:
            write_image(os.path.join(a.out, f"{label}_{tag}_stack.fits"), ev.stack,
                        dict(hdr, STACKOF=",".join(str(p) for p in (ev.partitions or selected))))
        print(f"{label} {tag}: {time.time() - t1:.0f} s, finite {np.isfinite(ev.image).mean():.3f}")
    if a.inject:
        with open(src_json, "x") as f:
            json.dump([list(r) for r in source_rows(rec, a.extra_sources, a.rotate_sources)], f)
    print(f"done in {time.time() - t0:.0f} s")


# ----------------------------------------------------------------------------- the comparison

def relative_noise(img, cov, fwhm, rmin, rmax, cov_edges=(0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.97, 1.01),
                   band=None, min_px=200):
    """Return a relative noise map: the image's robust sigma by radius and coverage over each radius'.

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        The clean image, NaN where it has no value.
    cov : ndarray, shape (ny, nx)
        Coverage, the fraction of frames with data at each pixel.
    fwhm : float
        FWHM of the PSF, in pixels.
    rmin, rmax : float
        Radii, in pixels, between which the map is made; 1 elsewhere.
    cov_edges : sequence of float, default (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.97, 1.01)
        Coverage bin edges.
    band : float or None, default None
        Width of the radius bands, in pixels; None takes one FWHM.
    min_px : int, default 200
        Fewest pixels of a (band, coverage) bin for its own sigma; sparser bins take the band's.

    Returns
    -------
    ndarray, shape (ny, nx)
        The ratio, 1 where no estimate is made.
    """
    from scipy import ndimage

    def rsig(v):
        """Robust sigma of at least 20 finite values, else NaN."""
        return robust_sigma(v, min_n=20)

    ny, nx = img.shape
    yy, xx = np.mgrid[:ny, :nx]
    r = np.hypot(xx - (nx - 1) / 2.0, yy - (ny - 1) / 2.0)
    band = float(band or fwhm)
    out = np.ones_like(img, dtype=float)
    fin = np.isfinite(img)
    for r0 in np.arange(rmin, rmax, band):
        ring = (r >= r0) & (r < r0 + band) & fin
        s_ring = rsig(img[ring])
        if not np.isfinite(s_ring) or s_ring <= 0:
            continue
        for c0, c1 in zip(cov_edges, cov_edges[1:]):
            m = ring & (cov >= c0) & (cov < c1)
            if m.sum() >= min_px:
                s = rsig(img[m])
                if np.isfinite(s) and s > 0:
                    out[(r >= r0) & (r < r0 + band) & (cov >= c0) & (cov < c1)] = s / s_ring
    return ndimage.gaussian_filter(out, fwhm / 2.355)


#: The ``COMBINE`` card of a reduction and the partition weights it stands for (`partition_weights`);
#: a reduction without the card used KLIP-TPE's one equal weight per partition.
COMBINE_MODES = {"pixel inverse variance": "pixel", "pixel coverage": "coverage", "equal": "equal"}


def combine_mode(header):
    """Return how a reduction's partitions were combined, from its ``COMBINE`` card.

    Parameters
    ----------
    header : astropy.io.fits.Header or dict
        The clean image's header.

    Returns
    -------
    str
        "pixel", "coverage" or "equal" (`partition_weights`); "equal" without the card.
    """
    return COMBINE_MODES.get(str(header.get("COMBINE", "equal")).strip(), "equal")


def partition_weights(counts, sig, mode):
    """Return each partition's weight at each pixel for a combination mode.

    "pixel" is the inverse of each partition's variance there, n_p / s_p**2; "coverage" is the
    fraction of the partition's frames with data there, n_p / N_p, which is 1 wherever every frame
    has data, so data without NaN combine as with equal weights; "equal" is 1 wherever the
    partition has data. N_p is the largest count of partition p.

    Parameters
    ----------
    counts : ndarray, shape (npart, ny, nx)
        Each partition's number of derotated frames with data at each pixel, n_p; npart partitions.
    sig : sequence of float
        Each partition's per-frame noise s_p.
    mode : {"pixel", "coverage", "equal"}
        The weights.

    Returns
    -------
    ndarray, shape (npart, ny, nx)
        The weights, 0 where a partition has no data.

    Raises
    ------
    ValueError
        If the mode is none of these.
    """
    c = np.asarray(counts, float)
    if mode == "pixel":
        w = c / np.asarray(sig, float)[:, None, None] ** 2
    elif mode == "coverage":
        nmax = c.reshape(c.shape[0], -1).max(axis=1)[:, None, None]
        w = c / np.where(nmax > 0, nmax, 1.0)
    elif mode == "equal":
        w = np.ones_like(c)
    else:
        raise ValueError(f"unknown combination mode {mode!r}")
    return np.where(c > 0, w, 0.0)


def expected_noise(counts, sig, weights):
    """Return the combined image's expected noise at each pixel from the partitions' frame counts.

    Each partition p contributes an image of variance s_p**2 / n_p at a pixel where n_p of its
    derotated frames have data, and the weighted mean of the images has variance
    ``sum_p w_p**2 s_p**2 / n_p / (sum_p w_p)**2``. With per-pixel inverse-variance weights
    (`pixel_combine`) that is ``1 / sum_p (n_p / s_p**2)``; with one equal weight per partition, as
    KLIP-TPE's ``weighting="equal"`` combines them, ``sum_p (s_p**2 / n_p) / k**2`` over the k
    partitions with data there.

    Parameters
    ----------
    counts : ndarray, shape (npart, ny, nx)
        Each partition's number of derotated frames with data at each pixel; npart partitions.
    sig : sequence of float
        Each partition's per-frame noise s_p.
    weights : bool or {"pixel", "coverage", "equal"}
        The combination's weights (`partition_weights`); True means "pixel" and False "equal".

    Returns
    -------
    ndarray, shape (ny, nx)
        The expected noise, NaN where no partition has data.
    """
    mode = ("pixel" if weights else "equal") if isinstance(weights, (bool, np.bool_)) else str(weights)
    c = np.asarray(counts, float)
    s2 = np.asarray(sig, float)[:, None, None] ** 2
    w = partition_weights(c, sig, mode)
    have = c > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        num = np.where(have, w ** 2 * s2 / np.where(have, c, 1.0), 0.0).sum(axis=0)
        den = w.sum(axis=0) ** 2
        var = num / den
    return np.where(have.any(axis=0) & (den > 0), np.sqrt(var), np.nan)


def recombine(stack, counts, sig, mode):
    """Combine the partitions' images with a mode's per-pixel weights.

    Parameters
    ----------
    stack : ndarray, shape (npart, ny, nx)
        Each partition's image, NaN where it has no data.
    counts : ndarray, shape (npart, ny, nx)
        Each partition's number of derotated frames with data at each pixel.
    sig : sequence of float
        Each partition's per-frame noise s_p.
    mode : {"pixel", "coverage", "equal"}
        The weights (`partition_weights`).

    Returns
    -------
    ndarray, shape (ny, nx)
        The weighted mean of the images, NaN where no partition has data.
    """
    st = np.asarray(stack, float)
    w = np.where(np.isfinite(st), partition_weights(counts, sig, mode), 0.0)
    num = (np.where(w > 0, st, 0.0) * w).sum(axis=0)
    den = w.sum(axis=0)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def cmd_compare(a):
    """Compare the variants: reproduction, S/N maps, noise against coverage and the injected sources.

    Parameters
    ----------
    a : argparse.Namespace
        The parsed command line of the ``compare`` step.

    Returns
    -------
    None
        Writes ``compare.txt``, ``compare.png`` and each variant's ``*_snr.fits`` and
        ``*_snrcov.fits`` into ``--out``.
    """
    sys.path.insert(0, a.kt_src)
    from klip_tpe.products import snr_map
    setup = run_setup(a.run_dir)
    rec = load_winner(a.run_dir, a.annulus)
    fwhm, pxs = float(setup["fwhm_px"]), float(setup["pxscale"])
    a_in, a_out = (float(v) for v in rec["meta"]["zone"])
    cov = fits.getdata(os.path.join(a.run_dir, "coverage.fits")).astype(float)
    pmask = fits.getdata(os.path.join(a.run_dir, "pixel_mask.fits")).astype(bool)
    best = fits.getdata(os.path.join(a.run_dir, f"annulus{a.annulus:02d}", "best_clean.fits")).astype(float)
    ny, nx = best.shape
    yy, xx = np.mgrid[:ny, :nx]
    r = np.hypot(xx - (nx - 1) / 2.0, yy - (ny - 1) / 2.0)
    zone = (r >= a_in) & (r < a_out)
    lines = [f"Annulus {a.annulus} winner {rec['index']} ({', '.join(rec['config']['selected'])}), "
             f"zone {a_in:.2f}-{a_out:.2f} px; S/N maps by klip_tpe.products.snr_map (pixel mask of the run)"]

    rsig = robust_sigma

    labels = sorted(f[:-len("_clean.fits")] for f in os.listdir(a.out) if f.endswith("_clean.fits"))
    have = [v for v in VARIANTS if v in labels] + [v for v in labels if v not in VARIANTS]
    if a.only:
        have = [v for v in a.only if v in have]                  # in --only's order
    imgs, snrs, snrc, rels = {}, {}, {}, {}
    cmap = dict(c.split(":", 1) for c in (a.counts or []))
    lines.append("Noise-map S/N: the image divided by its expected noise (expected_noise, from each partition's "
                 "frame counts and per-frame sigma) relative to the zone's median, then klip_tpe's snr_map; "
                 "without counts, the clean image's sigma by radius and coverage (relative_noise)")
    for v in have:
        img = fits.getdata(os.path.join(a.out, f"{v}_clean.fits")).astype(float)
        img[~zone] = np.nan
        imgs[v] = img
        snrs[v] = snr_map(img, fwhm, pixel_mask=pmask)
        src = cmap.get(v, v)
        cpath = os.path.join(a.out, f"{src}_clean_counts.fits")
        if os.path.exists(cpath):
            ch = fits.getheader(cpath)
            sig = [float(x) for x in ch["FRAMESIG"].split(",")]
            mode = combine_mode(fits.getheader(os.path.join(a.out, f"{v}_clean.fits")))
            en = expected_noise(fits.getdata(cpath), sig, mode)
            rel = en / np.nanmedian(en[zone])
            rel = np.where(np.isfinite(rel), rel, 1.0)
            lines.append(f"  {v}: counts of {src}, {'per-pixel' if mode == 'pixel' else mode} partition weights, "
                         f"expected noise {np.nanmin(en[zone]):.3g}-{np.nanmax(en[zone]):.3g} over the zone")
        else:
            rel = relative_noise(img, cov, fwhm, a_in, a_out)
            lines.append(f"  {v}: no counts; relative_noise")
        rels[v] = rel
        snrc[v] = snr_map(img / rel, fwhm, pixel_mask=pmask)
        for tag, m in (("snr", snrs[v]), ("snrcov", snrc[v])):
            p = os.path.join(a.out, f"{v}_{tag}.fits")
            if not os.path.exists(p):
                fits.writeto(p, np.asarray(m, np.float32))
    if "base" in imgs:
        b = imgs["base"]
        both = zone & np.isfinite(b) & np.isfinite(best)
        d = np.abs(b[both] - best[both])
        lines.append(f"Reproduction: base vs the run's best_clean.fits over {both.sum()} zone pixels: max |diff| "
                     f"{d.max():.3g} = {d.max() / rsig(best[zone]):.2e} robust sigma; NaN pattern agrees on "
                     f"{np.mean(np.isfinite(b[zone]) == np.isfinite(best[zone])):.4f} of the zone")
    lines.append("")
    lines.append("Variant       finite  |S/N|>5 (klip-tpe / noise-map)   noise ratio by coverage "
                 "(0.5-0.6, 0.6-0.7, 0.7-0.8, 0.8-0.9, 0.9-0.97, >=0.97)   S/N spread by coverage "
                 "(klip-tpe | noise-map)")
    cbins = ((0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.97), (0.97, 1.01))
    for v in have:
        img, s, sc = imgs[v], snrs[v], snrc[v]
        full = zone & (cov >= 0.97)
        s_full = rsig(img[full])
        nr = [rsig(img[zone & (cov >= c0) & (cov < c1)]) / s_full for c0, c1 in cbins]
        ss = [rsig(s[zone & (cov >= c0) & (cov < c1)]) for c0, c1 in cbins]
        sn = [rsig(sc[zone & (cov >= c0) & (cov < c1)]) for c0, c1 in cbins]
        lines.append(f"{v:12s}  {np.isfinite(img[zone]).mean():.3f}   {int(np.nansum(np.abs(s[zone]) > 5)):6d} / "
                     f"{int(np.nansum(np.abs(sc[zone]) > 5)):6d}                 "
                     + " ".join(f"{x:4.2f}" for x in nr) + "      " + " ".join(f"{x:4.2f}" for x in ss)
                     + " | " + " ".join(f"{x:4.2f}" for x in sn))
    inj = [v for v in have if os.path.exists(os.path.join(a.out, f"{v}_inj.fits"))]
    if inj:
        srcs = json.load(open(os.path.join(a.out, f"{inj[0]}_sources.json")))
        lines.append("")
        lines.append("Injected sources: matched-filter S/N at each (peak within 0.5 FWHM), klip-tpe map / "
                     "noise-map S/N; coverage at the source")
        hdr = "rho\"   PA     cov  " + "  ".join(f"{v:>14s}" for v in inj)
        lines.append(hdr)
        rows = []
        for v in inj:
            im = fits.getdata(os.path.join(a.out, f"{v}_inj.fits")).astype(float)
            im[~zone] = np.nan
            ex = [source_xy(rho, th, pxs, (ny, nx)) for rho, th, _c in srcs]
            rel = rels[v]
            s1 = snr_map(im, fwhm, pixel_mask=pmask, exclude_xy=ex)
            s2 = snr_map(im / rel, fwhm, pixel_mask=pmask, exclude_xy=ex)
            vals = []
            for x0, y0 in ex:
                near = np.hypot(xx - x0, yy - y0) <= 0.5 * fwhm
                vals.append((np.nanmax(s1[near]) if np.isfinite(s1[near]).any() else np.nan,
                             np.nanmax(s2[near]) if np.isfinite(s2[near]).any() else np.nan))
            rows.append(vals)
        for i, (rho, th, _c) in enumerate(srcs):
            x0, y0 = ex[i]
            cv = cov[int(round(y0)), int(round(x0))]
            lines.append(f"{rho:5.2f} {th:6.1f}  {cv:4.2f}  " + "  ".join(
                f"{rows[j][i][0]:6.1f} / {rows[j][i][1]:5.1f}" for j in range(len(inj))))
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(a.out, "compare.txt"), "w") as f:
        f.write(text + "\n")
    plot_compare(a.out, have, imgs, snrs, snrc, zone, a.annulus)


def plot_compare(out, have, imgs, snrs, snrc, zone, annulus):
    """Draw each variant's clean image, KLIP-TPE S/N map and coverage-aware S/N map.

    Parameters
    ----------
    out : str
        Output folder; ``compare.png`` is written there.
    have : sequence of str
        The variants, one column each.
    imgs, snrs, snrc : dict of str to ndarray
        Each variant's clean image, S/N map and coverage-aware S/N map, shape (ny, nx).
    zone : ndarray of bool, shape (ny, nx)
        The annulus' pixels, for the crop and the stretch.
    annulus : int
        Annulus, counted from 1, for the title.

    Returns
    -------
    None
        Writes ``compare.png``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ys, xs = np.where(zone)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    names = {"base": "KLIP-TPE As Is", "hpmask": "High-Pass Edge Mask", "hpmask2": "High-Pass Mask, 2x Band",
             "hpmask_klip": "Edge Mask + Masked KLIP Fit", "base_pix": "Pixel Weights (PW)",
             "hm_pix": "PW + High-Pass Mask", "mk_pix": "PW + Masked KLIP", "nv_pix": "PW + NaN Variance",
             "mkn_pix": "PW + Masked KLIP + NaN Var.", "hk_pix": "PW + HP Mask + Masked KLIP",
             "hkn_pix": "PW + HP Mask + MK + NaN Var."}
    fig, ax = plt.subplots(3, len(have), figsize=(4.2 * len(have), 12.4), squeeze=False)
    v0 = imgs[have[0]][zone]
    s = 1.4826 * np.nanmedian(np.abs(v0 - np.nanmedian(v0)))
    for j, v in enumerate(have):
        rows = ((imgs[v], "RdBu_r", -5 * s, 5 * s, "Clean Image (±5σ of the First)"),
                (snrs[v], "RdBu_r", -5, 5, "S/N, KLIP-TPE Map"),
                (snrc[v], "RdBu_r", -5, 5, "S/N, Noise-Map Normalized"))
        for i, (m, cmap, lo, hi, what) in enumerate(rows):
            axx = ax[i, j]
            axx.imshow(m[y0:y1, x0:x1], origin="lower", cmap=cmap, vmin=lo, vmax=hi)
            axx.set_xticks([])
            axx.set_yticks([])
            for sp in axx.spines.values():
                sp.set_visible(True)
            axx.set_title(f"{names.get(v, v)}\n{what}" if i == 0 else what, fontsize=10)
    fig.suptitle(f"Annulus {annulus} Winner: Edge-Artifact Fixes (North Up, East Left)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "compare.png"), dpi=90)


# ----------------------------------------------------------------------------- sensitivity and figure

#: Coverage bins of the noise model's check and calibration in the ``sensitivity`` step.
COV_BINS = (0.0, 0.25, 0.5, 0.75, 0.9, 1.01)

#: Categorical colors in fixed order (blue, orange, aqua, yellow, magenta), checked for color-vision
#: deficiency with the dataviz validator; markers differ too, so color never carries identity alone.
SERIES = (("#2a78d6", "o"), ("#eb6834", "s"), ("#1baf7a", "^"), ("#eda100", "D"), ("#e87ba4", "v"))


def robust_sigma(v, min_n=1):
    """Return the MAD-based sigma of the finite values, NaN for fewer than ``min_n``.

    Parameters
    ----------
    v : ndarray
        Values; NaN are left out.
    min_n : int, default 1
        Fewest finite values for an estimate.

    Returns
    -------
    float
        ``1.4826 * median(|v - median(v)|)``.
    """
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size < max(int(min_n), 1):
        return float("nan")
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def source_xy(rho, theta, pxs, shape):
    """Return a source's pixel position, north up and east left, from its separation and position angle.

    Parameters
    ----------
    rho : float
        Separation, in arcsec.
    theta : float
        Position angle, in degrees east of north.
    pxs : float
        Plate scale, in arcsec per pixel.
    shape : tuple of 2 ints
        Image shape (ny, nx); the star is at ``((nx - 1) / 2, (ny - 1) / 2)``.

    Returns
    -------
    tuple of 2 floats
        ``(x, y)``, in pixels.
    """
    ny, nx = shape
    rp = float(rho) / float(pxs)
    return ((nx - 1) / 2.0 - rp * np.sin(np.deg2rad(theta)), (ny - 1) / 2.0 + rp * np.cos(np.deg2rad(theta)))


def matched_filter(img, kernel):
    """Return an image's matched-filter map, NaN as zeros as ``snr_map`` makes it, and its kernel support.

    Parameters
    ----------
    img : ndarray, shape (ny, nx)
        The image, NaN where it has no value.
    kernel : ndarray, shape (m, m)
        The matched filter.

    Returns
    -------
    A : ndarray, shape (ny, nx)
        The correlation of the zero-filled image with the kernel.
    support : ndarray, shape (ny, nx)
        Fraction of the kernel's weight on finite pixels at each pixel.
    """
    from scipy import ndimage
    fin = np.isfinite(img)
    A = ndimage.correlate(np.where(fin, img, 0.0), kernel, mode="nearest")
    support = ndimage.correlate(fin.astype(float), kernel, mode="nearest") / float(np.sum(kernel))
    return A, support


def noise_map_of(out, label, source):
    """Return a variant's expected noise from the frame counts of a pixel-combined reduction.

    Parameters
    ----------
    out : str
        The folder of the reduce step.
    label : str
        The variant whose ``LABEL_clean.fits`` header says how its partitions were combined.
    source : str
        The label whose ``SOURCE_clean_counts.fits`` gives the counts and per-frame noise (the
        same variant reduced with ``--pixcomb``, or the variant itself).

    Returns
    -------
    noise : ndarray, shape (ny, nx)
        `expected_noise`, NaN where no partition has data.
    mode : str
        How the variant's partitions were combined (`combine_mode`).

    Raises
    ------
    SystemExit
        If the counts file is missing.
    """
    cpath = os.path.join(out, f"{source}_clean_counts.fits")
    if not os.path.exists(cpath):
        raise SystemExit(f"{cpath} is missing: reduce {source} with --pixcomb, or pass --counts {label}:SOURCE")
    sig = [float(x) for x in fits.getheader(cpath)["FRAMESIG"].split(",")]
    mode = combine_mode(fits.getheader(os.path.join(out, f"{label}_clean.fits")))
    return expected_noise(fits.getdata(cpath), sig, mode), mode


def sources_file(out, label, tag):
    """Return the path of an injected pass' sources, as `cmd_reduce` names it.

    Parameters
    ----------
    out : str
        The folder of the reduce step.
    label : str
        The variant's label.
    tag : str
        The injected pass' tag.

    Returns
    -------
    str
        ``LABEL_sources.json`` for the tag ``inj``, ``LABEL_TAG_sources.json`` otherwise.
    """
    return os.path.join(out, f"{label}_sources.json" if tag == "inj" else f"{label}_{tag}_sources.json")


def coverage_calibration(mf, ok, cov, bins=COV_BINS, min_px=2000):
    """Return each coverage bin's robust sigma of a whitened matched-filter map over the whole map's.

    With a right noise model every bin gives 1; the ratio is the noise model's error there.

    Parameters
    ----------
    mf : ndarray, shape (ny, nx)
        Matched-filter map of the clean image divided by its expected noise.
    ok : ndarray of bool, shape (ny, nx)
        The pixels used.
    cov : ndarray, shape (ny, nx)
        The run's coverage, the fraction of frames with data at each pixel.
    bins : sequence of float, default COV_BINS
        Coverage bin edges.
    min_px : int, default 2000
        Fewest pixels for a bin's own ratio; sparser bins give NaN.

    Returns
    -------
    list of tuple
        ``(c0, c1, ratio, npix)`` per bin.
    """
    s_all = robust_sigma(mf[ok])
    out = []
    for c0, c1 in zip(bins, bins[1:]):
        m = ok & (cov >= c0) & (cov < c1)
        n = int(m.sum())
        out.append((c0, c1, robust_sigma(mf[m]) / s_all if n >= min_px else float("nan"), n))
    return out


def measure_sources(clean, inj, noise, srcs, kernel, fwhm, pxs, cov, pmask, zone, band=0.5):
    """Measure injected sources: signal from the injected image minus the clean one, noise from the clean.

    Both images are divided by the expected noise and matched-filtered (`matched_filter`). A source's
    signal is the largest value of the difference's map within 0.5 FWHM of its position, so the
    clean image's noise does not enter it. The noise is the robust sigma of the clean image's map
    over the zone's pixels within ``band`` FWHM of the source's radius, off ``pmask`` and with at
    least 90 % of the kernel on data, times the source's coverage bin's calibration
    (`coverage_calibration`), so an error of the noise model with coverage is corrected. The 5 sigma
    contrast is ``5 * contrast / (signal / noise)``. ``pmask`` should leave the low-coverage pixels
    in, or the calibration has no pixels at low coverage and takes 1 there.

    Parameters
    ----------
    clean, inj : ndarray, shape (ny, nx)
        The clean and injected images, NaN outside the zone.
    noise : ndarray, shape (ny, nx)
        The expected noise (`noise_map_of`).
    srcs : sequence of tuple of 3 floats
        ``(rho_arcsec, theta_deg, contrast)`` of each source.
    kernel : ndarray, shape (m, m)
        The matched filter.
    fwhm : float
        FWHM of the PSF, in pixels.
    pxs : float
        Plate scale, in arcsec per pixel.
    cov : ndarray, shape (ny, nx)
        The run's coverage.
    pmask : ndarray of bool, shape (ny, nx)
        Pixels left out of the noise, True to exclude, such as the known companions' keep-outs.
    zone : ndarray of bool, shape (ny, nx)
        The annulus' pixels.
    band : float, default 0.5
        Half-width of the noise band, in FWHM.

    Returns
    -------
    rows : list of dict
        Per source: ``rho``, ``theta``, ``contrast``, ``x``, ``y``, ``cov`` (the run's coverage there),
        ``support`` (kernel fraction on data), ``signal``, ``noise``, ``kappa`` (the calibration
        applied), ``snr`` (signal over noise), ``snr_real`` (the injected image's map at its peak,
        less the band's median, over the noise) and ``c5`` (5 sigma contrast).
    calib : list of tuple
        `coverage_calibration` of the clean map.
    """
    ny, nx = clean.shape
    yy, xx = np.mgrid[:ny, :nx]
    r = np.hypot(xx - (nx - 1) / 2.0, yy - (ny - 1) / 2.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        wc = clean / noise
        wi = inj / noise
    mc, sup = matched_filter(np.where(zone, wc, np.nan), kernel)
    md, _ = matched_filter(np.where(zone, wi - wc, np.nan), kernel)
    mi, _ = matched_filter(np.where(zone, wi, np.nan), kernel)
    ok = zone & np.isfinite(wc) & ~pmask & (sup >= 0.9)
    calib = coverage_calibration(mc, ok, cov)
    rows = []
    for rho, th, c in srcs:
        x0, y0 = source_xy(rho, th, pxs, (ny, nx))
        near = np.hypot(xx - x0, yy - y0) <= 0.5 * fwhm
        near &= np.isfinite(wi)
        rs = float(rho) / float(pxs)
        bnd = ok & (np.abs(r - rs) <= band * fwhm)
        sig = robust_sigma(mc[bnd], min_n=200)
        cv = float(cov[int(round(y0)), int(round(x0))])
        kap = next((k for c0, c1, k, _n in calib if c0 <= cv < c1), float("nan"))
        kap = kap if np.isfinite(kap) else 1.0
        if near.any() and np.isfinite(sig) and sig > 0:
            s = float(np.max(md[near]))
            s_real = float(np.max(mi[near]) - np.median(mc[bnd]))
            snr, snr_real = s / (sig * kap), s_real / (sig * kap)
        else:
            s = snr = snr_real = float("nan")
        rows.append({"rho": float(rho), "theta": float(th), "contrast": float(c), "x": x0, "y": y0, "cov": cv,
                     "support": float(sup[int(round(y0)), int(round(x0))]), "signal": s, "noise": sig,
                     "kappa": kap, "snr": snr, "snr_real": snr_real,
                     "c5": 5.0 * float(c) / snr if np.isfinite(snr) and snr > 0 else float("nan")})
    return rows, calib


def cmd_sensitivity(a):
    """Measure every injected source's 5 sigma contrast in each variant, against a reference variant.

    For each label of ``--labels`` and each injected pass of ``--tags``, the sources are measured
    with `measure_sources`, the noise model from the frame counts (`noise_map_of`, ``--counts``
    naming another label's counts where the variant has none). The noise leaves out the run's
    pixel mask only where the run's coverage is at least its ``min_coverage``, that is the known
    companions' keep-outs and not the low-coverage pixels, whose noise is what is measured. The
    table gives each source's S/N and 5 sigma contrast per variant, each variant's median contrast
    ratio to ``--ref`` overall and by coverage, and each variant's noise-model check by coverage.

    Parameters
    ----------
    a : argparse.Namespace
        The parsed command line of the ``sensitivity`` step.

    Returns
    -------
    None
        Writes ``PREFIX.txt``, ``PREFIX.csv`` and ``PREFIX.png`` into ``--out``, PREFIX ``--prefix``
        (default ``sensitivity``).
    """
    sys.path.insert(0, a.kt_src)
    from klip_tpe.metrics import gaussian_kernel
    setup = run_setup(a.run_dir)
    rec = load_winner(a.run_dir, a.annulus)
    fwhm, pxs = float(setup["fwhm_px"]), float(setup["pxscale"])
    a_in, a_out = (float(v) for v in rec["meta"]["zone"])
    cov = fits.getdata(os.path.join(a.run_dir, "coverage.fits")).astype(float)
    pmask = fits.getdata(os.path.join(a.run_dir, "pixel_mask.fits")).astype(bool)
    min_cov_run = float(setup.get("config", {}).get("min_coverage") or 0.0)
    known = pmask & (cov >= min_cov_run)          # the mask without its low-coverage part
    ny, nx = cov.shape
    yy, xx = np.mgrid[:ny, :nx]
    r = np.hypot(xx - (nx - 1) / 2.0, yy - (ny - 1) / 2.0)
    zone = (r >= a_in) & (r < a_out)
    kernel = gaussian_kernel(fwhm)
    cmap = dict(c.split(":", 1) for c in (a.counts or []))
    ref = a.ref or a.labels[0]
    if ref not in a.labels:
        raise SystemExit("--ref must be one of --labels")
    res, calibs = {}, {}
    for v in a.labels:
        clean = fits.getdata(os.path.join(a.out, f"{v}_clean.fits")).astype(float)
        clean[~zone] = np.nan
        noise, _mode = noise_map_of(a.out, v, cmap.get(v, v))
        res[v] = []
        for t in a.tags:
            inj = fits.getdata(os.path.join(a.out, f"{v}_{t}.fits")).astype(float)
            inj[~zone] = np.nan
            srcs = json.load(open(sources_file(a.out, v, t)))
            rows, calibs[v] = measure_sources(clean, inj, noise, srcs, kernel, fwhm, pxs, cov, known, zone,
                                              band=a.band)
            for row in rows:
                row["tag"] = t
            res[v] += rows
    keys = [(row["tag"], round(row["theta"], 3)) for row in res[ref]]
    for v in a.labels:
        if [(row["tag"], round(row["theta"], 3)) for row in res[v]] != keys:
            raise SystemExit(f"{v}'s injected sources differ from {ref}'s")
    names = {v: v for v in a.labels}
    lines = [f"Annulus {a.annulus} winner {rec['index']}, zone {a_in:.2f}-{a_out:.2f} px, FWHM {fwhm:.2f} px; "
             f"passes {', '.join(a.tags)}; reference {ref}",
             "Signal: the injected image minus the clean one, over the expected noise, matched-filtered (Gaussian "
             "of the FWHM), largest within 0.5 FWHM of the source. Noise: robust sigma of the clean image's map "
             f"within {a.band:g} FWHM of the source's radius, kernel >= 90 % on data, off the pixel mask where the "
             f"coverage is >= {min_cov_run:g} ({int((known & zone).sum())} zone pixels; the low-coverage pixels stay "
             "in), times its coverage bin's calibration. c5 = 5 contrast / (S/N). S/N real: the injected image's "
             "map at the peak.",
             "", "Noise-model check: robust sigma of the clean map by coverage over the whole map's "
             f"(bins {', '.join(f'{c0:g}-{c1:g}' for c0, c1 in zip(COV_BINS, COV_BINS[1:]))}; pixels)"]
    for v in a.labels:
        lines.append(f"  {v:12s} " + "  ".join(
            (f"{k:4.2f}" if np.isfinite(k) else " -- ") + f" ({n})" for _c0, _c1, k, n in calibs[v]))
    lines += ["", "Per source: S/N (signal / noise), [S/N real], 5 sigma contrast", "pass      rho\"    PA    cov  "
              + "  ".join(f"{v:>26s}" for v in a.labels)]
    for i, row0 in enumerate(res[ref]):
        cells = []
        for v in a.labels:
            q = res[v][i]
            cells.append(f"{q['snr']:6.1f} [{q['snr_real']:5.1f}] {q['c5']:9.3e}")
        lines.append(f"{row0['tag']:8s} {row0['rho']:5.2f} {row0['theta']:6.1f}  {row0['cov']:4.2f}  "
                     + "  ".join(f"{c:>26s}" for c in cells))
    groups = (("all", 0.0, 1.01), ("cov >= 0.75", 0.75, 1.01), ("0.4 <= cov < 0.75", 0.4, 0.75),
              ("cov < 0.4", 0.0, 0.4))
    lines += ["", f"Median 5 sigma contrast and median ratio to {ref} (below 1 is deeper), by coverage at the source"]
    for gname, c0, c1 in groups:
        sel = [i for i, q in enumerate(res[ref]) if c0 <= q["cov"] < c1]
        if not sel:
            continue
        parts = []
        for v in a.labels:
            c5 = np.array([res[v][i]["c5"] for i in sel])
            rat = c5 / np.array([res[ref][i]["c5"] for i in sel])
            parts.append(f"{v} {np.nanmedian(c5):.3e} ({np.nanmedian(rat):.2f})")
        lines.append(f"  {gname:18s} n={len(sel):2d}: " + ", ".join(parts))
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(a.out, f"{a.prefix}.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(a.out, f"{a.prefix}.csv"), "w") as f:
        cols = ["label", "tag", "rho", "theta", "contrast", "x", "y", "cov", "support", "signal", "noise", "kappa",
                "snr", "snr_real", "c5"]
        f.write(",".join(cols) + "\n")
        for v in a.labels:
            for q in res[v]:
                f.write(",".join([v] + [str(q[c]) if isinstance(q[c], str) else f"{q[c]:.6g}" for c in cols[1:]])
                        + "\n")
    plot_sensitivity(os.path.join(a.out, f"{a.prefix}.png"), res, a.labels, ref, a.annulus,
                     {**names, **dict(n.split(":", 1) for n in (a.names or []))}, a.legend_title)


def plot_sensitivity(path, res, labels, ref, annulus, names, legend_title=None):
    """Draw each variant's 5 sigma contrast and its ratio to the reference against coverage.

    Parameters
    ----------
    path : str
        Output PNG.
    res : dict of str to list of dict
        `measure_sources` rows per label, in the same source order.
    labels : sequence of str
        The variants, at most ``len(SERIES)``, in the order of the colors.
    ref : str
        The reference label.
    annulus : int
        Annulus, counted from 1, for the title.
    names : dict of str to str
        Legend names per label.
    legend_title : str or None, default None
        Title of both legends.

    Returns
    -------
    None
        Writes the PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if len(labels) > len(SERIES):
        raise SystemExit(f"at most {len(SERIES)} labels in the figure")
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 5.4))
    c_ref = np.array([q["c5"] for q in res[ref]])
    for j, v in enumerate(labels):
        col, mk = SERIES[j]
        cv = np.array([q["cov"] for q in res[v]])
        c5 = np.array([q["c5"] for q in res[v]])
        ax[0].plot(cv, c5, mk, ms=7, mfc=col, mec="white", mew=0.8, color=col, label=names.get(v, v), ls="none")
        if v != ref:
            ax[1].plot(cv, c5 / c_ref, mk, ms=7, mfc=col, mec="white", mew=0.8, color=col, label=names.get(v, v),
                       ls="none")
    ax[1].axhline(1.0, color="#8a8a86", lw=1.0)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("Coverage at the Source")
    ax[0].set_ylabel("5σ Contrast")
    ax[0].set_title("5σ Contrast at the Injected Sources")
    ax[1].set_xlabel("Coverage at the Source")
    ax[1].set_ylabel(f"5σ Contrast / {names.get(ref, ref)}")
    ax[1].set_title(f"Relative to {names.get(ref, ref)} (Below 1 Is Deeper)")
    for axx in ax:
        axx.grid(True, color="#e4e3df", lw=0.6)
        axx.set_axisbelow(True)
        for sp in axx.spines.values():
            sp.set_visible(True)
        axx.set_xlim(0, 1.02)
    handles, names_ = ax[0].get_legend_handles_labels()
    fig.legend(handles, names_, loc="lower center", ncol=min(len(labels), 3), fontsize=8.5, frameon=True,
               title=legend_title, title_fontsize=8.5)
    fig.suptitle(f"Annulus {annulus} Winner: Injection-Recovery Sensitivity", fontsize=12)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def cmd_recombine(a):
    """Combine a ``--pixcomb`` reduction's partition images with another mode's weights, without reducing.

    For each pass of ``--tags``, ``SOURCE_TAG_stack.fits`` is combined with `recombine`, the counts
    and per-frame noise of ``SOURCE_clean_counts.fits`` (the clean pass' noise for every pass, as
    a run would weight its injected images like its clean one), and written as ``LABEL_TAG.fits``
    with its ``COMBINE`` card; the clean pass also writes ``LABEL_clean_counts.fits`` and every
    injected pass its sources' file, so the other steps read the new label as a reduction.

    Parameters
    ----------
    a : argparse.Namespace
        The parsed command line of the ``recombine`` step.

    Returns
    -------
    None
        Writes the files above into ``--out``, refusing to overwrite.
    """
    import shutil
    card = {v: k for k, v in COMBINE_MODES.items()}[a.mode]
    cpath = os.path.join(a.out, f"{a.source}_clean_counts.fits")
    ch = fits.getheader(cpath)
    counts = fits.getdata(cpath)
    sig = [float(x) for x in ch["FRAMESIG"].split(",")]
    for tag in a.tags:
        spath = os.path.join(a.out, f"{a.source}_{tag}_stack.fits")
        sh = fits.getheader(spath)
        img = recombine(fits.getdata(spath), counts, sig, a.mode)
        hdr = {k: sh[k] for k in ("ANNULUS", "WINNER", "PARTS", "INRAD", "OUTRAD") if k in sh}
        write_image(os.path.join(a.out, f"{a.label}_{tag}.fits"), img,
                    dict(hdr, VARIANT=str(sh.get("VARIANT", "")), LABEL=a.label, SOURCE=a.source, COMBINE=card))
        if tag == "clean":
            write_image(os.path.join(a.out, f"{a.label}_clean_counts.fits"), counts,
                        dict(hdr, LABEL=a.label, SOURCE=a.source, FRAMESIG=ch["FRAMESIG"]))
        else:
            src, dst = sources_file(a.out, a.source, tag), sources_file(a.out, a.label, tag)
            if os.path.exists(dst):
                raise SystemExit(f"{dst} exists; not overwriting")
            shutil.copyfile(src, dst)
        print(f"{a.label} {tag}: {a.mode} weights from {a.source}, finite {np.isfinite(img).mean():.3f}")


def cmd_figure(a):
    """Draw one variant before and another after: clean image, its S/N map, the injected image's S/N map.

    The before column shows KLIP-TPE's S/N map (``snr_map`` of the image), as the run's products
    show it; the after column the noise-map S/N (``snr_map`` of the image over its expected noise
    relative to the zone's median), blank where the run's coverage is below ``--min-cov``. Both
    clean images are in units of the before image's robust sigma where the run's coverage is at
    least 0.97, with the run's coverage contour at ``--cov-contour``, and the injected sources are
    circled.

    Parameters
    ----------
    a : argparse.Namespace
        The parsed command line of the ``figure`` step.

    Returns
    -------
    None
        Writes ``before_after.png`` into ``--out``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, a.kt_src)
    from klip_tpe.products import snr_map
    setup = run_setup(a.run_dir)
    rec = load_winner(a.run_dir, a.annulus)
    fwhm, pxs = float(setup["fwhm_px"]), float(setup["pxscale"])
    a_in, a_out = (float(v) for v in rec["meta"]["zone"])
    cov = fits.getdata(os.path.join(a.run_dir, "coverage.fits")).astype(float)
    pmask = fits.getdata(os.path.join(a.run_dir, "pixel_mask.fits")).astype(bool)
    ny, nx = cov.shape
    yy, xx = np.mgrid[:ny, :nx]
    r = np.hypot(xx - (nx - 1) / 2.0, yy - (ny - 1) / 2.0)
    zone = (r >= a_in) & (r < a_out)
    cmap = dict(c.split(":", 1) for c in (a.counts or []))
    cols = []
    for v, noise_norm in ((a.before, False), (a.after, True)):
        clean = fits.getdata(os.path.join(a.out, f"{v}_clean.fits")).astype(float)
        inj = fits.getdata(os.path.join(a.out, f"{v}_inj.fits")).astype(float)
        clean[~zone] = np.nan
        inj[~zone] = np.nan
        srcs = json.load(open(sources_file(a.out, v, "inj")))
        ex = [source_xy(rho, th, pxs, (ny, nx)) for rho, th, _c in srcs]
        if noise_norm:
            en, _mode = noise_map_of(a.out, v, cmap.get(v, v))
            rel = en / np.nanmedian(en[zone])
            rel = np.where(np.isfinite(rel), rel, 1.0)
            low = cov < float(a.min_cov)
        else:
            rel = np.ones_like(clean)
            low = np.zeros(cov.shape, bool)
        s_clean = snr_map(np.where(low, np.nan, clean) / rel, fwhm, pixel_mask=pmask)
        s_inj = snr_map(np.where(low, np.nan, inj) / rel, fwhm, pixel_mask=pmask, exclude_xy=ex)
        cols.append((v, clean, s_clean, s_inj, ex, noise_norm))
    s0 = robust_sigma(cols[0][1][zone & (cov >= 0.97)])
    ys, xs = np.where(zone)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    fig, ax = plt.subplots(3, 2, figsize=(9.6, 14.2))
    titles = (a.before_title or f"Before: {a.before}", a.after_title or f"After: {a.after}")
    for j, (v, clean, s_clean, s_inj, ex, nn) in enumerate(cols):
        what = "S/N, Noise-Map Normalized" if nn else "S/N, KLIP-TPE Map"
        panels = ((clean / s0, -5, 5, f"{titles[j]}\nClean Image (Dashed: Coverage {a.cov_contour:g})",
                   "Image / Before's σ at Full Coverage"),
                  (s_clean, -5, 5, f"{what}, Clean", "S/N"),
                  (s_inj, -5, 5, f"{what}, Injected Sources", "S/N"))
        for i, (m, lo, hi, title, cbl) in enumerate(panels):
            axx = ax[i, j]
            im = axx.imshow(m[y0:y1, x0:x1], origin="lower", cmap="RdBu_r", vmin=lo, vmax=hi,
                            extent=(x0 - 0.5, x1 - 0.5, y0 - 0.5, y1 - 0.5))
            if i == 0 and a.cov_contour > 0:
                cz = np.where(zone, cov, np.nan)[y0:y1, x0:x1]
                axx.contour(np.arange(x0, x1), np.arange(y0, y1), cz, levels=[a.cov_contour], colors="#3d3d3a",
                            linewidths=0.9, linestyles="--")
            if i == 2:
                for ex_x, ex_y in ex:
                    axx.add_patch(plt.Circle((ex_x, ex_y), 1.5 * fwhm, fill=False, ec="black", lw=0.8))
            axx.set_xticks([])
            axx.set_yticks([])
            for sp in axx.spines.values():
                sp.set_visible(True)
            axx.set_title(title, fontsize=10)
            cb = fig.colorbar(im, ax=axx, fraction=0.046, pad=0.02)
            cb.set_label(cbl, fontsize=9)
    fig.suptitle(f"Annulus {a.annulus} Winner: Edge Artifacts Before and After (North Up, East Left)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "before_after.png"), dpi=100)
    plt.close(fig)


def stage_profile(frames, a_in, a_out, margin=15.0, dmax=30, dint=40.0):
    """Profile a stack of frames against the distance to their NaN inside the KLIP zone.

    Each frame's pixels between ``a_in + margin`` and ``a_out - margin`` from the star are binned by
    their distance to the nearest NaN pixel of the frame that lies inside the zone (the detector's
    edge, not the zone's own complement), and normalized by the frame's pixels at ``dint`` or more.

    Parameters
    ----------
    frames : ndarray, shape (n, ny, nx)
        The frames, NaN where they have no value.
    a_in, a_out : float
        The zone's radii, in pixels.
    margin : float, default 15.0
        Pixels kept away from the zone's radii.
    dmax : int, default 30
        Distance bins are 1 px wide up to ``dmax``.
    dint : float, default 40.0
        Edge distance, in pixels, of the interior used for the normalization.

    Returns
    -------
    list of tuple
        (bin, count, median / sigma, robust sigma ratio) for bins 1 to ``dmax``.
    """
    from scipy import ndimage
    n, ny, nx = frames.shape
    yy, xx = np.mgrid[:ny, :nx]
    r = np.hypot(xx - (nx - 1) / 2.0, yy - (ny - 1) / 2.0)
    zone = (r >= a_in) & (r <= a_out)
    band = (r >= a_in + margin) & (r <= a_out - margin)
    bins = [[] for _ in range(dmax + 1)]
    for i in range(n):
        f = np.asarray(frames[i], float)
        fin = np.isfinite(f)
        if fin[zone].all():
            continue
        d = ndimage.distance_transform_edt(fin | ~zone)
        sel = band & fin
        v, dd = f[sel], d[sel]
        inner = dd >= dint
        if inner.sum() < 100:
            continue
        med = np.median(v[inner])
        sig = 1.4826 * np.median(np.abs(v[inner] - med))
        if not sig > 0:
            continue
        v = (v - med) / sig
        b = np.floor(dd).astype(int)
        for k in range(1, dmax + 1):
            bins[k].append(v[b == k])
    out = []
    for k in range(1, dmax + 1):
        v = np.concatenate(bins[k]) if bins[k] else np.zeros(0)
        if v.size:
            out.append((k, v.size, float(np.median(v)), float(1.4826 * np.median(np.abs(v - np.median(v))))))
    return out


def cmd_stages(a):
    """Profile each step of one partition's reduction against the distance to the detector's edge.

    The reduction of the winner's partition ``--partition`` is run with hooks on klip-tpe's
    ``klip_annular``, the residuals' second high-pass and the derotation, and each step's frames
    are profiled with `stage_profile`: the high-passed binned frames, the KLIP residuals, the
    residuals after the second high-pass, and the derotated frames.

    Parameters
    ----------
    a : argparse.Namespace
        The parsed command line of the ``stages`` step.

    Returns
    -------
    None
        Prints the profiles and writes ``LABEL_stages.txt`` into ``--out``.
    """
    sys.path.insert(0, a.kt_src)
    install(a.variant)
    from klip_tpe import reducer as kr
    setup = run_setup(a.run_dir)
    rec = load_winner(a.run_dir, a.annulus)
    sel = [a.partition]
    red = build_reducer(a.inputs, setup, sel, a.workers)
    cfg = winner_config(rec, setup, sel, parse_params(a.param))
    cap = {}
    k0, h0, d0 = kr.klip_annular, kr.highpass_stack, kr.derotate

    def k_hook(cube, angles, *args, **kw):
        """Run ``klip_annular`` and keep its input frames and residuals."""
        out = k0(cube, angles, *args, **kw)
        cap["1 high-passed binned frames"] = np.array(cube, float)
        cap["2 KLIP residuals"] = np.array(out[0], float)
        return out

    def h_hook(cube, *args, **kw):
        """Run ``highpass_stack`` and keep the residuals' second high-pass."""
        out = h0(cube, *args, **kw)
        if "2 KLIP residuals" in cap and "3 after the second high-pass" not in cap:
            cap["3 after the second high-pass"] = np.array(out, float)
        return out

    def d_hook(cube, *args, **kw):
        """Derotate as klip-tpe does and keep the first derotated cube."""
        out = d0(cube, *args, **kw)
        if "4 derotated" not in cap:
            cap["4 derotated"] = np.array(out, float)
        return out

    kr.klip_annular, kr.highpass_stack, kr.derotate = k_hook, h_hook, d_hook
    red.reduce_config(cfg, None)
    a_in, a_out = (float(v) for v in rec["meta"]["zone"])
    label = a.label or a.variant
    lines = [f"{label}: partition {a.partition}, zone {a_in:.2f}-{a_out:.2f} px; pixels 15 px or more inside the "
             f"zone, by distance to the frame's NaN inside the zone: robust sigma / interior, median / interior sigma"]
    for name in sorted(cap):
        prof = stage_profile(cap[name], a_in, a_out)
        lines.append(f"  {name} ({cap[name].shape[0]} frames): " + " ".join(
            f"{k}:{sg:.2f}/{md:+.2f}" for k, n, md, sg in prof if k <= 16))
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(a.out, f"{label}_stages.txt"), "x") as f:
        f.write(text + "\n")


def main(argv=None):
    """Parse the command line and run its step: ``reduce``, ``stages``, ``compare``, ``sensitivity`` or ``figure``.

    Parameters
    ----------
    argv : list of str or None, default None
        Command-line arguments; None reads ``sys.argv[1:]``.

    Returns
    -------
    None
        The step's files are written.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="step", required=True)
    p = sub.add_parser("reduce", help="reduce the winner with one variant")
    p.add_argument("--kt-src", required=True, help="directory holding the klip_tpe package")
    p.add_argument("--run-dir", required=True, help="the finished run")
    p.add_argument("--inputs", required=True, help="the run's inputs folder, with manifest.json")
    p.add_argument("--annulus", type=int, required=True, help="annulus, counted from 1")
    p.add_argument("--variant", required=True,
                   help=f"one of {', '.join(VARIANTS)}, or patches of {', '.join(PATCHES)} joined with '+'")
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--inject", action="store_true", help="also reduce with the winner's injected sources")
    p.add_argument("--inj-tag", default="inj", help="name of the injected pass' files (default inj)")
    p.add_argument("--rotate-sources", type=float, default=0.0, metavar="DEG",
                   help="degrees added to the injected sources' position angles (default 0)")
    p.add_argument("--skip-clean", action="store_true", help="reduce only the injected pass")
    p.add_argument("--extra-sources", default=None, help="file of extra sources: rho_arcsec theta_deg contrast")
    p.add_argument("--workers", type=int, default=2, help="worker threads (default 2)")
    p.add_argument("--param", action="append", default=None, metavar="KEY=VALUE",
                   help="a reducer parameter over the winner's, e.g. post_hp=false or comb_type=mean")
    p.add_argument("--label", default=None, help="output name in place of the variant's")
    p.add_argument("--pixcomb", action="store_true",
                   help="combine the partitions with per-pixel inverse-variance weights (pixel_combine)")
    p.set_defaults(func=cmd_reduce)
    p = sub.add_parser("stages", help="profile each step of one partition's reduction near the detector edge")
    p.add_argument("--kt-src", required=True, help="directory holding the klip_tpe package")
    p.add_argument("--run-dir", required=True, help="the finished run")
    p.add_argument("--inputs", required=True, help="the run's inputs folder, with manifest.json")
    p.add_argument("--annulus", type=int, required=True, help="annulus, counted from 1")
    p.add_argument("--partition", required=True, help="one of the winner's partitions")
    p.add_argument("--variant", required=True, help="as reduce's --variant")
    p.add_argument("--param", action="append", default=None, metavar="KEY=VALUE")
    p.add_argument("--label", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=2)
    p.set_defaults(func=cmd_stages)
    p = sub.add_parser("compare", help="compare the variants written to --out")
    p.add_argument("--kt-src", required=True, help="directory holding the klip_tpe package")
    p.add_argument("--run-dir", required=True, help="the finished run")
    p.add_argument("--annulus", type=int, required=True, help="annulus, counted from 1")
    p.add_argument("--out", required=True, help="the folder of the reduce step")
    p.add_argument("--only", nargs="+", default=None, help="compare only these labels")
    p.add_argument("--counts", action="append", default=None, metavar="LABEL:SOURCE",
                   help="take LABEL's frame counts from SOURCE's *_clean_counts.fits (same partition images)")
    p.set_defaults(func=cmd_compare)
    p = sub.add_parser("sensitivity", help="5 sigma contrast at the injected sources, per variant")
    p.add_argument("--kt-src", required=True, help="directory holding the klip_tpe package")
    p.add_argument("--run-dir", required=True, help="the finished run")
    p.add_argument("--annulus", type=int, required=True, help="annulus, counted from 1")
    p.add_argument("--out", required=True, help="the folder of the reduce step")
    p.add_argument("--labels", nargs="+", required=True, help="the variants, the figure's colors in this order")
    p.add_argument("--tags", nargs="+", default=["inj"], help="the injected passes (default inj)")
    p.add_argument("--ref", default=None, help="the reference label (default the first)")
    p.add_argument("--band", type=float, default=0.5, help="half-width of the noise band, in FWHM (default 0.5)")
    p.add_argument("--counts", action="append", default=None, metavar="LABEL:SOURCE",
                   help="take LABEL's frame counts from SOURCE's *_clean_counts.fits")
    p.add_argument("--names", action="append", default=None, metavar="LABEL:NAME", help="legend name of a label")
    p.add_argument("--legend-title", default=None, help="title of the figure's legends")
    p.add_argument("--prefix", default="sensitivity", help="name of the output files (default sensitivity)")
    p.set_defaults(func=cmd_sensitivity)
    p = sub.add_parser("recombine", help="combine a --pixcomb reduction's partition images with other weights")
    p.add_argument("--out", required=True, help="the folder of the reduce step")
    p.add_argument("--source", required=True, help="label of a --pixcomb reduction (its stacks and counts)")
    p.add_argument("--label", required=True, help="label of the new images")
    p.add_argument("--mode", required=True, choices=("pixel", "coverage", "equal"), help="the weights")
    p.add_argument("--tags", nargs="+", default=["clean", "inj"], help="passes to combine (default clean inj)")
    p.set_defaults(func=cmd_recombine)
    p = sub.add_parser("figure", help="one variant before and another after")
    p.add_argument("--kt-src", required=True, help="directory holding the klip_tpe package")
    p.add_argument("--run-dir", required=True, help="the finished run")
    p.add_argument("--annulus", type=int, required=True, help="annulus, counted from 1")
    p.add_argument("--out", required=True, help="the folder of the reduce step")
    p.add_argument("--before", required=True, help="label of the before column (KLIP-TPE's S/N map)")
    p.add_argument("--after", required=True, help="label of the after column (noise-map S/N)")
    p.add_argument("--before-title", default=None, help="the before column's title")
    p.add_argument("--after-title", default=None, help="the after column's title")
    p.add_argument("--min-cov", type=float, default=0.05,
                   help="the after column's S/N maps are blank where the run's coverage is below this (default 0.05)")
    p.add_argument("--cov-contour", type=float, default=0.5,
                   help="coverage of the dashed contour on the clean images, 0 for none (default 0.5)")
    p.add_argument("--counts", action="append", default=None, metavar="LABEL:SOURCE",
                   help="take LABEL's frame counts from SOURCE's *_clean_counts.fits")
    p.set_defaults(func=cmd_figure)
    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
