# reduce_nomic

`reduce_nomic` is a Python port of `reduce_nomic.pro`, the IDL imaging reduction for LBTI/NOMIC (2026-09-14 version). It reduces one night of raw frames to aligned, centered, frame-selected and time-binned cubes for each chop position. Each cube comes with parallactic angles and median PSFs, under the IDL product names that KLIP-TPE reads.

## Install

```
pip install -e .            # numpy, scipy, astropy
```

## Run

```
reduce-nomic Procyon_17 --workers 16 --archive /mnt/RAID64TB/NOMIC
reduce-nomic Procyon_17 --dry-run                     # settings and frame selection only
reduce-nomic Procyon_17 --set radius=400 --set sky_dirs='["skys1"]'
```

- **Settings:** start from the IDL defaults, then apply the night's object block from `reduce_nomic.pro`, which `tools/extract_idl_settings.py` extracts to `config/nights_idl.json`. `--set` overrides any setting by name.
- **Outputs:** go to `<archive>/<night>/reduced_py/` and never overwrite an existing file. They use the IDL names:
  - `<night>_<side>_cube<n>_align_cen_clean_bin.fits`
  - `_angles<n>_clean_bin.fits` / `.csv`
  - `_cube<n>_align_cen_clean_median_PSF.fits`
  - `_summary.json`
- **Intermediates:** memory-mapped files in `reduced_py/work_<side>/`, which can be deleted after a run.

## Stages

| Module | IDL step | What it does |
|---|---|---|
| `combine` | `do_combine` | Bad-pixel map, chop state of each frame from the star's position, coadds within each chop |
| `chopsub` | `do_chop_sub` | Subtraction of the neighboring chops, channel-seam and polynomial background, `fixpix`, channel-edge NaNs, sky subtraction |
| `geometry` | `do_split`, `do_align`, `do_center` | Cut 2·radius around each chop's star, cross-correlation alignment, centroid centering at (radius − 0.5, radius − 0.5) |
| `select` | `do_pre_clean`, `do_clean` | Rejection by correlation, noise and peak; peak scaling |
| `binning` | `do_pre_binning`, `do_replace` | Median bins of `prebin` frames, `fixpix` of interior NaNs |
| `idl` | helper routines | `smooth`, `median`, `fshift`, `crosscorr`, `cntrd`, `fixpix`, `sfit`, `psf_gaussian` |

**Parallelism.** Blocks of frames, segments and chunks of `frame_limit` frames run in parallel worker processes (`--workers`).

**Frame selection.** Frames are selected by frame number. An IDL position cut is translated through the archive's sorting log, as `docs/PORTING.md` explains.

## Tests

```
python3 -m pytest -q
```

The tests cover each IDL primitive and an end-to-end synthetic night with a chopping star. In that night, the final PSFs are centered within 0.05 px of (499.5, 499.5).
