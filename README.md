# reduce_nomic

`reduce_nomic` is a Python port of `reduce_nomic.pro`, the IDL imaging reduction for LBTI/NOMIC (2026-09-14 version). It reduces one night of raw frames to aligned, centered, frame-selected and time-binned cubes for each chop position. Each cube comes with parallactic angles and median PSFs, under the IDL product names that KLIP-TPE reads.

## Install

```
pip install -e .            # numpy, scipy, astropy
```

## Run

```
reduce-nomic Procyon_20230406 --workers 16 --archive /mnt/RAID64TB/NOMIC
reduce-nomic Procyon_17 --dry-run                     # the same night by its old name; settings and frames only
reduce-nomic Procyon_20230406 --set radius=400 --set sky_dirs='["skys1"]'
reduce-nomic Procyon_20230406 --workers 16 --display  # live view (pip install -e '.[display]')
```

- **Night names:** the archive's night folders were renamed on 2026-11-20 from the IDL predecessor's names (`Procyon_17`) to `<target>_<UT date>` (`Procyon_20230406`); `<archive>/folder_name_map.csv` lists both and `FOLDER_NAMES.md` beside it explains the convention. A night can be given by either name (`reduce_nomic.names`). Its settings and the sorting's move log are looked up under the old name, and new products are named with the current one.

- **Progress:** every stage prints progress lines, in frames or segments, with an estimate of the time remaining.
- **Live view:** `--display` opens a window of the KLIP-TPE live window's size, 1850 x 990 pixels (shrunk to fit a smaller screen, fixed size). The top row follows the chop identification of each round: the image the latest frame's chop state was decided from (rolling background subtracted, destriped, masked) with its detected star, the detected star positions colored by chop state, the value that decides the chop state over the last 600 frames, and every frame's chop-ID peak S/N (frames with no star in the searched region show as a dip and are skipped, in gray). The bottom row shows the chop-subtracted frame and its high-pass, each chunk's centered median and its frame selection; a strip below lists every stage's progress. The window is redrawn in place at most once per second without being raised or taking the focus, and the final view is written to `<night>_<side>_display.png`. It is drawn by its own process, so it stays responsive, and closing it or losing the X connection (which ends any Tk program) does not stop the reduction. Without an X display (or Tk) only that PNG is written, at most every 10 s. `python3 tools/preview_display.py NEW_DIR` renders the view for a synthetic night.
- **Reruns:** a run refuses to start if its outputs already exist.

- **Settings:** start from the IDL defaults, then apply the night's object block from `reduce_nomic.pro`, which `tools/extract_idl_settings.py` extracts to `config/nights_idl.json`. `--set` overrides any setting by name.
- **Outputs:** go to `<archive>/<night>/reduced_py/`, `<night>` being the folder's current name, and never overwrite an existing file.
  - **For KLIP-TPE**, which makes the final frame selection itself with its `corr_thresh`, `noise_max` and `coronoise_max`: every binned frame after the pre-clean.
    - `<night>_<side>_cube<n>_align_cen_bin.fits`, peak-scaled.
    - `_angles<n>_bin.fits`.
    - `_frametags<n>_bin.fits` / `.csv`, the frame tags. `corrs`, `noises` and `coronoise` are in KLIP-TPE's names and units; the raw measures and the IDL clean's decision (`idl_clean_keep`) are also included.
  - **With the IDL names**, after the IDL clean stage, for comparison: `_cube<n>_align_cen_clean_bin.fits`, `_angles<n>_clean_bin.fits` / `.csv` and `_cube<n>_align_cen_clean_median_PSF.fits`.
  - `_summary.json`.
- **Intermediates:** memory-mapped files in `reduced_py/work_<side>/`, which can be deleted after a run.

## Stages

| Module | IDL step | What it does |
|---|---|---|
| `combine` | `do_combine` | Bad-pixel map, chop state of each frame from the star's position, coadds within each chop |
| `chopsub` | `do_chop_sub` | Subtraction of the neighboring chops, channel-seam and polynomial background, `fixpix`, channel-edge NaNs, sky subtraction |
| `geometry` | `do_split`, `do_align`, `do_center` | Cut 2·radius around each chop's star, cross-correlation alignment, centroid centering at (radius − 0.5, radius − 0.5) |
| `select` | `do_pre_clean`, `do_clean` | Pre-clean rejection by correlation, noise and peak; the clean's measures as frame tags; peak scaling |
| `binning` | `do_pre_binning`, `do_replace` | Median bins of `prebin` frames, `fixpix` of interior NaNs |
| `idl` | helper routines | `smooth`, `median`, `fshift`, `crosscorr`, `cntrd`, `fixpix`, `sfit`, `psf_gaussian` |

**Parallelism.** Blocks of frames, segments and chunks of `frame_limit` frames run in parallel worker processes (`--workers`). Each worker uses one core, so the worker count sets the speed. A split-to-pre-bin task takes about 2.5 GB at radius 500 (see `docs/PORTING.md`), and that stage lowers its worker count, with a log line, when the workers would need more than 75 % of the memory available. On patrick (96 cores, 384 GB), `--workers 64` leaves a third of the cores free.

**Frame selection.** Frames are selected by frame number. An IDL position cut is translated through the archive's sorting log, as `docs/PORTING.md` explains.

## Tests

```
python3 -m pytest -q
```

The tests cover each IDL primitive and an end-to-end synthetic night with a chopping star. In that night, the final PSFs are centered within 0.05 px of (499.5, 499.5).
