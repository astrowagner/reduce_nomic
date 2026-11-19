# Porting Notes: reduce_nomic.pro to Python

These notes list how the port matches the IDL predecessor (`reduce_nomic.pro`, 2026-09-14 version) and where it differs.

## Reproduced as in IDL

These quirks change results, so the port keeps them by default:

| Behavior | IDL lines | Port |
|---|---|---|
| `remove_oddeven_noise` parses `i+j mod 2` as `i + (j mod 2)`, so it acts as a 2×2 box smoothing | 4939–4958 | `chopsub.remove_oddeven(idl_mode=True)`; `idl_oddeven=False` applies the intended rule |
| `smooth_bg` is assigned to the misspelled `begsubframe` and has no effect | 4128 | not applied |
| Every combine round rewrites the bad-pixel map, so the cosmetics of all rounds use the last sky round's map | 3920, 4450 | `pipeline.reduce` uses `rounds[-1]["badmap"]` |
| Sky rounds 4–6 reread `skys3/` | 3779–3782 | default `sky_dirs` reproduces it; set `sky_dirs` explicitly to fix |
| Chop A frames get the chop B mean sky unless `flip_sky_chop` | 5161–5167 | `chopsub.subtract_sky` |
| A segment's `INUM`, `DATE` and `EFF_EXPT` come from the frame that closes it (the next segment's first) | 4332–4344 | `combine.build_segments` |
| The last segment of a round is never written | 4281 | same |
| In sky rounds, segments of two or more frames are set to NaN (`limit_sky_to_one`) | 4316–4326 | same |
| `filter_auto_pos` uses the first frame's zero map for both frames | 5636–5637 | same |
| The align reference is built with `cntrd` near the hard-coded (500, 500). For radius 400 this fails, and the reference is centered at (400.5, 400.5) | 6151–6164 | `geometry.gaussian_reference` |
| Positions, centers, pre-clean statistics and bins are computed per chunk of `frame_limit` frames | throughout | same chunk boundaries |
| `fixpix` looks up distances with the clipped box's indices near the image edges | `fixpix.pro` | `idl.FixPix` |
| `coadds` are not binned or matched to the binned frames | 7295–7304, 7684 | same; written to `_coadds<n>_clean_bin.csv` |
| IDL `median` without `/even` takes the upper middle value; `medarr` uses `/even` | | `idl.median(even=...)` as each call site does |

## Differences

**Frame selection.** IDL cuts the file list by position (`start_frame_left`, ...). The archive was sorted on 2026-11-17/18, which moved files into and out of several `raw/` folders. The port therefore rebuilds the pre-sorting list from the move log, finds the frame number at the IDL position, and keeps the frames still in the folder from that number on. `--frames FIRST LAST` selects explicitly.

**File order.** IDL's `FILE_SEARCH(/nosort)` lists files unsorted on patrick's RAID. The port always sorts by stem and frame number, as the Mac's volume did.

**Random draws.** The center stage takes the median of 100 frames drawn with IDL `randomu`. The port draws with a seeded numpy generator, so its centers differ from IDL's by the scatter between random medians.

**`fix_pa`.** On the five nights that set it (β UMa_5, β UMa_6, Procyon_7–9), the port recomputes the parallactic angle from DATE-OBS and TIME-END with astropy's mean sidereal time and IDL `parangle`'s formula. IDL's `ct2lst` takes the local hour and the time zone, which gives back the same UTC instant, so the two agree to well under a second of sidereal time.

**Pauses.** No step pauses for a key (`hak`). `checksky` logs a warning instead of pausing.

**Overwrites.** Outputs go to `reduced_py/`, and existing files are never overwritten. IDL deletes and rewrites `chop/` and `reduced/`.

**Precision.** The port computes in double precision and stores cubes as float32. IDL mostly computes in single precision.

**Not ported.** These options are not ported because no night turns them on: `priorpoly_bgsub`, `midpoly_bgsub`, `fix_bad_channel`, `polyfit_edges`, `subtract_crosstalk`, `flatten_*`, `mask_stripes`, `zero_edges`, `nan_bad_region`, `pca_background`, `destripe_*_at_split`, `rot_center` and the clean stage's `coronoise_clean`. The pre-clean's `pre_coronoise_clean` is ported (five nights use it). The pipeline refuses settings that turn one on.

## Validation Status

**Tests.**
- `tests/test_idl.py` checks every primitive against its IDL definition.
- `tests/test_pipeline_synthetic.py` reduces a synthetic chopping star end to end. The final PSFs come out centered at (499.505, 499.544) and (499.496, 499.500), against an expected (499.5, 499.5), with FWHM 15.1 px against 15.0 px injected.

**Real data (2026-11-18, the sandbox on patrick).**
- **`fix_pa`:** the recomputed parallactic angles of 25 Procyon_17 frames match LBT_PARA to within 0.001–0.010° over 11–54°.
- **Frame selection:** the IDL start positions translate to the expected frame numbers. ε Eri starts at 3900, β Leo_9 at 3000, Procyon_22 at 2760, τ Cet_2 at 638 (its acquisition frames were moved out) and β UMa_8 at 156895.
- **Smoke run:** Procyon_13, left aperture (double-sided, left-right chop), 120 raw frames (20000–20119) with `prebin=5`, ran end to end in 11 s on 2 cores.
  - The median-PSF cores (64×64) correlate at 0.9965 (chop A) and 0.9978 (chop B) with the IDL full-night PSFs.
  - Gaussian FWHM: 20.8 and 19.1 px in the port, against 21.9 and 20.0 px in IDL. The port's are sharper, as expected from 10 s of data against a whole night.

**Still to do.** Reduce full nights on patrick and compare with the IDL products: frame counts, angles, the median PSF, and the frames kept by the selection.

**Memory.** A worker holds one chunk of up to 1000 frames (1 GB) plus one chop's cut cube (up to about 2 GB at radius 500). Plan about 4 GB per worker.
