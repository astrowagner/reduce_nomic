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

**Final frame selection.** In this pipeline, KLIP-TPE makes the final frame selection: its optimizer searches `corr_thresh`, `noise_max` and `coronoise_max` on per-frame tags. So the port runs the IDL clean stage's measures on every binned frame but does not drop frames.
- It writes all binned frames (`*_cube<n>_align_cen_bin.fits`, replaced and peak-scaled as the clean stage would) with their angles.
- It writes the frame tags (`*_frametags<n>_bin.fits` / `.csv`):
  - `corrs`: the clean's correlation with the mean of the frames within 50;
  - `noises`: the clean's noise outside the central 100 columns, over its median;
  - `coronoise`: the standard deviation from 0.3 to 0.65″, the pre-clean's coronoise region, over its median;
  - `idl_clean_keep`: the IDL clean's decision with the night's thresholds.
- The IDL-named `*_clean_bin` products, which apply that decision, are still written for comparison with IDL.
- The pre-clean, on the unbinned frames before the pre-binning, still rejects frames as IDL does.

**Chop-ID destriping** (`chopid_destripe`, on by default). Before the masks, the port subtracts from each row of the chop-identification image its median over each channel half (x < 256 and x ≥ 256, skipping the border and blocked columns). This removes the horizontal stripes and channel-edge offsets that the rolling background leaves. The IDL predecessor has a commented-out `destripe` call at this point. The stripe-mean image of `set_chop_via_bg` is not destriped. `--set chopid_destripe=0` reproduces the IDL chop identification exactly.

On 1,600 sampled Procyon_13 frames (40 blocks of 20 over the night, both apertures):
- **Positions:** 1,550 star positions are unchanged and 33 move by 1 px. Seven move by 2–4 px, mostly in y, from y ≈ 140 next to the channel-edge rows to the star's center at y ≈ 142–143.
- **Labels:** every chop label is the same.
- **Not fixed:** the 10 frames of an SX excursion near frame 91,700 stay wrong. Their star sat at y ≈ 205–225, in the rows the double-sided mask blanks for SX, so no star was in the searched region at all.
- **Contrast:** the star's peak over the image's robust noise rises from about 140 to about 500 (SX) and from about 240 to about 620 (DX).

**Frames without a star** (`chopid_nostar_frac`, 0.2 by default). A frame whose aperture has no star in the searched region still has a chop-ID maximum: a hot pixel, a channel edge or noise. IDL would give the frame a chop state from it.
- **Rule:** the port measures each frame's peak S/N in the chop-ID image (`combine.chopid_snr`: the maximum over the robust noise of the unmasked pixels). It skips a science-round frame whose S/N is below 0.2× the round's median, as it skips frames over the background limit (`combine.reject_no_star`).
- **Procyon_13, SX:** about 2,200 frames (~91,700–93,900) have the star at y ≈ 205–225, inside the rows the SX mask blanks. Their maxima sit on hot pixels near (25–45, 91–116), at S/N ~8, or 0.02× the median of ~360. Every frame with the star is at ≥ 0.65× the median. A 5×5 median filter (`chopid_median`, off by default) does not change that separation.
- **Not applied** with `set_chop_via_bg` or in sky rounds. `--set chopid_nostar_frac=0` reproduces IDL.

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
