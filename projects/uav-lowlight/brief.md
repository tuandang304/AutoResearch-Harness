# Small-object detection from UAVs in low light

## Question and hypothesis
Does illumination-aware training transfer from synthetic to real night UAV imagery
for small objects, and does test-time enhancement help or hurt them?

- H1 (primary): fine-tuning YOLO11s on daytime VisDrone images, half of them
  replaced by synthetic low-light copies (linearize, exposure scaling,
  Poisson-Gaussian noise, re-gamma) whose darkness is calibrated to real night
  images, raises COCO AP for objects under 32^2 px (at original resolution) on
  held-out real night clips relative to the same detector trained on unchanged
  daytime images. Gap closure is measured against arm C, which replaces day images
  with real night images from disjoint night clips (a reference upper bound).
- H2: test-time enhancement (gamma, CLAHE) of night images gives smaller gains, or
  losses, for small objects than for medium objects because it amplifies noise.
- H3 (stage 4, exploratory): which part of the synthesis carries any real-night gain:
  darkening, noise, calibration of the darkness, spatially varying light pools or a
  sodium colour cast.
- Guard: arm B does not reduce daytime AP by more than 1 point.

Baseline (arm A): the same detector, image count, schedule and seeds with unchanged
day images. Arm B: 50% calibrated synthetic low light. Arm C: real night-train
images. Negative or null results are reported as such.

## Data and evaluation
VisDrone2019-DET (Zhu et al., TPAMI 2022) from the Ultralytics GitHub release
assets (train 1.55 GB, val 82 MB, test-dev 311 MB). It is for non-commercial
academic research; do not redistribute images. The data has no illumination labels.

**Pre-registered groups** (`uav_splits.json`, built locally by `make_splits.py` on
2026-09-29, before any experiment):

- Per-image median luma of the grayscale image resized to a 256 px long side, in
  [0, 1]: night < 0.16, twilight 0.16-0.30, day >= 0.30.
- A visual audit of every clip with a frame below 0.16 found real lit-night scenes
  in 39 clips. In 8 mostly-daylight clips (9999940, 9999955, 9999956, 9999966,
  9999976, 9999982, 9999997, 9999999) the 31 dark frames were overcast, shaded or
  low-sun daylight; they are reclassified as twilight.
- The clip id is the first file-name field. The official splits share clips: 24
  clip ids occur in both train and val, and 35 clips mix night and non-night
  frames. So all three splits are pooled and split **by clip** (seed 0, greedy by
  image count). Every frame of a test clip is withheld from training.
- Night clips: 40/60 into night-train/night-test. Other clips: 75/10/15 into
  day-train/day-tune/day-test by day-image count. Day frames of night-test clips
  join day-test; twilight frames of test clips form twilight-test; other twilight
  frames are unused.

| Group | Images | Clips | Use |
|---|---|---|---|
| day_train | 3945 | 216 | training (all arms) |
| day_tune | 482 | 25 | selection |
| day_test | 1443 | 48 | report, day guard |
| night_train | 496 | 15 | arm C only; calibrates synthetic darkness |
| night_test | 744 | 24 | final report only |
| twilight_test | 393 | 32 | secondary report |
| synth_dark_tune / _test | 482 / 1443 | | fixed-seed degraded copies of day_tune / day_test |

Five clips hold 677 of the 744 night-test images. 64% of night-test objects are
under 32^2 px and 32% under 16^2 px.

**Synthetic low light** (`uavlib.synth_lowlight`): sRGB to linear (gamma 2.2),
exposure k ~ LogUniform(0.005, 0.1), shot variance a*x with a ~ LogUniform(1e-4, 1e-2),
read std s ~ LogUniform(1e-3, 1e-2), clip, re-gamma, JPEG 95. The exposure range was
calibrated so the median-luma quantiles of synthetic day-train copies (10/50/90%:
0.043/0.071/0.137) match night-train (0.039/0.078/0.141). The MAET-like range
k ~ LogUniform(0.05, 0.3) gives 0.110/0.161/0.256, so most images would count as twilight.
It is kept as a stage-4 ablation. The light-pool variant multiplies by a smooth
map with ambient level k plus 2-8 Gaussian pools (0.051/0.078/0.135).

**Evaluation** (`uavlib.coco_eval`): pycocotools COCO AP (IoU 0.50:0.95) and AP50 at
original resolution, at most 500 detections per image (conf >= 0.001) and maxDets 500
(as in VisDrone), per area bin: tiny < 16^2,
small 16^2-32^2, small_all < 32^2 (primary), medium 32^2-96^2, large. VisDrone
ignored regions and "others" are crowd regions for every class, so detections in
them are neither true nor false positives. Test-time enhancement: none; gamma
mapping the image median luma to 0.35 (exponent in [0.3, 1]); CLAHE on LAB L
(clip 2.0, 8x8). Paired clip-level bootstrap: 1000 resamples of night-test clips
with all their frames, per-image weights applied to COCOeval's own matches. On
synthetic detections it reproduced COCOeval AP exactly (all/small_all/medium).
Selection uses only day_tune and synth_dark_tune (mean small_all AP).

## Resources
Colab through the Colab CLI (auto-provisioned; model-selected cpu/T4/L4/A100
tiers; T4 is discouraged because its VMs have few CPU cores). Hard spend cap 185
compute units of the ~198-unit balance (expected ~110). Per-script targets:
stage 1 ~40 min, stage 2 ~60 min, stage 3 ~2 h, stage 4 ~90 min (guidance, not pass/fail);
hard limit 2.5 h.

Search budget: 2 parallel workers. Stage 1 ends at its first working node (at most
4 nodes). Stages 2, 3 and 4 have 4 new nodes each. Seeds 0-2 of the best stage-3
script are run by the harness (seed evaluation only in stage 3). Stage 2 selects
imgsz (640/960/1280) and epochs. Fine-tuning budgets remain below published
300-600 epoch schedules, so absolute AP is not comparable to the literature; only
within-study comparisons are claimed.

Experiment code imports `uavlib.py` and `uav_splits.json`, which the harness copies
into every node workspace (`exec.support_files`). `ideas.py` is the reference pilot
(`--load_code`). Launch:

```bash
.venv/bin/python -m autoresearch --project projects/uav-lowlight --load_code \
  --exec_backend colab 2>&1 | tee projects/uav-lowlight/runs/launch-$(date +%Y%m%d-%H%M%S).log
```

Earlier launches (2026-09-29 16:34 and 17:12, old concept) failed before any training.
Their `ProcessPoolExecutor` pickling errors, sequence-leakage check and lost jobs
after tunnel drops were fixed in the harness or avoided by this design.

## Interpretation
- A luminance threshold plus a visual audit is a proxy for night; counts and the
  audit list are recorded.
- Night images come from few clips; clip variation dominates. Report clip-bootstrap
  intervals and seed ranges; do not claim significance from three seeds.
- Arm C is a reference, not a method; gap estimates depend on night-train clips.
- Synthetic-dark gains alone do not support H1; only real night-test results do.
- Stage-4 ablations use seed 0 only and are compared with stage-3 seed-0 arms.
- DroneVehicle (mostly medium-sized vehicles, Baidu/HF-gated labels) and UAVDT
  (unclear redistribution) were considered and not used.
