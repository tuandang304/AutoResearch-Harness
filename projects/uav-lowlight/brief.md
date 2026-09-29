# Small-object detection from UAVs in low light

## Question and hypothesis
Does illumination-aware training transfer from synthetic to real night UAV imagery
for small objects, and does test-time enhancement help or hurt them?

- H1 (primary): fine-tuning YOLOv8n on daytime VisDrone images, where half of the
  training images are replaced by physics-inspired synthetic low-light versions
  (linearize, exposure scaling, Poisson-Gaussian noise, re-gamma), raises COCO AP
  for small objects (area < 32^2 px at original resolution) on real night images
  relative to the same detector trained on unchanged daytime images.
- H2: test-time enhancement (gamma, CLAHE) of night images gives smaller gains, or
  losses, for small objects than for medium objects because it amplifies noise.
- Guard: neither intervention reduces daytime AP by more than 1 point.

Baseline: the same detector, data, epochs and seeds with unchanged images.
Negative or null results are reported as such.

## Data and evaluation
VisDrone2019-DET (Zhu et al., TPAMI 2021) from the Ultralytics GitHub release
assets (train 1.55 GB, val 82 MB, test-dev 311 MB). It is for non-commercial
academic research; do not redistribute images. The data has no illumination
labels, so illumination groups are pre-registered on per-image median luma
(grayscale, 256 px thumbnail, [0, 1]). The thresholds were chosen from a visual
check of val (2026-09-29): the darkest images below 0.16 were all street-lit
night scenes; 0.16-0.30 mixed dusk and shaded daylight; >= 0.30 was daylight.

| Group | Rule | Use |
|---|---|---|
| day-train / day-tune | train split, median >= 0.30, 90/10 split by sequence id | training / model selection |
| day-test | val split, median >= 0.30 | report |
| night-test (real) | median < 0.16 from train, val and test-dev | final report only |
| twilight | 0.16-0.30 from val and test-dev | secondary report |
| synthetic-dark tune/test | fixed-seed degraded copies of day-tune / day-test | selection / report |

Real night images are never used for training or selection. Metrics: pycocotools
COCOeval AP, AP50 and per size bin (tiny < 16^2, small 16^2-32^2, medium 32^2-96^2,
large) computed at original resolution, 3 training seeds with paired differences.
The Ultralytics conversion drops VisDrone ignored regions, so detections there count
as false positives in every arm equally.

## Resources
Colab through the Colab CLI (auto-provisioned; model-selected cpu/T4/L4 tiers,
A100 excluded). Each script targets <= 40 minutes (hard limit 50). Hard spend cap
50 compute units, about a quarter of the 200-unit balance at launch. Search budget:
stages of 6/4/5/4 nodes, 2 parallel workers, 3 seeds for each stage's best node.
Fine-tuning budgets are far below published 300-600 epoch schedules, so absolute
AP is not comparable to the literature; only within-study comparisons are claimed.

## Interpretation
- A luminance threshold is a proxy for night; counts and montages must be checked.
- Night scenes share locations with daytime training scenes.
- Three seeds bound seed variance, not dataset variance; do not claim significance.
- Synthetic-dark gains alone do not support H1; only real night-test results do.
- DroneVehicle (mostly medium-sized vehicles, Baidu/HF-gated labels) and UAVDT
  (unclear redistribution) were considered and not used.
