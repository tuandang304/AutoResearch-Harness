# Night UAV small objects v2: unlabeled real night vs realistic synthetic night

Pre-registered on 2026-10-01, before any v2 training. Follows the v1 study in
`projects/uav-lowlight` (run 2026-09-30_00-44-30; re-analysis in its `runs/..._writeup3`).

## What v1 found and what v2 changes

v1 replaced 50% of day images with calibrated globally darkened copies (B). Over
three seeds B lowered real night small-object AP (B − A = −0.0053), while 496
labelled night images (C) raised it (+0.0133). Its recovered ablations showed that
the night deficit remained at a 12.5% replacement share, so the reduced number of
day images did not explain it. Its review asked for a realistic generator, matched
replacement shares, per-image results for intervals, and working ablations.

v2 changes:
- **Matched share.** B, C and D each replace exactly 496 of the 3945 day images (12.6%).
- **Realistic synthesis.** B_full adds spatially varying light pools, a sodium colour
  cast and noise. v1 pre-registered this as an ablation, but it never ran. B_global
  (v1's generator) is kept for comparison.
- **A method that needs no night labels.** D is single-round self-training. The
  seed's day model (A) pseudo-labels the 496 night_train images, with no night labels
  and a fixed confidence of 0.25, and D trains on them in place of the same 496 day
  images that C replaces. C and D differ only in the labels of those images.
- **Saved per-image results.** Night_test states are saved for every model, so every
  comparison, including ablations, has paired clip-bootstrap intervals,
  leave-one-clip-out checks and a seed-pooled clip bootstrap.
- **Working ablations.** The harness copies the stage-3 seed-0 results into each
  stage-4 workspace (`parent_results/`).
- **No tuning.** The v1 schedule is fixed (YOLO11s, 16 epochs, 1280 px, batch 16).
  v1's selection metric ranked models poorly against real night AP, so v2 selects
  nothing with it.

## Hypotheses (night_test small_all AP, objects < 32² px)

- **H1 (primary):** D > A, with gap closure (D − A)/(C − A) > 0.
- **H2:** B_full > A and B_full > B_global.
- **Guard:** no arm loses more than 0.01 day_test small_all AP against A.
- **Reference:** C − A > 0, which replicates v1.
- **Secondary:** the effect of test-time gamma and CLAHE, per arm, for small vs medium objects.

Decision rule: a hypothesis is supported when the seed-pooled clip-bootstrap 95%
interval excludes zero and all three seeds agree in sign. Otherwise we report "no
detectable effect". Per-seed bootstraps and leave-one-clip-out results are always reported.

## Data, splits and evaluation

These are identical to v1 (`uav_splits.json` is byte-identical, manifest bcb01db885f3):
- clip-level groups: day_train 3945, day_tune 482, day_test 1443, night_train 496
  (15 clips), night_test 744 (24 clips), twilight_test 393;
- COCO AP at original resolution, with maxDets 500 and VisDrone crowd handling.

night_test and twilight_test are report-only.

## Disclosure of a known weakness

night_test is the same set v1 reported on, and these hypotheses were written after
v1's results were known. B_full was pre-registered in v1. D was not. The paper must
state this.

## Budget

There is a hard cap of 75 Colab compute units out of a balance of 102.8 (checked
2026-10-01). The estimate is about 45:

| Stage | Plan | Est. units |
|---|---|---|
| 1 | five-arm L4 pilot | ~1 |
| 2 | full-schedule check of arm A | ~2 |
| 3 | one or two five-arm nodes | 8–17 |
| Seeds | seeds 1 and 2 (seed 0 reused) | ~17 |
| 4 | three ablation nodes | ~11 |

The run uses one worker, so nodes run one at a time.

```bash
.venv/bin/python -m autoresearch --project projects/uav-lowlight-v2 --load_code \
  --exec_backend colab 2>&1 | tee projects/uav-lowlight-v2/runs/launch-$(date +%Y%m%d-%H%M%S).log
```

## v2.1 extension: stronger teachers for self-training (pre-registered 2026-10-02)

Written after the v2 results were known, before any v2.1 training. Script:
`extension_v21.py`, one seed per call, seeds 0–2, same schedule, splits and evaluation.

- **D2 (second round):** D's model pseudo-labels night_train, and D2 trains on the same
  496 replaced day images with those labels.
- **D_B (synthetic-trained teacher):** B_full's model is the teacher instead of A.
- Both use the fixed threshold 0.25. A, D and B_full are retrained as teachers. Training
  is deterministic at a fixed seed, so A's pseudo-label counts and D's night_test AP must
  equal the v2 values. A mismatch is reported, not hidden.

Hypotheses (night_test small_all AP, paired with the saved v2 states of the same seed):
- **H3:** D2 > A.
- **H4:** D_B > A.
- Secondary: D2 > D and D_B > D.

The decision rule is the v2 one (seed-pooled clip-bootstrap 95% interval excludes zero
and all three seeds agree in sign). Day guard: day_test small_all loss ≤ 0.01 against A.
Budget cap: 24 Colab compute units.

Data check done before this extension: one night_test image (clip 0000112) shows the same
intersection as night_train clip 0000111 (580 RANSAC-consistent ORB matches; every other
cross-split pair has ≤ 10). Leaving that clip out changes no v2 comparison at four
decimals.
