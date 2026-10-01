# Low-light UAV detection, v2

Does self-training on unlabeled real night images, or a realistic synthetic night
model, improve small-object detection on real night UAV clips when every arm replaces
the same share of training images? See brief.md for the hypotheses and budget.

| File | Role |
|---|---|
| brief.md | Pre-registration: changes from v1, hypotheses, decision rule, budget |
| ideas.json | Pipeline input (protocol for every stage) |
| ideas.py | Reference pilot passed with `--load_code` |
| config.yaml | Run configuration (Colab CLI, one worker, 75-unit cap, seed-0 reuse) |
| uavlib.py, uav_splits.json | Support files copied into every node workspace (uavlib 2.0; splits identical to v1) |
| make_splits.py | Rebuilds uav_splits.json from local VisDrone data (copy of v1's) |

v1 lives in `projects/uav-lowlight` and is unchanged. New outputs go to runs/ (ignored).
