# Low-light UAV detection

Does calibrated synthetic low-light training transfer to real night UAV imagery for
small objects, and does test-time enhancement help them? See brief.md for the
hypotheses, the pre-registered clip-level split and budgets.

| File | Role |
|---|---|
| brief.md | Planning document: hypotheses, data, evaluation, budget |
| ideas.json | Pipeline input (reviewed) |
| ideas.py | Reference pilot passed with `--load_code` |
| config.yaml | Full run configuration (Colab CLI, stage budgets) |
| uavlib.py, uav_splits.json | Support files copied into every node workspace |
| make_splits.py | Rebuilds uav_splits.json from local VisDrone data (data/, ignored) |

New outputs belong in runs/ (ignored). Earlier trial runs of the previous concept
were removed from this folder; they produced no training results.
