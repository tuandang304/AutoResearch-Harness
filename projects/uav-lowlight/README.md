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
| reanalyze_v1.py | CPU-only re-analysis of the recorded v1 metrics (no training or inference) |

New outputs belong in runs/ (ignored). Earlier trial runs of the previous concept
were removed from this folder; they produced no training results.

## v1.1 re-analysis (2026-10-01)

`reanalyze_v1.py` re-analyses run 2026-09-30_00-44-30 from its stored metrics, with
no new GPU work. It adds the two stage-4 ablations that trained but were flagged
buggy (they could not load the stage-3 arms), reports seed-paired differences, and
checks the selection proxy against real night results. Output and provenance are in
`runs/..._writeup3/` (`analysis.json`, `PROVENANCE.md`). For a future GPU run, save
per-image night_test states so ablations get clip-bootstrap intervals, and let stage-4
scripts read the stage-3 results instead of failing without them.
