"""CPU-only re-analysis of the recorded v1 results (no training, no inference).

Reads the experiment_data.npy files of run 2026-09-30_00-44-30_545669 and writes
analysis.json plus experiment_results/reanalysis_v1_1/experiment_data.npy into a
write-up folder. Usage: python reanalyze_v1.py <run_dir> <writeup_dir>
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

T95 = {2: 4.303, 1: 12.706}  # two-sided 95% t quantiles, by degrees of freedom
BINS = ["all", "tiny", "small", "small_all", "medium", "large"]
GROUPS = ["night_test", "day_test", "twilight_test", "synth_dark_test", "day_tune", "synth_dark_tune"]

run, out = Path(sys.argv[1]), Path(sys.argv[2])
results = run / "logs/0-run/experiment_results"
SEEDS = {  # stage-3 seed evaluation of the best node (see writeup2/PROVENANCE.md)
    0: results / "experiment_683d687734514cde86ec78c0cde07005_proc_411054",
    1: Path(sys.argv[2]) / "experiment_results/experiment_a31ff4e8b66645d0828b72242137d240_seed1_recovered",
    2: results / "experiment_b5e835509c0e4af28d327fbd6505211f_proc_411054",
}
SEED0_RERUNS = [results / f"experiment_{x}" for x in (
    "3a8e5ed100744905a204596ecf486244_proc_411055", "409d7ee97c744db29b8c16736dc95ba2_proc_411054",
    "683d687734514cde86ec78c0cde07005_proc_411054", "8a75838793ba4dcdbec1ba6a6d501606_proc_411055")]
ABLATIONS = {  # stage-4 nodes flagged buggy only for missing stage-3 comparisons
    "61c02ec1": run / "0-run/process_ForkProcess-8_5f139990a20f/working",
    "f40f2c64": run / "0-run/process_ForkProcess-8_40a88c53b62f/working",
}


def load(folder):
    return np.load(Path(folder) / "experiment_data.npy", allow_pickle=True).item()


def ap(arm, group, bin_="small_all", method="none", kind="AP"):
    return arm["metrics"][group][method][kind][bin_]


def interval(values):
    values = np.asarray(values, float)
    mean = float(values.mean())
    if len(values) < 2:
        return {"mean": mean, "n": 1}
    sd = float(values.std(ddof=1))
    half = T95[len(values) - 1] * sd / math.sqrt(len(values))
    return {"mean": mean, "sd": sd, "t95": [mean - half, mean + half], "n": len(values),
            "values": values.tolist(), "n_negative": int((values < 0).sum())}


def ranks(x):
    order = np.argsort(x)
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    return r


def spearman(x, y):
    return float(np.corrcoef(ranks(np.asarray(x)), ranks(np.asarray(y)))[0, 1])


seeds = {s: load(p)["arms"] for s, p in SEEDS.items()}
for s, arms in seeds.items():
    assert all(arms[a]["train"]["seed"] == s for a in "ABC"), s
analysis = {"note": "Derived only from recorded metrics; no model was trained or evaluated."}

# 1. Determinism: independent seed-0 retrainings in separate stage-3 nodes.
reruns = [load(p)["arms"] for p in SEED0_RERUNS]
max_diff = 0.0
for arm in "ABC":
    for other in reruns[1:]:
        for g in ("night_test", "day_test", "twilight_test", "synth_dark_test"):
            for b in BINS:
                max_diff = max(max_diff, abs(ap(other[arm], g, b) - ap(reruns[0][arm], g, b)))
analysis["determinism"] = {
    "nodes": [p.name for p in SEED0_RERUNS],
    "train_seconds_A": [r["A"]["train"]["train_seconds"] for r in reruns],
    "max_abs_AP_difference": max_diff,
    "meaning": "Each node retrained A, B and C from scratch at seed 0 (about 710 s each) and "
               "produced identical AP in every group and size bin, so a fixed seed reproduces "
               "a training run exactly on this setup (A100, ultralytics 8.4.166, torch 2.11).",
}

# 2. Seed-paired differences.
paired = {}
for diff, (x, y) in {"B-A": ("B", "A"), "C-A": ("C", "A"), "B-C": ("B", "C")}.items():
    paired[diff] = {}
    for g in ("night_test", "day_test", "twilight_test", "synth_dark_test"):
        paired[diff][g] = {}
        for kind in ("AP", "AP50"):
            for b in BINS:
                paired[diff][g][f"{kind}_{b}"] = interval(
                    [ap(seeds[s][x], g, b, kind=kind) - ap(seeds[s][y], g, b, kind=kind) for s in seeds])
analysis["seed_paired_differences"] = paired
analysis["per_arm_seed_spread"] = {
    arm: {g: interval([ap(seeds[s][arm], g) for s in seeds]) for g in ("night_test", "day_test", "twilight_test")}
    for arm in "ABC"}
gap = [(ap(seeds[s]["B"], "night_test") - ap(seeds[s]["A"], "night_test")) /
       (ap(seeds[s]["C"], "night_test") - ap(seeds[s]["A"], "night_test")) for s in seeds]
analysis["gap_closure_small_all"] = interval(gap)
analysis["recorded_clip_bootstrap_ci95_small_all"] = {  # computed on Colab, per seed
    s: load(p)["comparisons"]["report_only_night_test_paired_clip_bootstrap"] for s, p in SEEDS.items()}

# 3. Recovered stage-4 ablations, paired with stage-3 seed 0.
a0 = seeds[0]
variants = {"B (50%, calibrated, darken+noise)": seeds[0]["B"], "C (real night)": seeds[0]["C"]}
configs = {}
for node, folder in ABLATIONS.items():
    data = load(folder)
    seen = {}
    for name, arm in data["arms"].items():
        cfg = arm["arm"]
        key = (cfg["fraction"], cfg["darken"], cfg["noise"], cfg["cast"], cfg["pools"], tuple(cfg["exposure"]))
        if key in seen:  # the decomposition node stored each model under two names
            continue
        seen[key] = name
        assert arm["train"]["seed"] == 0 and arm["train"]["epochs"] == 16 and arm["train"]["imgsz"] == 1280
        variants[name] = arm
        configs[name] = {"node": node, **{k: cfg[k] for k in ("fraction", "darken", "noise", "cast", "pools", "exposure",
                                                              "n_synthetic")}}
abl = {}
for name, arm in variants.items():
    row = {"vs_A_seed0": {}}
    for g in ("night_test", "day_test", "twilight_test", "synth_dark_test"):
        for b in ("small_all", "tiny", "small", "medium", "all"):
            row.setdefault(g, {})[b] = ap(arm, g, b)
            row["vs_A_seed0"].setdefault(g, {})[b] = ap(arm, g, b) - ap(a0["A"], g, b)
    row["selection_AP_small_all"] = arm.get("selection_AP_small_all")
    row["night_enhancement_change_small_all"] = {
        m: ap(arm, "night_test", method=m) - ap(arm, "night_test") for m in ("gamma", "clahe")}
    row["config"] = configs.get(name)
    abl[name] = row
abl["A (day only)"] = {g: {b: ap(a0["A"], g, b) for b in ("small_all", "tiny", "small", "medium", "all")}
                       for g in ("night_test", "day_test", "twilight_test", "synth_dark_test")}
abl["A (day only)"]["selection_AP_small_all"] = a0["A"].get("selection_AP_small_all")
analysis["seed0_ablations"] = {
    "rows": abl,
    "noise_scale": {
        "sd_of_seed_paired_B-A_night_small_all": paired["B-A"]["night_test"]["AP_small_all"]["sd"],
        "sd_of_seed_paired_C-A_night_small_all": paired["C-A"]["night_test"]["AP_small_all"]["sd"],
        "meaning": "Spread across seeds of a paired difference; a rough scale for one seed-0 "
                   "difference. It covers training noise only, not test-clip sampling.",
    },
    "caveat": "Seed 0 only; per-image detections were not saved, so no clip bootstrap exists for these rows.",
}

# 4. Does the selection proxy rank models like real night data?
models = [(f"seed{s} {arm}", seeds[s][arm]) for s in seeds for arm in "ABC"]
models += [(f"seed0 {n}", v) for n, v in variants.items() if n in configs]
night = [ap(m, "night_test") for _, m in models]
proxies = {
    "selection_AP_small_all (day_tune+synth_dark_tune)": [m["selection_AP_small_all"] for _, m in models],
    "synth_dark_test small_all": [ap(m, "synth_dark_test") for _, m in models],
    "day_test small_all": [ap(m, "day_test") for _, m in models],
    "day_tune small_all": [ap(m, "day_tune") for _, m in models],
    "twilight_test small_all": [ap(m, "twilight_test") for _, m in models],
}
analysis["proxy_validity"] = {
    "models": [n for n, _ in models],
    "night_test_small_all": night,
    "spearman_with_night_test_small_all": {k: spearman(v, night) for k, v in proxies.items()},
    "values": proxies,
    "best_by_proxy": {k: models[int(np.argmax(v))][0] for k, v in proxies.items()},
    "best_on_night_test": models[int(np.argmax(night))][0],
}

# 5. Per-class small-object AP on night_test (mean over seeds).
classes = list(seeds[0]["A"]["metrics"]["night_test"]["none"]["per_class_AP_small_all"])
per_class = {}
for c in classes:
    get = lambda s, arm: seeds[s][arm]["metrics"]["night_test"]["none"]["per_class_AP_small_all"].get(c)
    vals = {arm: [get(s, arm) for s in seeds] for arm in "ABC"}
    if any(v is None or (isinstance(v, float) and math.isnan(v)) for vs in vals.values() for v in vs):
        continue
    per_class[c] = {"A": float(np.mean(vals["A"])),
                    "B-A": interval(np.subtract(vals["B"], vals["A"])),
                    "C-A": interval(np.subtract(vals["C"], vals["A"]))}
analysis["night_per_class_small_all"] = per_class

# 6. Test-time enhancement (change from no enhancement), per arm, over seeds.
enh = {}
for arm in "ABC":
    for g in ("night_test", "twilight_test"):
        for m in ("gamma", "clahe"):
            for b in ("small_all", "tiny", "small", "medium", "all"):
                enh.setdefault(arm, {}).setdefault(g, {}).setdefault(m, {})[b] = interval(
                    [ap(seeds[s][arm], g, b, m) - ap(seeds[s][arm], g, b) for s in seeds])
analysis["enhancement_change"] = enh

out.mkdir(parents=True, exist_ok=True)
(out / "analysis.json").write_text(json.dumps(analysis, indent=1))
target = out / "experiment_results/reanalysis_v1_1"
target.mkdir(parents=True, exist_ok=True)
np.save(target / "experiment_data.npy", analysis, allow_pickle=True)
for node, folder in ABLATIONS.items():
    dest = out / f"experiment_results/experiment_{node}_stage4_recovered"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "experiment_data.npy").write_bytes((folder / "experiment_data.npy").read_bytes())
print(json.dumps({
    "determinism_max_diff": max_diff,
    "B-A night": paired["B-A"]["night_test"]["AP_small_all"],
    "C-A night": paired["C-A"]["night_test"]["AP_small_all"],
    "B-A day": paired["B-A"]["day_test"]["AP_small_all"],
    "gap": analysis["gap_closure_small_all"],
    "ablations night vs A": {k: v["vs_A_seed0"]["night_test"]["small_all"] for k, v in abl.items() if "vs_A_seed0" in v},
    "ablations day vs A": {k: v["vs_A_seed0"]["day_test"]["small_all"] for k, v in abl.items() if "vs_A_seed0" in v},
    "spearman": analysis["proxy_validity"]["spearman_with_night_test_small_all"],
    "best": (analysis["proxy_validity"]["best_by_proxy"], analysis["proxy_validity"]["best_on_night_test"]),
}, indent=1))
