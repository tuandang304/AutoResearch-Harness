# COMPUTE: L4  # stage-1 pilot: five short YOLO11s trainings at 640 px plus evaluation
# Reference usage of uavlib 2.0 (a support file next to this script). Later stages use
# the fixed full schedule (16 epochs, 1280 px) and add seeds and analyses, but keep
# this protocol: arms, replacement share, evaluation, saved states and comparisons.
import os
import time

import numpy as np

import uavlib

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
seed = int(os.environ.get("AUTORESEARCH_SEED", "0"))
t0 = time.time()
env = uavlib.environment()
print("environment", env, "cpus", os.cpu_count())

data = uavlib.prepare()  # downloads once per VM; verifies the pre-registered group counts
print("group counts", data.counts(), "clips", data.clips())

EPOCHS, IMGSZ = 3, 640  # pilot only; stages 2-4 use 16 epochs at 1280 px
EVAL_GROUPS = ["day_tune", "synth_dark_tune", "day_test", "synth_dark_test",
               "night_test", "twilight_test"]
ARMS = {  # every arm trains on 3945 images; B, C and D replace the same share (496)
    "A": dict(arm="A"),
    "B_global": dict(arm="B"),                          # v1 calibrated global darkening + noise
    "B_full": dict(arm="B", pools=True, cast=True),     # light pools + sodium cast + noise
    "C": dict(arm="C"),                                 # real night images with labels
}
experiment_data = {
    "settings": {"epochs": EPOCHS, "imgsz": IMGSZ, "seed": seed, "model": "yolo11s.pt",
                 "pseudo_conf": uavlib.PSEUDO_CONF, "matched_fraction": uavlib.matched_fraction(data)},
    "environment": env, "counts": data.counts(), "clips": data.clips(), "arms": {},
}
results, weights = {}, {}


def run(name, data_yaml):
    w, info = uavlib.train(data_yaml, f"{name}_s{seed}", epochs=EPOCHS, imgsz=IMGSZ, seed=seed)
    weights[name] = w
    results[name] = uavlib.evaluate(w, data, EVAL_GROUPS, imgsz=IMGSZ)
    metrics = uavlib.strip_states(results[name])
    experiment_data["arms"][name] = {"train": info, "arm": uavlib.arm_info(data_yaml), "metrics": metrics}
    uavlib.save_state(results[name]["night_test"]["none"]["state"],
                      os.path.join(working_dir, f"state_{name}_night_test.npy"))
    print(f"arm {name}: day_tune AP_small_all={metrics['day_tune']['none']['AP']['small_all']:.4f} "
          f"day_test AP_small_all={metrics['day_test']['none']['AP']['small_all']:.4f} "
          f"report_only night_test AP_small_all={metrics['night_test']['none']['AP']['small_all']:.4f}")


for name, kwargs in ARMS.items():
    run(name, uavlib.build_training_set(data, seed=seed, **kwargs))
# Arm D: the same seed's arm-A model pseudo-labels the night_train images (no night labels).
pseudo, pseudo_stats = uavlib.pseudo_label(weights["A"], data, imgsz=IMGSZ)
experiment_data["pseudo_labels"] = pseudo_stats
print("pseudo labels (diagnostic_vs_ground_truth is report_only):", pseudo_stats)
run("D", uavlib.build_training_set(data, "D", seed, pseudo=pseudo))

night = {name: results[name]["night_test"]["none"]["state"] for name in results}
pairs = {"D-A": ("D", "A"), "B_full-A": ("B_full", "A"), "B_global-A": ("B_global", "A"),
         "B_full-B_global": ("B_full", "B_global"), "C-A": ("C", "A"), "D-C": ("D", "C")}
experiment_data["report_only_night_test_paired_clip_bootstrap_small_all"] = {
    k: uavlib.paired_clip_bootstrap(night[x], night[y], n=1000) for k, (x, y) in pairs.items()}
experiment_data["report_only_night_test_leave_one_clip_out_small_all"] = {
    k: uavlib.leave_one_clip_out(night[x], night[y]) for k, (x, y) in pairs.items() if k in ("D-A", "B_full-A", "C-A")}
print("report_only paired clip bootstrap:", experiment_data["report_only_night_test_paired_clip_bootstrap_small_all"])
experiment_data["runtime_minutes"] = round((time.time() - t0) / 60, 1)
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True)
print(f"total runtime {experiment_data['runtime_minutes']} min")
