# COMPUTE: L4  # stage-1 pilot: three short YOLO11s trainings at 640 px plus evaluation
# Reference usage of uavlib (a support file next to this script). Later stages
# change the schedule, add seeds and analyses, but keep this protocol.
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

EPOCHS, IMGSZ = 3, 640
EVAL_GROUPS = ["day_tune", "synth_dark_tune", "day_test", "synth_dark_test",
               "night_test", "twilight_test"]
experiment_data = {
    "settings": {"epochs": EPOCHS, "imgsz": IMGSZ, "seed": seed, "model": "yolo11s.pt"},
    "environment": env,
    "counts": data.counts(),
    "clips": data.clips(),
    "arms": {},
}
results = {}
for arm in ["A", "B", "C"]:
    data_yaml = uavlib.build_training_set(data, arm, seed)
    weights, info = uavlib.train(data_yaml, f"{arm}_s{seed}", epochs=EPOCHS, imgsz=IMGSZ, seed=seed)
    results[arm] = uavlib.evaluate(weights, data, EVAL_GROUPS, imgsz=IMGSZ)
    metrics = uavlib.strip_states(results[arm])
    selection = float(np.mean([metrics[g]["none"]["AP"]["small_all"]
                               for g in ("day_tune", "synth_dark_tune")]))
    experiment_data["arms"][arm] = {"train": info, "arm": uavlib.arm_info(data_yaml),
                                    "metrics": metrics, "selection_AP_small_all": selection}
    print(f"arm {arm}: selection_AP_small_all={selection:.4f} "
          f"day_test AP={metrics['day_test']['none']['AP']['all']:.4f} "
          f"report_only night_test AP_small_all={metrics['night_test']['none']['AP']['small_all']:.4f}")

night = {arm: results[arm]["night_test"]["none"]["state"] for arm in results}
experiment_data["night_test_bootstrap_small_all"] = {
    f"{x}-A": uavlib.paired_clip_bootstrap(night[x], night["A"], n=1000) for x in ("B", "C")}
print("report_only paired clip bootstrap:", experiment_data["night_test_bootstrap_small_all"])
experiment_data["runtime_minutes"] = round((time.time() - t0) / 60, 1)
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data, allow_pickle=True)
print(f"total runtime {experiment_data['runtime_minutes']} min")
