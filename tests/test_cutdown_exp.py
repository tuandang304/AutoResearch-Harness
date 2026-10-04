"""End-to-end protocol smoke test of cutdown_exp on coco8 (CPU, tiny). Opt-in: CUTDOWN_SMOKE=1."""

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects" / "cut-down"
HAVE = (all(importlib.util.find_spec(m) for m in ("torch", "ultralytics", "pycocotools", "cv2")) and PROJECT.is_dir()
        and os.environ.get("CUTDOWN_SMOKE") == "1")


def coco8_records(root: Path, split: str, clip: str):
    import numpy as np
    from PIL import Image
    recs = []
    for img in sorted((root / "images" / split).glob("*.jpg")):
        w, h = Image.open(img).size
        lab = root / "labels" / split / (img.stem + ".txt")
        rows = [list(map(float, l.split())) for l in lab.read_text().splitlines() if l.strip()] if lab.exists() else []
        boxes = [[(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h] for _, cx, cy, bw, bh in rows]
        recs.append(dict(key=f"{split}/{img.name}", path=str(img), clip=clip, luma=0.5, width=w, height=h,
                         boxes=np.array(boxes, np.float32).reshape(-1, 4), labels=np.array([int(r[0]) % 10 for r in rows], np.int64),
                         ignore=np.zeros((0, 4), np.float32)))
    return recs


@unittest.skipUnless(HAVE, "set CUTDOWN_SMOKE=1 with torch, ultralytics, pycocotools, opencv")
class ProtocolSmoke(unittest.TestCase):
    def test_pipeline_runs_end_to_end_at_tiny_scale(self):
        import torch
        from ultralytics import settings
        sys.path.insert(0, str(PROJECT))
        import cutdown_exp as ce
        import uavlib
        data_dir = PROJECT / "data"
        data_dir.mkdir(exist_ok=True)
        settings.update({"datasets_dir": str(data_dir)})
        from ultralytics.utils.downloads import download
        if not (data_dir / "coco8").exists():
            download("https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip", dir=data_dir, unzip=True)
        coco = data_dir / "coco8"
        tmp = Path(tempfile.mkdtemp())
        ce.RUNS = uavlib.RUNS = tmp
        train, val = coco8_records(coco, "train", "c1"), coco8_records(coco, "val", "c2")
        data = uavlib.Data(tmp, {"day_train": train, "day_tune": val, "night_test": val}, {})
        ds = ce.build_datasets(data, seed=0, dark_fraction=0.5, probe_fraction=0.5, root=tmp / "ds")
        self.assertEqual(ds["n_train"] + ds["n_probe"], len(train))
        kw = dict(imgsz=64, batch=2, device="cpu", workers=0, amp=False)
        warm = ce.warmup(ds, 0, 1, 64, weights=None, **{k: v for k, v in kw.items() if k not in ("imgsz",)})
        caches, diag = ce.probe_cache(warm, ds, 64, batch=2, device="cpu", cfg=dict(k=4), gate_batches=1)
        self.assertEqual(set(caches), {"scup", "gradient", "snr", "global_delta"})
        self.assertEqual(len(caches["scup"]), ds["n_probe"])
        passed, report = ce.reliability_gate(diag)
        self.assertIn("split_half", report)
        w = ce.train_condition("scup", warm, ds, caches, 0, 64, ft_epochs=1, distill_epochs=1, batch=2, device="cpu", workers=0, amp=False)
        res = ce.evaluate(w, data, ["night_test"], 64, policies=("router", "stable", "oracle"), device="cpu", batch=2,
                          oracle_cfg=dict(k=4))
        for pol in ("router", "stable", "oracle"):
            self.assertIn("AP", res["night_test"][pol]["metrics"])
        sr, ss = res["night_test"]["router"]["state"], res["night_test"]["stable"]["state"]
        uavlib.save_state(sr, str(tmp / "s.npy"))
        boot = uavlib.paired_clip_bootstrap(sr, ss, n=20)
        self.assertIsNotNone(boot)
        self.assertEqual(set(ce.oracle_recovery({"stable": .1, "detail": .12, "half": .11}, .2, .15)), {"best_fixed", "p_fixed", "p_oracle", "p_router", "oracle_gap", "recovery"})
        stats = ce.gate_statistics(torch.nn.Identity() if False else __import__("cutdown_yolo").load_detector(w), val, 64, "cpu")
        self.assertIn("background", stats)


if __name__ == "__main__":
    unittest.main()
