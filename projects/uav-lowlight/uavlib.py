"""uavlib: data, degradation, training and evaluation helpers for the UAV low-light study.

Experiment scripts import this module from their working directory. It fixes the
parts of the protocol that must not vary between nodes: the pre-registered image
groups (uav_splits.json), annotation handling, the synthetic low-light model, the
test-time enhancements and COCO evaluation by object size at original resolution.

    import uavlib
    data = uavlib.prepare()                         # cached per VM
    yaml_a = uavlib.build_training_set(data, "A", seed)
    weights, info = uavlib.train(yaml_a, "A_s0", epochs=30, imgsz=960, seed=seed)
    result = uavlib.evaluate(weights, data, ["night_test", "day_test"], imgsz=960)
    result["night_test"]["none"]["metrics"]["AP"]["small_all"]
    uavlib.paired_clip_bootstrap(result_b["night_test"]["none"]["state"],
                                 result_a["night_test"]["none"]["state"])

Everything heavy is cached under UAV_CACHE (default /content/cache/visdrone) with a
completion marker per step, so a second script on the same VM reuses it.
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request
import zipfile
import zlib

import numpy as np

__version__ = "1.0"

CACHE = Path(os.environ.get("UAV_CACHE", "/content/cache/visdrone"))
RUNS = Path(os.environ.get("UAV_RUNS", "/content/cache/runs"))
MANIFEST = Path(__file__).with_name("uav_splits.json")
URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-{}.zip"
SPLITS = ("train", "val", "test-dev")
CLASSES = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle",
           "awning-tricycle", "bus", "motor"]
GROUPS = ("day_train", "day_tune", "day_test", "night_train", "night_test", "twilight_test")
SYNTH_GROUPS = {"synth_dark_tune": "day_tune", "synth_dark_test": "day_test"}
EVAL_SEED = 12345  # fixed degradation of the synthetic-dark evaluation sets
# COCO-style area bins in original-resolution pixels; small_all (< 32^2) is primary.
AREA_BINS = {"all": [0, 1e10], "tiny": [0, 16 ** 2], "small": [16 ** 2, 32 ** 2],
             "small_all": [0, 32 ** 2], "medium": [32 ** 2, 96 ** 2], "large": [96 ** 2, 1e10]}
ENHANCEMENTS = ("none", "gamma", "clahe")
THREADS = max(4, 2 * (os.cpu_count() or 2))


def log(message):
    print(f"[uavlib] {message}", flush=True)


def _cv2():
    import cv2  # imported lazily so the module can be inspected without OpenCV

    cv2.setNumThreads(1)
    return cv2


def _seed_for(name, seed):
    return [int(seed), zlib.crc32(name.encode())]


def _parallel(fn, items):
    with ThreadPoolExecutor(THREADS) as pool:  # OpenCV and I/O release the GIL
        return list(pool.map(fn, items))


def _step(name, cache=None):
    """Completion marker for a cached preparation step."""
    return Path(cache or CACHE) / f".done_{name}"


# ------------------------------------------------------------------ data


class Data:
    """Prepared dataset: records per group, each {key, path, clip, luma, width,
    height, boxes (N,4 xywh), labels (N,), ignore (M,4 xywh)}."""

    def __init__(self, cache, groups, manifest):
        self.cache, self.groups, self.manifest = Path(cache), groups, manifest

    def counts(self):
        return {g: len(r) for g, r in self.groups.items()}

    def clips(self):
        return {g: len({r["clip"] for r in recs}) for g, recs in self.groups.items()}


def _download(split, cache):
    target = cache / f"VisDrone2019-DET-{split}.zip"
    for attempt in range(4):
        try:
            tmp = target.with_suffix(".part")
            with urllib.request.urlopen(URL.format(split), timeout=60) as resp, open(tmp, "wb") as out:
                shutil.copyfileobj(resp, out, 1 << 22)
            tmp.replace(target)
            return target
        except OSError as exc:
            log(f"download {split} failed ({exc}); retry {attempt + 1}/3")
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"could not download VisDrone {split}")


def _parse_annotation(path, width, height):
    """VisDrone DET txt -> boxes/labels of the 10 classes plus ignore regions.

    Rows are x,y,w,h,score,category,truncation,occlusion. Category 0 (ignored
    region, score 0) and 11 (others) become ignore regions for every class.
    """
    boxes, labels, ignore = [], [], []
    text = path.read_text().strip() if path.exists() else ""
    for line in text.splitlines():
        v = [int(float(t)) for t in line.strip().rstrip(",").split(",")[:6]]
        x, y, w, h, score, cat = v
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(width, x + w), min(height, y + h)
        if x1 <= x0 or y1 <= y0:
            continue
        box = [x0, y0, x1 - x0, y1 - y0]
        if cat in (0, 11) or score == 0:
            ignore.append(box)
        elif 1 <= cat <= 10:
            boxes.append(box)
            labels.append(cat - 1)
    return (np.array(boxes, np.float32).reshape(-1, 4), np.array(labels, np.int64),
            np.array(ignore, np.float32).reshape(-1, 4))


def prepare(cache=None, source=None):
    """Download (once per VM), index the pre-registered groups and build the
    fixed synthetic-dark evaluation sets. `source` is an existing directory with
    extracted VisDrone2019-DET-* folders (skips the download)."""
    cache = Path(cache or CACHE)
    cache.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text())
    root = Path(source) if source else cache
    t0 = time.time()
    if source is None and not _step("extract", cache).exists():
        with ThreadPoolExecutor(3) as pool:
            zips = list(pool.map(lambda s: _download(s, cache), SPLITS))
        for z in zips:
            with zipfile.ZipFile(z) as f:
                f.extractall(cache)
            z.unlink()
        _step("extract", cache).touch()
        log(f"downloaded and extracted VisDrone in {time.time() - t0:.0f}s")
    index_path = cache / f"index_{manifest_hash()}.json"
    if not index_path.exists():
        from PIL import Image

        def record(item):
            key, luma = item
            split, name = key.split("/")
            img = root / f"VisDrone2019-DET-{split}" / "images" / name
            with Image.open(img) as im:  # header only
                width, height = im.size
            boxes, labels, ignore = _parse_annotation(
                root / f"VisDrone2019-DET-{split}" / "annotations" / (Path(name).stem + ".txt"),
                width, height)
            return dict(key=key, path=str(img), clip=name.split("_")[0], luma=luma,
                        width=width, height=height, boxes=boxes.tolist(),
                        labels=labels.tolist(), ignore=ignore.tolist())

        index = {g: _parallel(record, manifest["groups"][g]) for g in GROUPS}
        index_path.write_text(json.dumps(index))
        log(f"indexed {sum(map(len, index.values()))} images in {time.time() - t0:.0f}s")
    groups = json.loads(index_path.read_text())
    for recs in groups.values():
        for r in recs:
            r["boxes"] = np.array(r["boxes"], np.float32).reshape(-1, 4)
            r["labels"] = np.array(r["labels"], np.int64)
            r["ignore"] = np.array(r["ignore"], np.float32).reshape(-1, 4)
    expected = {g: manifest["counts"][g] for g in GROUPS}
    got = {g: len(groups[g]) for g in GROUPS}
    if got != expected:
        raise RuntimeError(f"group counts {got} differ from the manifest {expected}")
    data = Data(cache, groups, manifest)
    for synth, base in SYNTH_GROUPS.items():
        data.groups[synth] = _synthetic_eval_set(data, base, synth)
    log(f"prepared in {time.time() - t0:.0f}s: {data.counts()}")
    return data


def manifest_hash():
    return hashlib.sha256(MANIFEST.read_bytes()).hexdigest()[:12]


def _synthetic_eval_set(data, base, name):
    out = data.cache / name
    marker = _step(name, data.cache)
    records = [dict(r, path=str(out / Path(r["path"]).name), key=f"{name}/{Path(r['path']).name}")
               for r in data.groups[base]]
    if not marker.exists():
        out.mkdir(exist_ok=True)
        cv2 = _cv2()

        def make(pair):
            src, dst = pair
            rng = np.random.default_rng(_seed_for(Path(src["path"]).name, EVAL_SEED))
            cv2.imwrite(dst["path"], synth_lowlight(cv2.imread(src["path"]), rng),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

        _parallel(make, list(zip(data.groups[base], records)))
        marker.touch()
    for r in records:
        r["luma"] = None
    return records


# ------------------------------------------------------------ degradation


SODIUM_RGB = (1.0, 0.72, 0.38)  # linear gains of a warm sodium-vapour cast


def synth_lowlight(img, rng, darken=True, noise=True, cast=False,
                   exposure=(0.05, 0.3), shot=(1e-4, 1e-2), read=(1e-3, 1e-2), gamma=2.2):
    """Physics-inspired low light on a BGR uint8 image (returns BGR uint8).

    sRGB -> linear (gamma 2.2); optional warm cast; exposure k ~ LogUniform;
    Poisson-Gaussian noise in linear space (shot variance a*x, a ~ LogUniform;
    read std s ~ LogUniform); clip; re-apply gamma; the caller JPEG-encodes.
    darken=False keeps normal exposure (noise-only ablation); noise=False
    darkens without noise (darkening-only ablation).
    """
    def loguniform(lo_hi):
        lo, hi = lo_hi
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))

    x = (img.astype(np.float32) / 255.0) ** gamma
    if cast:
        x = x * np.array(SODIUM_RGB[::-1], np.float32)  # BGR order
    if darken:
        x = x * loguniform(exposure)
    if noise:
        a, s = loguniform(shot), loguniform(read)
        x = x + np.sqrt(np.clip(a * x, 0, None) + s * s) * rng.standard_normal(x.shape, dtype=np.float32)
    x = np.clip(x, 0.0, 1.0) ** (1.0 / gamma)
    return np.round(x * 255.0).astype(np.uint8)


def enhance(img, method):
    """Test-time enhancement of a BGR uint8 image: none | gamma | clahe.

    gamma: exponent that maps the image's median luma to 0.35, clipped to [0.3, 1].
    clahe: CLAHE on the LAB L channel, clipLimit 2.0, 8x8 tiles.
    """
    if method == "none":
        return img
    cv2 = _cv2()
    if method == "gamma":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        s = 256 / max(h, w)
        small = cv2.resize(gray, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_LINEAR)
        median = float(np.median(small)) / 255.0
        g = 1.0 if median <= 0 else float(np.clip(np.log(0.35) / np.log(max(median, 1e-3)), 0.3, 1.0))
        lut = np.round(((np.arange(256) / 255.0) ** g) * 255.0).astype(np.uint8)
        return cv2.LUT(img, lut)
    if method == "clahe":
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    raise ValueError(f"unknown enhancement {method!r}")


# ------------------------------------------------------------- training


def _yolo_label(record):
    lines = []
    for (x, y, w, h), c in zip(record["boxes"], record["labels"]):
        cx, cy = (x + w / 2) / record["width"], (y + h / 2) / record["height"]
        lines.append(f"{c} {cx:.6f} {cy:.6f} {w / record['width']:.6f} {h / record['height']:.6f}")
    return "\n".join(lines) + ("\n" if lines else "")


def _link_split(records, root, split, image_for):
    img_dir, lbl_dir = root / "images" / split, root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    def one(record):
        src = image_for(record)
        name = Path(src).name
        dst = img_dir / name
        if not dst.exists():
            os.symlink(src, dst)
        (lbl_dir / (Path(name).stem + ".txt")).write_text(_yolo_label(record))

    _parallel(one, records)


def build_training_set(data, arm, seed, fraction=None, darken=True, noise=True, cast=False,
                       tag=None):
    """Write an Ultralytics dataset for one arm and training seed; returns data.yaml.

    Every arm has exactly len(day_train) training images; only the images differ.
      A: day_train unchanged.
      B: a seeded random `fraction` (default 0.5) of day_train replaced by synthetic
         low-light copies (darken/noise/cast switches select ablations).
      C: randomly chosen day_train images replaced by all night_train images
         (a real-night reference, not a proposed method).
    Validation during training (if enabled) uses day_tune only.
    """
    day = data.groups["day_train"]
    rng = np.random.default_rng([int(seed), 7])
    options = dict(arm=arm, seed=int(seed), fraction=fraction, darken=darken, noise=noise,
                   cast=cast, manifest=manifest_hash(), version=__version__)
    tag = tag or "_".join(f"{k}{v}" for k, v in options.items() if k not in ("manifest", "version"))
    root = RUNS / "datasets" / hashlib.sha1(json.dumps(options, sort_keys=True).encode()).hexdigest()[:10]
    marker = root / ".done"
    if marker.exists():
        return str(root / "data.yaml")
    if root.exists():
        shutil.rmtree(root)
    cv2 = _cv2()
    if arm == "A":
        train = [(r, None) for r in day]
    elif arm == "B":
        f = 0.5 if fraction is None else float(fraction)
        chosen = set(rng.choice(len(day), size=int(round(f * len(day))), replace=False).tolist())
        train = [(r, "synth" if i in chosen else None) for i, r in enumerate(day)]
    elif arm == "C":
        night = data.groups["night_train"]
        n = len(night) if fraction is None else int(round(float(fraction) * len(day)))
        n = min(n, len(night))
        keep = set(rng.choice(len(day), size=len(day) - n, replace=False).tolist())
        nights = sorted(rng.choice(len(night), size=n, replace=False).tolist())
        train = [(r, None) for i, r in enumerate(day) if i in keep] + [(night[i], "night") for i in nights]
    else:
        raise ValueError("arm must be A, B or C")
    synth_dir = root / "synth"
    synth_dir.mkdir(parents=True, exist_ok=True)

    def image_for(pair):
        record, kind = pair
        if kind != "synth":
            return record["path"]
        name = Path(record["path"]).name
        out = synth_dir / name
        img_rng = np.random.default_rng(_seed_for(name, seed))
        cv2.imwrite(str(out), synth_lowlight(cv2.imread(record["path"]), img_rng, darken=darken,
                                             noise=noise, cast=cast),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        return str(out)

    paths = _parallel(image_for, train)
    records = [dict(r, path=p) for (r, _), p in zip(train, paths)]
    _link_split(records, root, "train", lambda r: r["path"])
    _link_split(data.groups["day_tune"], root, "val", lambda r: r["path"])
    (root / "data.yaml").write_text(
        f"path: {root}\ntrain: images/train\nval: images/val\n"
        f"names: {json.dumps(dict(enumerate(CLASSES)))}\n")
    (root / "arm.json").write_text(json.dumps(dict(options, tag=tag, n_train=len(records),
                                                   n_synthetic=sum(k == "synth" for _, k in train),
                                                   n_night=sum(k == "night" for _, k in train))))
    marker.touch()
    log(f"arm {arm} seed {seed}: {len(records)} training images at {root}")
    return str(root / "data.yaml")


def arm_info(data_yaml):
    return json.loads((Path(data_yaml).parent / "arm.json").read_text())


def train(data_yaml, name, epochs, imgsz, seed, batch=16, weights="yolo11s.pt", **overrides):
    """Fine-tune an Ultralytics detector; returns (path to last.pt, info dict).

    Per-epoch validation is off (val=False) and the final weights are evaluated,
    so nothing is selected on held-out data. Pass extra Ultralytics arguments
    (lr0, optimizer, cos_lr, close_mosaic, ...) through overrides.
    """
    from ultralytics import YOLO
    import torch

    RUNS.mkdir(parents=True, exist_ok=True)
    try:
        model = YOLO(weights)
    except Exception as exc:  # e.g. a weights name unknown to this Ultralytics version
        log(f"could not load {weights} ({exc}); falling back to yolov8s.pt")
        weights, model = "yolov8s.pt", YOLO("yolov8s.pt")
    args = dict(data=data_yaml, epochs=int(epochs), imgsz=int(imgsz), batch=batch, seed=int(seed),
                device=0 if torch.cuda.is_available() else "cpu", workers=min(8, os.cpu_count() or 2),
                project=str(RUNS / "train"), name=name, exist_ok=True, val=False, plots=False,
                verbose=False, deterministic=False, amp=torch.cuda.is_available())
    args.update(overrides)
    t0 = time.time()
    model.train(**args)  # never call model.to(device) first: Ultralytics 8.4 then fails
    seconds = time.time() - t0
    last = Path(model.trainer.save_dir) / "weights" / "last.pt"
    info = dict(weights=weights, epochs=int(epochs), imgsz=int(imgsz), batch=batch, seed=int(seed),
                train_seconds=round(seconds, 1), seconds_per_epoch=round(seconds / max(1, int(epochs)), 1),
                overrides={k: v for k, v in overrides.items()})
    log(f"trained {name}: {seconds / 60:.1f} min ({info['seconds_per_epoch']} s/epoch)")
    return str(last), info


# ------------------------------------------------------------ evaluation


def predict(weights, records, imgsz, method="none", batch=16, conf=0.001, max_det=1000):
    """Detections per record as arrays (x, y, w, h, score, class) at original resolution."""
    from ultralytics import YOLO
    import torch

    cv2 = _cv2()
    model = YOLO(weights)
    device = 0 if torch.cuda.is_available() else "cpu"
    out = []
    for start in range(0, len(records), batch):
        chunk = records[start:start + batch]
        images = _parallel(lambda r: enhance(cv2.imread(r["path"]), method), chunk)
        results = model.predict(images, imgsz=int(imgsz), conf=conf, iou=0.7, max_det=max_det,
                                device=device, verbose=False)
        for res in results:
            b = res.boxes
            xyxy = b.xyxy.cpu().numpy()
            det = np.concatenate([xyxy[:, :2], xyxy[:, 2:] - xyxy[:, :2],
                                  b.conf.cpu().numpy()[:, None], b.cls.cpu().numpy()[:, None]], 1)
            out.append(det.astype(np.float32))
    return out


def coco_eval(records, detections, max_det=500):
    """COCO AP/AP50 per area bin (original resolution, maxDets=max_det as in VisDrone).

    Ignore regions and 'others' are crowd boxes for every class, so detections in
    them are neither true nor false positives. Returns (metrics, state); state
    feeds paired_clip_bootstrap.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import contextlib
    import io

    images, anns, dets = [], [], []
    for i, (r, d) in enumerate(zip(records, detections)):
        images.append(dict(id=i, width=r["width"], height=r["height"]))
        for (x, y, w, h), c in zip(r["boxes"], r["labels"]):
            anns.append(dict(id=len(anns) + 1, image_id=i, category_id=int(c), bbox=[float(x), float(y), float(w), float(h)],
                             area=float(w * h), iscrowd=0))
        for x, y, w, h in r["ignore"]:
            for c in range(len(CLASSES)):
                anns.append(dict(id=len(anns) + 1, image_id=i, category_id=c, bbox=[float(x), float(y), float(w), float(h)],
                                 area=float(w * h), iscrowd=1))
        for x, y, w, h, s, c in np.asarray(d, np.float32).reshape(-1, 6):
            dets.append(dict(image_id=i, category_id=int(c), bbox=[float(x), float(y), float(w), float(h)],
                             score=float(s)))
    gt = COCO()
    gt.dataset = dict(images=images, annotations=anns,
                      categories=[dict(id=c, name=n) for c, n in enumerate(CLASSES)])
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
        dt = gt.loadRes(dets) if dets else COCO()
        if not dets:
            dt.dataset = dict(images=images, annotations=[], categories=gt.dataset["categories"])
            dt.createIndex()
        ev = COCOeval(gt, dt, "bbox")
        ev.params.maxDets = [1, 100, max_det]
        ev.params.areaRng = list(AREA_BINS.values())
        ev.params.areaRngLbl = list(AREA_BINS)
        ev.evaluate()
        ev.accumulate()
    prec = ev.eval["precision"]  # [T, R, K, A, M]

    def ap(p):
        return float(np.mean(p[p > -1])) if np.any(p > -1) else float("nan")

    metrics = {"AP": {}, "AP50": {}, "n_gt": {}, "per_class_AP_small_all": {}}
    for a, label in enumerate(AREA_BINS):
        metrics["AP"][label] = ap(prec[:, :, :, a, -1])
        metrics["AP50"][label] = ap(prec[0, :, :, a, -1])
    for k, name in enumerate(CLASSES):
        metrics["per_class_AP_small_all"][name] = ap(prec[:, :, k, list(AREA_BINS).index("small_all"), -1])
    for label, (lo, hi) in AREA_BINS.items():
        metrics["n_gt"][label] = int(sum(((r["boxes"][:, 2] * r["boxes"][:, 3] >= lo) &
                                          (r["boxes"][:, 2] * r["boxes"][:, 3] < hi)).sum() for r in records))
    metrics["n_images"] = len(records)
    metrics["n_detections"] = len(dets)
    return metrics, _bootstrap_state(ev, records, max_det)


def _bootstrap_state(ev, records, max_det):
    """Per class and area bin: detections in COCOeval's accumulate order with their
    image index, match/ignore flags per IoU threshold, and non-ignored GT per image."""
    p = ev.params
    n_img, n_area = len(p.imgIds), len(p.areaRng)
    state = {"clips": [r["clip"] for r in records], "iou": p.iouThrs.tolist(),
             "rec": p.recThrs.tolist(), "areas": list(AREA_BINS), "cells": {}}
    for k in range(len(p.catIds)):
        for a in range(n_area):
            es = [ev.evalImgs[k * n_area * n_img + a * n_img + i] for i in range(n_img)]
            npig = np.zeros(n_img, np.int32)
            scores, imgs, dtm, dtig = [], [], [], []
            for i, e in enumerate(es):
                if e is None:
                    continue
                npig[i] = int(np.count_nonzero(e["gtIgnore"] == 0))
                s = np.asarray(e["dtScores"][:max_det])
                scores.append(s)
                imgs.append(np.full(len(s), i, np.int32))
                dtm.append(e["dtMatches"][:, :max_det])
                dtig.append(e["dtIgnore"][:, :max_det])
            if not scores:
                continue
            s = np.concatenate(scores)
            order = np.argsort(-s, kind="mergesort")
            state["cells"][(k, a)] = dict(
                img=np.concatenate(imgs)[order],
                tp=np.logical_and(np.concatenate(dtm, 1)[:, order] != 0, ~np.concatenate(dtig, 1)[:, order]),
                fp=np.logical_and(np.concatenate(dtm, 1)[:, order] == 0, ~np.concatenate(dtig, 1)[:, order]),
                npig=npig)
    return state


def weighted_ap(state, area="small_all", weights=None):
    """COCO AP (IoU 0.50:0.95) recomputed with per-image weights (1 = COCOeval)."""
    a = state["areas"].index(area)
    rec = np.asarray(state["rec"])
    values = []
    for (k, cell_area), cell in state["cells"].items():
        if cell_area != a:
            continue
        w = np.ones(len(cell["npig"])) if weights is None else np.asarray(weights, float)
        npig = float(np.dot(w, cell["npig"]))
        if npig == 0:
            continue
        dw = w[cell["img"]]
        tp = np.cumsum(cell["tp"] * dw, axis=1)
        fp = np.cumsum(cell["fp"] * dw, axis=1)
        for t in range(tp.shape[0]):
            q = np.zeros(len(rec))
            if tp.shape[1]:
                rc = tp[t] / npig
                pr = tp[t] / (tp[t] + fp[t] + np.spacing(1))
                pr = np.maximum.accumulate(pr[::-1])[::-1]
                idx = np.searchsorted(rc, rec, side="left")
                ok = idx < len(pr)
                q[ok] = pr[idx[ok]]
            values.append(q)
    return float(np.mean(values)) if values else float("nan")


def paired_clip_bootstrap(state_x, state_y, area="small_all", n=1000, seed=0):
    """AP(x) - AP(y) on the same images, with a percentile 95% CI from resampling
    clips (all frames of a clip together) with replacement."""
    clips = np.asarray(state_x["clips"])
    if list(clips) != list(state_y["clips"]):
        raise ValueError("paired bootstrap needs both states on the same images")
    unique = sorted(set(clips))
    member = np.array([unique.index(c) for c in clips])
    rng = np.random.default_rng(seed)
    point = weighted_ap(state_x, area) - weighted_ap(state_y, area)
    diffs = []
    for _ in range(n):
        counts = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        w = counts[member]
        diffs.append(weighted_ap(state_x, area, w) - weighted_ap(state_y, area, w))
    diffs = np.asarray(diffs)
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    return dict(diff=point, ci95=[float(lo), float(hi)], p_diff_le_0=float(np.mean(diffs <= 0)),
                n_boot=n, n_clips=len(unique), area=area)


def evaluate(weights, data, groups, imgsz, methods=("none",), night_methods=ENHANCEMENTS,
             batch=16, keep_state=True):
    """Evaluate one model on groups; night_test and twilight_test also get every
    method in night_methods. Returns {group: {method: {"metrics", "state"}}}."""
    out = {}
    for group in groups:
        records = data.groups[group]
        chosen = night_methods if group in ("night_test", "twilight_test") else methods
        out[group] = {}
        for method in chosen:
            t0 = time.time()
            dets = predict(weights, records, imgsz, method, batch=batch)
            metrics, state = coco_eval(records, dets)
            metrics["seconds"] = round(time.time() - t0, 1)
            out[group][method] = {"metrics": metrics, "state": state if keep_state else None}
            log(f"{group}/{method}: AP {metrics['AP']['all']:.4f} "
                f"AP_small_all {metrics['AP']['small_all']:.4f} ({metrics['seconds']}s)")
    return out


def strip_states(result):
    """Metrics only (for np.save); bootstrap states are large."""
    return {g: {m: v["metrics"] for m, v in methods.items()} for g, methods in result.items()}


def environment():
    import platform

    info = {"uavlib": __version__, "manifest": manifest_hash(), "python": platform.python_version()}
    try:
        import torch
        import ultralytics

        info.update(torch=str(torch.__version__), ultralytics=str(ultralytics.__version__),
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
    except ImportError:
        pass
    return info
