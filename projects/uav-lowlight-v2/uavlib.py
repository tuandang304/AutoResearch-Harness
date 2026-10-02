"""uavlib: data, degradation, training and evaluation helpers for the UAV low-light study.

Experiment scripts import this module from their working directory. It fixes the
parts of the protocol that must not vary between nodes: the pre-registered image
groups (uav_splits.json), annotation handling, the synthetic low-light model, the
test-time enhancements and COCO evaluation by object size at original resolution.

    import uavlib
    data = uavlib.prepare()                         # cached per VM
    yaml_a = uavlib.build_training_set(data, "A", seed)
    weights_a, info = uavlib.train(yaml_a, "A_s0", epochs=16, imgsz=1280, seed=seed)
    result_a = uavlib.evaluate(weights_a, data, ["night_test", "day_test"], imgsz=1280)
    result_a["night_test"]["none"]["metrics"]["AP"]["small_all"]
    pseudo, stats = uavlib.pseudo_label(weights_a, data, imgsz=1280)   # arm D teacher
    yaml_d = uavlib.build_training_set(data, "D", seed, pseudo=pseudo)
    uavlib.paired_clip_bootstrap(result_d["night_test"]["none"]["state"],
                                 result_a["night_test"]["none"]["state"])
    uavlib.save_state(result_a["night_test"]["none"]["state"], "working/state_A_night_test.npy")

v2.0 (2026-10-01): B replaces the same share of day images as C and D (496, 12.6%)
by default; arm D (self-training on unlabeled night_train images); compact saving of
bootstrap states; leave-one-clip-out and seed-pooled clip bootstrap; a per-VM cache
of trained weights (training is deterministic at a fixed seed on one setup).

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

__version__ = "2.0"
SYNTH_VERSION = "1.0"  # synthetic-dark evaluation sets (unchanged since 1.0)

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
# pycocotools is single-threaded Python: evaluations run in subprocesses, overlapped
# with GPU prediction (each needs a few GB of RAM for the larger groups)
EVAL_WORKERS = int(os.environ.get("UAV_EVAL_WORKERS", max(1, min(6, (os.cpu_count() or 2) - 2))))
# Bootstrap resamples run in this many fresh subprocesses (results are identical to a
# sequential run: the resampling draws are made in the caller, in the same order).
BOOT_WORKERS = int(os.environ.get("UAV_BOOT_WORKERS", max(1, min(4, (os.cpu_count() or 2) - 1))))  # memory-bound: 4 was fastest
STATE_GROUPS = ("night_test", "twilight_test")  # groups whose bootstrap state is kept
STATE_AREAS = ("all", "small_all", "medium")  # area bins kept by save_state
ENHANCED_GROUPS = ("night_test",)  # groups also evaluated with gamma and CLAHE
# Pre-registered pseudo-label confidence threshold for arm D (not tuned on night data).
PSEUDO_CONF = 0.25


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
        tmp = index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(index))
        tmp.replace(index_path)  # atomic: an interrupted job never leaves half a file
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
    out = data.cache / f"{name}_v{SYNTH_VERSION}"
    marker = _step(f"{name}_v{SYNTH_VERSION}", data.cache)
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
# Calibrated (2026-09-29) so the median-luma quantiles of synthetic day_train copies
# match night_train (10/50/90%: 0.043/0.071/0.137 vs 0.039/0.078/0.141). The
# MAET-like range (0.05, 0.3) gives 0.110/0.161/0.256, mostly brighter than night.
EXPOSURE = (0.005, 0.1)
UNCALIBRATED_EXPOSURE = (0.05, 0.3)


def light_pools(shape, rng, ambient, count=(2, 8), amplitude=(0.05, 0.6), width=(0.02, 0.06)):
    """Spatially varying illumination: ambient level plus Gaussian pools of light
    (street lamps, shop fronts), as a linear-intensity gain map of `shape`. The
    defaults give median-luma quantiles close to night_train (10/50/90%:
    0.051/0.078/0.141 on 100 day_train images)."""
    h, w = shape[:2]
    gh, gw = max(2, h // 16), max(2, w // 16)
    yy, xx = np.mgrid[0:gh, 0:gw].astype(np.float32)
    gain = np.full((gh, gw), ambient, np.float32)
    diag = float(np.hypot(gh, gw))
    for _ in range(int(rng.integers(count[0], count[1] + 1))):
        cy, cx = rng.uniform(0, gh), rng.uniform(0, gw)
        sigma = rng.uniform(*width) * diag
        amp = float(np.exp(rng.uniform(np.log(amplitude[0]), np.log(amplitude[1]))))
        gain += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma * sigma))
    return _cv2().resize(np.clip(gain, 0, 1.0), (w, h), interpolation=_cv2().INTER_CUBIC)


def synth_lowlight(img, rng, darken=True, noise=True, cast=False, pools=False,
                   exposure=EXPOSURE, shot=(1e-4, 1e-2), read=(1e-3, 1e-2), gamma=2.2):
    """Physics-inspired low light on a BGR uint8 image (returns BGR uint8).

    sRGB -> linear (gamma 2.2); optional warm cast; global exposure k ~ LogUniform
    (or, with pools=True, ambient k plus Gaussian light pools); Poisson-Gaussian
    noise in linear space (shot variance a*x, a ~ LogUniform; read std s ~
    LogUniform); clip; re-apply gamma; the caller JPEG-encodes.
    darken=False keeps normal exposure (noise-only ablation); noise=False
    darkens without noise (darkening-only ablation).
    """
    def loguniform(lo_hi):
        lo, hi = lo_hi
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))

    x = (img.astype(np.float32) / 255.0) ** gamma
    if cast:
        x = x * np.array(SODIUM_RGB[::-1], np.float32)  # BGR order
    if darken and pools:
        x = x * light_pools(x.shape, rng, loguniform(exposure))[..., None]
    elif darken:
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


def matched_fraction(data):
    """Share of day_train that C and D replace (496 / 3945 = 0.126)."""
    return len(data.groups["night_train"]) / len(data.groups["day_train"])


def build_training_set(data, arm, seed, fraction=None, darken=True, noise=True, cast=False,
                       pools=False, exposure=EXPOSURE, pseudo=None, tag=None):
    """Write an Ultralytics dataset for one arm and training seed; returns data.yaml.

    Every arm has exactly len(day_train) training images; only the images differ.
      A: day_train unchanged.
      B: a seeded random `fraction` of day_train replaced by synthetic low-light copies
         of the same images (darken/noise/cast/pools/exposure select variants). The
         default fraction is matched_fraction(data), the share that C and D replace.
      C: randomly chosen day_train images replaced by all night_train images with
         their labels (a real-night reference, not a proposed method).
      D: the same day images as C replaced by the same night_train images, labelled
         only with `pseudo` (from pseudo_label); no night ground truth is used.
    Validation during training (if enabled) uses day_tune only.
    """
    day = data.groups["day_train"]
    rng = np.random.default_rng([int(seed), 7])
    if arm == "D" and not pseudo:
        raise ValueError("arm D needs pseudo labels from pseudo_label()")
    pseudo_digest = None
    if pseudo:
        pseudo_digest = hashlib.sha1(json.dumps(
            {k: [r["boxes"].round(1).tolist(), r["labels"].tolist()] for k, r in sorted(pseudo.items())}
        ).encode()).hexdigest()[:12]
    options = dict(arm=arm, seed=int(seed), fraction=fraction, darken=darken, noise=noise,
                   cast=cast, pools=pools, exposure=list(exposure), pseudo=pseudo_digest,
                   manifest=manifest_hash(), version=__version__)
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
        f = matched_fraction(data) if fraction is None else float(fraction)
        chosen = set(rng.choice(len(day), size=int(round(f * len(day))), replace=False).tolist())
        train = [(r, "synth" if i in chosen else None) for i, r in enumerate(day)]
    elif arm in ("C", "D"):
        # identical draws for C and D, so they replace the same day images with the
        # same night images and differ only in the labels of those night images
        night = data.groups["night_train"]
        n = len(night) if fraction is None else int(round(float(fraction) * len(day)))
        n = min(n, len(night))
        keep = set(rng.choice(len(day), size=len(day) - n, replace=False).tolist())
        nights = sorted(rng.choice(len(night), size=n, replace=False).tolist())
        chosen = [night[i] for i in nights]
        if arm == "D":
            chosen = [pseudo[r["key"]] for r in chosen]
        train = [(r, None) for i, r in enumerate(day) if i in keep] + [(r, "night") for r in chosen]
    else:
        raise ValueError("arm must be A, B, C or D")
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
                                             noise=noise, cast=cast, pools=pools,
                                             exposure=exposure),
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
    import logging

    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
    import torch

    LOGGER.setLevel(logging.WARNING)  # keep the argument dump out of the reviewed output
    RUNS.mkdir(parents=True, exist_ok=True)
    # Training is deterministic at a fixed seed on one setup, so a model trained
    # earlier on this VM with identical inputs is reused instead of retrained.
    key = hashlib.sha1(json.dumps(dict(arm=arm_info(data_yaml), epochs=int(epochs), imgsz=int(imgsz),
                                       batch=batch, seed=int(seed), weights=weights,
                                       overrides=overrides, version=__version__),
                                  sort_keys=True, default=str).encode()).hexdigest()[:12]
    cached = RUNS / "train_cache" / key / "last.pt"
    if cached.exists():
        info = json.loads((cached.parent / "info.json").read_text())
        log(f"reusing trained {name} from this VM's cache ({key})")
        return str(cached), dict(info, cached=True)
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
    cached.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(last, cached)
    (cached.parent / "info.json").write_text(json.dumps(info))
    return str(last), dict(info, cached=False)


# ------------------------------------------------------------ evaluation


def predict(weights, records, imgsz, method="none", batch=16, conf=0.001, max_det=500):
    """Detections per record as arrays (x, y, w, h, score, class) at original
    resolution; at most max_det per image (500, the VisDrone convention).
    `weights` is a path or an already loaded YOLO model."""
    from ultralytics import YOLO
    import torch

    cv2 = _cv2()
    model = YOLO(weights) if isinstance(weights, (str, Path)) else weights
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


def _resample_weights(clips, n, seed):
    """Per-image weights of n clip resamples (all frames of a clip together)."""
    unique = sorted(set(clips))
    member = np.array([unique.index(c) for c in clips])
    rng = np.random.default_rng(seed)
    return [np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))[member]
            for _ in range(n)], unique


def _slim(state, area):
    """The cells of one area bin only (smaller to send to a worker process)."""
    a = state["areas"].index(area)
    return dict(state, cells={key: c for key, c in state["cells"].items() if key[1] == a})


def _mean_diffs(pairs, area, weights):
    return [float(np.mean([weighted_ap(x, area, w) - weighted_ap(y, area, w) for x, y in pairs]))
            for w in weights]


def _mean_diffs_parallel(pairs, area, weights):
    """_mean_diffs split over BOOT_WORKERS fresh interpreters (never forked after CUDA)."""
    import pickle
    import subprocess
    import sys
    import tempfile

    workers = min(BOOT_WORKERS, max(1, len(weights) // 50))
    if workers <= 1:
        return _mean_diffs(pairs, area, weights)
    tmp = RUNS / "boot_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    fd, shared = tempfile.mkstemp(dir=tmp, suffix=".states.pkl")
    os.close(fd)
    files = [shared]
    code = ("import pickle, sys; sys.path.insert(0, sys.argv[4]); import uavlib; "
            "pairs, area = pickle.load(open(sys.argv[1], 'rb')); w = pickle.load(open(sys.argv[2], 'rb')); "
            "pickle.dump(uavlib._mean_diffs(pairs, area, w), open(sys.argv[3], 'wb'))")
    try:
        with open(shared, "wb") as f:
            pickle.dump(([(_slim(x, area), _slim(y, area)) for x, y in pairs], area), f, protocol=4)
        chunks = np.array_split(np.arange(len(weights)), workers)
        procs = []
        for i, idx in enumerate(chunks):
            inp, out = shared.replace(".states.pkl", f".w{i}.pkl"), shared.replace(".states.pkl", f".d{i}.pkl")
            files += [inp, out]
            with open(inp, "wb") as f:
                pickle.dump([weights[j] for j in idx], f, protocol=4)
            procs.append((out, subprocess.Popen([sys.executable, "-c", code, shared, inp, out,
                                                 str(Path(__file__).resolve().parent)],
                                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)))
        diffs = []
        for out, proc in procs:
            _, err = proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(f"bootstrap worker failed: {err[-2000:]}")
            with open(out, "rb") as f:
                diffs += pickle.load(f)
        return diffs
    finally:
        for path in files:
            Path(path).unlink(missing_ok=True)


def paired_clip_bootstrap(state_x, state_y, area="small_all", n=1000, seed=0):
    """AP(x) - AP(y) on the same images, with a percentile 95% CI from resampling
    clips (all frames of a clip together) with replacement."""
    clips = np.asarray(state_x["clips"])
    if list(clips) != list(state_y["clips"]):
        raise ValueError("paired bootstrap needs both states on the same images")
    weights, unique = _resample_weights(list(clips), n, seed)
    point = weighted_ap(state_x, area) - weighted_ap(state_y, area)
    diffs = np.asarray(_mean_diffs_parallel([(state_x, state_y)], area, weights))
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    return dict(diff=point, ci95=[float(lo), float(hi)], p_diff_le_0=float(np.mean(diffs <= 0)),
                n_boot=n, n_clips=len(unique), area=area)


def _coco_eval_subprocess(records, dets, keep_state):
    """coco_eval in a fresh interpreter, so several run in parallel. A fresh process
    (not fork) is safe after CUDA initialisation and never re-runs the caller."""
    import pickle
    import subprocess
    import sys
    import tempfile

    tmp = RUNS / "eval_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    fd, inp = tempfile.mkstemp(dir=tmp, suffix=".in.pkl")
    os.close(fd)
    outp = inp.replace(".in.pkl", ".out.pkl")
    slim = [{k: r[k] for k in ("width", "height", "boxes", "labels", "ignore", "clip")} for r in records]
    code = ("import pickle, sys; sys.path.insert(0, sys.argv[3]); import uavlib; "
            "r, d, k = pickle.load(open(sys.argv[1], 'rb')); m, s = uavlib.coco_eval(r, d); "
            "pickle.dump((m, s if k else None), open(sys.argv[2], 'wb'), protocol=4)")
    try:
        with open(inp, "wb") as f:
            pickle.dump((slim, dets, keep_state), f, protocol=4)
        done = subprocess.run([sys.executable, "-c", code, inp, outp, str(Path(__file__).resolve().parent)],
                              capture_output=True, text=True)
        if done.returncode != 0:
            raise RuntimeError(f"coco_eval subprocess failed: {done.stderr[-2000:]}")
        with open(outp, "rb") as f:
            return pickle.load(f)
    finally:
        for path in (inp, outp):
            Path(path).unlink(missing_ok=True)


def evaluate(weights, data, groups, imgsz, methods=("none",), night_methods=ENHANCEMENTS,
             batch=16, state_groups=STATE_GROUPS, enhanced_groups=ENHANCED_GROUPS):
    """Evaluate one model on groups; enhanced_groups (night_test) also get every
    method in night_methods. Returns {group: {method: {"metrics", "state"}}}; the
    bootstrap state is kept for state_groups only (None elsewhere).

    Prediction runs on the GPU while earlier COCO evaluations run in up to
    EVAL_WORKERS parallel subprocesses."""
    from ultralytics import YOLO

    model = YOLO(weights)
    jobs = []
    t_start = time.time()
    with ThreadPoolExecutor(EVAL_WORKERS) as pool:
        for group in groups:
            records = data.groups[group]
            chosen = night_methods if group in enhanced_groups else methods
            for method in chosen:
                t0 = time.time()
                dets = predict(model, records, imgsz, method, batch=batch)
                jobs.append((group, method, round(time.time() - t0, 1),
                             pool.submit(_coco_eval_subprocess, records, dets, group in state_groups)))
        out = {}
        for group, method, predict_seconds, future in jobs:
            metrics, state = future.result()
            metrics["predict_seconds"] = predict_seconds
            out.setdefault(group, {})[method] = {"metrics": metrics, "state": state}
            log(f"{group}/{method}: AP {metrics['AP']['all']:.4f} "
                f"AP_small_all {metrics['AP']['small_all']:.4f} (predict {predict_seconds}s)")
    log(f"evaluated {len(jobs)} group/enhancement pairs in {time.time() - t_start:.0f}s")
    return out


def pseudo_label(weights, data, imgsz, conf=PSEUDO_CONF, group="night_train", batch=16):
    """Pseudo-labels for arm D: the teacher's detections with score >= conf become the
    boxes of each `group` image. No ground truth and no ignore region is used.

    Returns ({record key: pseudo record}, stats). stats["diagnostic_vs_ground_truth"]
    compares the pseudo boxes with the real labels (IoU >= 0.5, same class); it is
    reported only and must never be used to choose conf or anything else."""
    records = data.groups[group]
    dets = predict(weights, records, imgsz, "none", batch=batch, conf=conf)
    pseudo, n_boxes, n_small = {}, 0, 0
    tp = tp_small = n_gt = n_gt_small = 0
    for r, d in zip(records, dets):
        d = d[d[:, 4] >= conf]
        boxes, labels = d[:, :4].astype(np.float32), d[:, 5].astype(np.int64)
        pseudo[r["key"]] = dict(r, boxes=boxes, labels=labels, ignore=np.zeros((0, 4), np.float32),
                                pseudo=True)
        n_boxes += len(boxes)
        n_small += int((boxes[:, 2] * boxes[:, 3] < 32 ** 2).sum())
        gt_small = r["boxes"][:, 2] * r["boxes"][:, 3] < 32 ** 2
        n_gt += len(r["boxes"])
        n_gt_small += int(gt_small.sum())
        used = np.zeros(len(r["boxes"]), bool)
        for b, c in zip(boxes, labels):  # greedy matching, highest score first
            iou = _iou(b, r["boxes"]) * (r["labels"] == c) * ~used
            if len(iou) and iou.max() >= 0.5:
                j = int(iou.argmax())
                used[j] = True
                tp += 1
                tp_small += int(gt_small[j])
    stats = dict(conf=conf, group=group, n_images=len(records), n_boxes=n_boxes,
                 n_small_boxes=n_small, diagnostic_vs_ground_truth=dict(
                     precision=tp / max(1, n_boxes), recall=tp / max(1, n_gt),
                     recall_small=tp_small / max(1, n_gt_small), n_gt=n_gt, n_gt_small=n_gt_small))
    log(f"pseudo-labelled {len(records)} {group} images: {n_boxes} boxes at conf >= {conf}")
    return pseudo, stats


def _iou(box, boxes):
    """IoU of one xywh box with an (N, 4) xywh array."""
    if len(boxes) == 0:
        return np.zeros(0)
    x0 = np.maximum(box[0], boxes[:, 0])
    y0 = np.maximum(box[1], boxes[:, 1])
    x1 = np.minimum(box[0] + box[2], boxes[:, 0] + boxes[:, 2])
    y1 = np.minimum(box[1] + box[3], boxes[:, 1] + boxes[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    return inter / (box[2] * box[3] + boxes[:, 2] * boxes[:, 3] - inter + 1e-9)


def save_state(state, path, areas=STATE_AREAS):
    """Save a bootstrap state compactly (bit-packed match flags, chosen area bins), so
    later scripts and offline analyses can compare models without re-evaluating.
    Use a .npy file in ./working; load it with load_state."""
    keep = [state["areas"].index(a) for a in areas]
    cells = {}
    for (k, a), c in state["cells"].items():
        if a in keep:
            cells[f"{k},{state['areas'][a]}"] = dict(
                img=c["img"].astype(np.int32), npig=c["npig"], n=int(c["tp"].shape[1]),
                tp=np.packbits(c["tp"], axis=1), fp=np.packbits(c["fp"], axis=1))
    np.save(path, dict(kind="uavlib_state", version=__version__, clips=list(state["clips"]),
                       iou=state["iou"], rec=state["rec"], areas=list(areas), cells=cells),
            allow_pickle=True)


def load_state(path):
    """Inverse of save_state; the result works with weighted_ap and the bootstraps."""
    d = np.load(path, allow_pickle=True).item()
    cells = {}
    for key, c in d["cells"].items():
        k, area = key.split(",")
        cells[(int(k), d["areas"].index(area))] = dict(
            img=c["img"].astype(np.int64), npig=c["npig"],
            tp=np.unpackbits(c["tp"], axis=1, count=c["n"]).astype(bool),
            fp=np.unpackbits(c["fp"], axis=1, count=c["n"]).astype(bool))
    return dict(clips=d["clips"], iou=d["iou"], rec=d["rec"], areas=d["areas"], cells=cells)


def leave_one_clip_out(state_x, state_y, area="small_all"):
    """AP(x) - AP(y) with each clip dropped in turn (all its frames)."""
    clips = list(state_x["clips"])
    if clips != list(state_y["clips"]):
        raise ValueError("leave_one_clip_out needs both states on the same images")
    full = weighted_ap(state_x, area) - weighted_ap(state_y, area)
    drop = {}
    for clip in sorted(set(clips)):
        w = np.array([0.0 if c == clip else 1.0 for c in clips])
        drop[clip] = weighted_ap(state_x, area, w) - weighted_ap(state_y, area, w)
    values = np.array(list(drop.values()))
    return dict(full=full, drop_one=drop, min=float(values.min()), max=float(values.max()),
                n_sign_changes=int(np.sum(np.sign(values) != np.sign(full))), area=area)


def pooled_clip_bootstrap(pairs, area="small_all", n=1000, seed=0):
    """Mean over seeds of AP(x) - AP(y), from a list of (state_x, state_y) pairs (one
    per training seed) on the same images. Each resample draws night clips once and
    applies the same draw to every seed, so the interval reflects test-clip sampling
    of the seed-averaged effect (seed variation is not resampled)."""
    clips = list(pairs[0][0]["clips"])
    if any(list(x["clips"]) != clips or list(y["clips"]) != clips for x, y in pairs):
        raise ValueError("pooled_clip_bootstrap needs every state on the same images")
    weights, unique = _resample_weights(clips, n, seed)
    point = float(np.mean([weighted_ap(x, area) - weighted_ap(y, area) for x, y in pairs]))
    diffs = np.asarray(_mean_diffs_parallel(pairs, area, weights))
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    return dict(diff=point, ci95=[float(lo), float(hi)], p_diff_le_0=float(np.mean(diffs <= 0)),
                n_boot=n, n_clips=len(unique), n_seeds=len(pairs), area=area)


def strip_states(result):
    """Metrics only (for np.save); bootstrap states are large and not saved."""
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
