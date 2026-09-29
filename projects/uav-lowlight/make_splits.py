"""Build the pre-registered VisDrone2019-DET illumination/clip split (uav_splits.json).

Run locally once, before any experiment:
    .venv/bin/python projects/uav-lowlight/make_splits.py projects/uav-lowlight/data/visdrone

The data directory holds the extracted Ultralytics release folders
VisDrone2019-DET-{train,val,test-dev}/images. Only file names, sizes, luminance
values and group labels are written; no image content is redistributed.
"""

import json
from multiprocessing import Pool
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image

SPLITS = ("train", "val", "test-dev")
NIGHT, DAY = 0.16, 0.30  # thresholds on per-image median luma, from a visual check of val
# Visual audit (2026-09-29) of every clip containing a frame below NIGHT: in these
# clips the dark frames are overcast, shaded or low-sun daylight, not night.
DAYLIGHT_CLIPS = {"9999940", "9999955", "9999956", "9999966",
                  "9999976", "9999982", "9999997", "9999999"}
NIGHT_TEST_FRACTION = 0.6
DAY_TEST_FRACTION, DAY_TUNE_FRACTION = 0.15, 0.10
SEED = 0


def median_luma(path):
    """Median of the grayscale image resized to a 256 px long side, in [0, 1]."""
    im = Image.open(path)
    w, h = im.size
    s = 256 / max(w, h)
    small = im.convert("L").resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    return path, w, h, float(np.median(np.asarray(small, dtype=np.float32) / 255.0))


def assign(clips, weights, fractions, rng):
    """Greedy clip assignment: largest clips first, each to the group furthest
    below its target share of the total weight."""
    order = sorted(clips, key=lambda c: (-weights[c], rng.random()))
    total = sum(weights.values())
    got = {g: 0 for g in fractions}
    out = {}
    for clip in order:
        group = min(fractions, key=lambda g: (got[g] - fractions[g] * total) / max(fractions[g], 1e-9))
        out[clip] = group
        got[group] += weights[clip]
    return out


def main(root):
    root = Path(root)
    paths = [p for s in SPLITS for p in sorted((root / f"VisDrone2019-DET-{s}" / "images").glob("*.jpg"))]
    with Pool() as pool:
        measured = pool.map(median_luma, paths, chunksize=32)
    images = []
    for path, w, h, luma in measured:
        clip = path.name.split("_")[0]
        light = "night" if luma < NIGHT else "twilight" if luma < DAY else "day"
        if light == "night" and clip in DAYLIGHT_CLIPS:
            light = "twilight"  # dim daylight: never used as night
        images.append(dict(split=path.parent.parent.name.replace("VisDrone2019-DET-", ""),
                           name=path.name, clip=clip, width=w, height=h,
                           luma=round(luma, 4), light=light))
    rng = random.Random(SEED)
    night_count, day_count = {}, {}
    for im in images:
        night_count.setdefault(im["clip"], 0)
        day_count.setdefault(im["clip"], 0)
        night_count[im["clip"]] += im["light"] == "night"
        day_count[im["clip"]] += im["light"] == "day"
    night_clips = sorted(c for c, n in night_count.items() if n)
    other_clips = sorted(c for c in night_count if c not in night_clips)
    role = assign(night_clips, {c: night_count[c] for c in night_clips},
                  {"night_test": NIGHT_TEST_FRACTION, "night_train": 1 - NIGHT_TEST_FRACTION}, rng)
    role |= assign(other_clips, {c: day_count[c] for c in other_clips},
                   {"day_test": DAY_TEST_FRACTION, "day_tune": DAY_TUNE_FRACTION,
                    "day_train": 1 - DAY_TEST_FRACTION - DAY_TUNE_FRACTION}, rng)
    for im in images:
        r, light = role[im["clip"]], im["light"]
        test_clip = r in ("night_test", "day_test")
        if light == "night":
            im["group"] = r  # night_train or night_test
        elif light == "twilight":
            im["group"] = "twilight_test" if test_clip else "unused"
        elif test_clip:
            im["group"] = "day_test"  # all frames of a test clip stay out of training
        else:
            im["group"] = "day_tune" if r == "day_tune" else "day_train"
    counts = {}
    for im in images:
        counts[im["group"]] = counts.get(im["group"], 0) + 1
    clips = {g: len({im["clip"] for im in images if im["group"] == g}) for g in counts}
    manifest = {
        "description": "Pre-registered VisDrone2019-DET groups. Clip = first file-name field, "
                       "pooled over train/val/test-dev; every frame of a test clip is withheld "
                       "from training. luma = median grayscale value at a 256 px long side.",
        "thresholds": {"night_below": NIGHT, "day_from": DAY},
        "daylight_clips_audited": sorted(DAYLIGHT_CLIPS),
        "seed": SEED,
        "counts": dict(sorted(counts.items())),
        "clip_counts": dict(sorted(clips.items())),
        # group -> [["<split>/<file name>", luma], ...]; "unused" images are omitted
        "groups": {g: [[f"{im['split']}/{im['name']}", im["luma"]]
                       for im in images if im["group"] == g]
                   for g in sorted(counts) if g != "unused"},
    }
    out = Path(__file__).with_name("uav_splits.json")
    out.write_text(json.dumps(manifest, separators=(",", ":")))
    print(json.dumps({"counts": manifest["counts"], "clip_counts": manifest["clip_counts"]}, indent=1))
    print("night-test clips:", sorted({(im["clip"]) for im in images if im["group"] == "night_test"}))
    print("night-train clips:", sorted({(im["clip"]) for im in images if im["group"] == "night_train"}))


if __name__ == "__main__":
    main(sys.argv[1])
