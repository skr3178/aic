#!/usr/bin/env python3
"""Offline down-res cache for fast 224 training. NO sim / docker — pure image resize of the
already-collected native PNGs. Labels (parquet/index/meta) are resolution-independent -> copied
as-is. Resizes each cam PNG to short-side SHORT (preserving aspect, so both squash and keep_aspect
transforms work downstream). ~1/16 the pixels -> decode negligible -> training goes GPU-bound.

Usage: python make_cache.py --src ~/aic_data/perception_v2 --dst ~/aic_data/perception_v2_c256 [--short 256]
"""
import argparse, os, shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

CAMS = ["left", "center", "right"]

def resize_one(args):
    src_png, dst_png, short = args
    if os.path.exists(dst_png) and os.path.getsize(dst_png) > 0:
        return True                                  # resumable: skip already-cached
    try:
        im = Image.open(src_png).convert("RGB")
        w, h = im.size
        s = short / min(w, h)
        im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
        im.save(dst_png)
        return True
    except Exception as e:
        print("  fail", src_png, e); return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--short", type=int, default=256)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    # copy top-level labels (index.parquet/json)
    for f in ("index.parquet", "index.json"):
        if (src / f).exists(): shutil.copy2(src / f, dst / f)
    jobs = []
    eps = sorted(src.glob("episode_*"))
    for epdir in eps:
        dep = dst / epdir.name
        dep.mkdir(exist_ok=True)
        # copy per-episode labels unchanged
        for f in ("frames.parquet", "frames.jsonl", "meta.json"):
            if (epdir / f).exists(): shutil.copy2(epdir / f, dep / f)
        for cam in CAMS:
            (dep / cam).mkdir(exist_ok=True)
            for png in (epdir / cam).glob("*.png"):
                jobs.append((str(png), str(dep / cam / png.name), args.short))
    print(f"{src.name}: {len(eps)} episodes, {len(jobs)} PNGs -> {dst} (short={args.short})", flush=True)
    ok = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(resize_one, jobs)):
            ok += r
            if (i + 1) % 20000 == 0: print(f"  {i+1}/{len(jobs)}", flush=True)
    print(f"done: {ok}/{len(jobs)} PNGs cached -> {dst}", flush=True)

if __name__ == "__main__":
    main()
