#!/usr/bin/env python3
"""verify_labels.py — the make-or-break check: does the recorded port pose project onto the port
in the image, and is the port-in-base label stable across an episode while the views vary?

Projects the GT port position (in the center-camera optical frame) into the downscaled center image
using the (scaled) intrinsics, and draws a dot. If the dot lands on the port, the label chain
(TF -> camera frame -> pixel, with the right intrinsics scaling) is correct.

Usage:
  aicrun python verify_labels.py --ep ~/aic_data/smoke/episode_0000 --n 5
"""
import argparse, json, os
import numpy as np
import cv2
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", required=True)
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()
    ep = os.path.expanduser(a.ep)
    meta = json.load(open(os.path.join(ep, "meta.json")))
    df = pd.read_parquet(os.path.join(ep, "frames.parquet"))
    s = meta["image_scale"]
    K = np.array(meta["intrinsics"]["center"], dtype=float).reshape(3, 3)
    fx, fy, cx, cy = K[0, 0] * s, K[1, 1] * s, K[0, 2] * s, K[1, 2] * s
    print(f"episode {meta['episode']} ({meta['type']}) frames={len(df)} img={meta['image_size']} port={meta['port_frame']}")

    # 1) port-in-base stability across the episode (should be ~constant; views change, label fixed)
    pb = np.array([r for r in df.get("port_base_pos", []) if r is not None], dtype=float)
    if len(pb):
        print(f"  port_base_pos  mean={pb.mean(0).round(4).tolist()}  std(mm)={(pb.std(0)*1000).round(2).tolist()}")

    # 2) project port (center-cam optical) -> pixel, overlay a dot
    os.makedirs(os.path.join(ep, "overlay"), exist_ok=True)
    if "port_center_pos" not in df.columns:
        print("  !! no port_center_pos column — camera-frame lookup failed"); return
    idxs = np.linspace(0, len(df) - 1, min(a.n, len(df))).astype(int)
    hits = 0
    for i in idxs:
        r = df.iloc[i]
        p = r["port_center_pos"]
        if p is None:
            print(f"  frame {i}: no port_center_pos"); continue
        X, Y, Z = p
        if Z <= 0:
            print(f"  frame {i}: Z<=0 ({Z:.3f}) — behind camera?"); continue
        u, v = fx * X / Z + cx, fy * Y / Z + cy
        f = int(r["frame"])
        img = cv2.imread(os.path.join(ep, "center", f"{f:04d}.png"))
        H, W = img.shape[:2]
        inb = 0 <= u < W and 0 <= v < H
        hits += inb
        cv2.circle(img, (int(round(u)), int(round(v))), 4, (0, 0, 255), -1)
        cv2.imwrite(os.path.join(ep, "overlay", f"{f:04d}.png"), img)
        print(f"  frame {i}: port@cam Z={Z:.3f}m -> pixel ({u:.1f},{v:.1f})  {'IN' if inb else 'OUT of'} {W}x{H}")
    print(f"  {hits}/{len(idxs)} projections landed inside the image; overlays -> {ep}/overlay/")


if __name__ == "__main__":
    main()
