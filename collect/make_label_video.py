#!/usr/bin/env python3
"""Render a continuous L|C|R video with the GT port projected as a red dot in each view —
the keypoint-label overlay across a whole episode (same projection as the smoke test, every frame).
Uses the cached (288x256) training images so it shows exactly what the detector trains on.

Usage: make_label_video.py --ep <episode_dir> --out <mp4> [--fps 20] [--point port_center|entrance]
"""
import argparse, json, os, subprocess, tempfile, shutil
import numpy as np, cv2, pandas as pd

CAMS = ["left", "center", "right"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--point", default="port_center", choices=["port_center", "entrance"])
    a = ap.parse_args()
    ep = a.ep
    meta = json.load(open(os.path.join(ep, "meta.json")))
    df = pd.read_parquet(os.path.join(ep, "frames.parquet"))
    # cache images are 288x256; intrinsics are native (1152x1024) -> scale by cached_w/native_w
    sample = cv2.imread(os.path.join(ep, "center", sorted(os.listdir(os.path.join(ep, "center")))[0]))
    Hc, Wc = sample.shape[:2]
    scale = Wc / meta["image_size"][0]
    Ks = {c: np.array(meta["intrinsics"][c], float).reshape(3, 3) for c in CAMS}
    # column per camera for the chosen point
    if a.point == "port_center":
        cols = {"left": "port_left_pos", "center": "port_center_pos", "right": "port_right_pos"}
    else:  # entrance is only stored in base frame -> fall back to port_center per-cam (visual only)
        cols = {"left": "port_left_pos", "center": "port_center_pos", "right": "port_right_pos"}

    tmp = tempfile.mkdtemp()
    n = 0
    for _, r in df.iterrows():
        f = int(r["frame"])
        tiles = []
        for c in CAMS:
            img = cv2.imread(os.path.join(ep, c, f"{f:04d}.png"))
            if img is None:
                img = np.zeros((Hc, Wc, 3), np.uint8)
            K = Ks[c]; fx, fy, cx, cy = K[0,0]*scale, K[1,1]*scale, K[0,2]*scale, K[1,2]*scale
            p = r[cols[c]]
            if p is not None:
                X, Y, Z = [float(v) for v in p]
                if Z > 0:
                    u, v = fx*X/Z + cx, fy*Y/Z + cy
                    cv2.circle(img, (int(round(u)), int(round(v))), 5, (0, 0, 255), -1)
                    cv2.circle(img, (int(round(u)), int(round(v))), 7, (0, 0, 255), 1)
            cv2.putText(img, c, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            tiles.append(img)
        cv2.imwrite(os.path.join(tmp, f"{n:04d}.png"), np.hstack(tiles))
        n += 1
    # ffmpeg -> mp4, upscale to ~1200 wide for viewing
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps),
                    "-i", os.path.join(tmp, "%04d.png"),
                    "-vf", "scale=1200:-2", "-pix_fmt", "yuv420p", "-crf", "24", a.out], check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    tgt = f"{meta.get('type','?')} -> {meta.get('port_frame','?')}"
    print(f"wrote {a.out}  ({n} frames @ {a.fps}fps, {tgt})")

if __name__ == "__main__":
    main()
