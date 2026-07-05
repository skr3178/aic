#!/usr/bin/env python3
"""Model-prediction overlay video: run the trained keypoint detector on an episode's cached frames
and draw PREDICTED port keypoint (red) vs GT port (green), tiled L|C|R, every frame. This is the
apples-to-apples counterpart to make_label_video.py (which shows GT only) — it shows what the MODEL
outputs, so its smoothness/jitter is directly comparable to the live eval video.

Also prints frame-to-frame jitter (median |Δpixel| between consecutive frames) per camera — the
quantitative 'jaggedness' number, comparable to the eval log's jitter.

Usage: make_pred_video.py --ep <episode_dir> --ckpt ~/aic_data/kp_v1_run/best.pt --out <mp4>
"""
import argparse, json, os, subprocess, tempfile, shutil
import numpy as np, cv2, pyarrow.parquet as pq
from PIL import Image
import torch
from torchvision.transforms import functional as VF
from kp_train import KPNet, CAMS, COLS, IMAGENET_MEAN, IMAGENET_STD

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", required=True)
    ap.add_argument("--ckpt", default="/home/skr/aic_data/kp_v1_run/best.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=12)
    a = ap.parse_args()
    dev = "cuda"
    model = KPNet(False).to(dev).eval()
    model.load_state_dict(torch.load(a.ckpt, map_location=dev, weights_only=False)["model"])

    meta = json.load(open(os.path.join(a.ep, "meta.json")))
    df = pq.read_table(os.path.join(a.ep, "frames.parquet")).to_pandas()
    sample = cv2.imread(os.path.join(a.ep, "center", sorted(os.listdir(os.path.join(a.ep, "center")))[0]))
    Hc, Wc = sample.shape[:2]; scale = Wc / 1152.0
    Ks = {c: np.array(meta["intrinsics"][c], float).reshape(3, 3) for c in CAMS}

    tmp = tempfile.mkdtemp(); n = 0
    jit = {c: [] for c in CAMS}; prev = {c: None for c in CAMS}
    err = {c: [] for c in CAMS}
    for _, r in df.iterrows():
        f = int(r["frame"]); tiles = []
        for c in CAMS:
            path = os.path.join(a.ep, c, f"{f:04d}.png")
            img = cv2.imread(path)
            if img is None: img = np.zeros((Hc, Wc, 3), np.uint8)
            # model prediction (red)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            t = VF.normalize(VF.to_tensor(rgb), IMAGENET_MEAN, IMAGENET_STD).unsqueeze(0).to(dev)
            with torch.no_grad():
                u, v = model(t)[0].float().cpu().numpy()
            cv2.circle(img, (int(round(u)), int(round(v))), 4, (0, 0, 255), 2)   # red (BGR)
            if prev[c] is not None:
                jit[c].append(float(np.hypot(u - prev[c][0], v - prev[c][1])))
            prev[c] = (u, v)
            # GT port (green)
            p = r[COLS[c]]
            if p is not None:
                X, Y, Z = [float(x) for x in p]
                if Z > 0:
                    K = Ks[c]; fx, fy, cx, cy = K[0,0]*scale, K[1,1]*scale, K[0,2]*scale, K[1,2]*scale
                    ug, vg = fx*X/Z+cx, fy*Y/Z+cy
                    cv2.circle(img, (int(round(ug)), int(round(vg))), 4, (0, 255, 0), 2)  # green
                    err[c].append(float(np.hypot(u-ug, v-vg)))
            cv2.putText(img, c, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            tiles.append(img)
        cv2.imwrite(os.path.join(tmp, f"{n:05d}.png"), np.hstack(tiles)); n += 1
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps),
                    "-i", os.path.join(tmp, "%05d.png"), "-vf", "scale=1400:-2",
                    "-pix_fmt", "yuv420p", "-crf", "24", a.out], check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"wrote {a.out}  ({n} frames, {meta.get('type')})")
    for c in CAMS:
        j = np.array(jit[c]); e = np.array(err[c])
        print(f"  {c:6s} frame-to-frame jitter med {np.median(j):.1f}px | pred-vs-GT err med {np.median(e):.1f}px")

if __name__ == "__main__":
    main()
