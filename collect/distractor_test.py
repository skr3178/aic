#!/usr/bin/env python3
"""Ablation #2 — distractor injection. On clean in-domain v1 frames (where the detector is sharp),
paste a COPY of the port region at a second location, creating two identical ports. If the prediction
jumps to the pasted copy, the detector has no notion of WHICH port is the target -> wrong-object
lock-on -> target-conditioning is the fix. If it stays on the true port, conditioning isn't the issue."""
import json, glob, os
import numpy as np, cv2
import torch
from torchvision.transforms import functional as VF
from kp_train import KPNet, CAMS, COLS, IMAGENET_MEAN, IMAGENET_STD

CKPT = "/home/skr/aic_data/kp_v1_run/best.pt"
V1 = "/home/skr/aic_data/perception_v1"
STRIDE = 6
PATCH = 56                       # port-region patch size (px) in the 288x256 frame
# distractor placements relative to the true port (dx,dy) in px — a few spots around the frame
OFFSETS = [(90, 0), (-90, 0), (0, 70), (-70, -60), (70, 60)]
THRESH = 20                      # px: "landed on" a port if within this

def predict(model, dev, rgb):
    t = VF.normalize(VF.to_tensor(rgb), IMAGENET_MEAN, IMAGENET_STD).unsqueeze(0).to(dev)
    with torch.no_grad():
        u, v = model(t)[0].float().cpu().numpy()
    return float(u), float(v)

def pick_sfp():
    for ep in sorted(glob.glob(f"{V1}/episode_*")):
        if json.load(open(f"{ep}/meta.json")).get("type") == "sfp":
            return ep

def main():
    import pyarrow.parquet as pq
    dev = "cuda"
    model = KPNet(False).to(dev).eval()
    model.load_state_dict(torch.load(CKPT, map_location=dev, weights_only=False)["model"])
    ep = pick_sfp()
    meta = json.load(open(f"{ep}/meta.json"))
    df = pq.read_table(f"{ep}/frames.parquet").to_pandas()
    sample = cv2.imread(glob.glob(f"{ep}/center/*.png")[0]); H, W = sample.shape[:2]; scale = W/1152.0
    K = np.array(meta["intrinsics"]["center"], float).reshape(3, 3)
    fx, fy, cx, cy = K[0,0]*scale, K[1,1]*scale, K[0,2]*scale, K[1,2]*scale
    print(f"episode {os.path.basename(ep)} ({meta['type']}), frame {W}x{H}, patch {PATCH}px\n")

    jumped = stayed = other = 0; base_errs = []; comp_errs = []; total = 0
    for i in range(0, len(df), STRIDE):
        r = df.iloc[i]; f = int(r["frame"]); p = r[COLS["center"]]
        if p is None: continue
        X, Y, Z = [float(x) for x in p]
        if Z <= 0: continue
        ug, vg = fx*X/Z+cx, fy*Y/Z+cy
        if not (PATCH//2 <= ug < W-PATCH//2 and PATCH//2 <= vg < H-PATCH//2):
            continue
        img = cv2.cvtColor(cv2.imread(f"{ep}/center/{f:04d}.png"), cv2.COLOR_BGR2RGB)
        ub, vb = predict(model, dev, img)
        base_errs.append(np.hypot(ub-ug, vb-vg))
        pu, pv = int(round(ug)), int(round(vg)); h = PATCH//2
        patch = img[pv-h:pv+h, pu-h:pu+h].copy()
        for dx, dy in OFFSETS:
            du, dv = pu+dx, pv+dy
            if not (h <= du < W-h and h <= dv < H-h):
                continue
            comp = img.copy(); comp[dv-h:dv+h, du-h:du+h] = patch      # paste 2nd identical port
            up, vp = predict(model, dev, comp)
            d_true = np.hypot(up-ug, vp-vg); d_dist = np.hypot(up-du, vp-dv)
            comp_errs.append(d_true); total += 1
            if d_dist < THRESH and d_dist < d_true:   jumped += 1
            elif d_true < THRESH:                      stayed += 1
            else:                                      other += 1
    print(f"clean baseline pred-vs-GT error: med {np.median(base_errs):.1f}px")
    print(f"with a 2nd identical port pasted in ({total} trials):")
    print(f"  STAYED on true port : {stayed:3d}  ({100*stayed/total:.0f}%)")
    print(f"  JUMPED to distractor: {jumped:3d}  ({100*jumped/total:.0f}%)")
    print(f"  other/in-between    : {other:3d}  ({100*other/total:.0f}%)")
    print(f"  pred-vs-TRUE error with distractor present: med {np.median(comp_errs):.1f}px "
          f"(baseline {np.median(base_errs):.1f}px)")
    print(f"\n=> high JUMPED % confirms no-target-conditioning wrong-object lock-on.")

if __name__ == "__main__":
    main()
