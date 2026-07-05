#!/usr/bin/env python3
"""Lighting-sensitivity probe: take in-domain v1 episodes (where the detector is ~1-2 px from GT),
apply increasing brightness / reduced-contrast (mimicking the washed-out eval SC scene), and measure
how the predicted-vs-GT pixel error degrades. If error explodes with brightness -> lighting is the
culprit and domain randomization is the fix. If it stays low -> the eval gap is geometry/distractors,
not lighting."""
import json, os, glob
import numpy as np, cv2
import torch
from torchvision.transforms import functional as VF
from kp_train import KPNet, CAMS, COLS, IMAGENET_MEAN, IMAGENET_STD

CKPT = "/home/skr/aic_data/kp_v1_run/best.pt"
V1 = "/home/skr/aic_data/perception_v1"
STRIDE = 4
# (label, transform-fn) — the full appearance sweep: brightness, contrast, hue, saturation, blur, dark
LEVELS = [
    ("baseline", lambda t: t),
    ("bright x2.5", lambda t: VF.adjust_brightness(t, 2.5)),
    ("dark x0.4", lambda t: VF.adjust_brightness(t, 0.4)),
    ("washout b2.5,c0.5", lambda t: VF.adjust_contrast(VF.adjust_brightness(t, 2.5), 0.5)),
    ("hue +0.15", lambda t: VF.adjust_hue(t, 0.15)),
    ("hue -0.15", lambda t: VF.adjust_hue(t, -0.15)),
    ("saturation 0.2", lambda t: VF.adjust_saturation(t, 0.2)),
    ("saturation 2.0", lambda t: VF.adjust_saturation(t, 2.0)),
    ("blur k5", lambda t: VF.gaussian_blur(t, 5)),
]

def pick_eps():
    """first SFP and first SC episode in v1."""
    got = {}
    for ep in sorted(glob.glob(f"{V1}/episode_*")):
        t = json.load(open(f"{ep}/meta.json")).get("type")
        if t not in got:
            got[t] = ep
        if len(got) >= 2 and "sfp" in got and "sc" in got:
            break
    return got

def run(model, dev, ep):
    import pyarrow.parquet as pq
    meta = json.load(open(f"{ep}/meta.json"))
    df = pq.read_table(f"{ep}/frames.parquet").to_pandas()
    sample = cv2.imread(glob.glob(f"{ep}/center/*.png")[0]); W = sample.shape[1]; scale = W / 1152.0
    Ks = {c: np.array(meta["intrinsics"][c], float).reshape(3, 3) for c in CAMS}
    out = {lab: {"err": [], "conf": []} for lab, _ in LEVELS}
    idx = range(0, len(df), STRIDE)
    for i in idx:
        r = df.iloc[i]; f = int(r["frame"])
        for c in CAMS:
            p = r[COLS[c]]
            if p is None: continue
            X, Y, Z = [float(x) for x in p]
            if Z <= 0: continue
            K = Ks[c]; fx, fy, cx, cy = K[0,0]*scale, K[1,1]*scale, K[0,2]*scale, K[1,2]*scale
            ug, vg = fx*X/Z+cx, fy*Y/Z+cy
            img = cv2.cvtColor(cv2.imread(f"{ep}/{c}/{f:04d}.png"), cv2.COLOR_BGR2RGB)
            base = VF.to_tensor(img)
            for lab, fn in LEVELS:
                t = VF.normalize(fn(base).clamp(0, 1), IMAGENET_MEAN, IMAGENET_STD).unsqueeze(0).to(dev)
                with torch.no_grad():
                    hm = model.head(model.dec(model.enc(t)))          # (1,1,h,w) pre-softmax heatmap
                    B, _, H, W = hm.shape
                    pmap = torch.softmax(hm.flatten(2), dim=2).view(1, 1, H, W)
                    maxp = pmap.max().item()                          # peak confidence (uniform ~ 1/4608)
                    gy = torch.linspace(0, H-1, H, device=hm.device).view(1, 1, H, 1)
                    gx = torch.linspace(0, W-1, W, device=hm.device).view(1, 1, 1, W)
                    u = (pmap*gx).sum().item()*4; v = (pmap*gy).sum().item()*4
                out[lab]["err"].append(float(np.hypot(u-ug, v-vg)))
                out[lab]["conf"].append(maxp)
    return out, meta.get("type")

def main():
    dev = "cuda"
    model = KPNet(False).to(dev).eval()
    model.load_state_dict(torch.load(CKPT, map_location=dev, weights_only=False)["model"])
    eps = pick_eps()
    print(f"probing: {[(t, os.path.basename(e)) for t,e in eps.items()]}\n")
    for t, ep in eps.items():
        out, typ = run(model, dev, ep)
        print(f"===== {typ.upper()}  {os.path.basename(ep)} =====")
        base_med = np.median(out["baseline"]["err"])
        print(f"  {'level':22s} {'err_med':>8s} {'err_p90':>8s} {'x':>5s} {'conf(peak)':>11s}")
        for lab, _ in LEVELS:
            e = np.array(out[lab]["err"]); conf = np.array(out[lab]["conf"])
            mult = np.median(e) / max(base_med, 0.1)
            bar = "#" * int(min(50, np.median(e)))
            print(f"  {lab:22s} {np.median(e):7.1f}px {np.quantile(e,0.9):7.1f}px x{mult:4.1f} {np.median(conf):10.3f}  {bar}")
        print()

if __name__ == "__main__":
    main()
