#!/usr/bin/env python3
"""Cross-type distractor test — the faithful version of ablation #2. Instead of pasting a copy of the
SAME SC port, paste a real GREEN NIC-CARD region (cropped from an SFP frame, where the NIC card IS the
target) into clean SC frames. If the SC prediction now jumps to the NIC card, the detector confuses
NIC cards (its SFP target) for the target on SC boards -> confirms the eval SC failure mechanism, which
same-type injection (#2, 95% stayed) could not reproduce."""
import json, glob, os
import numpy as np, cv2
import torch
from torchvision.transforms import functional as VF
from kp_train import KPNet, CAMS, COLS, IMAGENET_MEAN, IMAGENET_STD

CKPT = "/home/skr/aic_data/kp_v1_run/best.pt"
V1 = "/home/skr/aic_data/perception_v1"
STRIDE = 6
PATCH = 60
OFFSETS = [(90, 0), (-90, 0), (0, 70), (-70, -60), (70, 60)]
THRESH = 22

def predict(model, dev, rgb):
    t = VF.normalize(VF.to_tensor(rgb), IMAGENET_MEAN, IMAGENET_STD).unsqueeze(0).to(dev)
    with torch.no_grad():
        u, v = model(t)[0].float().cpu().numpy()
    return float(u), float(v)

def eps_of(typ):
    return [ep for ep in sorted(glob.glob(f"{V1}/episode_*"))
            if json.load(open(f"{ep}/meta.json")).get("type") == typ]

def port_px(ep, r, cam="center"):
    meta = json.load(open(f"{ep}/meta.json"))
    W = cv2.imread(glob.glob(f"{ep}/{cam}/*.png")[0]).shape[1]; scale = W/1152.0
    K = np.array(meta["intrinsics"][cam], float).reshape(3, 3)
    p = r[COLS[cam]]
    if p is None: return None
    X, Y, Z = [float(x) for x in p]
    if Z <= 0: return None
    return K[0,0]*scale*X/Z + K[0,2]*scale, K[1,1]*scale*Y/Z + K[1,2]*scale

def main():
    import pyarrow.parquet as pq
    dev = "cuda"
    model = KPNet(False).to(dev).eval()
    model.load_state_dict(torch.load(CKPT, map_location=dev, weights_only=False)["model"])

    # --- grab a real green NIC-card patch from an SFP frame (the NIC card = SFP target region) ---
    sfp = eps_of("sfp")[0]
    sdf = pq.read_table(f"{sfp}/frames.parquet").to_pandas(); sr = sdf.iloc[len(sdf)//2]
    su, sv = port_px(sfp, sr); f = int(sr["frame"]); h = PATCH//2
    simg = cv2.cvtColor(cv2.imread(f"{sfp}/center/{f:04d}.png"), cv2.COLOR_BGR2RGB)
    su, sv = int(np.clip(su, h, simg.shape[1]-h)), int(np.clip(sv, h, simg.shape[0]-h))
    nic_patch = simg[sv-h:sv+h, su-h:su+h].copy()
    print(f"NIC-card patch from SFP {os.path.basename(sfp)} @ ({su},{sv}), {PATCH}px\n")

    sc = eps_of("sc")[0]
    df = pq.read_table(f"{sc}/frames.parquet").to_pandas()
    W = cv2.imread(glob.glob(f"{sc}/center/*.png")[0]).shape[1]
    H = cv2.imread(glob.glob(f"{sc}/center/*.png")[0]).shape[0]
    jumped = stayed = other = 0; base_errs = []; total = 0
    for i in range(0, len(df), STRIDE):
        r = df.iloc[i]; pp = port_px(sc, r)
        if pp is None: continue
        ug, vg = pp; f = int(r["frame"])
        if not (h <= ug < W-h and h <= vg < H-h): continue
        img = cv2.cvtColor(cv2.imread(f"{sc}/center/{f:04d}.png"), cv2.COLOR_BGR2RGB)
        ub, vb = predict(model, dev, img); base_errs.append(np.hypot(ub-ug, vb-vg))
        pu, pv = int(round(ug)), int(round(vg))
        for dx, dy in OFFSETS:
            du, dv = pu+dx, pv+dy
            if not (h <= du < W-h and h <= dv < H-h): continue
            comp = img.copy(); comp[dv-h:dv+h, du-h:du+h] = nic_patch   # paste NIC card near SC port
            up, vp = predict(model, dev, comp)
            d_true = np.hypot(up-ug, vp-vg); d_dist = np.hypot(up-du, vp-dv); total += 1
            if d_dist < THRESH and d_dist < d_true:  jumped += 1
            elif d_true < THRESH:                     stayed += 1
            else:                                     other += 1
    print(f"SC scene {os.path.basename(sc)}, clean baseline SC-port error: med {np.median(base_errs):.1f}px")
    print(f"with a real NIC card pasted near the SC port ({total} trials):")
    print(f"  STAYED on SC port    : {stayed:3d}  ({100*stayed/total:.0f}%)")
    print(f"  JUMPED to NIC card   : {jumped:3d}  ({100*jumped/total:.0f}%)")
    print(f"  other/in-between     : {other:3d}  ({100*other/total:.0f}%)")
    print(f"\ncompare to same-type injection (#2): 95% stayed. High JUMPED here => cross-type NIC cards")
    print(f"are the eval SC failure mechanism -> fix = cross-type board population + conditioning.")

if __name__ == "__main__":
    main()
