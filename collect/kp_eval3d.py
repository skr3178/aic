#!/usr/bin/env python3
"""Stage 4-5 of the keypoint+PnP plan: run the trained detector on the 3 wrist cameras, back-project
each predicted pixel into base_link, triangulate the 3 rays -> predicted 3D port position, and compare
to GT port_base_pos. Reports the number that decides everything: 3D position error vs the ~80mm
regression baseline (and val 14.5mm) — same episode split as training (in-distribution here; a true
OOD set comes from eval-scene collection).

Extrinsics trick: we KNOW the port pose in base_link (port_base_pos/quat) AND in each optical frame
(port_{cam}_pos/quat). A full 6-DoF pose correspondence gives T_base_cam = T_base_port @ inv(T_cam_port)
exactly, per frame — no URDF / mount calibration. (Live policy uses TF for the same transform.)

Self-check: triangulating the GT-projected pixels must recover port_base_pos to ~0mm; if it doesn't,
the quaternion convention is wrong. We assert this before trusting the predicted-pixel numbers.

Usage:
  python kp_eval3d.py --data <native_c256>,<v2_c256> --ckpt ~/aic_data/kp_v1_run/best.pt --stride 6
"""
import argparse, json
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
from PIL import Image
import torch
from torchvision.transforms import functional as VF
from kp_train import KPNet, build_split, CAMS, COLS, IMAGENET_MEAN, IMAGENET_STD

def R_of(q):  # wxyz -> 3x3
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
                     [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])

def T_of(p, q):
    T = np.eye(4); T[:3,:3] = R_of(q); T[:3,3] = p; return T

def triangulate(origins, dirs):
    """least-squares closest point to a set of base-frame rays (origin + t*dir)."""
    A = np.zeros((3,3)); b = np.zeros(3)
    for o, d in zip(origins, dirs):
        d = d/np.linalg.norm(d); P = np.eye(3) - np.outer(d, d)
        A += P; b += P @ o
    return np.linalg.solve(A, b)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--split", default="val", choices=["val","train","all"])
    a = ap.parse_args()
    dev = "cuda"
    roots = [Path(p) for p in a.data.split(",") if p.strip()]
    tr, va = build_split(roots)
    eps = va if a.split == "val" else (tr if a.split == "train" else tr | va)
    model = KPNet(False).to(dev).eval()
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"])
    print(f"loaded {a.ckpt} (val px med at save: {ck.get('metrics',{}).get('px_med','?')})", flush=True)

    err3d_pred, err3d_gt, per_cam_px = [], [], {c: [] for c in CAMS}
    depths = []
    by_root = {}
    for root in roots:
        idx = pq.read_table(root/"index.parquet").to_pandas()
        idx = idx[idx["split"] != "reject"]
        for ep in sorted(int(e) for e in idx["episode"]):
            if (root, ep) not in eps: continue
            epdir = root/f"episode_{ep:04d}"
            meta = json.load(open(epdir/"meta.json"))
            sample = next((epdir/"center").glob("*.png")); W = Image.open(sample).size[0]
            scale = W/1152.0   # intrinsics are native-1152 in ALL datasets (v1 mislabels image_size as 288)
            Ks = {c: np.array(meta["intrinsics"][c], float).reshape(3,3)*np.array([[scale,1,scale],[1,scale,scale],[1,1,1]]) for c in CAMS}
            df = pq.read_table(epdir/"frames.parquet").to_pandas()
            rkey = root.name
            for i in range(0, len(df), a.stride):
                r = df.iloc[i]; f = int(r["frame"])
                P_base = np.asarray(r["port_base_pos"], float)
                q_base = np.asarray(r["port_base_quat"], float)
                origins, dirs_pred, dirs_gt, ok = [], [], [], True
                for c in CAMS:
                    pc = r[COLS[c]]; qc = r[f"port_{c}_quat"]
                    if pc is None or qc is None: ok = False; break
                    pc = np.asarray(pc, float); qc = np.asarray(qc, float)
                    if pc[2] <= 0: ok = False; break
                    # T_base_cam from pose correspondence (port is the shared rigid body)
                    T_base_cam = T_of(P_base, q_base) @ np.linalg.inv(T_of(pc, qc))
                    Rbc, tbc = T_base_cam[:3,:3], T_base_cam[:3,3]
                    K = Ks[c]; fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
                    # detector pixel
                    im = Image.open(epdir/c/f"{f:04d}.png").convert("RGB")
                    t = VF.normalize(VF.to_tensor(im), IMAGENET_MEAN, IMAGENET_STD).unsqueeze(0).to(dev)
                    with torch.no_grad(), torch.amp.autocast("cuda"):
                        u, v = model(t)[0].float().cpu().numpy()
                    # gt pixel (for self-check + per-cam px error)
                    ug, vg = fx*pc[0]/pc[2]+cx, fy*pc[1]/pc[2]+cy
                    per_cam_px[c].append(float(np.hypot(u-ug, v-vg)))
                    d_pred = Rbc @ np.array([(u-cx)/fx, (v-cy)/fy, 1.0])
                    d_gt   = Rbc @ np.array([(ug-cx)/fx, (vg-cy)/fy, 1.0])
                    origins.append(tbc); dirs_pred.append(d_pred); dirs_gt.append(d_gt)
                if not ok: continue
                Ppred = triangulate(origins, dirs_pred)
                Pgt   = triangulate(origins, dirs_gt)
                e_pred = float(np.linalg.norm(Ppred - P_base)); e_gt = float(np.linalg.norm(Pgt - P_base))
                err3d_pred.append(e_pred); err3d_gt.append(e_gt); depths.append(float(r["port_center_pos"][2]))
                by_root.setdefault(rkey, []).append(e_pred)

    def stats(x): x = np.array(x); return dict(med=np.median(x), mean=x.mean(), p90=np.quantile(x,0.9), n=len(x))
    eg = stats(err3d_gt)
    print(f"\n[self-check] GT-pixel triangulation 3D error: med {eg['med']*1000:.1f}mm mean {eg['mean']*1000:.1f}mm "
          f"(must be ~0; validates extrinsics + quat convention)")
    assert eg["med"] < 0.01, f"GT-pixel triangulation off by {eg['med']*1000:.0f}mm — quaternion/extrinsics bug"
    for c in CAMS:
        pc = np.array(per_cam_px[c]); print(f"  {c:6s} 2D px err: med {np.median(pc):.1f} p90 {np.quantile(pc,0.9):.1f}")
    ep_ = stats(err3d_pred)
    print(f"\n=== PREDICTED-pixel triangulated 3D port error ({a.split} split, n={ep_['n']}) ===")
    print(f"  median {ep_['med']*1000:.1f} mm | mean {ep_['mean']*1000:.1f} mm | p90 {ep_['p90']*1000:.1f} mm")
    for rk, v in by_root.items():
        v = np.array(v); print(f"  [{rk}] med {np.median(v)*1000:.1f}mm p90 {np.quantile(v,0.9)*1000:.1f}mm (n={len(v)})")
    d = np.array(depths); e = np.array(err3d_pred); far = d > np.median(d)
    print(f"  near-half (depth<med) med {np.median(e[~far])*1000:.1f}mm | far-half med {np.median(e[far])*1000:.1f}mm")
    print(f"\n  vs regression baseline: val 14.5mm / eval ~80mm.  (this = in-distribution triangulated 3D)")

if __name__ == "__main__":
    main()
