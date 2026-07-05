#!/usr/bin/env python3
"""Standalone keypoint perception module — the deployable "eyes".

KeypointPerceiver loads the trained detector and turns 3 wrist images + their live camera intrinsics
and extrinsics (T_base_cam, from TF at eval time) into a single triangulated port position in
base_link. This is the ONLY perception primitive the insertion stack needs: port_xyz in the control
frame. It is import-only and does NOT modify PerceptionInsert.py or any live policy — a standalone
building block for offline eval (kp_eval3d) and a future standalone live re-score node.

Design mirrors the validated offline path (kp_eval3d): per-camera soft-argmax pixel -> back-project
to a base-frame ray -> least-squares intersection of the 3 rays. At eval time T_base_cam comes from
TF (base_link <- {cam}/optical); offline it is reconstructed from the GT port pose correspondence.
"""
import numpy as np, torch
from torchvision.transforms import functional as VF
from kp_train import KPNet, CAMS, IMAGENET_MEAN, IMAGENET_STD


def triangulate(origins, dirs):
    """Least-squares closest point to base-frame rays (origin + t*dir). >=2 rays."""
    A = np.zeros((3, 3)); b = np.zeros(3)
    for o, d in zip(origins, dirs):
        d = np.asarray(d, float); d = d / np.linalg.norm(d)
        P = np.eye(3) - np.outer(d, d)
        A += P; b += P @ np.asarray(o, float)
    return np.linalg.solve(A, b)


class KeypointPerceiver:
    def __init__(self, ckpt, device="cuda"):
        self.device = device
        self.model = KPNet(False).to(device).eval()
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        self.model.load_state_dict(ck["model"])
        self.ckpt_metrics = ck.get("metrics", {})

    @torch.no_grad()
    def detect(self, img_rgb):
        """img_rgb: HxWx3 uint8 (native or cached res). Returns (u,v) in THAT image's pixels."""
        t = VF.normalize(VF.to_tensor(img_rgb), IMAGENET_MEAN, IMAGENET_STD).unsqueeze(0).to(self.device)
        with torch.amp.autocast(self.device if self.device != "cpu" else "cpu"):
            u, v = self.model(t)[0].float().cpu().numpy()
        return float(u), float(v)

    def detect_batch(self, imgs_rgb):
        """List of HxWx3 uint8 -> list of (u,v). One forward pass."""
        ts = torch.stack([VF.normalize(VF.to_tensor(i), IMAGENET_MEAN, IMAGENET_STD) for i in imgs_rgb]).to(self.device)
        with torch.no_grad(), torch.amp.autocast(self.device if self.device != "cpu" else "cpu"):
            uv = self.model(ts).float().cpu().numpy()
        return [(float(u), float(v)) for u, v in uv]

    def perceive(self, images, Ks, T_base_cam, return_detail=False):
        """images/Ks/T_base_cam: dict keyed by 'left'/'center'/'right'.
          images[c]     : HxWx3 uint8 (K must match this image's resolution)
          Ks[c]         : 3x3 intrinsics at the image resolution
          T_base_cam[c] : 4x4 base_link <- camera-optical transform (from TF live)
        Returns port_xyz in base_link (np.array 3). Uses all cameras with a valid image."""
        cams = [c for c in CAMS if images.get(c) is not None]
        uvs = self.detect_batch([images[c] for c in cams])
        origins, dirs, pix = [], [], {}
        for c, (u, v) in zip(cams, uvs):
            K = np.asarray(Ks[c], float); fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            T = np.asarray(T_base_cam[c], float); R, t = T[:3, :3], T[:3, 3]
            d = R @ np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
            origins.append(t); dirs.append(d); pix[c] = (u, v)
        xyz = triangulate(origins, dirs)
        if return_detail:
            return xyz, dict(pixels=pix, n_views=len(cams))
        return xyz


if __name__ == "__main__":
    # tiny self-test: reconstruct T_base_cam from GT poses on one perception_v1 frame and confirm
    # perceive() matches kp_eval3d's triangulation (sanity that the shared core is consistent).
    import json, sys, pyarrow.parquet as pq
    from pathlib import Path
    from PIL import Image
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "/home/skr/aic_data/kp_v1_run/best.pt"
    ep = Path("/home/skr/aic_data/perception_v1/episode_0000")
    meta = json.load(open(ep / "meta.json"))
    df = pq.read_table(ep / "frames.parquet").to_pandas(); r = df.iloc[len(df) // 2]

    def R_of(q):
        w, x, y, z = q
        return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    def T_of(p, q):
        T = np.eye(4); T[:3, :3] = R_of(q); T[:3, 3] = p; return T

    W = Image.open(next((ep / "center").glob("*.png"))).size[0]; scale = W / 1152.0
    Pb, qb = np.asarray(r["port_base_pos"], float), np.asarray(r["port_base_quat"], float)
    imgs, Ks, Tbc = {}, {}, {}
    for c in CAMS:
        f = int(r["frame"])
        imgs[c] = np.array(Image.open(ep / c / f"{f:04d}.png").convert("RGB"))
        Ks[c] = np.array(meta["intrinsics"][c], float).reshape(3, 3) * np.array([[scale, 1, scale], [1, scale, scale], [1, 1, 1]])
        Tbc[c] = T_of(Pb, qb) @ np.linalg.inv(T_of(np.asarray(r[f"port_{c}_pos"], float), np.asarray(r[f"port_{c}_quat"], float)))
    perc = KeypointPerceiver(ckpt)
    xyz, det = perc.perceive(imgs, Ks, Tbc, return_detail=True)
    print("GT port_base :", Pb)
    print("perceived    :", xyz)
    print(f"error        : {np.linalg.norm(xyz - Pb)*1000:.1f} mm  ({det['n_views']} views, pixels={ {k:tuple(round(x,1) for x in v) for k,v in det['pixels'].items()} })")
