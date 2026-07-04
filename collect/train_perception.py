#!/usr/bin/env python3
"""M3 — perception net: 3 wrist images -> port 6-DoF pose (in center-camera frame).

The "semantic anchor" of the CoStream-style #7+#3 plan. GT labels are the free simulator
port pose (from TF), so this is a supervised regression, not end-to-end IL.

Target : port_center_pos (3, meters) + port_center_quat (4, unit) from frames.parquet.
Input  : left/center/right wrist RGB (288x256) -> shared ResNet18 -> concat -> MLP heads.
Split  : whole-episode holdout, stratified by connector type (SFP/SC) -> measures generalization
         to unseen scenes, not memorized frames.

Usage:
  python train_perception.py --data <perception_v1> --out <rundir> [--epochs 15 --stride 2]
"""
import argparse, json, os, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision as tv
from torchvision.transforms import functional as TF

CAMS = ["left", "center", "right"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def set_seed(s=0):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ----------------------------- data -----------------------------
class PortPoseDataset(Dataset):
    def __init__(self, data_root, episodes, stride=1, train=True, img_size=224):
        self.root = Path(data_root)
        self.train = train
        self.img_size = img_size
        self.samples = []   # (episode_dir, frame_idx)
        pos_all = []
        for ep in episodes:
            epdir = self.root / f"episode_{ep:04d}"
            df = pq.read_table(epdir / "frames.parquet",
                               columns=["frame", "port_center_pos", "port_center_quat"]).to_pandas()
            for i in range(0, len(df), stride):
                r = df.iloc[i]
                self.samples.append((epdir, int(r["frame"]),
                                     np.asarray(r["port_center_pos"], np.float32),
                                     np.asarray(r["port_center_quat"], np.float32)))
                pos_all.append(r["port_center_pos"])
        self.pos_all = np.asarray(pos_all, np.float32)

    def set_pos_stats(self, mean, std):
        self.pos_mean = mean.astype(np.float32); self.pos_std = std.astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, k):
        epdir, frame, pos, quat = self.samples[k]
        imgs = []
        for cam in CAMS:
            im = Image.open(epdir / cam / f"{frame:04d}.png").convert("RGB")
            t = TF.resize(TF.to_tensor(im), [self.img_size, self.img_size], antialias=True)
            t = TF.normalize(t, IMAGENET_MEAN, IMAGENET_STD)
            imgs.append(t)
        x = torch.stack(imgs, 0)                       # (3cam, 3, H, W)
        pos_n = (pos - self.pos_mean) / self.pos_std   # standardized target
        q = quat / (np.linalg.norm(quat) + 1e-9)
        return x, torch.from_numpy(pos_n), torch.from_numpy(q.astype(np.float32)), torch.from_numpy(pos)


# ----------------------------- model -----------------------------
class TrinocularPoseNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = tv.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        bb = tv.models.resnet18(weights=weights)
        self.feat_dim = bb.fc.in_features        # 512
        bb.fc = nn.Identity()
        self.backbone = bb                       # shared across the 3 cameras
        self.head = nn.Sequential(
            nn.Linear(self.feat_dim * 3, 512), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
        )
        self.pos_head = nn.Linear(256, 3)
        self.quat_head = nn.Linear(256, 4)

    def forward(self, x):                         # x: (B, 3cam, 3, H, W)
        B, C = x.shape[:2]
        f = self.backbone(x.flatten(0, 1)).view(B, C, -1).flatten(1)  # (B, 3*512)
        h = self.head(f)
        pos = self.pos_head(h)
        quat = F.normalize(self.quat_head(h), dim=-1)
        return pos, quat


def quat_geodesic_deg(qp, qt):
    d = (qp * qt).sum(-1).abs().clamp(max=1.0)
    return torch.rad2deg(2 * torch.acos(d))


# ----------------------------- train -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/media/skr/storage/aic/perception_v1")
    ap.add_argument("--out", default="/home/skr/aic_data/m3_perception_run")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--quat_w", type=float, default=1.0)
    args = ap.parse_args()
    set_seed(0)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"

    idx = pq.read_table(Path(args.data) / "index.parquet").to_pandas()
    idx = idx[idx["split"] != "reject"].reset_index(drop=True)

    # episode-wise, type-stratified holdout
    val_eps = []
    for typ, g in idx.groupby("type"):
        eps = sorted(g["episode"].tolist())
        random.Random(0).shuffle(eps)
        k = max(1, int(round(len(eps) * args.val_frac)))
        val_eps += eps[:k]
    val_eps = set(val_eps)
    train_eps = [e for e in idx["episode"] if e not in val_eps]
    print(f"episodes: {len(idx)} kept  |  train {len(train_eps)}  val {len(val_eps)}={sorted(val_eps)}")

    ds_tr = PortPoseDataset(args.data, train_eps, stride=args.stride, train=True)
    ds_va = PortPoseDataset(args.data, sorted(val_eps), stride=args.stride, train=False)
    mean = ds_tr.pos_all.mean(0); std = ds_tr.pos_all.std(0) + 1e-6
    ds_tr.set_pos_stats(mean, std); ds_va.set_pos_stats(mean, std)
    std_t = torch.tensor(std, device=dev)
    print(f"frames: train {len(ds_tr)}  val {len(ds_va)}  | pos_mean={np.round(mean,3)} pos_std={np.round(std,3)}")

    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                       pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                       pin_memory=True, persistent_workers=args.workers > 0)

    try:
        model = TrinocularPoseNet(pretrained=True).to(dev)
        print("backbone: ImageNet-pretrained resnet18")
    except Exception as e:
        print("pretrained download failed -> random init:", e)
        model = TrinocularPoseNet(pretrained=False).to(dev)

    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 1e-4},
        {"params": list(model.head.parameters()) + list(model.pos_head.parameters())
                    + list(model.quat_head.parameters()), "lr": 1e-3},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()

    def evaluate():
        model.eval()
        pos_err, ang_err = [], []
        with torch.no_grad():
            for x, pos_n, q, pos_m in dl_va:
                x = x.to(dev, non_blocking=True)
                with torch.cuda.amp.autocast():
                    pp, pq_ = model(x)
                pp_m = pp.float() * std_t + torch.tensor(mean, device=dev)
                pos_err.append((pp_m.cpu() - pos_m).norm(dim=-1) * 1000.0)   # mm
                ang_err.append(quat_geodesic_deg(pq_.float().cpu(), q))
        pe = torch.cat(pos_err); ae = torch.cat(ang_err)
        return dict(pos_mm_med=pe.median().item(), pos_mm_mean=pe.mean().item(),
                    ang_deg_med=ae.median().item(), ang_deg_mean=ae.mean().item())

    best = 1e9; hist = []
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); run = 0.0
        for x, pos_n, q, pos_m in dl_tr:
            x = x.to(dev, non_blocking=True); pos_n = pos_n.to(dev); q = q.to(dev)
            with torch.cuda.amp.autocast():
                pp, pq_ = model(x)
                loss_pos = F.smooth_l1_loss(pp, pos_n)
                loss_q = (1 - (pq_ * q).sum(-1).abs()).mean()
                loss = loss_pos + args.quat_w * loss_q
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run += loss.item()
        sched.step()
        m = evaluate(); m["epoch"] = ep; m["train_loss"] = run / len(dl_tr); m["sec"] = time.time() - t0
        hist.append(m)
        print(f"ep {ep:02d}  loss {m['train_loss']:.4f}  |  val pos med {m['pos_mm_med']:.1f}mm "
              f"mean {m['pos_mm_mean']:.1f}mm  ang med {m['ang_deg_med']:.1f}deg  ({m['sec']:.0f}s)")
        if m["pos_mm_med"] < best:
            best = m["pos_mm_med"]
            torch.save({"model": model.state_dict(), "pos_mean": mean, "pos_std": std,
                        "val_eps": sorted(val_eps), "metrics": m}, out / "best.pt")
    json.dump(hist, open(out / "history.json", "w"), indent=2)
    print(f"\nBEST val median position error: {best:.1f} mm   -> {out/'best.pt'}")


if __name__ == "__main__":
    main()
