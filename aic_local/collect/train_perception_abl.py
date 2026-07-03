#!/usr/bin/env python3
"""Resolution ablation of the M3 perception net (port-pose regression).
Same net/arch/split as train_perception.py, but exposes --img_size (network input resolution)
and reports depth-binned position error (||port_center_pos|| = camera->port range) so we can
see whether higher camera quality tightens the CLOSE-APPROACH localization.

Run 3x on the native dataset with --img_size 224 / 384 / 512 and compare.
Usage: python train_perception_abl.py --data <perception_native> --out <rundir> --img_size 384
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
IMAGENET_MEAN = [0.485, 0.456, 0.406]; IMAGENET_STD = [0.229, 0.224, 0.225]

def set_seed(s=0):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

class PortPoseDataset(Dataset):
    def __init__(self, data_root, episodes, stride=1, img_size=224):
        self.root = Path(data_root); self.img_size = img_size
        self.samples = []; pos_all = []
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
    def __len__(self): return len(self.samples)
    def __getitem__(self, k):
        epdir, frame, pos, quat = self.samples[k]
        imgs = []
        for cam in CAMS:
            im = Image.open(epdir / cam / f"{frame:04d}.png").convert("RGB")
            t = TF.resize(TF.to_tensor(im), [self.img_size, self.img_size], antialias=True)
            t = TF.normalize(t, IMAGENET_MEAN, IMAGENET_STD)
            imgs.append(t)
        x = torch.stack(imgs, 0)
        pos_n = (pos - self.pos_mean) / self.pos_std
        q = quat / (np.linalg.norm(quat) + 1e-9)
        return x, torch.from_numpy(pos_n), torch.from_numpy(q.astype(np.float32)), torch.from_numpy(pos)

class TrinocularPoseNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = tv.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        bb = tv.models.resnet18(weights=weights); self.feat_dim = bb.fc.in_features
        bb.fc = nn.Identity(); self.backbone = bb
        self.head = nn.Sequential(nn.Linear(self.feat_dim*3, 512), nn.ReLU(True), nn.Dropout(0.2),
                                  nn.Linear(512, 256), nn.ReLU(True))
        self.pos_head = nn.Linear(256, 3); self.quat_head = nn.Linear(256, 4)
    def forward(self, x):
        B, C = x.shape[:2]
        f = self.backbone(x.flatten(0,1)).view(B, C, -1).flatten(1)
        h = self.head(f)
        return self.pos_head(h), F.normalize(self.quat_head(h), dim=-1)

def quat_geodesic_deg(qp, qt):
    d = (qp*qt).sum(-1).abs().clamp(max=1.0); return torch.rad2deg(2*torch.acos(d))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/skr/aic_data/perception_native")
    ap.add_argument("--out", default="/home/skr/aic_data/abl_run")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    set_seed(0); out = Path(args.out); out.mkdir(parents=True, exist_ok=True); dev = "cuda"

    idx = pq.read_table(Path(args.data)/"index.parquet").to_pandas()
    idx = idx[idx["split"] != "reject"].reset_index(drop=True)
    val_eps = []
    for typ, g in idx.groupby("type"):
        eps = sorted(g["episode"].tolist()); random.Random(0).shuffle(eps)
        val_eps += eps[:max(1, int(round(len(eps)*args.val_frac)))]
    val_eps = set(val_eps); train_eps = [e for e in idx["episode"] if e not in val_eps]
    print(f"[img_size={args.img_size}] episodes {len(idx)} | train {len(train_eps)} val {sorted(val_eps)}")

    ds_tr = PortPoseDataset(args.data, train_eps, stride=args.stride, img_size=args.img_size)
    ds_va = PortPoseDataset(args.data, sorted(val_eps), stride=args.stride, img_size=args.img_size)
    mean = ds_tr.pos_all.mean(0); std = ds_tr.pos_all.std(0)+1e-6
    ds_tr.set_pos_stats(mean, std); ds_va.set_pos_stats(mean, std)
    std_t = torch.tensor(std, device=dev); mean_t = torch.tensor(mean, device=dev)
    print(f"frames: train {len(ds_tr)} val {len(ds_va)}")

    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                       pin_memory=True, drop_last=True, persistent_workers=args.workers>0)
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                       pin_memory=True, persistent_workers=args.workers>0)
    model = TrinocularPoseNet(pretrained=True).to(dev)
    opt = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": 1e-4},
        {"params": list(model.head.parameters())+list(model.pos_head.parameters())+list(model.quat_head.parameters()), "lr": 1e-3}],
        weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()

    def evaluate():
        model.eval(); pe_all, ae_all, depth_all = [], [], []
        with torch.no_grad():
            for x, pos_n, q, pos_m in dl_va:
                x = x.to(dev, non_blocking=True)
                with torch.cuda.amp.autocast():
                    pp, pq_ = model(x)
                pp_m = pp.float()*std_t + mean_t
                pe_all.append((pp_m.cpu()-pos_m).norm(dim=-1)*1000.0)
                ae_all.append(quat_geodesic_deg(pq_.float().cpu(), q))
                depth_all.append(pos_m.norm(dim=-1))          # ||port_center_pos|| = camera->port range
        pe = torch.cat(pe_all); ae = torch.cat(ae_all); dep = torch.cat(depth_all)
        # depth-binned median error
        bins = {}
        for lo, hi, name in [(0,0.28,"<0.28m(near)"),(0.28,0.32,"0.28-0.32"),(0.32,0.36,"0.32-0.36"),(0.36,9,">0.36m(far)")]:
            m = (dep>=lo)&(dep<hi)
            if m.any(): bins[name] = round(pe[m].median().item(),1)
        # closest 20%
        thr = torch.quantile(dep, 0.2); near = dep<=thr
        return dict(pos_mm_med=pe.median().item(), pos_mm_mean=pe.mean().item(),
                    ang_deg_med=ae.median().item(), close20_med=round(pe[near].median().item(),1),
                    depth_bins=bins)

    best = 1e9; hist = []
    for ep in range(args.epochs):
        model.train(); t0=time.time(); run=0.0
        for x, pos_n, q, pos_m in dl_tr:
            x=x.to(dev, non_blocking=True); pos_n=pos_n.to(dev); q=q.to(dev)
            with torch.cuda.amp.autocast():
                pp, pq_ = model(x)
                loss = F.smooth_l1_loss(pp, pos_n) + (1-(pq_*q).sum(-1).abs()).mean()
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run += loss.item()
        sched.step(); m = evaluate(); m["epoch"]=ep; m["train_loss"]=run/len(dl_tr); m["sec"]=time.time()-t0
        hist.append(m)
        print(f"ep {ep:02d} loss {m['train_loss']:.4f} | val med {m['pos_mm_med']:.1f}mm mean {m['pos_mm_mean']:.1f}mm "
              f"close20 {m['close20_med']}mm ang {m['ang_deg_med']:.1f}deg ({m['sec']:.0f}s)")
        if m["pos_mm_med"] < best:
            best = m["pos_mm_med"]
            torch.save({"model": model.state_dict(), "pos_mean": mean, "pos_std": std,
                        "img_size": args.img_size, "metrics": m}, out/"best.pt")
    json.dump({"img_size": args.img_size, "history": hist, "best_med_mm": best}, open(out/"result.json","w"), indent=2)
    print(f"[img_size={args.img_size}] BEST val median {best:.1f}mm | final depth_bins {hist[-1]['depth_bins']}")

if __name__ == "__main__":
    main()
