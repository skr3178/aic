#!/usr/bin/env python3
"""Keypoint detector — FIRST PASS of the keypoint+PnP plan.
Predicts the (verified) port-center keypoint pixel in each wrist image, on the merged 288x256 caches.
Heatmap head + soft-argmax -> (u,v); L2 loss on the pixel. Tests the core OOD hypothesis:
does keypoint DETECTION generalize better than the current global pose REGRESSION (~80mm eval)?

At the cache, 1 px ~= 1 mm at the port depth (Z~0.3m, fx_cache~309 -> Z/fx~0.97mm/px), so the
reported val pixel error ~= lateral mm error — directly comparable to the regression baseline.

Usage:
  python kp_train.py --data ~/aic_data/perception_native_c256,~/aic_data/perception_v2_c256 \
     --out ~/aic_data/kp_v1_run --epochs 15
"""
import argparse, json, os, random, time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision as tv
from torchvision.transforms import functional as VF

CAMS = ["left", "center", "right"]
COLS = {"left": "port_left_pos", "center": "port_center_pos", "right": "port_right_pos"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]; IMAGENET_STD = [0.229, 0.224, 0.225]
DOWN = 4                      # heatmap is input/DOWN

def set_seed(s=0):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

class KPDataset(Dataset):
    """One sample = (one camera image, its GT port-center pixel)."""
    def __init__(self, roots, split_eps, stride=3, train=True):
        self.samples = []; self.train = train
        for root in roots:
            root = Path(root)
            idx = pq.read_table(root / "index.parquet").to_pandas()
            idx = idx[idx["split"] != "reject"]
            keep = set(int(e) for e in idx["episode"])
            for ep in sorted(keep):
                if (root, ep) not in split_eps:  # split_eps holds (root,ep) tuples for THIS subset
                    continue
                epdir = root / f"episode_{ep:04d}"
                meta = json.load(open(epdir / "meta.json"))
                # cache is 288x256; intrinsics are native -> scale by cached_w/native_w
                sample_png = next((epdir / "center").glob("*.png"))
                W = Image.open(sample_png).size[0]
                scale = W / 1152.0   # intrinsics are native-1152 in ALL datasets (v1 mislabels image_size as 288)
                Ks = {c: (np.array(meta["intrinsics"][c], float).reshape(3, 3)) for c in CAMS}
                df = pq.read_table(epdir / "frames.parquet",
                                   columns=["frame"] + list(COLS.values())).to_pandas()
                for i in range(0, len(df), stride):
                    r = df.iloc[i]; f = int(r["frame"])
                    for c in CAMS:
                        p = r[COLS[c]]
                        if p is None: continue
                        X, Y, Z = [float(v) for v in p]
                        if Z <= 0: continue
                        K = Ks[c]; fx, fy, cx, cy = K[0,0]*scale, K[1,1]*scale, K[0,2]*scale, K[1,2]*scale
                        u, v = fx*X/Z + cx, fy*Y/Z + cy
                        self.samples.append((str(epdir / c / f"{f:04d}.png"), u, v))
    def __len__(self): return len(self.samples)
    def __getitem__(self, k):
        path, u, v = self.samples[k]
        im = Image.open(path).convert("RGB")
        t = VF.normalize(VF.to_tensor(im), IMAGENET_MEAN, IMAGENET_STD)
        return t, torch.tensor([u, v], dtype=torch.float32)

class KPNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        w = tv.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        bb = tv.models.resnet18(weights=w)
        self.enc = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool,
                                 bb.layer1, bb.layer2, bb.layer3, bb.layer4)   # -> 512, /32
        def up(ci, co):
            return nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                 nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(True))
        self.dec = nn.Sequential(up(512, 256), up(256, 128), up(128, 64))      # /32 -> /4
        self.head = nn.Conv2d(64, 1, 1)
    def forward(self, x):
        hm = self.head(self.dec(self.enc(x)))                                  # (B,1,H/4,W/4)
        B, _, H, W = hm.shape
        p = F.softmax(hm.flatten(2), dim=2).view(B, 1, H, W)                   # spatial softmax
        gy = torch.linspace(0, H-1, H, device=hm.device).view(1,1,H,1)
        gx = torch.linspace(0, W-1, W, device=hm.device).view(1,1,1,W)
        uy = (p*gy).sum((2,3)); ux = (p*gx).sum((2,3))                          # soft-argmax (heatmap coords)
        return torch.cat([ux, uy], 1) * DOWN                                   # -> image pixel (u,v)

def build_split(roots, val_frac=0.15, seed=0):
    """episode-wise split, stratified by (root, type); returns (train_set, val_set) of (root,ep)."""
    groups = {}
    for root in roots:
        root = Path(root); idx = pq.read_table(root / "index.parquet").to_pandas()
        idx = idx[idx["split"] != "reject"]
        for _, r in idx.iterrows():
            groups.setdefault((str(root), r.get("type", "?")), []).append((root, int(r["episode"])))
    val = set()
    for g in sorted(groups):
        lst = sorted(groups[g], key=lambda t: t[1]); random.Random(seed).shuffle(lst)
        val |= set(lst[:max(1, round(len(lst)*val_frac))])
    allset = set(x for g in groups.values() for x in g)
    return allset - val, val

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="/home/skr/aic_data/kp_v1_run")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    set_seed(0); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); dev = "cuda"
    roots = [Path(p) for p in a.data.split(",") if p.strip()]
    tr_eps, va_eps = build_split(roots)
    print(f"episodes: train {len(tr_eps)} val {len(va_eps)}", flush=True)
    ds_tr = KPDataset(roots, tr_eps, stride=a.stride, train=True)
    ds_va = KPDataset(roots, va_eps, stride=a.stride, train=False)
    print(f"samples (image-views): train {len(ds_tr)} val {len(ds_va)}", flush=True)
    dl_tr = DataLoader(ds_tr, a.batch, shuffle=True, num_workers=a.workers, pin_memory=True,
                       drop_last=True, persistent_workers=a.workers>0)
    dl_va = DataLoader(ds_va, a.batch, shuffle=False, num_workers=a.workers, pin_memory=True,
                       persistent_workers=a.workers>0)
    model = KPNet(True).to(dev)
    opt = torch.optim.AdamW([{"params": model.enc.parameters(), "lr": 1e-4},
                             {"params": list(model.dec.parameters())+list(model.head.parameters()), "lr": 1e-3}],
                            weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    scaler = torch.amp.GradScaler("cuda")

    def evaluate():
        model.eval(); errs = []
        with torch.no_grad():
            for x, gt in dl_va:
                x = x.to(dev)
                with torch.amp.autocast("cuda"):
                    pr = model(x)
                errs.append((pr.float().cpu() - gt).norm(dim=1))   # px error
        e = torch.cat(errs)
        return dict(px_med=e.median().item(), px_mean=e.mean().item(), px_p90=e.quantile(0.9).item())

    best, hist, start = 1e9, [], 0
    last = out / "last.pt"
    if a.resume and last.exists():
        ck = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"]); best, hist, start = ck["best"], ck["hist"], ck["epoch"]+1
        print(f"RESUMED at epoch {start}, best {best:.1f}px", flush=True)
    for ep in range(start, a.epochs):
        model.train(); t0 = time.time(); run = 0.0
        for x, gt in dl_tr:
            x, gt = x.to(dev), gt.to(dev)
            with torch.amp.autocast("cuda"):
                pr = model(x); loss = F.smooth_l1_loss(pr, gt)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run += loss.item()
        sched.step(); m = evaluate(); m.update(epoch=ep, train_loss=run/len(dl_tr), sec=time.time()-t0)
        hist.append(m)
        print(f"ep {ep:02d} loss {m['train_loss']:.2f} | val px med {m['px_med']:.1f} "
              f"mean {m['px_mean']:.1f} p90 {m['px_p90']:.1f}  (~mm at port depth)  ({m['sec']:.0f}s)", flush=True)
        if m["px_med"] < best:
            best = m["px_med"]; torch.save({"model": model.state_dict(), "metrics": m}, out/"best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": ep, "best": best, "hist": hist}, last)
    json.dump(hist, open(out/"history.json", "w"), indent=2)
    print(f"\nBEST val median pixel error: {best:.1f} px  (~{best:.0f} mm lateral at port depth) -> {out/'best.pt'}", flush=True)

if __name__ == "__main__":
    main()
