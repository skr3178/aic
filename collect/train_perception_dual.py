#!/usr/bin/env python3
"""M7 — conditioned dual-anchor perception net: 3 wrist images + TARGET IDENTITY
-> port and plug pose (in center-camera optical frame).

Why conditioning: with multi-rail targets (M6), the same-looking scene appears in training
with different target ports -> pixels-only regression is ambiguous from neutral viewpoints
(measured: 43 mm at home-pose frames vs 18 mm at expert-steered frames; 69-220 mm at eval
where the policy steers itself). The target identity (target_module_name + port_name) is in
every index.parquet, and comes free at eval from the Task msg.

Also here: multi-root training (comma-separated --data, e.g. v1-native + v2), orientation-aware
checkpoint selection, and per-epoch FIRST-frame / CLOSE-range gates (the decision metrics).

Usage (merged, fast 224 iteration loop):
  python train_perception_dual.py \
    --data /home/skr/aic_data/perception_native,/home/skr/aic_data/perception_v2 \
    --img_size 224 --epochs 15 --out /home/skr/aic_data/m7_cond224
Final polish (after gates pass): same but --img_size 1024 --keep_aspect --batch 12 --workers 6.
"""
import argparse, json, os, random, time
from pathlib import Path
import numpy as np
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
IMAGENET_STD = [0.229, 0.224, 0.225]
ORI_W = 1.0          # checkpoint score = port_mm + plug_mm + ORI_W*(port_deg + plug_deg)
FIRST_N = 5          # "first frames" = frame index < FIRST_N (home pose, pre-steering)
EMB_DIM = 16


def set_seed(s=0):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def R_of(q):  # wxyz -> 3x3
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def T_of(p, q):
    M = np.eye(4); M[:3, :3] = R_of(q); M[:3, 3] = p; return M


def quat_of_R(Rm):  # 3x3 -> wxyz
    tr = np.trace(Rm)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2; w = 0.25 * s
        x = (Rm[2, 1] - Rm[1, 2]) / s; y = (Rm[0, 2] - Rm[2, 0]) / s; z = (Rm[1, 0] - Rm[0, 1]) / s
    else:
        i = np.argmax([Rm[0, 0], Rm[1, 1], Rm[2, 2]])
        if i == 0:
            s = np.sqrt(1 + Rm[0, 0] - Rm[1, 1] - Rm[2, 2]) * 2
            w = (Rm[2, 1] - Rm[1, 2]) / s; x = 0.25 * s
            y = (Rm[0, 1] + Rm[1, 0]) / s; z = (Rm[0, 2] + Rm[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1 + Rm[1, 1] - Rm[0, 0] - Rm[2, 2]) * 2
            w = (Rm[0, 2] - Rm[2, 0]) / s; x = (Rm[0, 1] + Rm[1, 0]) / s
            y = 0.25 * s; z = (Rm[1, 2] + Rm[2, 1]) / s
        else:
            s = np.sqrt(1 + Rm[2, 2] - Rm[0, 0] - Rm[1, 1]) * 2
            w = (Rm[1, 0] - Rm[0, 1]) / s; x = (Rm[0, 2] + Rm[2, 0]) / s
            y = (Rm[1, 2] + Rm[2, 1]) / s; z = 0.25 * s
    q = np.array([w, x, y, z]); return q / (np.linalg.norm(q) + 1e-9)


class DualPoseDataset(Dataset):
    """ep_specs: list of (episode_dir: Path, target_id: int)."""
    def __init__(self, ep_specs, stride=1, img_size=224, keep_aspect=False):
        self.img_size = img_size; self.keep_aspect = keep_aspect
        self.samples = []
        cols = ["frame", "port_center_pos", "port_center_quat", "port_base_pos", "port_base_quat",
                "plug_base_pos", "plug_base_quat"]
        for epdir, tid in ep_specs:
            df = pq.read_table(Path(epdir) / "frames.parquet", columns=cols).to_pandas()
            for i in range(0, len(df), stride):
                r = df.iloc[i]
                port_pos = np.asarray(r["port_center_pos"], np.float32)
                port_quat = np.asarray(r["port_center_quat"], np.float32)
                # derive plug-in-optical from the port correspondence
                Tbo = T_of(np.asarray(r["port_base_pos"]), np.asarray(r["port_base_quat"])) @ \
                    np.linalg.inv(T_of(np.asarray(r["port_center_pos"]), np.asarray(r["port_center_quat"])))
                Tplug = np.linalg.inv(Tbo) @ T_of(np.asarray(r["plug_base_pos"]), np.asarray(r["plug_base_quat"]))
                plug_pos = Tplug[:3, 3].astype(np.float32)
                plug_quat = quat_of_R(Tplug[:3, :3]).astype(np.float32)
                self.samples.append((Path(epdir), int(r["frame"]), port_pos, port_quat,
                                     plug_pos, plug_quat, int(tid)))
        self.port_pos = np.stack([s[2] for s in self.samples])
        self.plug_pos = np.stack([s[4] for s in self.samples])

    def set_stats(self, pm, ps, gm, gs):
        self.pm, self.ps, self.gm, self.gs = [a.astype(np.float32) for a in (pm, ps, gm, gs)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, k):
        epdir, frame, port_pos, port_quat, plug_pos, plug_quat, tid = self.samples[k]
        imgs = []
        S = self.img_size
        for cam in CAMS:
            im = Image.open(epdir / cam / f"{frame:04d}.png").convert("RGB")
            t = TF.to_tensor(im)
            if self.keep_aspect:            # resize short side -> S, center-crop SxS (true pixels)
                t = TF.center_crop(TF.resize(t, S, antialias=True), [S, S])
            else:                            # squash to SxS
                t = TF.resize(t, [S, S], antialias=True)
            imgs.append(TF.normalize(t, IMAGENET_MEAN, IMAGENET_STD))
        x = torch.stack(imgs, 0)
        port_n = (port_pos - self.pm) / self.ps
        plug_n = (plug_pos - self.gm) / self.gs
        pq_ = port_quat / (np.linalg.norm(port_quat) + 1e-9)
        gq_ = plug_quat / (np.linalg.norm(plug_quat) + 1e-9)
        return (x, torch.from_numpy(port_n), torch.from_numpy(pq_.astype(np.float32)),
                torch.from_numpy(plug_n), torch.from_numpy(gq_.astype(np.float32)),
                torch.from_numpy(port_pos), torch.from_numpy(plug_pos),
                torch.tensor(tid, dtype=torch.long), torch.tensor(frame, dtype=torch.long))


class DualPoseNet(nn.Module):
    """Conditioned: forward(x, tid). tid = vocab index of (target_module_name|port_name); 0 = <unk>."""
    def __init__(self, pretrained=True, n_targets=16, emb_dim=EMB_DIM):
        super().__init__()
        weights = tv.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        bb = tv.models.resnet18(weights=weights)
        fd = bb.fc.in_features; bb.fc = nn.Identity(); self.backbone = bb
        self.target_emb = nn.Embedding(n_targets, emb_dim)
        self.trunk = nn.Sequential(nn.Linear(fd * 3 + emb_dim, 512), nn.ReLU(True), nn.Dropout(0.2),
                                   nn.Linear(512, 256), nn.ReLU(True))
        self.port_pos = nn.Linear(256, 3); self.port_quat = nn.Linear(256, 4)
        self.plug_pos = nn.Linear(256, 3); self.plug_quat = nn.Linear(256, 4)

    def forward(self, x, tid):
        B, C = x.shape[:2]
        f = self.backbone(x.flatten(0, 1)).view(B, C, -1).flatten(1)
        h = self.trunk(torch.cat([f, self.target_emb(tid)], dim=1))
        return (self.port_pos(h), F.normalize(self.port_quat(h), dim=-1),
                self.plug_pos(h), F.normalize(self.plug_quat(h), dim=-1))


def ang_deg(qp, qt):
    return torch.rad2deg(2 * torch.acos((qp * qt).sum(-1).abs().clamp(max=1.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/skr/aic_data/perception_native,/home/skr/aic_data/perception_v2",
                    help="comma-separated dataset roots (each with index.parquet + episode_* dirs)")
    ap.add_argument("--out", default="/home/skr/aic_data/m7_cond224")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--keep_aspect", action="store_true")
    ap.add_argument("--val_eps", default="", help='force val episodes as "rootIdx:ep,..." (e.g. "1:2,1:3")')
    ap.add_argument("--resume", action="store_true", help="resume from <out>/last.pt if present")
    args = ap.parse_args()
    set_seed(0); out = Path(args.out); out.mkdir(parents=True, exist_ok=True); dev = "cuda"

    # ---- roots, vocab, episode table ----
    roots = [Path(p) for p in args.data.split(",") if p.strip()]
    vocab = {"<unk>": 0}
    entries = []            # (root_i, ep, epdir, key, type)
    for ri, root in enumerate(roots):
        idx = pq.read_table(root / "index.parquet").to_pandas()
        idx = idx[idx["split"] != "reject"].reset_index(drop=True)
        has_target = "target_module_name" in idx.columns
        for _, r in idx.iterrows():
            epdir = root / f"episode_{int(r['episode']):04d}"
            if has_target:
                key = f"{r['target_module_name']}|{r['port_name']}"
            else:
                # older index (perception_native): derive from meta.json's port_frame,
                # e.g. "task_board/nic_card_mount_0/sfp_port_0_link"
                meta = json.load(open(epdir / "meta.json"))
                _, module, port_link = meta["port_frame"].split("/")
                key = f"{module}|{port_link.removesuffix('_link')}"
            vocab.setdefault(key, len(vocab))
            entries.append((ri, int(r["episode"]), epdir, key, r["type"]))
    print(f"roots: {[str(r) for r in roots]} | episodes {len(entries)} | vocab {len(vocab)}: "
          f"{sorted(vocab)[1:]}", flush=True)

    # ---- episode-wise split, stratified by (root, type) ----
    if args.val_eps.strip():
        forced = set()
        for tok in args.val_eps.split(","):
            ri, e = tok.split(":"); forced.add((int(ri), int(e)))
        val_set = forced
    else:
        val_set = set()
        groups = {}
        for ent in entries:
            groups.setdefault((ent[0], ent[4]), []).append(ent)
        for gkey, g in sorted(groups.items()):
            g = sorted(g, key=lambda t: t[1]); random.Random(0).shuffle(g)
            k = max(1, round(len(g) * args.val_frac))
            val_set |= {(ent[0], ent[1]) for ent in g[:k]}
    tr_specs = [(ent[2], vocab[ent[3]]) for ent in entries if (ent[0], ent[1]) not in val_set]
    va_specs = [(ent[2], vocab[ent[3]]) for ent in entries if (ent[0], ent[1]) in val_set]
    print(f"split: train {len(tr_specs)} eps | val {len(va_specs)} eps = "
          f"{sorted(val_set)}", flush=True)

    ds_tr = DualPoseDataset(tr_specs, args.stride, args.img_size, args.keep_aspect)
    ds_va = DualPoseDataset(va_specs, args.stride, args.img_size, args.keep_aspect)
    pm, ps = ds_tr.port_pos.mean(0), ds_tr.port_pos.std(0) + 1e-6
    gm, gs = ds_tr.plug_pos.mean(0), ds_tr.plug_pos.std(0) + 1e-6
    ds_tr.set_stats(pm, ps, gm, gs); ds_va.set_stats(pm, ps, gm, gs)
    print(f"frames: train {len(ds_tr)} val {len(ds_va)} | port z~{pm[2]:.2f} plug z~{gm[2]:.2f}", flush=True)

    dl_tr = DataLoader(ds_tr, args.batch, shuffle=True, num_workers=args.workers, pin_memory=True,
                       drop_last=True, persistent_workers=args.workers > 0)
    dl_va = DataLoader(ds_va, args.batch, shuffle=False, num_workers=args.workers, pin_memory=True,
                       persistent_workers=args.workers > 0)

    model = DualPoseNet(True, n_targets=len(vocab)).to(dev)
    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 1e-4},
        {"params": [p for n, m in model.named_children() if n != "backbone" for p in m.parameters()], "lr": 1e-3},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    def evaluate():
        model.eval(); P, Pa, G, Ga, Zs, Fi = [], [], [], [], [], []
        with torch.no_grad():
            for x, pn, pqt, gn, gqt, pm_m, gm_m, tid, fidx in dl_va:
                x = x.to(dev); tid = tid.to(dev)
                with torch.amp.autocast("cuda"):
                    ppos, pquat, gpos, gquat = model(x, tid)
                ppos = ppos.float().cpu() * torch.tensor(ps) + torch.tensor(pm)
                gpos = gpos.float().cpu() * torch.tensor(gs) + torch.tensor(gm)
                P.append((ppos - pm_m).norm(dim=-1) * 1000); Pa.append(ang_deg(pquat.float().cpu(), pqt))
                G.append((gpos - gm_m).norm(dim=-1) * 1000); Ga.append(ang_deg(gquat.float().cpu(), gqt))
                Zs.append(pm_m[:, 2]); Fi.append(fidx)
        P, Pa, G, Ga, Zs, Fi = map(torch.cat, (P, Pa, G, Ga, Zs, Fi))
        close = Zs <= torch.quantile(Zs, 0.2)          # nearest 20% by port depth = insertion range
        first = Fi < FIRST_N                            # home-pose frames = the ambiguity gate
        m = dict(port_mm_med=P.median().item(), port_deg_med=Pa.median().item(),
                 plug_mm_med=G.median().item(), plug_deg_med=Ga.median().item(),
                 port_mm_first=P[first].median().item() if first.any() else float("nan"),
                 port_mm_close=P[close].median().item(), plug_mm_close=G[close].median().item())
        return m

    best, hist, start_ep = 1e9, [], 0
    last_path = out / "last.pt"
    if args.resume and last_path.exists():
        ck = torch.load(last_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        best, hist, start_ep = ck["best"], ck["hist"], ck["epoch"] + 1
        print(f"RESUMED from {last_path}: epoch {start_ep}/{args.epochs}, best {best:.1f}", flush=True)

    meta = dict(port_mean=pm, port_std=ps, plug_mean=gm, plug_std=gs,
                vocab=vocab, img_size=args.img_size, keep_aspect=args.keep_aspect,
                val_eps=sorted(val_set), data_roots=[str(r) for r in roots])
    for ep in range(start_ep, args.epochs):
        model.train(); t0 = time.time(); run = 0.0
        for x, pn, pqt, gn, gqt, _, _, tid, _ in dl_tr:
            x, pn, pqt, gn, gqt, tid = (t.to(dev) for t in (x, pn, pqt, gn, gqt, tid))
            with torch.amp.autocast("cuda"):
                ppos, pquat, gpos, gquat = model(x, tid)
                loss = (F.smooth_l1_loss(ppos, pn) + F.smooth_l1_loss(gpos, gn)
                        + (1 - (pquat * pqt).sum(-1).abs()).mean() + (1 - (gquat * gqt).sum(-1).abs()).mean())
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run += loss.item()
        sched.step()
        m = evaluate(); m.update(epoch=ep, train_loss=run / max(1, len(dl_tr)), sec=time.time() - t0)
        hist.append(m)
        print(f"ep {ep:02d} loss {m['train_loss']:.3f} | PORT {m['port_mm_med']:.1f}mm/{m['port_deg_med']:.1f}d "
              f"| PLUG {m['plug_mm_med']:.1f}mm/{m['plug_deg_med']:.1f}d "
              f"| FIRST {m['port_mm_first']:.1f}mm | CLOSE p{m['port_mm_close']:.1f}/g{m['plug_mm_close']:.1f}mm "
              f"({m['sec']:.0f}s)", flush=True)
        score = (m["port_mm_med"] + m["plug_mm_med"]
                 + ORI_W * (m["port_deg_med"] + m["plug_deg_med"]))
        if score < best:
            best = score
            torch.save({"model": model.state_dict(), "metrics": m, **meta}, out / "best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": ep, "best": best, "hist": hist, **meta},
                   last_path)
    json.dump(hist, open(out / "history.json", "w"), indent=2)
    print(f"\nBEST (mm+deg score {best:.1f}) -> {out/'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
