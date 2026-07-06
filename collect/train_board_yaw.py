#!/usr/bin/env python3
"""Board-orientation estimator (Stage 2b-orient). Predicts board orientation in the CENTER-camera frame
(so it composes with FK at inference to give board pose in base_link — handles the moving camera).
Board is flat, so what we ultimately want is board YAW in base; port orientation then follows as
port_orient = board_yaw ⊗ offset[type].

Label (free, from existing data): R_center_board = R(port_center_quat) · inv(R_offset[type]).
Bet: the board is big/always-visible, so estimating its orientation should generalize better OOD than
the tiny port. We VALIDATE board-yaw error on held-out + the v1 OOD proxy to check that bet.

Usage: train_board_yaw.py --data <native_c256>,<v2_c256> --out ~/aic_data/boardyaw_run --epochs 12
"""
import argparse, json, math, os, random, time
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision as tv
from torchvision.transforms import functional as VF
from torch.utils.data import Dataset, DataLoader

IMAGENET_MEAN=[0.485,0.456,0.406]; IMAGENET_STD=[0.229,0.224,0.225]
CAMS=["left","center","right"]
# measured port-in-board orientation offset (R_board_port), quat wxyz, per type (M16)
R_OFFSET={"sfp":[0.00632,0.99998,0.0,0.0], "sc":[0.00028,-0.70682,0.70739,-0.00028]}

def R_of(q):
    w,x,y,z=q; return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def quat_of(R):
    w=math.sqrt(max(0,1+R[0,0]+R[1,1]+R[2,2]))/2
    if w<1e-8: return np.array([1,0,0,0.])
    x=(R[2,1]-R[1,2])/(4*w);y=(R[0,2]-R[2,0])/(4*w);z=(R[1,0]-R[0,1])/(4*w);q=np.array([w,x,y,z]);return q/np.linalg.norm(q)
def yaw_of(R): return math.degrees(math.atan2(R[1,0],R[0,0]))

def set_seed(s=0): random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)

class BoardDS(Dataset):
    """one sample = (3 cam images, board-orientation quat in center-cam frame, port_center_quat for val)."""
    def __init__(self, roots, split_eps, stride=4):
        self.s=[]
        for root in roots:
            root=Path(root); idx=pq.read_table(root/"index.parquet").to_pandas(); idx=idx[idx["split"]!="reject"]
            for ep in sorted(int(e) for e in idx["episode"]):
                if (root,ep) not in split_eps: continue
                epd=root/f"episode_{ep:04d}"; meta=json.load(open(epd/"meta.json")); typ=meta.get("type")
                if typ not in R_OFFSET: continue
                df=pq.read_table(epd/"frames.parquet",columns=["frame","port_center_quat"]).to_pandas()
                for i in range(0,len(df),stride):
                    r=df.iloc[i]; q=r["port_center_quat"]
                    if q is None: continue
                    self.s.append((str(epd),int(r["frame"]),np.asarray(q,float),typ))
    def __len__(self): return len(self.s)
    def __getitem__(self,k):
        epd,f,pcq,typ=self.s[k]
        imgs=[]
        for c in CAMS:
            im=Image.open(f"{epd}/{c}/{f:04d}.png").convert("RGB")
            imgs.append(VF.normalize(VF.to_tensor(im),IMAGENET_MEAN,IMAGENET_STD))
        # label: R_center_board = R(port_center_quat) @ inv(R_offset[type])
        Rcb=R_of(pcq)@np.linalg.inv(R_of(R_OFFSET[typ]))
        qcb=quat_of(Rcb)
        return torch.stack(imgs), torch.tensor(qcb,dtype=torch.float32), torch.tensor(pcq,dtype=torch.float32)

class BoardNet(nn.Module):
    def __init__(self):
        super().__init__()
        bb=tv.models.resnet18(weights=tv.models.ResNet18_Weights.IMAGENET1K_V1); fd=bb.fc.in_features; bb.fc=nn.Identity()
        self.bb=bb
        self.trunk=nn.Sequential(nn.Linear(fd*3,512),nn.ReLU(True),nn.Dropout(0.2),nn.Linear(512,256),nn.ReLU(True),nn.Linear(256,4))
    def forward(self,x):  # x: (B,3,3,H,W)
        B,C=x.shape[:2]; f=self.bb(x.flatten(0,1)).view(B,C,-1).flatten(1)
        return F.normalize(self.trunk(f),dim=-1)

def build_split(roots,val_frac=0.15,seed=0):
    groups={}
    for root in roots:
        root=Path(root); idx=pq.read_table(root/"index.parquet").to_pandas(); idx=idx[idx["split"]!="reject"]
        for _,r in idx.iterrows(): groups.setdefault((str(root),r.get("type","?")),[]).append((root,int(r["episode"])))
    val=set()
    for g in sorted(groups):
        lst=sorted(groups[g],key=lambda t:t[1]); random.Random(seed).shuffle(lst); val|=set(lst[:max(1,round(len(lst)*val_frac))])
    allset=set(x for g in groups.values() for x in g); return allset-val, val

def board_yaw_err(qcb_pred, pcq):
    """board-yaw error (deg): compose predicted center-frame board-orient with the GT camera pose
    (recovered as R_base_center via the port correspondence is NOT needed — board yaw in base equals
    yaw of R_base_center@R_center_board; but for a metric we compare pred vs label DIRECTLY in center
    frame as base-yaw is monotone in it). Simplest faithful metric: yaw of R_base_board where
    R_base_center is approximated identity-free via label consistency -> use geodesic of center-frame quats."""
    # geodesic angle between predicted and label center-frame board orientation (deg)
    d=torch.clamp((qcb_pred*pcq).sum(-1).abs(),0,1)   # placeholder; replaced below
    return d

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data",required=True); ap.add_argument("--out",default="/home/skr/aic_data/boardyaw_run")
    ap.add_argument("--epochs",type=int,default=12); ap.add_argument("--stride",type=int,default=4); ap.add_argument("--batch",type=int,default=48)
    ap.add_argument("--workers",type=int,default=10); a=ap.parse_args()
    set_seed(0); out=Path(a.out); out.mkdir(parents=True,exist_ok=True); dev="cuda"
    roots=[Path(p) for p in a.data.split(",") if p.strip()]; tr,va=build_split(roots)
    print(f"episodes: train {len(tr)} val {len(va)}",flush=True)
    dtr=BoardDS(roots,tr,a.stride); dva=BoardDS(roots,va,a.stride)
    print(f"samples: train {len(dtr)} val {len(dva)}",flush=True)
    ltr=DataLoader(dtr,a.batch,shuffle=True,num_workers=a.workers,pin_memory=True,drop_last=True,persistent_workers=True)
    lva=DataLoader(dva,a.batch,shuffle=False,num_workers=a.workers,pin_memory=True,persistent_workers=True)
    net=BoardNet().to(dev)
    opt=torch.optim.AdamW([{"params":net.bb.parameters(),"lr":1e-4},{"params":net.trunk.parameters(),"lr":1e-3}],weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs); scaler=torch.amp.GradScaler("cuda")

    def geo_deg(qp,ql):  # geodesic angle (deg) between quats
        d=torch.clamp((qp*ql).sum(-1).abs(),0,1); return torch.rad2deg(2*torch.arccos(d))
    def evaluate():
        net.eval(); es=[]
        with torch.no_grad():
            for x,qcb,pcq in lva:
                x=x.to(dev)
                with torch.amp.autocast("cuda"): p=net(x)
                es.append(geo_deg(p.float().cpu(),qcb))
        e=torch.cat(es); return dict(med=e.median().item(),mean=e.mean().item(),p90=e.quantile(0.9).item())
    best=1e9; hist=[]
    for ep in range(a.epochs):
        net.train(); t0=time.time(); run=0.
        for x,qcb,pcq in ltr:
            x,qcb=x.to(dev),qcb.to(dev)
            with torch.amp.autocast("cuda"):
                p=net(x); loss=(1-(p*qcb).sum(-1).abs()**2).mean()   # quaternion geodesic loss
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); run+=loss.item()
        sch.step(); m=evaluate(); m.update(epoch=ep,loss=run/len(ltr),sec=time.time()-t0); hist.append(m)
        print(f"ep {ep:02d} loss {m['loss']:.4f} | val board-orient geodesic med {m['med']:.1f} mean {m['mean']:.1f} p90 {m['p90']:.1f} deg ({m['sec']:.0f}s)",flush=True)
        if m["med"]<best: best=m["med"]; torch.save({"model":net.state_dict(),"metrics":m,"R_offset":R_OFFSET},out/"best.pt")
        torch.save({"model":net.state_dict(),"epoch":ep,"best":best,"hist":hist},out/"last.pt")
    json.dump(hist,open(out/"history.json","w"),indent=2)
    print(f"\nBEST val board-orient geodesic: {best:.1f} deg -> {out/'best.pt'}",flush=True)

if __name__=="__main__": main()
