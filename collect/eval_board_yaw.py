#!/usr/bin/env python3
"""Base-frame board-YAW error for the board-orientation net (the number that matters for the fix).
Predicts R_center_board, recovers R_base_center from the port correspondence (T_base_cam via the port
pose in base vs center-cam), composes -> R_base_board -> board yaw, compares to GT board yaw.
Reports on a split + on the v1 OOD proxy. Usage:
  eval_board_yaw.py --data <native_c256>,<v2_c256> --ckpt ~/aic_data/boardyaw_run/best.pt --split val
  eval_board_yaw.py --data <perception_v1> --ckpt ... --split all
"""
import argparse, json, math
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
from PIL import Image
import torch
from torchvision.transforms import functional as VF
from train_board_yaw import BoardNet, R_OFFSET, R_of, IMAGENET_MEAN, IMAGENET_STD, CAMS, build_split

def yaw_of(R): return math.degrees(math.atan2(R[1,0],R[0,0]))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data",required=True); ap.add_argument("--ckpt",required=True)
    ap.add_argument("--stride",type=int,default=8); ap.add_argument("--split",default="val",choices=["val","train","all"]); a=ap.parse_args()
    dev="cuda"; net=BoardNet().to(dev).eval()
    net.load_state_dict(torch.load(a.ckpt,map_location=dev,weights_only=False)["model"])
    roots=[Path(p) for p in a.data.split(",") if p.strip()]; tr,va=build_split(roots)
    eps=va if a.split=="val" else (tr if a.split=="train" else tr|va)
    errs=[]; by=({})
    for root in roots:
        idx=pq.read_table(root/"index.parquet").to_pandas(); idx=idx[idx["split"]!="reject"]
        for ep in sorted(int(e) for e in idx["episode"]):
            if (root,ep) not in eps: continue
            epd=root/f"episode_{ep:04d}"; meta=json.load(open(epd/"meta.json")); typ=meta.get("type")
            if typ not in R_OFFSET: continue
            df=pq.read_table(epd/"frames.parquet",columns=["frame","port_center_quat","port_base_quat"]).to_pandas()
            Roff_inv=np.linalg.inv(R_of(R_OFFSET[typ]))
            for i in range(0,len(df),a.stride):
                r=df.iloc[i]; pcq=r["port_center_quat"]; pbq=r["port_base_quat"]
                if pcq is None or pbq is None: continue
                pcq=np.asarray(pcq,float); pbq=np.asarray(pbq,float)
                # GT board yaw (base) = yaw(R(port_base_quat) @ inv(offset))
                gt_yaw=yaw_of(R_of(pbq)@Roff_inv)
                # R_base_center from correspondence: T_base_cam = T_base_port @ inv(T_cam_port)  (rotation)
                R_base_center=R_of(pbq)@np.linalg.inv(R_of(pcq))
                # predict R_center_board
                imgs=torch.stack([VF.normalize(VF.to_tensor(Image.open(f"{epd}/{c}/{int(r['frame']):04d}.png").convert("RGB")),IMAGENET_MEAN,IMAGENET_STD) for c in CAMS]).unsqueeze(0).to(dev)
                with torch.no_grad(), torch.amp.autocast("cuda"): q=net(imgs)[0].float().cpu().numpy()
                pred_yaw=yaw_of(R_base_center@R_of(q))
                e=abs(((pred_yaw-gt_yaw+180)%360)-180); errs.append(e); by.setdefault(typ,[]).append(e)
    errs=np.array(errs)
    print(f"[{a.split}] board-YAW error (base): med {np.median(errs):.1f} deg | mean {errs.mean():.1f} | p90 {np.quantile(errs,0.9):.1f}  (n={len(errs)})")
    for t,v in by.items(): v=np.array(v); print(f"   {t}: med {np.median(v):.1f} p90 {np.quantile(v,0.9):.1f} (n={len(v)})")

if __name__=="__main__": main()
