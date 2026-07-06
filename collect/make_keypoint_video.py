#!/usr/bin/env python3
"""Continuous L|C|R video overlaying the MESH-DERIVED cage-corner keypoints (the ones we'd train
the keypoint detector on) — the port mesh's 8 bounding-box corners, transformed by the GT port pose
into each camera frame, projected, drawn as colored dots + a wireframe box. Every frame.

Usage: make_keypoint_video.py --ep <episode_dir> --mesh <port.obj> --out <mp4> [--fps 20]
"""
import argparse, json, os, subprocess, tempfile, shutil, itertools
import numpy as np, cv2, pandas as pd, trimesh

CAMS = ["left", "center", "right"]
COLS = {"left": "port_left", "center": "port_center", "right": "port_right"}
# 8 distinct BGR colors for the 8 corners
KP_COLORS = [(0,0,255),(0,128,255),(0,255,255),(0,255,0),(255,255,0),(255,0,0),(255,0,255),(200,200,200)]

def R_of(q):  # wxyz -> 3x3
    w,x,y,z = q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=20)
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.ep, "meta.json")))
    df = pd.read_parquet(os.path.join(a.ep, "frames.parquet"))
    sample = cv2.imread(os.path.join(a.ep, "center", sorted(os.listdir(os.path.join(a.ep, "center")))[0]))
    Hc, Wc = sample.shape[:2]; scale = Wc / meta["image_size"][0]
    Ks = {c: np.array(meta["intrinsics"][c], float).reshape(3,3) for c in CAMS}

    lo, hi = trimesh.load(a.mesh, force="mesh").bounds
    corners = np.array(list(itertools.product([lo[0],hi[0]], [lo[1],hi[1]], [lo[2],hi[2]])))  # (8,3) port frame
    # box edges = corner index pairs differing in exactly one axis bit
    edges = [(i,j) for i in range(8) for j in range(i+1,8) if bin(i^j).count("1")==1]

    def project(cam, r):
        col = COLS[cam]
        p = np.asarray(r[col+"_pos"], float); q = np.asarray(r[col+"_quat"], float)  # port pose in this cam frame
        pts_cam = (R_of(q) @ corners.T).T + p            # 8 corners in camera frame
        K = Ks[cam]; fx,fy,cx,cy = K[0,0]*scale, K[1,1]*scale, K[0,2]*scale, K[1,2]*scale
        uv = []
        for X,Y,Z in pts_cam:
            uv.append((fx*X/Z+cx, fy*Y/Z+cy) if Z>0 else None)
        return uv

    tmp = tempfile.mkdtemp(); n = 0
    for _, r in df.iterrows():
        f = int(r["frame"]); tiles = []
        for cam in CAMS:
            img = cv2.imread(os.path.join(a.ep, cam, f"{f:04d}.png"))
            if img is None: img = np.zeros((Hc,Wc,3), np.uint8)
            uv = project(cam, r)
            for i,j in edges:                                    # wireframe box edges
                if uv[i] and uv[j]:
                    cv2.line(img, (int(uv[i][0]),int(uv[i][1])), (int(uv[j][0]),int(uv[j][1])), (60,220,60), 1)
            for k,pt in enumerate(uv):                            # 8 colored corner keypoints
                if pt: cv2.circle(img, (int(round(pt[0])),int(round(pt[1]))), 3, KP_COLORS[k], -1)
            cv2.putText(img, cam, (4,16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            tiles.append(img)
        cv2.imwrite(os.path.join(tmp, f"{n:04d}.png"), np.hstack(tiles)); n += 1
    subprocess.run(["ffmpeg","-y","-loglevel","error","-framerate",str(a.fps),
                    "-i",os.path.join(tmp,"%04d.png"),"-vf","scale=1200:-2","-pix_fmt","yuv420p",
                    "-crf","24",a.out], check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"wrote {a.out}  ({n} frames, 8 mesh cage-corner keypoints + wireframe, "
          f"{meta.get('type')} -> {meta.get('port_frame')})")

if __name__ == "__main__":
    main()
