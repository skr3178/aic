#
#  CheatVizSlide — DIAGNOSTIC video (no behavior change). CheatCode drives; a background thread runs the
#  ONLINE slide-estimation + fixed-identity SFP selector and draws it: dim-red = raw YOLO candidates (flicker),
#  GREEN cross = the FIXED-IDENTITY target lock (slide-aligned), white ring = GT. Shows the two-port fix.
#  Slide is accumulated from both-visible frames; identity is committed once >=3 such frames exist.
#  Run with ground_truth:=true.  Frames: ~/aic_data/yolo_viz_slide/trial{i}/
#
import math
import os
import threading
import time

import numpy as np
from rclpy.time import Time
from tf2_ros import TransformException

from aic_example_policies.ros.CheatCode import CheatCode

YOLO_WEIGHTS = "/home/skr/aic_data/yolo_runs/best_final.pt"; YOLO_CONF = 0.15
FX = FY = 1236.63; CX = 576.0; CY = 512.0
CADZ = {"sfp": 0.1335, "sc": 0.0145}; YCLASS = {"sfp": 0, "sc": 1}
CAMS = ["left", "center", "right"]; OUT_ROOT = "/home/skr/aic_data/yolo_viz_slide"; SAMPLE_DT = 0.12
P0X, P1X = -0.0708, -0.0926; CARD_Y0, CARD_DY = -0.188, 0.04; CAD_MID_X = 0.5*(P0X+P1X)
CARD_GATE, SPLIT_LO, SPLIT_HI, ACCEPT, MARGIN, FILT_N, MIN_BV = 0.020, 0.012, 0.033, 0.013, 0.004, 7, 3


def rot2(a): c, s = math.cos(a), math.sin(a); return np.array([[c, -s], [s, c]])


class CheatVizSlide(CheatCode):
    def __init__(self, parent_node):
        super().__init__(parent_node); self._yolo = None; self._trial = 0

    def _lazy_yolo(self):
        if self._yolo is None:
            from ultralytics import YOLO
            self._yolo = YOLO(YOLO_WEIGHTS); self.get_logger().info("CheatVizSlide: YOLO loaded")
        return self._yolo

    def _raw_rgb(self, raw): return np.frombuffer(raw.data, dtype=np.uint8).reshape(raw.height, raw.width, 3)

    def _tfvec(self, frame):
        tf = self._parent_node._tf_buffer.lookup_transform("base_link", frame, Time())
        t = tf.transform.translation; q = tf.transform.rotation
        return [t.x, t.y, t.z, q.w, q.x, q.y, q.z]

    def _mat(self, v):
        x, y, z, w, qx, qy, qz = v
        R = np.array([[1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*w), 2*(qx*qz+qy*w)],
                      [2*(qx*qy+qz*w), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*w)],
                      [2*(qx*qz-qy*w), 2*(qy*qz+qx*w), 1-2*(qx*qx+qy*qy)]])
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = [x, y, z]; return T

    def insert_cable(self, task, get_observation, move_robot, send_feedback):
        import cv2
        ti = self._trial; self._trial += 1
        outdir = f"{OUT_ROOT}/trial{ti}"; os.makedirs(outdir, exist_ok=True)
        yolo = self._lazy_yolo()
        typ = (getattr(task, "plug_type", "") or ("sc" if "sc" in task.port_name.lower() else "sfp")).lower()
        want = YCLASS[typ]; pz = CADZ[typ]; is_sfp = typ == "sfp"
        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        if is_sfp:
            k = int(task.target_module_name.split("_")[-1]); card_y = CARD_Y0 + CARD_DY*k
            is_p0 = task.port_name.endswith("port_0"); tgt_nom = P0X if is_p0 else P1X; sib_nom = P1X if is_p0 else P0X
        slides = []; lock = {"xy": None, "hist": []}
        stop = threading.Event(); cnt = {"n": 0}

        def logger():
            while not stop.is_set():
                t0 = time.time()
                try:
                    obs = get_observation(); pgt = np.array(self._tfvec(port_frame)[:3]); board = self._tfvec("task_board")
                    cen = np.array(board[:2]); yw = math.atan2(2*(board[4]*board[5]+board[6]*board[3]), 1-2*(board[5]**2+board[6]**2))
                    Tc = self._mat(self._tfvec("center_camera/optical"))
                    img = cv2.cvtColor(self._raw_rgb(obs.center_image), cv2.COLOR_RGB2BGR)
                    raws = {"left": obs.left_image, "center": obs.center_image, "right": obs.right_image}
                    base_c = []; board_c = []
                    for c in CAMS:
                        try: Tbc = self._mat(self._tfvec(f"{c}_camera/optical"))
                        except TransformException: continue
                        res = yolo.predict(cv2.cvtColor(self._raw_rgb(raws[c]), cv2.COLOR_RGB2BGR), conf=YOLO_CONF, verbose=False, device="cuda")[0]
                        if not len(res.boxes): continue
                        cls = res.boxes.cls.cpu().numpy().astype(int); xywh = res.boxes.xywh.cpu().numpy()
                        R, tt = Tbc[:3, :3], Tbc[:3, 3]
                        for m in range(len(cls)):
                            if cls[m] != want: continue
                            u, v, bw, bh = xywh[m]
                            d = R @ np.array([(u-CX)/FX, (v-CY)/FY, 1.0]); s = (pz-tt[2])/d[2] if abs(d[2]) > 1e-9 else -1
                            if s > 0:
                                p = (tt+s*d)[:2]; base_c.append(p); board_c.append((p-cen) @ rot2(yw))
                            if c == "center":
                                cv2.rectangle(img, (int(u-bw/2), int(v-bh/2)), (int(u+bw/2), int(v+bh/2)), (70, 70, 200), 1)
                    sel = None; status = "acquiring identity"
                    if is_sfp and base_c:
                        base_c = np.array(base_c); board_c = np.array(board_c)
                        on = np.abs(board_c[:, 1] - card_y) < CARD_GATE
                        bc = board_c[on]; bs = base_c[on]
                        if len(bc):
                            bxs = np.sort(bc[:, 0]); gi = int(np.argmax(np.diff(bxs))) if len(bxs) > 1 else -1
                            lo, hi = (bxs[:gi+1], bxs[gi+1:]) if gi >= 0 else (bxs, np.array([]))
                            sep = np.median(hi)-np.median(lo) if len(lo) and len(hi) else 0.0
                            if SPLIT_LO < sep < SPLIT_HI: slides.append(0.5*(np.median(lo)+np.median(hi)) - CAD_MID_X)
                            if len(slides) >= MIN_BV:                       # identity committed
                                sl = float(np.median(slides)); ta = tgt_nom+sl; sa = sib_nom+sl
                                dt = np.abs(bc[:, 0]-ta); ds = np.abs(bc[:, 0]-sa); keep = (dt < ACCEPT) & (dt+MARGIN < ds)
                                if keep.sum(): sel = np.median(bs[keep], axis=0); status = f"slide {sl*1000:+.0f}mm  LOCKED"
                    elif not is_sfp and base_c:
                        sel = np.median(np.array(base_c), axis=0); status = "SC (no selector)"
                    if sel is not None:
                        if lock["xy"] is None: lock["xy"] = sel; lock["hist"] = [sel]
                        elif np.linalg.norm(sel-lock["xy"]) < ACCEPT:
                            lock["hist"].append(sel); lock["hist"] = lock["hist"][-FILT_N:]; lock["xy"] = np.median(np.array(lock["hist"]), axis=0)
                    def proj(P):
                        inv = np.linalg.inv(Tc); pc = inv[:3, :3] @ P + inv[:3, 3]
                        return (int(FX*pc[0]/pc[2]+CX), int(FY*pc[1]/pc[2]+CY)) if pc[2] > 1e-3 else None
                    g = proj(pgt)
                    if g: cv2.circle(img, g, 16, (0, 0, 0), 5); cv2.circle(img, g, 16, (255, 255, 255), 2)
                    err = float('nan')
                    if lock["xy"] is not None:
                        l = proj(np.array([lock["xy"][0], lock["xy"][1], pz])); err = float(np.linalg.norm(lock["xy"]-pgt[:2]))*1000
                        if l:
                            u, v = l; r = 24
                            cv2.line(img, (u-r, v), (u+r, v), (0, 0, 0), 6); cv2.line(img, (u, v-r), (u, v+r), (0, 0, 0), 6)
                            cv2.line(img, (u-r, v), (u+r, v), (0, 255, 0), 3); cv2.line(img, (u, v-r), (u, v+r), (0, 255, 0), 3)
                    rng = float(np.linalg.norm(Tc[:3, 3]-pgt))*1000
                    cv2.rectangle(img, (0, 0), (img.shape[1], 48), (20, 20, 20), -1)
                    cv2.putText(img, f"{typ.upper()} {task.port_name}  range {rng:.0f}mm  {status}  err {err:.0f}mm",
                                (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(img, "dim red = raw YOLO candidates   GREEN = slide-fixed identity lock   white = GT",
                                (12, img.shape[0]-16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                    cv2.imwrite(f"{outdir}/{cnt['n']:04d}.png", img); cnt["n"] += 1
                except Exception as e:
                    if cnt["n"] % 40 == 0: self.get_logger().warn(f"vizslide skip: {e}")
                dt = SAMPLE_DT - (time.time() - t0)
                if dt > 0: time.sleep(dt)

        th = threading.Thread(target=logger, daemon=True); th.start()
        ok = super().insert_cable(task, get_observation, move_robot, send_feedback)
        stop.set(); th.join(timeout=3)
        self.get_logger().info(f"CheatVizSlide trial {ti}: saved {cnt['n']} frames -> {outdir}")
        return ok
