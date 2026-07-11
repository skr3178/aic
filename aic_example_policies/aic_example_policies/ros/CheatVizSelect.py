#
#  CheatVizSelect — DIAGNOSTIC video (no behavior change). CheatCode drives the plug in; a background thread
#  runs the FULL candidate→select→lock stack per frame and draws it: dim-red = raw YOLO candidates (they
#  flicker between the 2 SFP openings), GREEN cross = the GEOMETRY-SELECTED + temporally-gated lock (stable on
#  the port_name opening), white ring = GT. Shows the two-port fix live.  Run with ground_truth:=true.
#  Frames: ~/aic_data/yolo_viz_sel/trial{i}/
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
CAMS = ["left", "center", "right"]; OUT_ROOT = "/home/skr/aic_data/yolo_viz_sel"; SAMPLE_DT = 0.12
CARD_Y0, CARD_DY, CARD_GATE, LOCK_ACCEPT, FILT_N = -0.188, 0.04, 0.020, 0.012, 7


def rot2(a): c, s = math.cos(a), math.sin(a); return np.array([[c, -s], [s, c]])


class CheatVizSelect(CheatCode):
    def __init__(self, parent_node):
        super().__init__(parent_node); self._yolo = None; self._trial = 0

    def _lazy_yolo(self):
        if self._yolo is None:
            from ultralytics import YOLO
            self._yolo = YOLO(YOLO_WEIGHTS); self.get_logger().info("CheatVizSelect: YOLO loaded")
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
        want = YCLASS[typ]; pz = CADZ[typ]
        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        is_sfp = typ == "sfp"
        if is_sfp:
            k = int(task.target_module_name.split("_")[-1]); exp_by = CARD_Y0 + CARD_DY*k; is_p0 = task.port_name.endswith("port_0")
        lock = {"xy": None, "hist": []}
        stop = threading.Event(); cnt = {"n": 0}

        def select(cands, board):
            if not cands: return None
            cen = np.array(board[:2]); yaw = math.atan2(2*(board[4]*board[5]+board[6]*board[3]), 1-2*(board[5]**2+board[6]**2))
            C = np.array(cands); B = (C - cen) @ rot2(yaw)
            if not is_sfp: return np.median(C, axis=0)
            on = np.abs(B[:, 1] - exp_by) < CARD_GATE
            if on.sum() == 0: return None
            cc = C[on]; bx = B[on, 0]; order = np.argsort(bx); bxs = bx[order]
            gi = int(np.argmax(np.diff(bxs))) if len(bxs) > 1 else -1
            hi, lo = (order[gi+1:], order[:gi+1]) if gi >= 0 else (order, np.array([], int))
            sep = (np.median(bx[hi]) - np.median(bx[lo])) if len(lo) and len(hi) else 0.0
            sel = (hi if is_p0 else lo) if (len(lo) >= 1 and len(hi) >= 1 and 0.012 < sep < 0.033) else np.arange(len(cc))
            return np.median(cc[sel], axis=0)

        def logger():
            while not stop.is_set():
                t0 = time.time()
                try:
                    obs = get_observation()
                    pgt = np.array(self._tfvec(port_frame)[:3]); board = self._tfvec("task_board")
                    Tc = self._mat(self._tfvec("center_camera/optical"))
                    img = cv2.cvtColor(self._raw_rgb(obs.center_image), cv2.COLOR_RGB2BGR)
                    raws = {"left": obs.left_image, "center": obs.center_image, "right": obs.right_image}
                    cands = []
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
                            if s > 0: cands.append((tt+s*d)[:2])
                            if c == "center":  # dim-red raw candidate box
                                cv2.rectangle(img, (int(u-bw/2), int(v-bh/2)), (int(u+bw/2), int(v+bh/2)), (70, 70, 200), 1)
                    sel = select(cands, board)
                    if sel is not None:
                        if lock["xy"] is None: lock["xy"] = sel; lock["hist"] = [sel]
                        elif np.linalg.norm(sel - lock["xy"]) < LOCK_ACCEPT:
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
                    cv2.putText(img, f"{typ.upper()} {task.port_name}  range {rng:.0f}mm  SELECTED-LOCK err {err:.0f}mm  ({len(cands)} cand)",
                                (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
                    cv2.putText(img, "dim red = raw YOLO candidates   GREEN = geometry-selected + gated lock   white = GT",
                                (12, img.shape[0]-16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                    cv2.imwrite(f"{outdir}/{cnt['n']:04d}.png", img); cnt["n"] += 1
                except Exception as e:
                    if cnt["n"] % 40 == 0: self.get_logger().warn(f"vizsel skip: {e}")
                dt = SAMPLE_DT - (time.time() - t0)
                if dt > 0: time.sleep(dt)

        th = threading.Thread(target=logger, daemon=True); th.start()
        ok = super().insert_cable(task, get_observation, move_robot, send_feedback)
        stop.set(); th.join(timeout=3)
        self.get_logger().info(f"CheatVizSelect trial {ti}: saved {cnt['n']} frames -> {outdir}")
        return ok
