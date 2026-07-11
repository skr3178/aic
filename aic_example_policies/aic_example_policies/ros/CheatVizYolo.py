#
#  CheatVizYolo — DIAGNOSTIC (pure logging, no behavior change). CheatCode drives the plug into the port;
#  a background thread samples the CENTER wrist camera, runs YOLO, and saves each frame with YOLO's
#  detection drawn on it (RED cross+box = what YOLO detects; GREEN ring = GT port; camera->port range).
#  -> assemble into a video of the gripper moving in for insertion with the live YOLO marker.
#  YOLO drives NOTHING. Run with ground_truth:=true.  Frames: ~/aic_data/yolo_viz/trial{i}/
#
import os
import threading
import time

import numpy as np
from rclpy.time import Time
from tf2_ros import TransformException

from aic_example_policies.ros.CheatCode import CheatCode

YOLO_WEIGHTS = "/home/skr/aic_data/yolo_runs/best_final.pt"
YOLO_CLASS = {"sfp": 0, "sc": 1}; YOLO_CONF = 0.25
FX = FY = 1236.63; CX = 576.0; CY = 512.0
OUT_ROOT = "/home/skr/aic_data/yolo_viz"
SAMPLE_DT = 0.12


class CheatVizYolo(CheatCode):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._yolo = None; self._trial = 0

    def _lazy_yolo(self):
        if self._yolo is None:
            from ultralytics import YOLO
            self._yolo = YOLO(YOLO_WEIGHTS)
            self.get_logger().info(f"CheatVizYolo: YOLO loaded {YOLO_WEIGHTS}")
        return self._yolo

    def _raw_rgb(self, raw):
        return np.frombuffer(raw.data, dtype=np.uint8).reshape(raw.height, raw.width, 3)

    def _mat(self, tf):
        tr = tf.transform.translation; q = tf.transform.rotation
        w, x, y, z = q.w, q.x, q.y, q.z
        R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                      [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                      [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = [tr.x, tr.y, tr.z]; return T

    def insert_cable(self, task, get_observation, move_robot, send_feedback):
        import cv2
        ti = self._trial; self._trial += 1
        outdir = f"{OUT_ROOT}/trial{ti}"; os.makedirs(outdir, exist_ok=True)
        yolo = self._lazy_yolo()
        typ = (getattr(task, "plug_type", "") or ("sc" if "sc" in task.port_name.lower() else "sfp")).lower()
        want = YOLO_CLASS.get(typ, 0)
        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        stop = threading.Event(); cnt = {"n": 0}

        def logger():
            while not stop.is_set():
                t0 = time.time()
                try:
                    obs = get_observation()
                    p = self._parent_node._tf_buffer.lookup_transform("base_link", port_frame, Time()).transform.translation
                    pgt = np.array([p.x, p.y, p.z])
                    Tbc = self._mat(self._parent_node._tf_buffer.lookup_transform("base_link", "center_camera/optical", Time()))
                    R, tt = Tbc[:3, :3], Tbc[:3, 3]; rng = float(np.linalg.norm(tt - pgt)) * 1000
                    img = cv2.cvtColor(self._raw_rgb(obs.center_image), cv2.COLOR_RGB2BGR)
                    # GT port projected -> green ring
                    inv = np.linalg.inv(Tbc); pc = inv[:3, :3] @ pgt + inv[:3, 3]
                    if pc[2] > 1e-3:
                        gu, gv = int(FX*pc[0]/pc[2]+CX), int(FY*pc[1]/pc[2]+CY)
                        cv2.circle(img, (gu, gv), 16, (0, 0, 0), 5); cv2.circle(img, (gu, gv), 16, (0, 255, 0), 2)
                    # YOLO detection -> red box + cross
                    res = yolo.predict(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), conf=YOLO_CONF, verbose=False, device="cuda")[0]
                    conf = 0.0; det = False
                    if len(res.boxes):
                        cls = res.boxes.cls.cpu().numpy().astype(int); cf = res.boxes.conf.cpu().numpy(); xywh = res.boxes.xywh.cpu().numpy()
                        sel = np.where(cls == want)[0]
                        if len(sel):
                            j = int(sel[np.argmax(cf[sel])]); det = True; conf = float(cf[j])
                            u, v, bw, bh = xywh[j]; u, v = int(u), int(v); x0, y0, x1, y1 = int(u-bw/2), int(v-bh/2), int(u+bw/2), int(v+bh/2)
                            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 3)
                            r = 22
                            cv2.line(img, (u-r, v), (u+r, v), (0, 0, 255), 3); cv2.line(img, (u, v-r), (u, v+r), (0, 0, 255), 3)
                    lab = f"{typ.upper()}  range {rng:.0f}mm  YOLO {'conf '+format(conf,'.2f') if det else 'NO DETECT'}"
                    cv2.rectangle(img, (0, 0), (img.shape[1], 48), (20, 20, 20), -1)
                    cv2.putText(img, lab, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255) if det else (0, 0, 255), 2)
                    cv2.putText(img, "red=YOLO  green=GT port", (12, img.shape[0]-16), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                    cv2.imwrite(f"{outdir}/{cnt['n']:04d}.png", img); cnt["n"] += 1
                except Exception as e:
                    if cnt["n"] % 40 == 0:
                        self.get_logger().warn(f"viz skip: {e}")
                dt = SAMPLE_DT - (time.time() - t0)
                if dt > 0:
                    time.sleep(dt)

        th = threading.Thread(target=logger, daemon=True); th.start()
        ok = super().insert_cable(task, get_observation, move_robot, send_feedback)
        stop.set(); th.join(timeout=3)
        self.get_logger().info(f"CheatVizYolo trial {ti}: saved {cnt['n']} annotated frames -> {outdir}")
        return ok
