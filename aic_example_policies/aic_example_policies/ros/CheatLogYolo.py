#
#  CheatLogYolo — DIAGNOSTIC (pure logging, no behavior change). Runs the WORKING CheatCode execution
#  (seats all 3 eval trials) and PASSIVELY logs YOLO's port prediction vs camera->port range ALL THE WAY TO
#  CONTACT. CheatCode never calls get_observation (it drives from GT/TF), so we run the YOLO logging in a
#  BACKGROUND SAMPLING THREAD that reads observations independently while CheatCode moves the arm.
#  YOLO drives NOTHING. Run with ground_truth:=true.  Output: ~/aic_data/yolo_range_log.jsonl
#
import json
import threading
import time

import numpy as np
from rclpy.time import Time
from tf2_ros import TransformException

from aic_example_policies.ros.CheatCode import CheatCode

YOLO_WEIGHTS = "/home/skr/aic_data/yolo_runs/best_final.pt"
YOLO_CLASS = {"sfp": 0, "sc": 1}
YOLO_CONF = 0.25
FX = FY = 1236.63; CX = 576.0; CY = 512.0          # NATIVE intrinsics
CAD_Z = {"sfp": 0.1335, "sc": 0.0145}
CAMS = ["left", "center", "right"]
LOG = "/home/skr/aic_data/yolo_range_log.jsonl"
SAMPLE_DT = 0.15                                    # sampling period (s) of the background logger


class CheatLogYolo(CheatCode):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._yolo = None
        self._trial = 0

    def _lazy_yolo(self):
        if self._yolo is None:
            from ultralytics import YOLO
            self._yolo = YOLO(YOLO_WEIGHTS)
            self.get_logger().info(f"CheatLogYolo: YOLO loaded {YOLO_WEIGHTS}")
        return self._yolo

    def _raw_rgb(self, raw):
        return np.frombuffer(raw.data, dtype=np.uint8).reshape(raw.height, raw.width, 3)

    def _mat(self, tf):
        tr = tf.transform.translation; q = tf.transform.rotation
        w, x, y, z = q.w, q.x, q.y, q.z
        R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                      [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                      [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = [tr.x, tr.y, tr.z]
        return T

    def insert_cable(self, task, get_observation, move_robot, send_feedback):
        import cv2
        ti = self._trial; self._trial += 1
        yolo = self._lazy_yolo()
        typ = (getattr(task, "plug_type", "") or ("sc" if "sc" in task.port_name.lower() else "sfp")).lower()
        want = YOLO_CLASS.get(typ, 0); pz = CAD_Z.get(typ, 0.0145)
        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        stop = threading.Event(); cnt = {"n": 0}

        def logger():
            while not stop.is_set():
                t0 = time.time()
                try:
                    obs = get_observation()
                    p = self._parent_node._tf_buffer.lookup_transform("base_link", port_frame, Time()).transform.translation
                    pgt = np.array([p.x, p.y, p.z])
                    raws = {"left": obs.left_image, "center": obs.center_image, "right": obs.right_image}
                    entry = {"trial": ti, "sample": cnt["n"], "plug_type": typ, "port_gt": pgt.tolist(), "cams": {}}
                    for c in CAMS:
                        try:
                            Tbc = self._mat(self._parent_node._tf_buffer.lookup_transform("base_link", f"{c}_camera/optical", Time()))
                        except TransformException:
                            continue
                        R, tt = Tbc[:3, :3], Tbc[:3, 3]
                        rng = float(np.linalg.norm(tt - pgt)) * 1000
                        res = yolo.predict(cv2.cvtColor(self._raw_rgb(raws[c]), cv2.COLOR_RGB2BGR),
                                           conf=YOLO_CONF, verbose=False, device="cuda")[0]
                        rec = {"range": round(rng, 1), "detected": False, "conf": 0.0, "box": None, "est": None, "err": None}
                        if len(res.boxes):
                            cls = res.boxes.cls.cpu().numpy().astype(int); cf = res.boxes.conf.cpu().numpy(); xywh = res.boxes.xywh.cpu().numpy()
                            sel = np.where(cls == want)[0]
                            if len(sel):
                                j = int(sel[np.argmax(cf[sel])]); u, v = float(xywh[j, 0]), float(xywh[j, 1])
                                rec.update(detected=True, conf=round(float(cf[j]), 3), box=[round(u, 1), round(v, 1)])
                                dv = R @ np.array([(u-CX)/FX, (v-CY)/FY, 1.0]); s = (pz-tt[2])/dv[2] if abs(dv[2]) > 1e-9 else -1
                                if s > 0:
                                    e = (tt + s*dv)[:2]; rec["est"] = [round(e[0], 4), round(e[1], 4)]
                                    rec["err"] = round(float(np.linalg.norm(e - pgt[:2])) * 1000, 1)
                        entry["cams"][c] = rec
                    if entry["cams"]:
                        with open(LOG, "a") as f:
                            f.write(json.dumps(entry) + "\n")
                        cnt["n"] += 1
                except Exception as e:
                    if cnt["n"] % 40 == 0:
                        self.get_logger().warn(f"yolo-log skip: {e}")
                dt = SAMPLE_DT - (time.time() - t0)
                if dt > 0:
                    time.sleep(dt)

        th = threading.Thread(target=logger, daemon=True); th.start()
        ok = super().insert_cable(task, get_observation, move_robot, send_feedback)   # unchanged CheatCode exec
        stop.set(); th.join(timeout=3)
        self.get_logger().info(f"CheatLogYolo trial {ti}: logged {cnt['n']} samples")
        return ok
