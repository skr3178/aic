#
#  CheatLogYoloAll — DIAGNOSTIC (pure logging). CheatCode drives the plug to contact (seats all 3 = valid);
#  a background thread logs, per frame per cam: ALL YOLO boxes (class+conf+xywh) + the camera extrinsic Tbc
#  + GT target port + GT board pose + task metadata. This is the raw CANDIDATE data for the offline
#  target-instance selector/evaluator. YOLO drives NOTHING. Run with ground_truth:=true.
#  Output: ~/aic_data/yolo_viz/yolo_candidates.jsonl
#
import json
import threading
import time

import numpy as np
from rclpy.time import Time
from tf2_ros import TransformException

from aic_example_policies.ros.CheatCode import CheatCode

YOLO_WEIGHTS = "/home/skr/aic_data/yolo_runs/best_final.pt"
YOLO_CONF = 0.15                                    # LOW conf: capture ALL plausible candidates
CAMS = ["left", "center", "right"]
LOG = "/home/skr/aic_data/yolo_viz/yolo_candidates.jsonl"
SAMPLE_DT = 0.12


class CheatLogYoloAll(CheatCode):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._yolo = None; self._trial = 0

    def _lazy_yolo(self):
        if self._yolo is None:
            from ultralytics import YOLO
            self._yolo = YOLO(YOLO_WEIGHTS)
            self.get_logger().info(f"CheatLogYoloAll: YOLO loaded {YOLO_WEIGHTS}")
        return self._yolo

    def _raw_rgb(self, raw):
        return np.frombuffer(raw.data, dtype=np.uint8).reshape(raw.height, raw.width, 3)

    def _tfmat(self, frame):
        tf = self._parent_node._tf_buffer.lookup_transform("base_link", frame, Time())
        tr = tf.transform.translation; q = tf.transform.rotation
        return [tr.x, tr.y, tr.z, q.w, q.x, q.y, q.z]

    def insert_cable(self, task, get_observation, move_robot, send_feedback):
        import cv2
        ti = self._trial; self._trial += 1
        yolo = self._lazy_yolo()
        typ = (getattr(task, "plug_type", "") or ("sc" if "sc" in task.port_name.lower() else "sfp")).lower()
        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        stop = threading.Event(); cnt = {"n": 0}

        def logger():
            while not stop.is_set():
                t0 = time.time()
                try:
                    obs = get_observation()
                    entry = {"trial": ti, "sample": cnt["n"], "plug_type": typ,
                             "target_module": task.target_module_name, "port_name": task.port_name,
                             "port_gt": self._tfmat(port_frame), "board_tf": self._tfmat("task_board"),
                             "cams": {}}
                    raws = {"left": obs.left_image, "center": obs.center_image, "right": obs.right_image}
                    for c in CAMS:
                        try:
                            Tbc = self._tfmat(f"{c}_camera/optical")
                        except TransformException:
                            continue
                        res = yolo.predict(cv2.cvtColor(self._raw_rgb(raws[c]), cv2.COLOR_RGB2BGR),
                                           conf=YOLO_CONF, verbose=False, device="cuda")[0]
                        boxes = []
                        if len(res.boxes):
                            cls = res.boxes.cls.cpu().numpy().astype(int); cf = res.boxes.conf.cpu().numpy(); xywh = res.boxes.xywh.cpu().numpy()
                            for k in range(len(cls)):
                                boxes.append([int(cls[k]), round(float(cf[k]), 3),
                                              round(float(xywh[k, 0]), 1), round(float(xywh[k, 1]), 1),
                                              round(float(xywh[k, 2]), 1), round(float(xywh[k, 3]), 1)])
                        entry["cams"][c] = {"Tbc": [round(v, 6) for v in Tbc], "boxes": boxes}   # boxes: [cls,conf,u,v,w,h]
                    if entry["cams"]:
                        with open(LOG, "a") as f:
                            f.write(json.dumps(entry) + "\n")
                        cnt["n"] += 1
                except Exception as e:
                    if cnt["n"] % 40 == 0:
                        self.get_logger().warn(f"cand-log skip: {e}")
                dt = SAMPLE_DT - (time.time() - t0)
                if dt > 0:
                    time.sleep(dt)

        th = threading.Thread(target=logger, daemon=True); th.start()
        ok = super().insert_cable(task, get_observation, move_robot, send_feedback)
        stop.set(); th.join(timeout=3)
        self.get_logger().info(f"CheatLogYoloAll trial {ti}: logged {cnt['n']} frames (all candidates)")
        return ok
