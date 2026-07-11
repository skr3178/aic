#
#  RecordingGeo — PerceptionInsertGeo that RECORDS the center wrist camera through the whole trial
#  (sweep + approach + descent) by transparently wrapping get_observation. For inspecting WHERE a
#  trial fails (perception lock vs insertion). Saves native center frames to geo_rec/trial{i}/.
#
import json
import os

import cv2
import numpy as np
from tf2_ros import TransformException

from aic_example_policies.ros.PerceptionInsertYOLO import FX, FY, CX, CY, KP_SCALE
from aic_example_policies.ros.PerceptionInsertGeo import PerceptionInsertGeo

OUT_ROOT = "/home/skr/aic_data/geo_rec_xo"
NFX, NFY, NCX, NCY = FX / KP_SCALE, FY / KP_SCALE, CX / KP_SCALE, CY / KP_SCALE   # native intrinsics


class RecordingGeo(PerceptionInsertGeo):
    def insert_cable(self, task, get_observation, move_robot, send_feedback):
        ti = getattr(self, "_rec_trial", 0); self._rec_trial = ti + 1
        rec_dir = f"{OUT_ROOT}/trial{ti}"; os.makedirs(rec_dir, exist_ok=True)
        self.get_logger().info(f"RecordingGeo: recording center cam -> {rec_dir}")
        state = {"i": 0}

        def wrapped():
            obs = get_observation()
            try:
                rgb = self._raw_rgb(obs.center_image)
                cv2.imwrite(f"{rec_dir}/{state['i']:04d}.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                try:                                                 # per-frame camera pose (base<-center cam)
                    Tbc = self._T_base_frame("center_camera/optical")
                    np.save(f"{rec_dir}/{state['i']:04d}_Tbc.npy", Tbc)
                except TransformException:
                    pass
                state["i"] += 1
            except Exception as e:  # never let recording break the run
                self.get_logger().warn(f"rec frame skip: {e}")
            return obs

        ok = super().insert_cable(task, wrapped, move_robot, send_feedback)
        # GT port (x) + the FROZEN prediction (o) + native intrinsics, for the x/o overlay
        pg = self._gt_transform(f"task_board/{task.target_module_name}/{task.port_name}_link")
        xo = {"port_gt": list(pg) if pg else None,
              "pred": self._sweep_port.tolist() if getattr(self, "_sweep_port", None) is not None else None,
              "plug_type": self._plug_type, "port_name": task.port_name,
              "intr": [NFX, NFY, NCX, NCY], "n_frames": state["i"]}
        json.dump(xo, open(f"{rec_dir}/xo.json", "w"))
        self.get_logger().info(f"RecordingGeo trial {ti}: {state['i']} frames, pred={xo['pred']}, gt={xo['port_gt']}")
        return ok
