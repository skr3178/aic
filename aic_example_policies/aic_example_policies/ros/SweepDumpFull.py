#
#  SweepDumpFull — like SweepDump, but saves NATIVE full-resolution frames (1024x1152) so a
#  detector trained on native yolo_ds frames can be gated at its TRAINING resolution
#  ("train 1024 -> deploy 1024" path). Dumps each cam's native RGB + base<-cam extrinsic Tbc,
#  plus gt.json with the GT port/board pose and the NATIVE intrinsics for offline scoring.
#  Standalone subclass; does NOT modify PerceptionInsertKP/CheatCode. Run with ground_truth:=true.
#
import json
import os

import numpy as np
import cv2
from geometry_msgs.msg import Point, Pose, Quaternion
from tf2_ros import TransformException

from aic_example_policies.ros.PerceptionInsertKP import (
    PerceptionInsertKP, CAMS, KP_SCALE, FX, FY, CX, CY, SWEEP_OFFSETS, _quat,
)

# native intrinsics = detector-scale intrinsics / KP_SCALE  (undo the 0.25x)
NFX, NFY, NCX, NCY = FX / KP_SCALE, FY / KP_SCALE, CX / KP_SCALE, CY / KP_SCALE
OUT_ROOT = "/home/skr/aic_data/sweep_dump_full"


class SweepDumpFull(PerceptionInsertKP):
    def insert_cable(self, task, get_observation, move_robot, send_feedback):
        self._task = task
        ti = getattr(self, "_sd_trial", 0)
        self._sd_trial = ti + 1
        out = f"{OUT_ROOT}/trial{ti}"
        os.makedirs(out, exist_ok=True)
        self.get_logger().info(f"SweepDumpFull trial {ti} (NATIVE res) -> {out}")

        T_home = self._T_base_frame("gripper/tcp")
        q_home = _quat(T_home[:3, :3]); t_home = T_home[:3, 3]
        j = 0
        for dx, dy, dz in SWEEP_OFFSETS:
            pose = Pose(position=Point(x=float(t_home[0] + dx), y=float(t_home[1] + dy),
                                       z=float(t_home[2] + dz)),
                        orientation=Quaternion(w=float(q_home[0]), x=float(q_home[1]),
                                               y=float(q_home[2]), z=float(q_home[3])))
            self.set_pose_target(move_robot=move_robot, pose=pose)
            for _ in range(30):
                self.sleep_for(0.05)
            obs = get_observation()
            raws = {"left": obs.left_image, "center": obs.center_image, "right": obs.right_image}
            for c in CAMS:
                try:
                    Tbc = self._T_base_frame(f"{c}_camera/optical")
                except TransformException:
                    continue
                rgb = self._raw_rgb(raws[c])                          # NATIVE, no resize
                cv2.imwrite(f"{out}/{j:03d}_{c}.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                np.save(f"{out}/{j:03d}_{c}_Tbc.npy", Tbc)
                j += 1

        gt = {
            "trial": ti, "n_frames": j,
            "board_tf": self._gt_transform("task_board"),
            "port_gt": self._gt_transform(
                f"task_board/{task.target_module_name}/{task.port_name}_link"),
            "port_type": task.port_type, "plug_type": task.plug_type,
            "target_module": task.target_module_name, "port_name": task.port_name,
            "intr": [NFX, NFY, NCX, NCY],                             # NATIVE intrinsics
        }
        json.dump(gt, open(f"{out}/gt.json", "w"))
        self.get_logger().info(f"SweepDumpFull trial {ti}: saved {j} native frames + gt.json")
        return True
