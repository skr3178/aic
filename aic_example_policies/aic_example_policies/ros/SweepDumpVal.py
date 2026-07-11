#
#  SweepDumpVal — identical to SweepDumpFull but dumps to a SEPARATE root so the held-out
#  validation sweep set does NOT clobber the 3-trial eval dump (sweep_dump_full). Used to
#  validate the [1.5] ensemble on many wider-slide / cluttered scenes. Run with ground_truth:=true.
#
import json
import os

import numpy as np
import cv2
from geometry_msgs.msg import Point, Pose, Quaternion
from tf2_ros import TransformException

from aic_example_policies.ros.PerceptionInsertKP import CAMS, SWEEP_OFFSETS, _quat
from aic_example_policies.ros.SweepDumpFull import SweepDumpFull, NFX, NFY, NCX, NCY

OUT_ROOT = "/home/skr/aic_data/sweep_dump_val"


class SweepDumpVal(SweepDumpFull):
    def insert_cable(self, task, get_observation, move_robot, send_feedback):
        self._task = task
        ti = getattr(self, "_sd_trial", 0)
        self._sd_trial = ti + 1
        out = f"{OUT_ROOT}/trial{ti}"
        os.makedirs(out, exist_ok=True)
        self.get_logger().info(f"SweepDumpVal trial {ti} (NATIVE res) -> {out}")

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
                rgb = self._raw_rgb(raws[c])
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
            "intr": [NFX, NFY, NCX, NCY],
        }
        json.dump(gt, open(f"{out}/gt.json", "w"))
        self.get_logger().info(f"SweepDumpVal trial {ti}: saved {j} native frames + gt.json")
        return True
