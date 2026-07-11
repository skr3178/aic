#
#  SFPDriveViz — DIAGNOSTIC video of the GT-FREE insert (no behavior change).
#  Subclasses PerceptionInsertSFPDrive, so CONTROL IS FULLY GT-FREE (perceived board pose +
#  slide-select lock + safe-clear motion). It only ANNOTATES frames during approach + descent:
#     GREEN X    = our FROZEN perceived target lock (what the arm is driving to)
#     WHITE ring = GT port  (reference ONLY — drawn, never used for control; needs ground_truth:=true)
#     YELLOW dot = plug tip (FK)
#  HUD: trial/port · board yaw · tip->target distance · force.  Frames: ~/aic_data/sfp_free_viz/trial{i}/
#
import math
import os

import numpy as np

from aic_example_policies.ros.PerceptionInsertSFPDrive import PerceptionInsertSFPDrive
from aic_example_policies.ros.PerceptionInsertSFP import FXN, FYN, CXN, CYN

OUT_ROOT = "/home/skr/aic_data/sfp_free_viz"
EVERY = 1          # save EVERY control step (0.05 s apart) -> encode at 20 fps = REAL TIME


class SFPDriveViz(PerceptionInsertSFPDrive):

    def _proj(self, Tbc_inv, P):
        """base-frame point -> center-cam pixel (native intrinsics). None if behind the camera."""
        pc = Tbc_inv[:3, :3] @ np.asarray(P, float) + Tbc_inv[:3, 3]
        if pc[2] <= 1e-3:
            return None
        return (int(FXN * pc[0] / pc[2] + CXN), int(FYN * pc[1] / pc[2] + CYN))

    def _perceive(self, get_observation):
        import cv2
        port_tf, plug_tf, info = super()._perceive(get_observation)   # GT-FREE control path (unchanged)
        self._vn = getattr(self, "_vn", 0) + 1
        if self._vn % EVERY or getattr(self, "_locked_xyz", None) is None:
            return port_tf, plug_tf, info
        try:
            obs = get_observation()
            img = cv2.cvtColor(self._raw_rgb(obs.center_image), cv2.COLOR_RGB2BGR)
            Tbc_inv = np.linalg.inv(self._T_base_frame("center_camera/optical"))
            tip = np.array([plug_tf.translation.x, plug_tf.translation.y, plug_tf.translation.z])

            # GT port — reference overlay ONLY (never used for control)
            g = self._gt_transform(f"task_board/{self._task.target_module_name}/{self._task.port_name}_link")
            if g:
                p = self._proj(Tbc_inv, g[:3])
                if p:
                    cv2.circle(img, p, 17, (0, 0, 0), 5)
                    cv2.circle(img, p, 17, (255, 255, 255), 2)

            # OUR frozen perceived lock -> GREEN X  (the target the arm is actually driving to)
            x = self._proj(Tbc_inv, self._locked_xyz)
            if x:
                u, v = x; r = 22
                for (c, t) in (((0, 0, 0), 7), ((0, 255, 0), 3)):
                    cv2.line(img, (u - r, v - r), (u + r, v + r), c, t)
                    cv2.line(img, (u - r, v + r), (u + r, v - r), c, t)

            # plug tip (FK) -> yellow dot
            pt = self._proj(Tbc_inv, tip)
            if pt:
                cv2.circle(img, pt, 6, (0, 0, 0), -1)
                cv2.circle(img, pt, 4, (0, 235, 235), -1)

            d = float(np.linalg.norm(tip - self._locked_xyz)) * 1000.0
            err = float(np.linalg.norm(np.array(g[:3]) - self._locked_xyz)) * 1000.0 if g else float("nan")
            cv2.rectangle(img, (0, 0), (img.shape[1], 52), (20, 20, 20), -1)
            cv2.putText(img, f"GT-FREE  {self._task.port_name}  yaw {math.degrees(self._board_yaw):+.1f}deg  "
                             f"tip->X {d:.0f}mm  lock err {err:.0f}mm  F {info['force']:.0f}N",
                        (12, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.74, (0, 255, 0), 2)
            cv2.putText(img, "GREEN X = our PERCEIVED locked target (drives the arm)   white ring = GT (reference only)"
                             "   yellow = plug tip",
                        (12, img.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)

            d_out = f"{OUT_ROOT}/trial{self._trial_idx}"
            os.makedirs(d_out, exist_ok=True)
            cv2.imwrite(f"{d_out}/{self._vn:05d}.png", img)
        except Exception as ex:
            if self._vn % 90 == 0:
                self.get_logger().warn(f"viz skip: {ex}")
        return port_tf, plug_tf, info
