#
#  SeatV2Viz — video of the SeatV2 seat (contact-trigger fix). CONTROL IS 100% GT-FREE.
#  `gt:=true` only draws the white reference ring; nothing reads it to drive.
#
#  RECORDING MUST NOT PERTURB THE RUN. The seat is timing-sensitive (a stalled control loop kills it:
#  PNG-in-loop cost us 132.2 -> 54.6 on identical code). So:
#    * get_observation() is CACHED per step — the base _perceive already fetches it; the old viz fetched
#      it a SECOND time every step (pure waste, ~half the overhead).
#    * frames are drawn + JPEG-encoded to RAM (~3 ms), never touched to disk inside the loop.
#    * everything is flushed to disk AFTER the trial.
#
#  HUD shows the mechanism that was dead before and is alive now:
#     CONTACT: force_abs / stall / -     ·  spiral_k  ·  stiffness  [90,90,90] STIFF -> [40,40,90] RCC
#     GREEN X = perceived lock · WHITE ring = GT (reference) · YELLOW = plug tip
#
import os

import numpy as np

from aic_example_policies.ros.PerceptionInsertSFPDriveSeatV2 import (
    PerceptionInsertSFPDriveSeatV2, F_ABS, STALL_CMD, STALL_ACT, LAT_STIFF, AX_STIFF,
)
from aic_example_policies.ros.PerceptionInsertSFP import FXN, FYN, CXN, CYN

OUT_ROOT = "/home/skr/aic_data/seat_v2_viz"


class SeatV2Viz(PerceptionInsertSFPDriveSeatV2):

    def _proj(self, Tbc_inv, P):
        pc = Tbc_inv[:3, :3] @ np.asarray(P, float) + Tbc_inv[:3, 3]
        if pc[2] <= 1e-3:
            return None
        return (int(FXN * pc[0] / pc[2] + CXN), int(FYN * pc[1] / pc[2] + CYN))

    def _perceive(self, get_observation):
        import cv2
        cache = {}

        def cached():                       # the base _perceive fetches once; we REUSE it (no 2nd fetch)
            if "o" not in cache:
                cache["o"] = get_observation()
            return cache["o"]

        port_tf, plug_tf, info = super()._perceive(cached)     # GT-FREE control path, UNCHANGED
        if getattr(self, "_locked_xyz", None) is None or "o" not in cache:
            return port_tf, plug_tf, info
        try:
            img = cv2.cvtColor(self._raw_rgb(cache["o"].center_image), cv2.COLOR_RGB2BGR)
            Tbc_inv = np.linalg.inv(self._T_base_frame("center_camera/optical"))
            tip = np.array([plug_tf.translation.x, plug_tf.translation.y, plug_tf.translation.z])

            g = self._gt_transform(f"task_board/{self._task.target_module_name}/{self._task.port_name}_link")
            if g:
                p = self._proj(Tbc_inv, g[:3])
                if p:
                    cv2.circle(img, p, 17, (0, 0, 0), 5); cv2.circle(img, p, 17, (255, 255, 255), 2)
            x = self._proj(Tbc_inv, self._locked_xyz)
            if x:
                u, v = x; r = 22
                for (c, t) in (((0, 0, 0), 7), ((0, 255, 0), 3)):
                    cv2.line(img, (u - r, v - r), (u + r, v + r), c, t)
                    cv2.line(img, (u - r, v + r), (u + r, v - r), c, t)
            pt = self._proj(Tbc_inv, tip)
            if pt:
                cv2.circle(img, pt, 6, (0, 0, 0), -1); cv2.circle(img, pt, 4, (0, 235, 235), -1)

            k = getattr(self, "_vk", 0)                      # spiral_k / contact reason published by SeatV2
            reason = getattr(self, "_vreason", None)
            above = (float(tip[2]) - float(self._locked_xyz[2])) * 1000.0
            soft = k > 0
            stiff_s = f"[{int(LAT_STIFF)},{int(LAT_STIFF)},{int(AX_STIFF)}] RCC-SOFT" if soft else "[90,90,90] STIFF"
            col = (0, 255, 255) if soft else (200, 200, 200)
            err = float(np.linalg.norm(np.array(g[:3]) - self._locked_xyz)) * 1000 if g else float("nan")

            cv2.rectangle(img, (0, 0), (img.shape[1], 104), (18, 18, 18), -1)
            cv2.putText(img, f"GT-FREE SEAT-V2  {self._task.port_name}  lock err {err:.0f}mm  F {info['force']:.0f}N",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 0), 2)
            cv2.putText(img, f"tip {above:5.1f}mm above port   CONTACT: {reason or '-':9s}  spiral_k {k:2d}",
                        (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (255, 255, 255), 2)
            cv2.putText(img, f"stiffness {stiff_s}", (12, 93), cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2)
            cv2.putText(img, "GREEN X = our PERCEIVED lock (drives the arm)   white ring = GT (reference only)",
                        (12, img.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)

            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ok:
                self._frames.append(buf.tobytes())          # RAM only
        except Exception:
            pass
        return port_tf, plug_tf, info

    def insert_cable(self, task, get_observation, move_robot, send_feedback):
        self._frames = []; self._vk = 0; self._vreason = None
        ok = super().insert_cable(task, get_observation, move_robot, send_feedback)
        d = f"{OUT_ROOT}/trial{self._trial_idx}"
        os.makedirs(d, exist_ok=True)
        for i, b in enumerate(self._frames):
            with open(f"{d}/{i:05d}.jpg", "wb") as fh:
                fh.write(b)
        self.get_logger().info(f"VIZ: flushed {len(self._frames)} frames -> {d}")
        self._frames = []
        return ok
