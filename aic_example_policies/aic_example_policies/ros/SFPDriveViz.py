#
#  SFPDriveViz — TRUTHFUL diagnostic video of the GT-FREE insert on our BEST policy (main, 92.2).
#  Subclasses PerceptionInsertSFPDrive -> CONTROL IS FULLY GT-FREE (perceived board pose +
#  slide-select lock + safe-clear motion).  `gt:=true` only draws the white reference ring.
#
#     GREEN X    = our PERCEIVED frozen lock (what actually drives the arm)
#     WHITE ring = GT port (reference ONLY — never read for control)
#     YELLOW dot = plug tip (FK)
#
#  HUD tells the TRUTH about what this policy does (earlier videos were narrated as a near-miss
#  insertion; they are not):
#     * height of the plug tip ABOVE THE PORT DATUM  -> it parks at ~46 mm, the SFP funnel depth
#     * PARKED banner once the tip stops descending  -> the plug is RESTING ON THE CAGE FACE
#     * FORCE STACK: INERT — the deployed contact test needs a force INCREASE, but landing on the
#       cage face UNLOADS the wrist (force DROPS), so it never fires and the search NEVER runs.
#       The scored "Partial insertion 0.05 m" IS this parked position, not an insertion.
#
#  PERF: frames are JPEG-encoded to RAM (~3 ms) and flushed to disk AFTER the trial.  Writing PNGs
#  inside the 50 ms control loop stalls the descent and changes the outcome (measured: 132.2 -> 54.6
#  on identical code), which would make the video a recording of a different run than the real one.
#
import os

import numpy as np

from aic_example_policies.ros.PerceptionInsertSFPDrive import PerceptionInsertSFPDrive
from aic_example_policies.ros.PerceptionInsertSFP import FXN, FYN, CXN, CYN

OUT_ROOT = "/home/skr/aic_data/sfp_free_viz"
FUNNEL = {"sfp": 0.0458, "sc": 0.0156}   # CAD entrance->seat funnel depth (GT-free)
STALL_N = 20                              # tip flat over 20 steps (1 s) => parked


class SFPDriveViz(PerceptionInsertSFPDrive):

    def _proj(self, Tbc_inv, P):
        pc = Tbc_inv[:3, :3] @ np.asarray(P, float) + Tbc_inv[:3, 3]
        if pc[2] <= 1e-3:
            return None
        return (int(FXN * pc[0] / pc[2] + CXN), int(FYN * pc[1] / pc[2] + CYN))

    def _perceive(self, get_observation):
        import cv2
        port_tf, plug_tf, info = super()._perceive(get_observation)   # GT-FREE control path, UNCHANGED
        if getattr(self, "_locked_xyz", None) is None:
            return port_tf, plug_tf, info
        try:
            obs = get_observation()
            img = cv2.cvtColor(self._raw_rgb(obs.center_image), cv2.COLOR_RGB2BGR)
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

            # ---- the truth: how high above the port is the tip, and has it STOPPED? ----
            above = (float(tip[2]) - float(self._locked_xyz[2])) * 1000.0     # mm above the port datum
            self._zh = getattr(self, "_zh", []); self._zh.append(float(tip[2]))
            parked = len(self._zh) >= STALL_N and (self._zh[-STALL_N] - self._zh[-1]) < 0.0005
            f0 = getattr(self, "_f0", None)
            if f0 is None and len(self._zh) == 1:
                self._f0 = f0 = info["force"]
            lock_err = float(np.linalg.norm(np.array(g[:3]) - self._locked_xyz)) * 1000 if g else float("nan")
            funnel = FUNNEL.get(self._plug_type, 0.046) * 1000

            cv2.rectangle(img, (0, 0), (img.shape[1], 104), (18, 18, 18), -1)
            cv2.putText(img, f"GT-FREE  {self._task.port_name}   lock err {lock_err:.0f}mm   F {info['force']:.0f}N"
                             f"{'' if f0 is None else f'  (baseline {f0:.0f}N)'}",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 0), 2)
            cv2.putText(img, f"plug tip is {above:6.1f} mm ABOVE the port  (funnel depth {funnel:.0f} mm)",
                        (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 255, 255), 2)
            if parked:
                cv2.putText(img, "PARKED ON THE CAGE FACE — not inserting.  FORCE STACK: INERT (never triggers)",
                            (12, 93), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (60, 60, 255), 2)
            else:
                cv2.putText(img, "descending...", (12, 93), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (180, 180, 180), 2)
            cv2.putText(img, "GREEN X = our PERCEIVED lock (drives the arm)   white ring = GT (reference only)",
                        (12, img.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)

            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            if ok:
                self._frames.append(buf.tobytes())          # RAM only — NO disk I/O in the control loop
        except Exception as ex:
            if len(getattr(self, "_frames", [])) % 120 == 0:
                self.get_logger().warn(f"viz skip: {ex}")
        return port_tf, plug_tf, info

    def insert_cable(self, task, get_observation, move_robot, send_feedback):
        self._frames = []; self._zh = []; self._f0 = None
        ok = super().insert_cable(task, get_observation, move_robot, send_feedback)
        d = f"{OUT_ROOT}/trial{self._trial_idx}"            # flush AFTER the trial — loop already done
        os.makedirs(d, exist_ok=True)
        for i, b in enumerate(self._frames):
            with open(f"{d}/{i:05d}.jpg", "wb") as fh:
                fh.write(b)
        self.get_logger().info(f"VIZ: flushed {len(self._frames)} frames -> {d} (post-trial, zero loop cost)")
        self._frames = []
        return ok
