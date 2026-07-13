#
#  FullViz — video of the SC (and SFP) insert on PerceptionInsertFull, same format as sfp_trial1_gtfree.mp4.
#  CONTROL IS 100% GT-FREE (module-gated lock + safe-clear + revived contact predicate).
#  `gt:=true` only draws the white reference ring — nothing reads it to drive.
#
#     GREEN X    = our PERCEIVED lock (what drives the arm)
#     WHITE ring = GT port (reference only)
#     YELLOW dot = plug tip (FK)
#
#  HUD shows the two things that were broken and are now fixed:
#     * the LOCK    — SC was 58mm off (median across all 3 SC modules); the module gate makes it 2mm
#     * the CONTACT — the predicate needed a force INCREASE, but a cage-face landing UNLOADS the wrist,
#                     so it never fired and the plug just parked.  Now: CONTACT: stall/force_abs, spiral_k,
#                     and the stiffness flipping to RCC-SOFT [40,40,90].
#
#  PERF: frames are JPEG-encoded to RAM (~3 ms) and flushed AFTER the trial. get_observation() is CACHED
#  (the base _perceive already fetches it) — writing/​fetching inside the 50 ms control loop stalls the
#  descent and changes the outcome (measured: 132.2 -> 54.6 on identical code).
#
import math
import os

import numpy as np

from aic_example_policies.ros.PerceptionInsertFull import (
    PerceptionInsertFull, ENTRANCE_H, LAT_STIFF, AX_STIFF, SEAT_TIP, SEAT_LAT,
)
from aic_example_policies.ros.PerceptionInsertSFP import FXN, FYN, CXN, CYN

OUT_ROOT = "/home/skr/aic_data/full_viz"


class FullViz(PerceptionInsertFull):

    def _proj(self, Tbc_inv, P):
        pc = Tbc_inv[:3, :3] @ np.asarray(P, float) + Tbc_inv[:3, 3]
        if pc[2] <= 1e-3:
            return None
        return (int(FXN * pc[0] / pc[2] + CXN), int(FYN * pc[1] / pc[2] + CYN))

    def _perceive(self, get_observation):
        import cv2
        cache = {}

        def cached():
            if "o" not in cache:
                cache["o"] = get_observation()
            return cache["o"]

        port_tf, plug_tf, info = super()._perceive(cached)      # GT-FREE control path, UNCHANGED
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

            typ = self._plug_type
            above = (float(tip[2]) - float(self._locked_xyz[2])) * 1000.0
            lat = math.hypot(float(tip[0]) - float(self._locked_xyz[0]),
                             float(tip[1]) - float(self._locked_xyz[1])) * 1000.0
            k = getattr(self, "_vk", 0); reason = getattr(self, "_vreason", None)
            soft = k > 0
            stiff_s = f"[{int(LAT_STIFF)},{int(LAT_STIFF)},{int(AX_STIFF)}] RCC-SOFT" if soft else "[90,90,90] STIFF"
            col = (0, 255, 255) if soft else (200, 200, 200)
            err = float(np.linalg.norm(np.array(g[:3]) - self._locked_xyz)) * 1000 if g else float("nan")
            funnel = ENTRANCE_H.get(typ, 0.046) * 1000
            seated = (above < SEAT_TIP * 1000) and (lat < SEAT_LAT * 1000)

            cv2.rectangle(img, (0, 0), (img.shape[1], 108), (18, 18, 18), -1)
            cv2.putText(img, f"GT-FREE  {typ.upper()}  {self._task.port_name}   LOCK ERR {err:.0f}mm"
                             f"   F {info['force']:.0f}N",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 0), 2)
            cv2.putText(img, f"tip {above:6.1f}mm above port ({lat:4.1f}mm lateral)   funnel {funnel:.0f}mm"
                             f"   CONTACT: {reason or '-'}  k {k}",
                        (12, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
            if seated:
                cv2.putText(img, f"*** SEATED — plug is IN ***  stiffness {stiff_s}",
                            (12, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2)
            else:
                cv2.putText(img, f"stiffness {stiff_s}", (12, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2)
            cv2.putText(img, "GREEN X = our PERCEIVED lock (drives the arm)   white ring = GT (reference only)",
                        (12, img.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)

            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ok:
                self._frames.append(buf.tobytes())
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
