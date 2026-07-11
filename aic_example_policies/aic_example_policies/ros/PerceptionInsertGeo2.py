#
#  PerceptionInsertGeo2 — two-stage lock: coarse sweep (from distance) THEN a closer REFINE sweep
#  centered on the coarse port, before the blind descent. The lock is frozen once perception goes
#  blind up close, so a second, nearer look (smaller world-error per pixel, port well-centered)
#  refines P0 -> P1 and corrects a bad coarse lock BEFORE committing.
#
#  Stage 1 (super): board yaw (magenta) + coarse CAD-z port lock P0.
#  Stage 2 (here) : hover over P0 at a lower altitude with the SAME viewing orientation the sweep used,
#                   5-pose mini-raster -> YOLO -> CAD-z -> P1 (+ SFP two-port selection). Keep stage-1 yaw.
#  Logs the refine detection count/spread and the P0->P1 shift so we can see if the closer look helps.
#
import math

import numpy as np
from geometry_msgs.msg import Point, Pose, Quaternion
from tf2_ros import TransformException

from aic_example_policies.ros.PerceptionInsertYOLO import (
    CAMS, KP_SCALE, FX, FY, CX, CY, PORT_FILTER_N, YOLO_CLASS, YOLO_CONF,
)
from aic_example_policies.ros.PerceptionInsertGeo import PerceptionInsertGeo, CAD_Z, _rot2

REFINE_ABOVE = 0.20                                        # hover this far above the port (closer than home)
REFINE_RASTER = [(0.0, 0.0), (0.05, 0.0), (-0.05, 0.0), (0.0, 0.05), (0.0, -0.05)]


class PerceptionInsertGeo2(PerceptionInsertGeo):

    def _sweep_board(self, get_observation, move_robot):
        import cv2
        pts = super()._sweep_board(get_observation, move_robot)     # Stage 1 -> coarse P0 in self._sweep_port
        if self._sweep_port is None:
            return pts
        P0 = self._sweep_port.copy()
        q = getattr(self, "_sweep_qhome", None)
        if q is None:
            return pts
        want = YOLO_CLASS.get(self._plug_type); pz = CAD_Z.get(self._plug_type, 0.0145)
        refine_z = float(P0[2] + REFINE_ABOVE)
        ray_o = []; ray_d = []
        for ddx, ddy in REFINE_RASTER:                              # Stage 2: closer raster centered on P0
            pose = Pose(position=Point(x=float(P0[0] + ddx), y=float(P0[1] + ddy), z=refine_z),
                        orientation=Quaternion(w=float(q[0]), x=float(q[1]), y=float(q[2]), z=float(q[3])))
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
                rgb = self._raw_rgb(raws[c]); R, tt = Tbc[:3, :3], Tbc[:3, 3]
                res = self._yolo.predict(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                         conf=YOLO_CONF, verbose=False, device=self._device)[0]
                if len(res.boxes):
                    cls = res.boxes.cls.cpu().numpy().astype(int)
                    conf = res.boxes.conf.cpu().numpy(); xywh = res.boxes.xywh.cpu().numpy()
                    sel = np.where(cls == want)[0]
                    j = int(sel[np.argmax(conf[sel])]) if len(sel) else int(np.argmax(conf))
                    dd = R @ np.array([(xywh[j, 0] * KP_SCALE - CX) / FX,
                                       (xywh[j, 1] * KP_SCALE - CY) / FY, 1.0])
                    ray_o.append(tt.copy()); ray_d.append(dd)

        # CAD-z back-projection of the refine rays
        port_xy = []
        for o, d in zip(ray_o, ray_d):
            if abs(d[2]) < 1e-9:
                continue
            s = (pz - o[2]) / d[2]
            if s > 0:
                port_xy.append((o + s * d)[:2])
        if len(port_xy) < 3:
            self.get_logger().warn(f"REFINE: only {len(port_xy)} rays at closer range -> keep coarse P0")
            return pts
        port_xy = np.array(port_xy); spread = float(np.hypot(*(port_xy.max(0) - port_xy.min(0))))
        m = self._robust_med(port_xy)
        # SFP two-port selection at the closer range too (port_0 = larger board-x)
        pn = getattr(self._task, "port_name", "") if self._task else ""
        if self._plug_type == "sfp" and self._board_yaw is not None and self._board_center is not None:
            B = (port_xy - self._board_center) @ _rot2(self._board_yaw); bx = B[:, 0]
            order = np.argsort(bx); bxs = bx[order]; gi = int(np.argmax(np.diff(bxs))) if len(bxs) > 1 else -1
            hi, lo = (order[gi + 1:], order[:gi + 1]) if gi >= 0 else (order, np.array([], int))
            sep = (np.median(bx[hi]) - np.median(bx[lo])) if len(lo) and len(hi) else 0.0
            if len(lo) >= 2 and len(hi) >= 2 and 0.012 < sep < 0.033:
                isel = hi if pn.endswith("port_0") else lo
                m = self._robust_med(port_xy[isel])
                self.get_logger().info(f"REFINE SFP 2-port resolved: {pn} sep {sep*1000:.0f}mm")
        P1 = np.array([m[0], m[1], pz])
        shift = float(np.linalg.norm(P1[:2] - P0[:2])) * 1000
        self._sweep_port = P1
        self._port_hist = [P1.copy() for _ in range(PORT_FILTER_N)]
        self.get_logger().info(f"REFINE LOCK P1 {np.round(P1,3)} ({len(port_xy)} rays, spread {spread*1000:.0f}mm, "
                               f"P0->P1 shift {shift:.0f}mm)")
        return pts
