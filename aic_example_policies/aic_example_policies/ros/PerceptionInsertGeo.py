#
#  PerceptionInsertGeo — fully GT-FREE board-first perception, wired for the closed loop.
#  Standalone subclass of PerceptionInsertYOLOSweep. Overrides ONLY _sweep_board with the three
#  fixes validated offline on sweep_dump_val (10/12 on the correct opening, SC 6/6 <=1mm):
#    (1) DEPTH  : YOLO port rays -> intersect a CONSTANT CAD z-plane (sfp .1335, sc .0145; board z=0)
#                 + robust median   [replaces the narrow-baseline triangulation that scored -51].
#    (2) YAW    : 90-deg quadrant from the MAGENTA anchor (-.070,+.121 board-frame) [replaces DISAMBIG_GT].
#    (3) SFP    : two ports/card 21.8mm apart on the rail -> select the Task's port by board-x ORDER
#                 (port_0 = larger board-x); if only one opening is seen -> keep median + flag LOW-conf.
#  No ground truth anywhere in the estimate (GT only ever used by the engine to score).
#
import math

import numpy as np
from geometry_msgs.msg import Point, Pose, Quaternion
from tf2_ros import TransformException

from aic_example_policies.ros.PerceptionInsertYOLO import (
    CAMS, KP_SCALE, FX, FY, CX, CY, SWEEP_OFFSETS,
    BOARD_GRAY_LO, BOARD_GRAY_HI, PORT_FILTER_N, YOLO_CLASS, YOLO_CONF, _quat,
)
from aic_example_policies.ros.PerceptionInsertYOLOSweep import PerceptionInsertYOLOSweep

CAD_Z = {"sfp": 0.1335, "sc": 0.0145}      # port opening height above board (board sits at base z=0)
MAG_BOARD = np.array([-0.070, 0.121])      # magenta pick-zone in the board frame (calibrated, stable)


def _rot2(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


class PerceptionInsertGeo(PerceptionInsertYOLOSweep):

    def _robust_med(self, P, thr=0.03):
        m = np.median(P, axis=0)
        for _ in range(4):
            keep = np.linalg.norm(P - m, axis=1) < thr
            if keep.sum() >= 3:
                m = np.median(P[keep], axis=0)
        return m

    def _sweep_board(self, get_observation, move_robot):
        """GT-free raster: board footprint -> yaw; magenta -> quadrant; YOLO rays -> CAD-z port lock."""
        import cv2
        T_home = self._T_base_frame("gripper/tcp"); q_home = _quat(T_home[:3, :3]); t_home = T_home[:3, 3]
        self._sweep_thome = t_home.copy(); self._sweep_qhome = q_home.copy()   # for the refine stage
        want = YOLO_CLASS.get(self._plug_type)
        pz = CAD_Z.get(self._plug_type, 0.0145)
        pts = []; ray_o = []; ray_d = []; mags = []
        for dx, dy, dz in SWEEP_OFFSETS:
            pose = Pose(position=Point(x=float(t_home[0] + dx), y=float(t_home[1] + dy), z=float(t_home[2] + dz)),
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
                rgb = self._raw_rgb(raws[c]); R, tt = Tbc[:3, :3], Tbc[:3, 3]
                small = cv2.resize(rgb, None, fx=KP_SCALE, fy=KP_SCALE)
                # board footprint -> z=0 (for yaw magnitude)
                mask = cv2.inRange(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), BOARD_GRAY_LO, BOARD_GRAY_HI)
                ys, xs = np.where(mask > 0)
                if len(xs):
                    idx = np.linspace(0, len(xs) - 1, min(300, len(xs))).astype(int)
                    for u, v in zip(xs[idx], ys[idx]):
                        d = R @ np.array([(u - CX) / FX, (v - CY) / FY, 1.0])
                        if abs(d[2]) < 1e-6:
                            continue
                        s = -tt[2] / d[2]
                        if s > 0:
                            p = tt + s * d
                            if -1 < p[0] < 1 and -1 < p[1] < 1:
                                pts.append([float(p[0]), float(p[1])])
                # magenta pick-zone centroid -> z=0 (for GT-free quadrant)
                r_, g_, b_ = small[:, :, 0].astype(int), small[:, :, 1].astype(int), small[:, :, 2].astype(int)
                mg = (r_ > 110) & (b_ > 110) & (g_ < 90)
                if mg.sum() > 8:
                    my, mx = np.where(mg); u, v = float(np.median(mx)), float(np.median(my))
                    d = R @ np.array([(u - CX) / FX, (v - CY) / FY, 1.0])
                    if abs(d[2]) > 1e-6:
                        s = -tt[2] / d[2]
                        if s > 0:
                            pm = tt + s * d
                            if -1 < pm[0] < 1 and -1 < pm[1] < 1:
                                mags.append(pm[:2])
                # YOLO port ray (target class) -> for the CAD-z position lock
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

        # ---- board yaw (magnitude from footprint, quadrant from MAGENTA) ----
        pts = np.array(pts) if pts else np.zeros((0, 2))
        self._board_yaw = None; self._board_center = None
        if len(pts) >= 50:
            cell = 0.02; q = np.floor(pts / cell).astype(int); keys = {}
            for i, k in enumerate(map(tuple, q)):
                keys.setdefault(k, []).append(i)
            thr = max(len(v) for v in keys.values()) * 0.25
            core = pts[[i for v in keys.values() if len(v) >= thr for i in v]]
            rect = cv2.minAreaRect((core * 1000).astype(np.float32)); (cx, cy), (w, h), ang = rect
            long_dir = ang if w >= h else ang + 90
            y = math.radians(((long_dir + 90) % 180) - 90)
            self._board_center = np.array([cx / 1000.0, cy / 1000.0])
            cands = [y, y + math.pi / 2, y + math.pi, y - math.pi / 2]
            if mags:
                M = np.median(np.array(mags), axis=0)
                y = min(cands, key=lambda a: float(np.linalg.norm(self._board_center + _rot2(a) @ MAG_BOARD - M)))
                self.get_logger().info(f"magenta quadrant: {len(mags)} obs, M={np.round(M,3)}")
            else:
                self.get_logger().warn("no magenta seen -> yaw quadrant UNRESOLVED (keeping raw)")
            self._board_yaw = y
            self.get_logger().info(f"board sweep yaw {math.degrees(y):.1f} deg, center {np.round(self._board_center,3)}")

        # ---- PORT LOCK: CAD-z z-plane back-projection (NOT triangulation) ----
        self._sweep_port = None; self._sweep_lowconf = False
        port_xy = []
        for o, d in zip(ray_o, ray_d):
            if abs(d[2]) < 1e-9:
                continue
            s = (pz - o[2]) / d[2]
            if s > 0:
                port_xy.append((o + s * d)[:2])
        if len(port_xy) >= 3:
            port_xy = np.array(port_xy)
            m = self._robust_med(port_xy)
            # ---- SFP two-port selection (board-x order; port_0 = larger board-x) ----
            pn = getattr(self._task, "port_name", "") if self._task else ""
            if self._plug_type == "sfp" and self._board_yaw is not None and self._board_center is not None:
                B = (port_xy - self._board_center) @ _rot2(self._board_yaw)   # board-frame coords
                bx = B[:, 0]; order = np.argsort(bx); bxs = bx[order]
                gi = int(np.argmax(np.diff(bxs))) if len(bxs) > 1 else -1
                hi, lo = (order[gi + 1:], order[:gi + 1]) if gi >= 0 else (order, np.array([], int))
                sep = (np.median(bx[hi]) - np.median(bx[lo])) if len(lo) and len(hi) else 0.0
                if len(lo) >= 2 and len(hi) >= 2 and 0.012 < sep < 0.033:     # two openings resolved
                    isel = hi if pn.endswith("port_0") else lo
                    m = self._robust_med(port_xy[isel])
                    self.get_logger().info(f"SFP 2-port: selected {pn} ({len(isel)} rays, sep {sep*1000:.0f}mm)")
                else:
                    self._sweep_lowconf = True
                    self.get_logger().warn(f"SFP 1-opening (sep {sep*1000:.0f}mm) -> LOW-conf, median kept")
            port = np.array([m[0], m[1], pz])
            self._sweep_port = port
            self._port_hist = [port.copy() for _ in range(PORT_FILTER_N)]
            self.get_logger().info(f"CAD-z PORT LOCK {np.round(port,3)} (z={pz}, {len(port_xy)} rays, "
                                   f"lowconf={self._sweep_lowconf})")
        else:
            self.get_logger().warn(f"CAD-z: only {len(port_xy)} port rays -> NO lock (will fail)")
        return pts
