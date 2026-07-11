#
#  PerceptionInsertYOLOSweep — sweep-lock variant of PerceptionInsertYOLO.
#
#  Why: YOLO detects the port only when the board is in view from DISTANCE (the home sweep);
#  at the close-up insertion hover/descent it returns no box, so per-step triangulation never
#  locks. Fix = the gated-lock (A5) pattern that the OFFLINE gate validated (2/3, SC 3.9mm):
#      during the SWEEP  -> YOLO boxes per cam -> back-project rays -> robust-triangulate -> LOCK
#      during insertion  -> HOLD the lock (no re-perception; YOLO is blind up close)
#
#  Standalone subclass: does NOT modify PerceptionInsertKP or PerceptionInsertYOLO.
#  Inherits all flags/machinery from PerceptionInsertYOLO (USE_GT_PORT_POS=False -> YOLO position,
#  SWEEP_ONLY=False -> insert, SWEEP_BOARD=True -> sweep runs, DISAMBIG_GT=True -> isolate the 90deg).
#
import math

import numpy as np
from geometry_msgs.msg import Point, Pose, Quaternion
from tf2_ros import TransformException

from aic_example_policies.ros.PerceptionInsertYOLO import (
    PerceptionInsertYOLO, CAMS, KP_SCALE, FX, FY, CX, CY, SWEEP_OFFSETS,
    BOARD_GRAY_LO, BOARD_GRAY_HI, DISAMBIG_GT, PORT_FILTER_N,
    YOLO_CLASS, YOLO_CONF, _quat, _R, _triangulate,
)


class PerceptionInsertYOLOSweep(PerceptionInsertYOLO):

    def _yolo_uv(self, obs):
        # once the sweep lock is set, insertion is HOLD-ONLY (YOLO can't see the port up close)
        if getattr(self, "_sweep_port", None) is not None:
            return np.full((3, 2), np.nan)
        return super()._yolo_uv(obs)

    def _robust_triangulate(self, origins, dirs, iters=4, thr=0.05):
        o = [np.asarray(x, float) for x in origins]
        d = [np.asarray(x, float) / (np.linalg.norm(x) + 1e-9) for x in dirs]
        keep = list(range(len(o)))
        p = _triangulate([o[i] for i in keep], [d[i] for i in keep])
        for _ in range(iters):
            nk = [i for i in range(len(o))
                  if np.linalg.norm((p - o[i]) - np.dot(p - o[i], d[i]) * d[i]) < thr]
            if len(nk) < 3 or len(nk) == len(keep):
                break
            keep = nk
            p = _triangulate([o[i] for i in keep], [d[i] for i in keep])
        return p, len(keep)

    def _sweep_board(self, get_observation, move_robot):
        """Raster once: board footprint (z=0) -> board yaw  AND  YOLO port rays -> port LOCK."""
        import cv2
        T_home = self._T_base_frame("gripper/tcp"); q_home = _quat(T_home[:3, :3]); t_home = T_home[:3, 3]
        want = YOLO_CLASS.get(self._plug_type)
        pts = []; ray_o = []; ray_d = []
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
                # board footprint -> z=0 (for yaw), same as PerceptionInsertKP
                small = cv2.resize(rgb, None, fx=KP_SCALE, fy=KP_SCALE)
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
                # YOLO port ray (native frame) -> for the position lock
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

        # board yaw from the footprint (identical to the base)
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
            gtf = self._gt_transform("task_board")
            if DISAMBIG_GT and gtf:
                gy = math.atan2(_R(gtf[3:])[1, 0], _R(gtf[3:])[0, 0])
                y = min(cands, key=lambda c: abs(((c - gy + math.pi) % (2 * math.pi)) - math.pi))
            self._board_yaw = y
            self.get_logger().info(f"board sweep yaw {math.degrees(y):.1f} deg, center {np.round(self._board_center,3)}")

        # PORT LOCK from the sweep YOLO rays (robust triangulation), pre-fill the running lock
        self._sweep_port = None
        if len(ray_o) >= 3:
            port, ninl = self._robust_triangulate(ray_o, ray_d)
            self._sweep_port = port
            self._port_hist = [port.copy() for _ in range(PORT_FILTER_N)]   # A5: perceive-once lock
            self.get_logger().info(f"YOLO sweep PORT LOCK {np.round(port,3)} "
                                   f"({ninl}/{len(ray_o)} inlier rays)")
        else:
            self.get_logger().warn(f"YOLO sweep: only {len(ray_o)} port rays -> NO lock (will fail)")
        return pts
