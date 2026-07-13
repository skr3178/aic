#
#  PerceptionInsertSCFix — STANDALONE. Fixes ONLY the SC port lock. SFP is bit-for-bit unchanged.
#
#  THE BUG (perception, not the seat): the eval's SC trial spawns THREE SC modules
#      sc_rail_0 : sc_mount_0   translation  0.00
#      sc_rail_1 : sc_mount_1   translation -0.04     <- the TARGET (sc_port_1 / sc_port_base) is here
#      (other)   : sc_mount_2   translation -0.03
#  ...and the SC lock takes a robust median over EVERY sc detection on the board:
#
#      if plug_type == "sfp":  m = self._sfp_slide_select(frame_cands)   # gates to the TARGET card
#      else:                   m = self._robust_med(ALL sc candidates)   # <-- averages ACROSS all 3 modules
#
#  SFP got a target-selection gate; SC never did. Averaging across the modules lands the lock ~58 mm
#  from the true port -- which is exactly the miss we measure (SC final plug-port distance 0.06 m).
#  A 36 mm spiral cannot rescue a 58 mm lock, so the seat work is irrelevant until this is fixed.
#
#  THE FIX: give SC the SAME step SFP already has -- a board-frame MODULE GATE. The SC rail slides
#  along board-X (+-60 mm), so the target module's board-Y is FIXED -> a board-Y gate cleanly separates
#  it from the distractor rails. SC has ONE port per module, so it needs NO slide/identity step (that
#  part of _sfp_slide_select exists only because a NIC card has TWO openings 21.8 mm apart).
#
#  STANDALONE: subclasses PerceptionInsertSFPDrive and re-implements the sweep with the SC gate added.
#  PerceptionInsertSFP.py / PerceptionInsertSFPDrive.py / PerceptionInsertYOLO.py are NOT modified.
#  The SFP branch here calls the inherited _sfp_slide_select, so SFP behaviour is identical.
#
import math

import numpy as np
from geometry_msgs.msg import Point, Pose, Quaternion
from tf2_ros import TransformException

from aic_example_policies.ros.PerceptionInsertYOLO import (
    CAMS, KP_SCALE, FX, FY, CX, CY, SWEEP_OFFSETS,
    BOARD_GRAY_LO, BOARD_GRAY_HI, PORT_FILTER_N, YOLO_CLASS, YOLO_CONF, _quat,
)
from aic_example_policies.ros.PerceptionInsertGeo import CAD_Z, MAG_BOARD, _rot2
from aic_example_policies.ros.PerceptionInsertSFP import FXN, FYN, CXN, CYN
from aic_example_policies.ros.PerceptionInsertSFPDrive import PerceptionInsertSFPDrive

# CAD nominal of the SC target in the BOARD frame (validated in ensemble_val_gtfree2.py).
# Rail slides along board-X, so board-Y is the module's fixed signature.
SC_CAD = {("sc_port_1", "sc_port_base"): (-0.075, 0.0705)}
SC_GATE_Y = 0.020        # board-Y tolerance -> keeps the target rail, drops the distractor rails


class PerceptionInsertSCFix(PerceptionInsertSFPDrive):

    # keep SFPDrive's per-trial reset + SAFE-CLEAR, but run OUR sweep (SC gate added)
    def _sweep_board(self, get_observation, move_robot):
        pts = self._sweep_board_scfix(get_observation, move_robot)
        self._locked_xyz = None                       # per-trial reset (as in PerceptionInsertSFPDrive)
        self._locked_quat = None
        self._safe_clear(move_robot)                  # joint-space IK-branch reset (the 92.2 fix)
        return pts

    def _sc_gate(self, frame_cands):
        """SC target selection — the step SFP has and SC never got.
        Gate candidates to the TARGET module by board-Y, then robust-median the survivors."""
        cen, yaw = self._board_center, self._board_yaw
        cad = SC_CAD.get((self._task.target_module_name, self._task.port_name))
        allc = [c for pc in frame_cands for c in pc]
        if not allc:
            return None, 0, 0
        A = np.array(allc)
        if cad is None or cen is None or yaw is None:
            self.get_logger().warn("SC: no CAD/board pose -> ungated median (old behaviour)")
            return self._robust_med(A), len(A), len(A)
        B = (A - cen) @ _rot2(yaw)                     # into the board frame
        on = np.abs(B[:, 1] - cad[1]) < SC_GATE_Y      # ★ MODULE GATE (board-Y = the target's rail)
        if on.sum() < 3:
            self.get_logger().warn(f"SC: gate kept only {int(on.sum())}/{len(A)} -> falling back to ungated median")
            return self._robust_med(A), len(A), len(A)
        return self._robust_med(A[on]), int(on.sum()), len(A)

    def _sweep_board_scfix(self, get_observation, move_robot):
        """Copy of PerceptionInsertSFP._sweep_board with ONE change: the SC branch gets a module gate."""
        import cv2
        T_home = self._T_base_frame("gripper/tcp"); q_home = _quat(T_home[:3, :3]); t_home = T_home[:3, 3]
        self._sweep_thome = t_home.copy(); self._sweep_qhome = q_home.copy()
        want = YOLO_CLASS.get(self._plug_type); pz = CAD_Z.get(self._plug_type, 0.0145)
        pts = []; mags = []; frame_cands = []
        for dx, dy, dz in SWEEP_OFFSETS:
            pose = Pose(position=Point(x=float(t_home[0] + dx), y=float(t_home[1] + dy), z=float(t_home[2] + dz)),
                        orientation=Quaternion(w=float(q_home[0]), x=float(q_home[1]),
                                               y=float(q_home[2]), z=float(q_home[3])))
            self.set_pose_target(move_robot=move_robot, pose=pose)
            for _ in range(30):
                self.sleep_for(0.05)
            obs = get_observation()
            raws = {"left": obs.left_image, "center": obs.center_image, "right": obs.right_image}
            pose_cands = []
            for c in CAMS:
                try:
                    Tbc = self._T_base_frame(f"{c}_camera/optical")
                except TransformException:
                    continue
                rgb = self._raw_rgb(raws[c]); R, tt = Tbc[:3, :3], Tbc[:3, 3]
                mask = cv2.inRange(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), BOARD_GRAY_LO, BOARD_GRAY_HI)
                ys, xs = np.where(mask > 0)
                if len(xs):
                    idx = np.linspace(0, len(xs) - 1, min(300, len(xs))).astype(int)
                    for u, v in zip(xs[idx], ys[idx]):
                        d = R @ np.array([(u - CXN) / FXN, (v - CYN) / FYN, 1.0])
                        if abs(d[2]) < 1e-6:
                            continue
                        s = -tt[2] / d[2]
                        if s > 0:
                            p = tt + s * d
                            if -1 < p[0] < 1 and -1 < p[1] < 1:
                                pts.append([float(p[0]), float(p[1])])
                r_, g_, b_ = rgb[:, :, 0].astype(int), rgb[:, :, 1].astype(int), rgb[:, :, 2].astype(int)
                mg = (r_ > 110) & (b_ > 110) & (g_ < 90)
                if mg.sum() > 8:
                    my, mx = np.where(mg); u, v = float(np.median(mx)), float(np.median(my))
                    d = R @ np.array([(u - CXN) / FXN, (v - CYN) / FYN, 1.0])
                    if abs(d[2]) > 1e-6:
                        s = -tt[2] / d[2]
                        if s > 0:
                            pm = tt + s * d
                            if -1 < pm[0] < 1 and -1 < pm[1] < 1:
                                mags.append(pm[:2])
                res = self._yolo.predict(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), conf=YOLO_CONF,
                                         verbose=False, device=self._device)[0]
                if len(res.boxes):
                    cls = res.boxes.cls.cpu().numpy().astype(int); xywh = res.boxes.xywh.cpu().numpy()
                    for j in np.where(cls == want)[0]:
                        dd = R @ np.array([(xywh[j, 0] * KP_SCALE - CX) / FX,
                                           (xywh[j, 1] * KP_SCALE - CY) / FY, 1.0])
                        if abs(dd[2]) > 1e-9:
                            s = (pz - tt[2]) / dd[2]
                            if s > 0:
                                pose_cands.append((tt + s * dd)[:2])
            frame_cands.append(pose_cands)

        # ---- BOARD POSE — magenta-anchored center + known-size yaw (UNCHANGED) ----
        pts = np.array(pts) if pts else np.zeros((0, 2))
        self._board_yaw = None; self._board_center = None
        slab = self._largest_cc(pts)
        if len(slab) >= 50 and mags:
            M = np.median(np.array(mags), axis=0)
            rect = cv2.minAreaRect((slab * 1000).astype(np.float32)); (rcx, rcy), (w, h), ang = rect
            long_dir = ang if w >= h else ang + 90
            yaw0 = math.radians(((long_dir + 90) % 180) - 90)
            rcen = np.array([rcx / 1000.0, rcy / 1000.0])
            yks = self._best_box_yaw(slab, yaw0)
            y = min([yks, yks + math.pi / 2, yks + math.pi, yks - math.pi / 2],
                    key=lambda a: float(np.linalg.norm(rcen + _rot2(a) @ MAG_BOARD - M)))
            self._board_yaw = y
            self._board_center = M - _rot2(y) @ MAG_BOARD
            self.get_logger().info(f"board FIX yaw {math.degrees(y):.1f} deg, "
                                   f"center {np.round(self._board_center,3)} (magenta-anchored)")

        # ---- PORT LOCK ----
        self._sweep_port = None; self._sweep_lowconf = False
        m = None
        if self._plug_type == "sfp" and self._board_center is not None:
            m, n_bv = self._sfp_slide_select(frame_cands)          # UNCHANGED (inherited)
            if m is not None:
                self.get_logger().info(f"SFP slide-select LOCK: {self._task.port_name} from {n_bv} both-visible poses")
            else:
                allc = [c for pc in frame_cands for c in pc]
                if len(allc) >= 3:
                    m = self._robust_med(np.array(allc)); self._sweep_lowconf = True
                    self.get_logger().warn("SFP: no both-visible pose -> AMBIGUOUS, LOW-conf median fallback")
        else:
            # ★★★ THE ONLY CHANGE: SC gets a MODULE GATE instead of a median over all 3 SC modules ★★★
            m, kept, total = self._sc_gate(frame_cands)
            if m is not None:
                self.get_logger().info(f"SC MODULE-GATE LOCK: {self._task.target_module_name}/"
                                       f"{self._task.port_name} — kept {kept}/{total} candidates "
                                       f"(board-Y gate ±{SC_GATE_Y*1000:.0f}mm)")

        if m is not None:
            port = np.array([m[0], m[1], pz]); self._sweep_port = port
            self._port_hist = [port.copy() for _ in range(PORT_FILTER_N)]
            gt = self._gt_transform(f"task_board/{self._task.target_module_name}/{self._task.port_name}_link")
            err = f" | vs GT {np.linalg.norm(np.array(gt[:2]) - port[:2])*1000:.0f}mm" if gt else ""
            self.get_logger().info(f"CAD-z PORT LOCK {np.round(port,3)} (z={pz}){err}")
        else:
            self.get_logger().warn("no candidates -> NO lock (will fail)")
        return pts
