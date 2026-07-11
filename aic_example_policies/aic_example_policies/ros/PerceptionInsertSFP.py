#
#  PerceptionInsertSFP — GT-free SFP policy. Same sweep + PERCEIVED board pose (gray-threshold fit + magenta)
#  + gated-lock + FK plug + force stack as PerceptionInsertGeo; the ONLY change is the port lock: it collects
#  ALL sfp YOLO candidates per sweep pose and runs the slide-estimation + fixed-identity selector
#  (validated offline: 0% wrong-port, 0.4 mm) IN THE PERCEIVED BOARD FRAME, instead of the fragile per-frame
#  board-x order select. SC falls back to the plain robust median (SC has a separate execution issue).
#
import math

import numpy as np
from geometry_msgs.msg import Point, Pose, Quaternion
from tf2_ros import TransformException

from aic_example_policies.ros.PerceptionInsertYOLO import (
    CAMS, KP_SCALE, FX, FY, CX, CY, SWEEP_OFFSETS,
    BOARD_GRAY_LO, BOARD_GRAY_HI, PORT_FILTER_N, YOLO_CLASS, YOLO_CONF, _quat,
)
from aic_example_policies.ros.PerceptionInsertGeo import PerceptionInsertGeo, CAD_Z, MAG_BOARD, _rot2

# SFP CAD nominal pair (board frame, slide=0): port_0 = larger board-x, port_1 = smaller.  card board-Y = CARD_Y0 + CARD_DY*k
P0X, P1X = -0.0708, -0.0926; CARD_Y0, CARD_DY = -0.188, 0.04; CAD_MID_X = 0.5*(P0X + P1X)
CARD_GATE, SPLIT_LO, SPLIT_HI, ACCEPT, MARGIN = 0.020, 0.012, 0.033, 0.013, 0.004
# board-pose fix (see board_pose.md): full-scale board mask -> native intrinsics; known-size box fit
FXN, FYN, CXN, CYN = FX / KP_SCALE, FY / KP_SCALE, CX / KP_SCALE, CY / KP_SCALE
BOARD_L, BOARD_W = 0.425, 0.30


class PerceptionInsertSFP(PerceptionInsertGeo):

    def _largest_cc(self, pts, cell=0.01):
        """keep only the biggest connected occupancy blob -> board slab (drops detached leaks)."""
        import cv2
        if len(pts) < 50:
            return pts
        lo = pts.min(0); gi = ((pts - lo) / cell).astype(int)
        gx, gy = gi[:, 0].max() + 1, gi[:, 1].max() + 1
        cnt = np.zeros((gx, gy), np.int32); np.add.at(cnt, (gi[:, 0], gi[:, 1]), 1)
        occ = (cnt >= max(2, int(0.05 * cnt.max()))).astype(np.uint8)
        n, lab = cv2.connectedComponents(occ, connectivity=8)
        if n <= 1:
            return pts
        big = 1 + int(np.argmax([(lab == k).sum() for k in range(1, n)]))
        return pts[lab[gi[:, 0], gi[:, 1]] == big]

    def _best_box_yaw(self, pts, yaw0, cell=0.01):
        """max-coverage placement of the FIXED 0.425x0.30 box; a fixed box can't stretch onto an
        ATTACHED NIC lobe, so it snaps to the true board yaw. Search around the minAreaRect seed."""
        best = None
        for dyaw in np.arange(-20, 20.001, 2.0):
            y = yaw0 + math.radians(dyaw); rp = pts @ _rot2(-y).T
            lo = rp.min(0); gi = ((rp - lo) / cell).astype(int)
            gx, gy = gi[:, 0].max() + 2, gi[:, 1].max() + 2
            H = np.zeros((gx, gy), np.int32); np.add.at(H, (gi[:, 0], gi[:, 1]), 1)
            ii = np.pad(H.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
            bw, bh = int(BOARD_L / cell), int(BOARD_W / cell)   # long edge -> x (rp aligned by yaw0)
            if gx <= bw or gy <= bh:
                continue
            cov = ii[bw:, bh:] - ii[:-bw, bh:] - ii[bw:, :-bh] + ii[:-bw, :-bh]
            i, j = np.unravel_index(int(np.argmax(cov)), cov.shape)
            if best is None or cov[i, j] > best[0]:
                best = (cov[i, j], y)
        return best[1] if best else yaw0

    def _sfp_slide_select(self, frame_cands):
        """slide-estimation + fixed-identity target selection in the PERCEIVED board frame.
        frame_cands = list over sweep poses of [base-xy candidates]. Returns (target_xy, n_both_visible)."""
        cen = self._board_center; yaw = self._board_yaw
        k = int(self._task.target_module_name.split("_")[-1]); card_y = CARD_Y0 + CARD_DY * k
        is_p0 = self._task.port_name.endswith("port_0")
        tgt_nom = P0X if is_p0 else P1X; sib_nom = P1X if is_p0 else P0X
        # PASS 1 — estimate the rail slide from both-openings-visible poses
        slides = []
        for pc in frame_cands:
            if len(pc) < 2:
                continue
            B = (np.array(pc) - cen) @ _rot2(yaw)
            bc = B[np.abs(B[:, 1] - card_y) < CARD_GATE]
            if len(bc) < 2:
                continue
            bx = np.sort(bc[:, 0]); gi = int(np.argmax(np.diff(bx)))
            lo, hi = bx[:gi + 1], bx[gi + 1:]
            sep = np.median(hi) - np.median(lo) if len(lo) and len(hi) else 0.0
            if SPLIT_LO < sep < SPLIT_HI:
                slides.append(0.5 * (np.median(lo) + np.median(hi)) - CAD_MID_X)
        if not slides:
            return None, 0
        slide = float(np.median(slides)); ta = tgt_nom + slide; sa = sib_nom + slide
        # PASS 2 — classify all candidates against the FIXED identity; keep the target ones
        tgt = []
        for pc in frame_cands:
            if not len(pc):
                continue
            P = np.array(pc); B = (P - cen) @ _rot2(yaw)
            on = np.abs(B[:, 1] - card_y) < CARD_GATE
            Pb = P[on]; bx = B[on, 0]
            dt = np.abs(bx - ta); ds = np.abs(bx - sa)
            keep = (dt < ACCEPT) & (dt + MARGIN < ds)
            tgt.extend(Pb[keep].tolist())
        if len(tgt) < 3:
            return None, len(slides)
        return self._robust_med(np.array(tgt)), len(slides)

    def _sweep_board(self, get_observation, move_robot):
        import cv2
        T_home = self._T_base_frame("gripper/tcp"); q_home = _quat(T_home[:3, :3]); t_home = T_home[:3, 3]
        self._sweep_thome = t_home.copy(); self._sweep_qhome = q_home.copy()
        want = YOLO_CLASS.get(self._plug_type); pz = CAD_Z.get(self._plug_type, 0.0145)
        pts = []; mags = []; frame_cands = []
        for dx, dy, dz in SWEEP_OFFSETS:
            pose = Pose(position=Point(x=float(t_home[0] + dx), y=float(t_home[1] + dy), z=float(t_home[2] + dz)),
                        orientation=Quaternion(w=float(q_home[0]), x=float(q_home[1]), y=float(q_home[2]), z=float(q_home[3])))
            self.set_pose_target(move_robot=move_robot, pose=pose)
            for _ in range(30):
                self.sleep_for(0.05)
            obs = get_observation()
            raws = {"left": obs.left_image, "center": obs.center_image, "right": obs.right_image}
            pose_cands = []                                       # ALL sfp candidates at THIS sweep pose (both openings)
            for c in CAMS:
                try:
                    Tbc = self._T_base_frame(f"{c}_camera/optical")
                except TransformException:
                    continue
                rgb = self._raw_rgb(raws[c]); R, tt = Tbc[:3, :3], Tbc[:3, 3]
                # BOARD MASK — FULL native scale (no 0.25 downscale: the blur merged board+NIC)
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
                res = self._yolo.predict(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), conf=YOLO_CONF, verbose=False, device=self._device)[0]
                if len(res.boxes):
                    cls = res.boxes.cls.cpu().numpy().astype(int); xywh = res.boxes.xywh.cpu().numpy()
                    for j in np.where(cls == want)[0]:            # ALL target-class boxes (candidate generator)
                        dd = R @ np.array([(xywh[j, 0] * KP_SCALE - CX) / FX, (xywh[j, 1] * KP_SCALE - CY) / FY, 1.0])
                        if abs(dd[2]) > 1e-9:
                            s = (pz - tt[2]) / dd[2]
                            if s > 0:
                                pose_cands.append((tt + s * dd)[:2])
            frame_cands.append(pose_cands)

        # ---- BOARD POSE — magenta-anchored center + known-size yaw (see board_pose.md) ----
        pts = np.array(pts) if pts else np.zeros((0, 2))
        self._board_yaw = None; self._board_center = None
        slab = self._largest_cc(pts)                          # drop the detached NIC/base blob
        if len(slab) >= 50 and mags:
            M = np.median(np.array(mags), axis=0)
            rect = cv2.minAreaRect((slab * 1000).astype(np.float32)); (rcx, rcy), (w, h), ang = rect
            long_dir = ang if w >= h else ang + 90
            yaw0 = math.radians(((long_dir + 90) % 180) - 90)
            rcen = np.array([rcx / 1000.0, rcy / 1000.0])
            yks = self._best_box_yaw(slab, yaw0)              # fixed-size box ignores attached NIC lobe
            y = min([yks, yks + math.pi / 2, yks + math.pi, yks - math.pi / 2],
                    key=lambda a: float(np.linalg.norm(rcen + _rot2(a) @ MAG_BOARD - M)))
            self._board_yaw = y
            self._board_center = M - _rot2(y) @ MAG_BOARD     # magenta anchor (immune to cloud-centroid bias)
            self.get_logger().info(f"board FIX yaw {math.degrees(y):.1f} deg, center {np.round(self._board_center,3)} (magenta-anchored, slab {len(slab)}/{len(pts)})")

        # ---- PORT LOCK: SFP slide-estimation selector (perceived board frame) ; SC = plain median ----
        self._sweep_port = None; self._sweep_lowconf = False
        m = None
        if self._plug_type == "sfp" and self._board_center is not None:
            m, n_bv = self._sfp_slide_select(frame_cands)
            if m is not None:
                self.get_logger().info(f"SFP slide-select LOCK: {self._task.port_name} from {n_bv} both-visible poses")
            else:
                allc = [c for pc in frame_cands for c in pc]
                if len(allc) >= 3:
                    m = self._robust_med(np.array(allc)); self._sweep_lowconf = True
                    self.get_logger().warn("SFP: no both-visible pose -> AMBIGUOUS, LOW-conf median fallback")
        else:
            allc = [c for pc in frame_cands for c in pc]
            if len(allc) >= 3:
                m = self._robust_med(np.array(allc))
        if m is not None:
            port = np.array([m[0], m[1], pz]); self._sweep_port = port
            self._port_hist = [port.copy() for _ in range(PORT_FILTER_N)]
            self.get_logger().info(f"CAD-z PORT LOCK {np.round(port,3)} (z={pz}, lowconf={self._sweep_lowconf})")
        else:
            self.get_logger().warn("no candidates -> NO lock (will fail)")
        return pts
