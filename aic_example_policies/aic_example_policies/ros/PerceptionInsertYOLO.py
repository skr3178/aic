#
#  PerceptionInsertKP — standalone A/B of PerceptionInsert with the PORT POSITION supplied by the
#  keypoint detector + multi-view triangulation (our "eyes"), instead of DualPoseNet's regression head.
#
#  Everything else is identical to PerceptionInsert (CheatCode insertion section ported + reactive
#  force-stop / spiral / yaw-dither + sanity gate + median filter). Port ORIENTATION and the PLUG pose
#  still come from DualPoseNet, so this isolates exactly ONE variable — port position:
#      DualPoseNet regression  ~80 mm on eval   ->   keypoint triangulation  ~1.9 mm proxy (OOD)
#  The score delta vs the documented M7b run (+1.4) is therefore attributable purely to the new eyes.
#
#  Perception per control step:
#     3 wrist images -> {KPNet per cam -> pixel -> base_link ray} -> triangulate 3 rays -> port xyz
#                    -> DualPoseNet -> port quat + plug pose (base_link via center-cam TF)
#     -> CheatCode alignment math (ported) -> compliant set_pose_target.
#  Port is static -> median-filtered + sanity-gated. Plug flexes -> fresh each step.
#
#  Runs at ground_truth:=false (real score). Under ground_truth:=true it also logs keypoint-port,
#  dualnet-port and GT-port each step -> the true live eval-scene perception number.
#
#  Standalone: does NOT modify CheatCode or PerceptionInsert. Reuses only the Policy base class.
#

import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torchvision.transforms import functional as VF

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion, Transform
from rclpy.time import Time
from tf2_ros import TransformException
from transforms3d._gohlketransforms import quaternion_multiply, quaternion_slerp

# --- checkpoints ---
DUAL_CKPT = "/home/skr/aic_data/m7_cond224/best.pt"   # port orientation + plug (unchanged from M7b)
KP_CKPT = "/home/skr/aic_data/kp_v1_run/best.pt"       # NEW: port-position keypoint detector
YOLO_WEIGHTS = "/home/skr/aic_data/yolo_runs/best_final.pt"  # fine-tuned port detector (native 1024)
YOLO_CLASS = {"sfp": 0, "sc": 1}
YOLO_CONF = 0.25
IMG_SIZE = 1024        # DualPoseNet fallback; overridden by its checkpoint
KEEP_ASPECT = True
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
PORT_FILTER_N = 15

# keypoint detector: short-side-256 input (the 288x256 cache it trained on) + native rig intrinsics
KP_SHORT = 256
KP_DOWN = 4                       # KPNet heatmap stride (must match kp_train)
NATIVE_SHORT = 1024.0
KP_SCALE = KP_SHORT / NATIVE_SHORT    # 0.25 : native intrinsics -> detector-image intrinsics
FX = 1236.63 * KP_SCALE; FY = 1236.63 * KP_SCALE       # identical across all 3 cams
CX = 576.0 * KP_SCALE;  CY = 512.0 * KP_SCALE
CAMS = ["left", "center", "right"]

# M14: plug pose = FK(gripper) . T_grasp[type] — grasp offset is TYPE-FIXED (gripper->plug), measured
# from GT (constant within-trial <=0.3 deg, identical across same-type trials). Selected by task.plug_type.
# (translation m, quat wxyz).  Fair source for a real submission = cable-template nominal grasp.
T_GRASP = {
    "sfp": ([0.0, -0.02069, 0.05405], [0.983663, 0.177851, 0.005053, -0.027406]),
    "sc":  ([-0.00413, -0.01123, 0.01165], [0.773221, 0.1229, -0.181047, 0.595187]),
}
# diagnostic: use GT port pose (needs gt:=true) to isolate whether port perception is the sole gap
# CONFIRMED 2026-07-05: GT-port + FK-plug = 227.8 (2 full inserts + 1 partial). Port perception = sole gap.
USE_GT_PORT = False
USE_GT_PORT_POS = False   # YOLO supplies the port POSITION (real, GT-free)
# Stage 2b-orient: port orientation from board-yaw net.  port_orient = board_yaw ⊗ R_offset[type]
ORIENT_FROM_BOARD = True
# Stage 2b (classical sweep): at home, raster the wrist, segment the dark board, back-project to the
# ground plane (z=0) via FK, accumulate a top-down footprint -> board pose. No learned features.
SWEEP_BOARD = True
SWEEP_ONLY = False       # DO the insertion (real scored eval)
DISAMBIG_GT = True        # TEST ONLY: pick the 90deg board-yaw candidate nearest GT (validate seating)
# TCP raster (dx,dy,dz) from home: raised (farther -> whole board fits) + wide, for full board coverage
SWEEP_OFFSETS = ([(dx, dy, 0.12) for dx in (-0.18, 0.0, 0.18) for dy in (-0.18, 0.0, 0.18)] +
                 [(dx, dy, 0.0) for dx in (-0.14, 0.14) for dy in (-0.14, 0.14)])
BOARD_GRAY_LO, BOARD_GRAY_HI = 35, 135    # dark-board segmentation on the downscaled image
BOARD_CKPT = "/home/skr/aic_data/boardyaw_run/best.pt"
R_BOARD_OFFSET = {   # R_board_port (port-in-board orientation), quat wxyz, measured (M16)
    "sfp": [0.00632, 0.99998, 0.0, 0.0],
    "sc":  [0.00028, -0.70682, 0.70739, -0.00028],
}

# --- reactive layer (identical to PerceptionInsert) ---
F_STOP = 8.0
Z_SEATED = -0.004
RETRACT = 0.004
SPIRAL_PITCH = 0.0012
SPIRAL_MAX = 30
GOLDEN = 2.399963
LOG_PATH = "/home/skr/aic_data/kp_pose_log.jsonl"
LOG_EVERY = 5
# --- red-dot eval video: dump detector-input frames with predicted (red) + GT (green) port dots ---
DUMP = False
DUMP_DIR = "/home/skr/aic_data/kp_frames"
DUMP_EVERY = 3
DITHER_SEQ = [0.0, 0.05, -0.05, 0.10, -0.10]
ENVELOPE_MIN = np.array([-0.80, -0.15, -0.05])
ENVELOPE_MAX = np.array([-0.10, 0.60, 0.35])
JUMP_REJECT = 0.08


# ----- DualPoseNet (unchanged; supplies port orientation + plug pose) -----
class DualPoseNet(nn.Module):
    def __init__(self, n_targets=16, emb_dim=16):
        super().__init__()
        bb = tv.models.resnet18(weights=None)
        fd = bb.fc.in_features
        bb.fc = nn.Identity()
        self.backbone = bb
        self.target_emb = nn.Embedding(n_targets, emb_dim)
        self.trunk = nn.Sequential(nn.Linear(fd * 3 + emb_dim, 512), nn.ReLU(True), nn.Dropout(0.2),
                                   nn.Linear(512, 256), nn.ReLU(True))
        self.port_pos = nn.Linear(256, 3); self.port_quat = nn.Linear(256, 4)
        self.plug_pos = nn.Linear(256, 3); self.plug_quat = nn.Linear(256, 4)

    def forward(self, x, tid):
        B, C = x.shape[:2]
        f = self.backbone(x.flatten(0, 1)).view(B, C, -1).flatten(1)
        h = self.trunk(torch.cat([f, self.target_emb(tid)], dim=1))
        return (self.port_pos(h), F.normalize(self.port_quat(h), dim=-1),
                self.plug_pos(h), F.normalize(self.plug_quat(h), dim=-1))


# ----- KPNet (our keypoint detector; supplies port position) — must match collect/kp_train.py -----
class KPNet(nn.Module):
    def __init__(self):
        super().__init__()
        bb = tv.models.resnet18(weights=None)
        self.enc = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool,
                                 bb.layer1, bb.layer2, bb.layer3, bb.layer4)

        def up(ci, co):
            return nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                 nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(True))
        self.dec = nn.Sequential(up(512, 256), up(256, 128), up(128, 64))
        self.head = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        hm = self.head(self.dec(self.enc(x)))
        B, _, H, W = hm.shape
        p = F.softmax(hm.flatten(2), dim=2).view(B, 1, H, W)
        gy = torch.linspace(0, H - 1, H, device=hm.device).view(1, 1, H, 1)
        gx = torch.linspace(0, W - 1, W, device=hm.device).view(1, 1, 1, W)
        uy = (p * gy).sum((2, 3)); ux = (p * gx).sum((2, 3))
        return torch.cat([ux, uy], 1) * KP_DOWN


# ----- BoardNet (Stage 2b-orient): 3 cams -> board orientation in center-cam frame (quat) -----
class BoardNet(nn.Module):
    def __init__(self):
        super().__init__()
        bb = tv.models.resnet18(weights=None); fd = bb.fc.in_features; bb.fc = nn.Identity()
        self.bb = bb
        self.trunk = nn.Sequential(nn.Linear(fd * 3, 512), nn.ReLU(True), nn.Dropout(0.2),
                                   nn.Linear(512, 256), nn.ReLU(True), nn.Linear(256, 4))

    def forward(self, x):  # x: (B,3,3,H,W)
        B, C = x.shape[:2]
        f = self.bb(x.flatten(0, 1)).view(B, C, -1).flatten(1)
        return F.normalize(self.trunk(f), dim=-1)


def _R(q):  # wxyz -> 3x3
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _T(p, q):
    M = np.eye(4); M[:3, :3] = _R(q); M[:3, 3] = p; return M


def _quat(Rm):  # 3x3 -> wxyz
    tr = np.trace(Rm)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (Rm[2, 1] - Rm[1, 2]) / s, (Rm[0, 2] - Rm[2, 0]) / s, (Rm[1, 0] - Rm[0, 1]) / s
    elif Rm[0, 0] > Rm[1, 1] and Rm[0, 0] > Rm[2, 2]:
        s = np.sqrt(1 + Rm[0, 0] - Rm[1, 1] - Rm[2, 2]) * 2
        w, x, y, z = (Rm[2, 1] - Rm[1, 2]) / s, 0.25 * s, (Rm[0, 1] + Rm[1, 0]) / s, (Rm[0, 2] + Rm[2, 0]) / s
    elif Rm[1, 1] > Rm[2, 2]:
        s = np.sqrt(1 + Rm[1, 1] - Rm[0, 0] - Rm[2, 2]) * 2
        w, x, y, z = (Rm[0, 2] - Rm[2, 0]) / s, (Rm[0, 1] + Rm[1, 0]) / s, 0.25 * s, (Rm[1, 2] + Rm[2, 1]) / s
    else:
        s = np.sqrt(1 + Rm[2, 2] - Rm[0, 0] - Rm[1, 1]) * 2
        w, x, y, z = (Rm[1, 0] - Rm[0, 1]) / s, (Rm[0, 2] + Rm[2, 0]) / s, (Rm[1, 2] + Rm[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z]); return q / (np.linalg.norm(q) + 1e-9)


def _triangulate(origins, dirs):
    A = np.zeros((3, 3)); b = np.zeros(3)
    for o, d in zip(origins, dirs):
        d = d / np.linalg.norm(d); P = np.eye(3) - np.outer(d, d)
        A += P; b += P @ o
    return np.linalg.solve(A, b)


class PerceptionInsertYOLO(Policy):
    def __init__(self, parent_node):
        self._tip_x_error_integrator = 0.0
        self._tip_y_error_integrator = 0.0
        self._max_integrator_windup = 0.05
        self._task = None
        self._port_hist = []
        self._trial_idx = 0
        self._fc = 0
        self._plug_type = "sfp"
        super().__init__(parent_node)

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        # DualPoseNet (port orientation + plug)
        ck = torch.load(DUAL_CKPT, map_location="cpu", weights_only=False)
        self._vocab = ck.get("vocab", {"<unk>": 0})
        self._img_size = int(ck.get("img_size", IMG_SIZE))
        self._keep_aspect = bool(ck.get("keep_aspect", KEEP_ASPECT))
        self._net = DualPoseNet(n_targets=len(self._vocab)).to(self._device).eval()
        self._net.load_state_dict(ck["model"])
        self._pm = np.asarray(ck["port_mean"], np.float32); self._ps = np.asarray(ck["port_std"], np.float32)
        self._gm = np.asarray(ck["plug_mean"], np.float32); self._gs = np.asarray(ck["plug_std"], np.float32)
        self._tid = torch.zeros(1, dtype=torch.long, device=self._device)
        # KPNet (port position — our eyes)
        kck = torch.load(KP_CKPT, map_location="cpu", weights_only=False)
        self._kp = KPNet().to(self._device).eval()
        self._kp.load_state_dict(kck["model"])
        # BoardNet (port orientation via board yaw)
        self._board = None
        if ORIENT_FROM_BOARD:
            bck = torch.load(BOARD_CKPT, map_location="cpu", weights_only=False)
            self._board = BoardNet().to(self._device).eval()
            self._board.load_state_dict(bck["model"])
            self.get_logger().info(f"BoardNet loaded {BOARD_CKPT} (val geo {bck.get('metrics',{}).get('med','?')} deg)")
        from ultralytics import YOLO
        self._yolo = YOLO(YOLO_WEIGHTS)
        self.get_logger().info(f"YOLO port detector loaded {YOLO_WEIGHTS}")
        self.get_logger().info(
            f"PerceptionInsertYOLO: DualPoseNet {DUAL_CKPT} (img {self._img_size}, vocab {len(self._vocab)}) "
            f"+ KPNet {KP_CKPT} (val px {kck.get('metrics', {}).get('px_med', '?')}) on {self._device}")

    # ---------- image preprocessing ----------
    def _raw_rgb(self, raw):
        return np.frombuffer(raw.data, dtype=np.uint8).reshape(raw.height, raw.width, 3)  # rgb8 HWC

    def _dual_tensor(self, raw):
        t = torch.from_numpy(self._raw_rgb(raw).copy()).permute(2, 0, 1).float().div(255.0)
        S = self._img_size
        if self._keep_aspect:
            t = VF.center_crop(VF.resize(t, S, antialias=True), [S, S])
        else:
            t = VF.resize(t, [S, S], antialias=True)
        return VF.normalize(t, IMAGENET_MEAN, IMAGENET_STD)

    def _kp_tensor(self, raw):
        t = torch.from_numpy(self._raw_rgb(raw).copy()).permute(2, 0, 1).float().div(255.0)
        t = VF.resize(t, KP_SHORT, antialias=True)          # short side -> 256 (== training cache)
        return VF.normalize(t, IMAGENET_MEAN, IMAGENET_STD)

    def _T_base_frame(self, frame):
        tf = self._parent_node._tf_buffer.lookup_transform("base_link", frame, Time())
        tr, ro = tf.transform.translation, tf.transform.rotation
        return _T([tr.x, tr.y, tr.z], [ro.w, ro.x, ro.y, ro.z])

    def _dump_frame(self, raw, u, v, cam, Tbc):
        """Save the detector-input frame (288x256) with predicted keypoint (red) and, if GT is
        published (ground_truth:=true), the GT port projected into this camera (green)."""
        try:
            import os
            from PIL import Image, ImageDraw
            native = self._raw_rgb(raw)
            H, W = native.shape[:2]
            s = KP_SHORT / min(H, W)
            im = Image.fromarray(native).resize((round(W * s), round(H * s)), Image.BILINEAR)
            dr = ImageDraw.Draw(im); r = 4
            dr.ellipse([u - r, v - r, u + r, v + r], outline=(255, 0, 0), width=2)   # predicted = red
            gt = self._gt_pose(f"task_board/{self._task.target_module_name}/{self._task.port_name}_link")
            if gt is not None:
                pc = np.linalg.inv(Tbc) @ np.array([gt[0], gt[1], gt[2], 1.0])
                if pc[2] > 0:
                    ug = FX * pc[0] / pc[2] + CX; vg = FY * pc[1] / pc[2] + CY
                    dr.ellipse([ug - r, vg - r, ug + r, vg + r], outline=(0, 255, 0), width=2)  # GT = green
            d = f"{DUMP_DIR}/t{self._trial_idx}"; os.makedirs(d, exist_ok=True)
            im.save(f"{d}/{self._fc:05d}_{CAMS.index(cam)}_{cam}.png")
        except Exception as ex:
            self.get_logger().warn(f"dump failed: {ex}")

    # ---------- perception ----------
    def _yolo_uv(self, obs):
        """Per-cam port pixel from the fine-tuned YOLO (native res) -> 288-scale. Geometry-select:
        with a running lock, pick the box whose back-projected ray passes nearest the estimate
        (rejects wrong-module boxes = the t1 fix); else target-class highest-conf. NaN row = no det."""
        import cv2
        raws = {"left": obs.left_image, "center": obs.center_image, "right": obs.right_image}
        want = YOLO_CLASS.get(self._plug_type)
        expected = (np.median(np.stack(self._port_hist), axis=0) if self._port_hist else None)
        uv = np.full((3, 2), np.nan)
        for i, c in enumerate(CAMS):
            try:
                Tbc = self._T_base_frame(f"{c}_camera/optical")
            except TransformException:
                continue
            img = cv2.cvtColor(self._raw_rgb(raws[c]), cv2.COLOR_RGB2BGR)
            res = self._yolo.predict(img, conf=YOLO_CONF, verbose=False, device=self._device)[0]
            if len(res.boxes) == 0:
                continue
            conf = res.boxes.conf.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            xywh = res.boxes.xywh.cpu().numpy()
            if expected is not None:
                o = Tbc[:3, 3]; best, bestd = 0, 1e9
                for j in range(len(xywh)):
                    d = Tbc[:3, :3] @ np.array([(xywh[j, 0]*KP_SCALE - CX)/FX,
                                                (xywh[j, 1]*KP_SCALE - CY)/FY, 1.0])
                    d = d/(np.linalg.norm(d)+1e-9); w = expected - o
                    dd = np.linalg.norm(w - np.dot(w, d)*d)
                    if dd < bestd:
                        bestd, best = dd, j
                j = best
            else:
                idx = np.where(cls == want)[0]
                j = int(idx[np.argmax(conf[idx])]) if len(idx) else int(np.argmax(conf))
            uv[i] = [xywh[j, 0]*KP_SCALE, xywh[j, 1]*KP_SCALE]
        return uv

    def _perceive(self, get_observation):
        obs = get_observation()

        # --- DualPoseNet: port orientation + plug pose (base_link via center-cam TF) ---
        x = torch.stack([self._dual_tensor(obs.left_image),
                         self._dual_tensor(obs.center_image),
                         self._dual_tensor(obs.right_image)], 0).unsqueeze(0).to(self._device)
        with torch.no_grad():
            pp, pquat, gp, gquat = self._net(x, self._tid)
        port_pos_dual = pp[0].cpu().numpy() * self._ps + self._pm    # DualPoseNet port (for #5 compare)
        port_q = pquat[0].cpu().numpy()
        Tbo = self._T_base_frame("center_camera/optical")
        port_R_b = (Tbo @ _T(np.zeros(3), port_q))[:3, :3]     # port orientation in base (DualPoseNet fallback)
        byaw = getattr(self, "_board_yaw", None)                       # Stage 2b: prefer the classical sweep yaw
        if byaw is None and self._board is not None:                   # learned BoardNet fallback
            bt = torch.stack([self._kp_tensor(obs.left_image), self._kp_tensor(obs.center_image),
                              self._kp_tensor(obs.right_image)], 0).unsqueeze(0).to(self._device)
            with torch.no_grad():
                qcb = self._board(bt)[0].float().cpu().numpy()
            R_base_board = Tbo[:3, :3] @ _R(qcb)
            byaw = math.atan2(R_base_board[1, 0], R_base_board[0, 0])
        if byaw is not None:
            cz, sz = math.cos(byaw), math.sin(byaw)
            R_board_flat = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])  # pure yaw (board flat)
            port_R_b = R_board_flat @ _R(R_BOARD_OFFSET[self._plug_type])   # port_orient = board_yaw ⊗ offset[type]
            self._board_yaw_pred = math.degrees(byaw)
        port_dual_b = Tbo @ _T(port_pos_dual, port_q)          # DualPoseNet port position in base
        # M14: plug pose from FK(gripper) . T_grasp[type] — exact, no vision (replaces DualPoseNet plug)
        gt_t, gt_q = T_GRASP[self._plug_type]
        plug_b = self._T_base_frame("gripper/tcp") @ _T(np.array(gt_t, float), np.array(gt_q, float))

        # --- KPNet + triangulation: port position (our eyes), base_link ---
        uv = self._yolo_uv(obs)      # (3,2) port pixels in 288-scale; NaN row = no detection this cam
        raws = {"left": obs.left_image, "center": obs.center_image, "right": obs.right_image}
        do_dump = DUMP and (self._fc % DUMP_EVERY == 0)
        gt_base = (self._gt_pose(f"task_board/{self._task.target_module_name}/{self._task.port_name}_link")
                   if self._task else None)   # only present at gt:=true
        origins, dirs, px_gt = [], [], []
        for c, (u, v) in zip(CAMS, uv):
            if np.isnan(u):
                px_gt.append(None); continue
            try:
                Tbc = self._T_base_frame(f"{c}_camera/optical")
            except TransformException:
                px_gt.append(None); continue
            d = Tbc[:3, :3] @ np.array([(u - CX) / FX, (v - CY) / FY, 1.0])
            origins.append(Tbc[:3, 3]); dirs.append(d)
            if do_dump:
                self._dump_frame(raws[c], u, v, c, Tbc)
            if gt_base is not None:                      # project the GT port into this camera
                pc = np.linalg.inv(Tbc) @ np.array([gt_base[0], gt_base[1], gt_base[2], 1.0])
                px_gt.append([round(float(FX*pc[0]/pc[2]+CX), 1), round(float(FY*pc[1]/pc[2]+CY), 1),
                              round(float(pc[2]), 3)] if pc[2] > 0 else "behind")
            else:
                px_gt.append(None)
        self._fc += 1
        if len(origins) < 2:
            if self._port_hist:
                port_kp = np.median(np.stack(self._port_hist), axis=0)   # YOLO missed -> HOLD the lock (A5 extrapolate)
            else:
                raise TransformException("insufficient camera TFs for triangulation")
        else:
            port_kp = _triangulate(origins, dirs)

        # sanity gate + running median on the (static) port position — identical policy to PerceptionInsert
        gated = ""
        if not (np.all(port_kp >= ENVELOPE_MIN) and np.all(port_kp <= ENVELOPE_MAX)):
            gated = "envelope"
        elif len(self._port_hist) >= 5 and np.linalg.norm(
                port_kp - np.median(np.stack(self._port_hist), axis=0)) > JUMP_REJECT:
            gated = "jump"
        if gated:
            self._gate_count += 1
            if self._gate_count % 20 == 1:
                self.get_logger().warn(f"sanity gate ({gated}): raw port {np.round(port_kp, 3)} rejected "
                                       f"({self._gate_count} total)")
        else:
            self._port_hist.append(port_kp)
            if len(self._port_hist) > PORT_FILTER_N:
                self._port_hist.pop(0)
        if not self._port_hist:
            raise TransformException("no valid port estimate yet (all gated)")
        port_xyz = np.median(np.stack(self._port_hist), axis=0)

        port_tf = Transform()
        port_tf.translation.x, port_tf.translation.y, port_tf.translation.z = map(float, port_xyz)
        pw = _quat(port_R_b)
        port_tf.rotation.w, port_tf.rotation.x, port_tf.rotation.y, port_tf.rotation.z = map(float, pw)
        if USE_GT_PORT:   # diagnostic override: perfect port pose (position + orientation) from GT
            g = self._gt_transform(f"task_board/{self._task.target_module_name}/{self._task.port_name}_link")
            if g:
                (port_tf.translation.x, port_tf.translation.y, port_tf.translation.z,
                 port_tf.rotation.w, port_tf.rotation.x, port_tf.rotation.y, port_tf.rotation.z) = g
        elif USE_GT_PORT_POS:   # diagnostic: GT position only, keep the board-yaw orientation (isolate orient fix)
            g = self._gt_transform(f"task_board/{self._task.target_module_name}/{self._task.port_name}_link")
            if g:
                port_tf.translation.x, port_tf.translation.y, port_tf.translation.z = g[0], g[1], g[2]

        plug_tf = Transform()
        plug_tf.translation.x, plug_tf.translation.y, plug_tf.translation.z = map(float, plug_b[:3, 3])
        gw = _quat(plug_b[:3, :3])
        plug_tf.rotation.w, plug_tf.rotation.x, plug_tf.rotation.y, plug_tf.rotation.z = map(float, gw)

        f = obs.wrist_wrench.wrench.force
        info = {
            "force": float(np.linalg.norm([f.x, f.y, f.z])),
            "port_kp_raw": [float(v) for v in port_kp],     # keypoint eyes, unfiltered (base_link)
            "port_dual": [float(v) for v in port_dual_b[:3, 3]],   # DualPoseNet port, same frame (#5)
            "port_filt": [float(v) for v in port_xyz],
            "plug": [float(v) for v in plug_b[:3, 3]],
            "n_views": len(origins),
            "gated": gated,
            "board_yaw_pred": getattr(self, "_board_yaw_pred", None),   # live board-yaw prediction (deg)
            "px": [[round(float(u), 1), round(float(v), 1)] for u, v in uv],   # per-cam L/C/R predicted pixel
            "px_gt": px_gt,                                                    # per-cam GT port pixel [u,v,Z] / "behind" / None
            "imhw": [int(obs.left_image.height), int(obs.left_image.width)],   # live image dims (detector: 288x256)
        }
        return port_tf, plug_tf, info

    # ---------- diagnostics ----------
    def _gt_pose(self, frame):
        try:
            tf = self._parent_node._tf_buffer.lookup_transform("base_link", frame, Time())
            t = tf.transform.translation
            return [t.x, t.y, t.z]
        except TransformException:
            return None

    def _gt_transform(self, frame):
        """Full pose [x,y,z, qw,qx,qy,qz] in base_link. gripper/tcp is FK (always); GT frames need gt:=true."""
        try:
            tf = self._parent_node._tf_buffer.lookup_transform("base_link", frame, Time())
            t = tf.transform.translation; r = tf.transform.rotation
            return [t.x, t.y, t.z, r.w, r.x, r.y, r.z]
        except TransformException:
            return None

    def _sweep_board(self, get_observation, move_robot):
        """Home raster sweep: segment dark board in the 3 cams, back-project to z=0 via FK, accumulate a
        top-down footprint of the (static) board. Returns Nx2 base_link ground points."""
        import cv2
        T_home = self._T_base_frame("gripper/tcp"); q_home = _quat(T_home[:3, :3]); t_home = T_home[:3, 3]
        pts = []
        for dx, dy, dz in SWEEP_OFFSETS:
            pose = Pose(position=Point(x=float(t_home[0] + dx), y=float(t_home[1] + dy), z=float(t_home[2] + dz)),
                        orientation=Quaternion(w=float(q_home[0]), x=float(q_home[1]),
                                               y=float(q_home[2]), z=float(q_home[3])))
            self.set_pose_target(move_robot=move_robot, pose=pose)
            for _ in range(30):                       # settle fully (FK must match the image)
                self.sleep_for(0.05)
            obs = get_observation()
            raws = {"left": obs.left_image, "center": obs.center_image, "right": obs.right_image}
            for c in CAMS:
                try:
                    Tbc = self._T_base_frame(f"{c}_camera/optical")
                except TransformException:
                    continue
                rgb = self._raw_rgb(raws[c])
                small = cv2.resize(rgb, None, fx=KP_SCALE, fy=KP_SCALE)   # ~288x256, matches FX/CX
                mask = cv2.inRange(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), BOARD_GRAY_LO, BOARD_GRAY_HI)
                ys, xs = np.where(mask > 0)
                if len(xs) == 0:
                    continue
                idx = np.linspace(0, len(xs) - 1, min(300, len(xs))).astype(int)
                R, tt = Tbc[:3, :3], Tbc[:3, 3]
                for u, v in zip(xs[idx], ys[idx]):
                    d = R @ np.array([(u - CX) / FX, (v - CY) / FY, 1.0])
                    if abs(d[2]) < 1e-6:
                        continue
                    s = -tt[2] / d[2]
                    if s <= 0:
                        continue
                    p = tt + s * d
                    if -1 < p[0] < 1 and -1 < p[1] < 1:
                        pts.append([float(p[0]), float(p[1])])
        pts = np.array(pts) if pts else np.zeros((0, 2))
        # fit board yaw from the footprint (density-filter to the core -> minAreaRect long edge)
        self._board_yaw = None; self._board_center = None
        if len(pts) >= 50:
            cell = 0.02; q = np.floor(pts / cell).astype(int); keys = {}
            for i, k in enumerate(map(tuple, q)):
                keys.setdefault(k, []).append(i)
            thr = max(len(v) for v in keys.values()) * 0.25
            core = pts[[i for v in keys.values() if len(v) >= thr for i in v]]
            rect = cv2.minAreaRect((core * 1000).astype(np.float32)); (cx, cy), (w, h), ang = rect
            long_dir = ang if w >= h else ang + 90
            y = math.radians(((long_dir + 90) % 180) - 90)                 # mod-90 board yaw (rad)
            self._board_center = np.array([cx / 1000.0, cy / 1000.0])
            cands = [y, y + math.pi / 2, y + math.pi, y - math.pi / 2]
            gtf = self._gt_transform("task_board")
            if DISAMBIG_GT and gtf:                                        # TEST: pick candidate nearest GT
                gy = math.atan2(_R(gtf[3:])[1, 0], _R(gtf[3:])[0, 0])
                y = min(cands, key=lambda c: abs(((c - gy + math.pi) % (2 * math.pi)) - math.pi))
            self._board_yaw = y
            self.get_logger().info(f"board sweep yaw {math.degrees(y):.1f} deg, center {np.round(self._board_center,3)} "
                                   f"(rect {w:.0f}x{h:.0f}mm, {len(core)} core pts)")
        return pts

    def _log(self, phase, step, info, z_offset, spiral_k=0):
        if step % LOG_EVERY:
            return
        try:
            entry = {"trial": self._trial_idx, "task": self._task.id if self._task else "?",
                     "phase": phase, "step": step,
                     "z_offset": round(z_offset, 4), "spiral_k": spiral_k, **info}
            pf = f"task_board/{self._task.target_module_name}/{self._task.port_name}_link"
            gf = f"{self._task.cable_name}/{self._task.plug_name}_link"
            entry["port_gt"] = self._gt_pose(pf)
            entry["plug_gt"] = self._gt_pose(gf)
            # full transforms for grasp-offset analysis (T_grasp = inv(gripper) @ plug)
            entry["gripper_tf"] = self._gt_transform("gripper/tcp")   # FK, always
            entry["plug_gt_tf"] = self._gt_transform(gf)              # GT plug pose
            entry["port_gt_tf"] = self._gt_transform(pf)              # GT port pose
            for bframe in ("task_board", "task_board/base_link", f"task_board/{self._task.target_module_name}"):
                bt = self._gt_transform(bframe)
                if bt: entry["board_tf"] = bt; entry["board_frame"] = bframe; break
            with open(LOG_PATH, "a") as fp:
                fp.write(json.dumps(entry) + "\n")
        except Exception as ex:
            self.get_logger().warn(f"pose-log failed: {ex}")

    # ---------- control (ported from CheatCode; plug from net, port position from keypoints) ----------
    def calc_gripper_pose(self, port_transform, plug_transform, slerp_fraction=1.0,
                          position_fraction=1.0, z_offset=0.1, reset_xy_integrator=False):
        q_port = (port_transform.rotation.w, port_transform.rotation.x,
                  port_transform.rotation.y, port_transform.rotation.z)
        q_plug = (plug_transform.rotation.w, plug_transform.rotation.x,
                  plug_transform.rotation.y, plug_transform.rotation.z)
        q_plug_inv = (-q_plug[0], q_plug[1], q_plug[2], q_plug[3])
        q_diff = quaternion_multiply(q_port, q_plug_inv)

        gripper = self._parent_node._tf_buffer.lookup_transform("base_link", "gripper/tcp", Time())
        q_gripper = (gripper.transform.rotation.w, gripper.transform.rotation.x,
                     gripper.transform.rotation.y, gripper.transform.rotation.z)
        q_gripper_target = quaternion_multiply(q_diff, q_gripper)
        q_gripper_slerp = quaternion_slerp(q_gripper, q_gripper_target, slerp_fraction)

        gripper_xyz = (gripper.transform.translation.x, gripper.transform.translation.y,
                       gripper.transform.translation.z)
        port_xy = (port_transform.translation.x, port_transform.translation.y)
        plug_xyz = (plug_transform.translation.x, plug_transform.translation.y, plug_transform.translation.z)
        plug_tip_gripper_offset_z = gripper_xyz[2] - plug_xyz[2]

        tip_x_error = port_xy[0] - plug_xyz[0]
        tip_y_error = port_xy[1] - plug_xyz[1]
        if reset_xy_integrator:
            self._tip_x_error_integrator = 0.0
            self._tip_y_error_integrator = 0.0
        else:
            self._tip_x_error_integrator = np.clip(self._tip_x_error_integrator + tip_x_error,
                                                   -self._max_integrator_windup, self._max_integrator_windup)
            self._tip_y_error_integrator = np.clip(self._tip_y_error_integrator + tip_y_error,
                                                   -self._max_integrator_windup, self._max_integrator_windup)
        i_gain = 0.15
        target_x = port_xy[0] + i_gain * self._tip_x_error_integrator
        target_y = port_xy[1] + i_gain * self._tip_y_error_integrator
        target_z = port_transform.translation.z + z_offset - plug_tip_gripper_offset_z
        blend = (position_fraction * target_x + (1.0 - position_fraction) * gripper_xyz[0],
                 position_fraction * target_y + (1.0 - position_fraction) * gripper_xyz[1],
                 position_fraction * target_z + (1.0 - position_fraction) * gripper_xyz[2])
        return Pose(position=Point(x=blend[0], y=blend[1], z=blend[2]),
                    orientation=Quaternion(w=q_gripper_slerp[0], x=q_gripper_slerp[1],
                                           y=q_gripper_slerp[2], z=q_gripper_slerp[3]))

    def insert_cable(self, task: Task, get_observation: GetObservationCallback,
                     move_robot: MoveRobotCallback, send_feedback: SendFeedbackCallback):
        self.get_logger().info(f"PerceptionInsertKP.insert_cable() task: {task}")
        self._task = task
        self._trial_idx += 1
        # M14: connector type (free from Task) selects the fixed grasp offset for the FK plug
        self._plug_type = (getattr(task, "plug_type", "") or
                           ("sc" if "sc" in task.port_name.lower() else "sfp")).lower()
        if self._plug_type not in T_GRASP:
            self._plug_type = "sc" if "sc" in task.port_name.lower() else "sfp"
        self.get_logger().info(f"plug_type={self._plug_type} -> FK grasp offset {T_GRASP[self._plug_type][0]}")
        key = f"{task.target_module_name}|{task.port_name}"
        tid = self._vocab.get(key, 0)
        if tid == 0:
            self.get_logger().warn(f"target '{key}' not in training vocab -> <unk> conditioning")
        self._tid = torch.tensor([tid], dtype=torch.long, device=self._device)
        self.get_logger().info(f"conditioning on target '{key}' (id {tid})")
        self._port_hist = []
        self._gate_count = 0
        self._tip_x_error_integrator = 0.0
        self._tip_y_error_integrator = 0.0

        if SWEEP_BOARD:
            self.get_logger().info("board sweep: rastering wrist to map the board footprint...")
            pts = self._sweep_board(get_observation, move_robot)
            bt = self._gt_transform("task_board")
            try:
                np.save(f"/home/skr/aic_data/sweep_pts_trial{self._trial_idx}.npy",
                        {"pts": pts, "board_tf": bt, "type": self._plug_type}, allow_pickle=True)
            except Exception as ex:
                self.get_logger().warn(f"sweep save failed: {ex}")
            self.get_logger().info(f"board sweep: {len(pts)} ground pts (GT board_tf={bt})")
            if SWEEP_ONLY:
                return True

        force_samples = []
        for i in range(PORT_FILTER_N):
            try:
                _, _, info = self._perceive(get_observation)
                force_samples.append(info["force"])
                self._log("warmup", i, info, 0.2)
            except TransformException as ex:
                self.get_logger().warn(f"TF not ready during warmup: {ex}")
            self.sleep_for(0.05)

        z_offset = 0.2
        for t in range(0, 100):
            frac = t / 100.0
            try:
                port_tf, plug_tf, info = self._perceive(get_observation)
                self._log("approach", t, info, z_offset)
                self.set_pose_target(move_robot=move_robot,
                                     pose=self.calc_gripper_pose(port_tf, plug_tf, slerp_fraction=frac,
                                                                 position_fraction=frac, z_offset=z_offset,
                                                                 reset_xy_integrator=True))
            except TransformException as ex:
                self.get_logger().warn(f"perceive/TF failed (approach): {ex}")
            self.sleep_for(0.05)

        try:
            _, _, info0 = self._perceive(get_observation)
            f_base = info0["force"]
        except TransformException:
            f_base = float(np.median(force_samples)) if force_samples else 20.0
        self.get_logger().info(f"descent: force baseline {f_base:.1f} N")

        spiral_k = 0
        sx = sy = 0.0
        step = 0
        while z_offset >= -0.015:
            z_offset -= 0.0005
            step += 1
            try:
                port_tf, plug_tf, info = self._perceive(get_observation)
                self._log("descent", step, info, z_offset, spiral_k)
                d_force = info["force"] - f_base

                if d_force > F_STOP:
                    if z_offset <= Z_SEATED:
                        self.get_logger().info(
                            f"contact at z_offset {z_offset:.4f} (dF {d_force:.1f} N) -> seated, stopping")
                        break
                    if spiral_k >= SPIRAL_MAX:
                        self.get_logger().warn("spiral exhausted -> stopping (no ram)")
                        break
                    spiral_k += 1
                    r = SPIRAL_PITCH * spiral_k
                    a = GOLDEN * spiral_k
                    sx, sy = r * math.cos(a), r * math.sin(a)
                    z_offset = min(z_offset + RETRACT, 0.02)
                    self.get_logger().info(
                        f"contact early (dF {d_force:.1f} N) -> spiral {spiral_k} offset ({sx*1000:.1f},{sy*1000:.1f}) mm")

                port_tf.translation.x += sx
                port_tf.translation.y += sy
                dyaw = DITHER_SEQ[spiral_k % len(DITHER_SEQ)]
                if dyaw != 0.0:
                    qz = (math.cos(dyaw / 2), 0.0, 0.0, math.sin(dyaw / 2))
                    qp = (port_tf.rotation.w, port_tf.rotation.x, port_tf.rotation.y, port_tf.rotation.z)
                    qn = quaternion_multiply(qz, qp)
                    (port_tf.rotation.w, port_tf.rotation.x,
                     port_tf.rotation.y, port_tf.rotation.z) = (float(qn[0]), float(qn[1]),
                                                                float(qn[2]), float(qn[3]))
                self.set_pose_target(move_robot=move_robot,
                                     pose=self.calc_gripper_pose(port_tf, plug_tf, z_offset=z_offset))
            except TransformException as ex:
                self.get_logger().warn(f"perceive/TF failed (descent): {ex}")
            self.sleep_for(0.05)

        self.get_logger().info("Waiting for connector to stabilize...")
        self.sleep_for(5.0)
        self.get_logger().info("PerceptionInsertKP.insert_cable() exiting...")
        return True
