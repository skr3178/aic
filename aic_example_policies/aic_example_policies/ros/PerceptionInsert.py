#
#  PerceptionInsert — M4 policy (GT-free).
#
#  Same insertion controller as CheatCode, but the port and plug poses come from a trained
#  perception net (images -> pose in center_camera/optical) instead of ground-truth TF.
#  Pipeline per control step:
#     3 wrist images (Observation) -> DualPoseNet -> port & plug pose in center_camera/optical
#       -> base_link  (via robot TF base_link<-center_camera/optical, published WITHOUT ground_truth)
#       -> CheatCode alignment math -> compliant set_pose_target.
#  Port is static in the world -> temporally median-filtered. Plug flexes -> taken fresh each step.
#
#  Standalone: does NOT modify CheatCode. Reuses only the Policy base class.
#

import json
import math
import os

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

CKPT = "/home/skr/aic_data/m7_cond224/best.pt"
IMG_SIZE = 1024        # fallback only — overridden by the checkpoint's img_size/keep_aspect
KEEP_ASPECT = True
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
PORT_FILTER_N = 15   # running median window for the (static) port

# --- v2: force-stop + spiral search (the #3 reactive layer) ---
F_STOP = 8.0         # N above descent-start baseline -> treat as contact
Z_SEATED = -0.004    # z_offset below this at contact -> consider seated
RETRACT = 0.004      # m to back off after early contact
SPIRAL_PITCH = 0.0012  # m radius growth per spiral step
SPIRAL_MAX = 30      # max spiral steps before giving up (stop, don't ram)
GOLDEN = 2.399963    # golden angle (rad) -> uniform spiral coverage
LOG_PATH = "/home/skr/aic_data/pi_pose_log.jsonl"
LOG_EVERY = 5        # log every Nth control step

# --- v3: yaw dither + sanity gate ---
# keyed connectors (SFP/SC) refuse entry on yaw error even at 0 lateral miss (M6b forensics, ep028):
# cycle a small plug-yaw offset about world z alongside the spiral.
DITHER_SEQ = [0.0, 0.05, -0.05, 0.10, -0.10]   # rad (~0/3/6 deg), indexed by spiral_k
# sanity gate: never chase a hallucinated target (M5: OOD estimates were 300+ mm off)
ENVELOPE_MIN = np.array([-0.80, -0.15, -0.05])  # plausible port region, base_link (generous)
ENVELOPE_MAX = np.array([-0.10,  0.60,  0.35])
JUMP_REJECT = 0.08   # m: raw estimate deviating this far from the running median is an outlier


# ----- model (must match collect/train_perception_dual.py DualPoseNet, conditioned) -----
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


class PerceptionInsert(Policy):
    def __init__(self, parent_node):
        self._tip_x_error_integrator = 0.0
        self._tip_y_error_integrator = 0.0
        self._max_integrator_windup = 0.05
        self._task = None
        self._port_hist = []
        super().__init__(parent_node)

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        ck = torch.load(CKPT, map_location="cpu", weights_only=False)
        self._vocab = ck.get("vocab", {"<unk>": 0})
        self._img_size = int(ck.get("img_size", IMG_SIZE))
        self._keep_aspect = bool(ck.get("keep_aspect", KEEP_ASPECT))
        self._net = DualPoseNet(n_targets=len(self._vocab)).to(self._device).eval()
        self._net.load_state_dict(ck["model"])
        self._pm = np.asarray(ck["port_mean"], np.float32); self._ps = np.asarray(ck["port_std"], np.float32)
        self._gm = np.asarray(ck["plug_mean"], np.float32); self._gs = np.asarray(ck["plug_std"], np.float32)
        self._tid = torch.zeros(1, dtype=torch.long, device=self._device)   # set per task
        self.get_logger().info(f"PerceptionInsert: net loaded from {CKPT} on {self._device} "
                               f"(img {self._img_size}, keep_aspect={self._keep_aspect}, "
                               f"vocab {len(self._vocab)})")

    # ---------- perception ----------
    def _img_tensor(self, raw):
        arr = np.frombuffer(raw.data, dtype=np.uint8).reshape(raw.height, raw.width, 3)  # rgb8
        t = torch.from_numpy(arr.copy()).permute(2, 0, 1).float().div(255.0)
        S = self._img_size
        if self._keep_aspect:
            t = VF.center_crop(VF.resize(t, S, antialias=True), [S, S])
        else:
            t = VF.resize(t, [S, S], antialias=True)
        return VF.normalize(t, IMAGENET_MEAN, IMAGENET_STD)

    def _T_base_optical(self):
        tf = self._parent_node._tf_buffer.lookup_transform("base_link", "center_camera/optical", Time())
        tr, ro = tf.transform.translation, tf.transform.rotation
        return _T([tr.x, tr.y, tr.z], [ro.w, ro.x, ro.y, ro.z])

    def _perceive(self, get_observation):
        """Run the net on the current observation; return (port, plug) as base_link Transforms."""
        obs = get_observation()
        x = torch.stack([self._img_tensor(obs.left_image),
                         self._img_tensor(obs.center_image),
                         self._img_tensor(obs.right_image)], 0).unsqueeze(0).to(self._device)
        with torch.no_grad():
            pp, pquat, gp, gquat = self._net(x, self._tid)
        port_pos = pp[0].cpu().numpy() * self._ps + self._pm
        plug_pos = gp[0].cpu().numpy() * self._gs + self._gm
        port_q = pquat[0].cpu().numpy(); plug_q = gquat[0].cpu().numpy()

        Tbo = self._T_base_optical()
        port_b = Tbo @ _T(port_pos, port_q)
        plug_b = Tbo @ _T(plug_pos, plug_q)

        # sanity gate: reject estimates outside the plausible board envelope, or that jump
        # far from the running median (outliers never enter the filter -> we keep steering
        # at the last good estimate instead of chasing a hallucination)
        raw = port_b[:3, 3]
        gated = ""
        if not (np.all(raw >= ENVELOPE_MIN) and np.all(raw <= ENVELOPE_MAX)):
            gated = "envelope"
        elif len(self._port_hist) >= 5 and np.linalg.norm(
                raw - np.median(np.stack(self._port_hist), axis=0)) > JUMP_REJECT:
            gated = "jump"
        if gated:
            self._gate_count += 1
            if self._gate_count % 20 == 1:
                self.get_logger().warn(
                    f"sanity gate ({gated}): raw port {np.round(raw,3)} rejected "
                    f"({self._gate_count} total)")
        else:
            self._port_hist.append(raw)
            if len(self._port_hist) > PORT_FILTER_N:
                self._port_hist.pop(0)
        if not self._port_hist:            # nothing trustworthy yet -> signal caller to hold
            raise TransformException("no valid port estimate yet (all gated)")
        port_xyz = np.median(np.stack(self._port_hist), axis=0)

        port_tf = Transform()
        port_tf.translation.x, port_tf.translation.y, port_tf.translation.z = map(float, port_xyz)
        pw = _quat(port_b[:3, :3])
        port_tf.rotation.w, port_tf.rotation.x, port_tf.rotation.y, port_tf.rotation.z = map(float, pw)

        plug_tf = Transform()
        plug_tf.translation.x, plug_tf.translation.y, plug_tf.translation.z = map(float, plug_b[:3, 3])
        gw = _quat(plug_b[:3, :3])
        plug_tf.rotation.w, plug_tf.rotation.x, plug_tf.rotation.y, plug_tf.rotation.z = map(float, gw)

        f = obs.wrist_wrench.wrench.force
        info = {
            "force": float(np.linalg.norm([f.x, f.y, f.z])),
            "port_raw": [float(v) for v in port_b[:3, 3]],       # unfiltered net estimate (base_link)
            "port_filt": [float(v) for v in port_xyz],
            "plug": [float(v) for v in plug_b[:3, 3]],
            "gated": gated,
        }
        return port_tf, plug_tf, info

    # ---------- diagnostics ----------
    def _gt_pose(self, frame):
        """GT TF lookup (only exists when eval runs ground_truth:=true). None otherwise."""
        try:
            tf = self._parent_node._tf_buffer.lookup_transform("base_link", frame, Time())
            t = tf.transform.translation
            return [t.x, t.y, t.z]
        except TransformException:
            return None

    def _log(self, phase, step, info, z_offset, spiral_k=0):
        if step % LOG_EVERY:
            return
        try:
            entry = {"task": self._task.id if self._task else "?", "phase": phase, "step": step,
                     "z_offset": round(z_offset, 4), "spiral_k": spiral_k, **info}
            pf = f"task_board/{self._task.target_module_name}/{self._task.port_name}_link"
            gf = f"{self._task.cable_name}/{self._task.plug_name}_link"
            entry["port_gt"] = self._gt_pose(pf)
            entry["plug_gt"] = self._gt_pose(gf)
            with open(LOG_PATH, "a") as fp:
                fp.write(json.dumps(entry) + "\n")
        except Exception as ex:  # logging must never kill the policy
            self.get_logger().warn(f"pose-log failed: {ex}")

    # ---------- control (ported from CheatCode; plug comes from the net, not TF) ----------
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
        self.get_logger().info(f"PerceptionInsert.insert_cable() task: {task}")
        self._task = task
        # resolve the target identity for conditioning (same vocab as training; 0 = <unk>)
        key = f"{task.target_module_name}|{task.port_name}"
        tid = self._vocab.get(key, 0)
        if tid == 0:
            self.get_logger().warn(f"target '{key}' not in training vocab -> <unk> conditioning")
        self._tid = torch.tensor([tid], dtype=torch.long, device=self._device)
        self.get_logger().info(f"conditioning on target '{key}' (id {tid})")
        # fresh per-trial state (persists across the 3 trials otherwise)
        self._port_hist = []
        self._gate_count = 0
        self._tip_x_error_integrator = 0.0
        self._tip_y_error_integrator = 0.0

        # warm up / seed the port filter with a few estimates while stationary
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
        # 5 s smooth approach to a point above the port
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

        # ---- descent with force-stop + expanding spiral search (#3 reactive layer) ----
        # contact = force rising F_STOP above the baseline measured at descent start
        try:
            _, _, info0 = self._perceive(get_observation)
            f_base = info0["force"]
        except TransformException:
            f_base = float(np.median(force_samples)) if force_samples else 20.0
        self.get_logger().info(f"descent: force baseline {f_base:.1f} N")

        spiral_k = 0
        sx = sy = 0.0        # current spiral xy offset applied to the port target
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
                        # contact at insertion depth -> seated; stop pushing
                        self.get_logger().info(
                            f"contact at z_offset {z_offset:.4f} (dF {d_force:.1f} N) -> seated, stopping")
                        break
                    if spiral_k >= SPIRAL_MAX:
                        self.get_logger().warn("spiral exhausted -> stopping (no ram)")
                        break
                    # early contact -> retract, step the spiral, try a new xy
                    spiral_k += 1
                    r = SPIRAL_PITCH * spiral_k
                    a = GOLDEN * spiral_k
                    sx, sy = r * math.cos(a), r * math.sin(a)
                    z_offset = min(z_offset + RETRACT, 0.02)
                    self.get_logger().info(
                        f"contact early (dF {d_force:.1f} N) -> spiral {spiral_k} offset ({sx*1000:.1f},{sy*1000:.1f}) mm")

                port_tf.translation.x += sx
                port_tf.translation.y += sy
                # yaw dither: rotate the port target about world z so the keyed plug tries
                # slightly different yaws as the spiral advances (M6b ep028 failure class)
                dyaw = DITHER_SEQ[spiral_k % len(DITHER_SEQ)]
                if dyaw != 0.0:
                    qz = (math.cos(dyaw / 2), 0.0, 0.0, math.sin(dyaw / 2))   # wxyz, about world z
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
        self.get_logger().info("PerceptionInsert.insert_cable() exiting...")
        return True
