#
#  PerceptionInsertFull — STANDALONE. Everything that works, for BOTH port types.
#  Working files (PerceptionInsertSFP/SFPDrive/YOLO/Geo) are NOT modified.
#
#  PERCEPTION  (inherited from PerceptionInsertSCFix — proven: SFP 2mm, SC 2mm vs GT)
#     board pose (magenta-anchor + known-size yaw)                      SHARED
#       -> board frame -> ★ MODULE GATE (board-Y)                       SHARED IDEA
#            SFP: + split the 2 openings by board-X, estimate the rail slide, fix identity  (SFP-only:
#                   a NIC card has TWO openings 21.8mm apart)
#            SC : nothing more — ONE port per module. The gate alone took SC from 58mm -> 2mm
#                   (it was taking a median across ALL THREE SC modules; the gate drops 27/43 candidates).
#
#  MOTION      (inherited) SAFE-CLEAR joint-space IK-branch reset -> approach -> descend.
#
#  SEAT        the force-insertion fix, now applied to BOTH TYPES (SC was previously delegated to the
#              DEAD path and just parked on its cage face 2mm from the right hole):
#     * CONTACT  = |force - f_base| > F_ABS  OR  STALL.  The deployed test needed a force INCREASE, but
#                  landing on a cage face UNLOADS the wrist (21N -> 8N) -> it NEVER fired, for EITHER type.
#     * STALL    = re-arming watchdog (reference advances as the tip moves -> stays local).
#     * COMPLY   = [90,90,90] free descent -> [40,40,90] RCC on contact (lateral soft, axial firm).
#     * SEAT EXIT= measured tip DEPTH **and** LATERAL proximity to the port.  Depth alone gives false
#                  positives (V3 fired seated=True at a 30mm offset: the plug had gone low BESIDE the
#                  cage, not into it -- 3 of 6 claims were wrong).
#     * SPIRAL   = the ORIGINAL golden spiral, FULL reach (V4's 10mm cap was WRONG: it removed the
#                  sweeps that actually capture the plug -> 0 seats. Reverted.)
#     * per-type constants: ENTRANCE_H (SFP 45.8mm vs SC 15.6mm), CAD_Z, T_GRASP.
#
import json
import math
import os

import numpy as np
import torch
from tf2_ros import TransformException
from transforms3d._gohlketransforms import quaternion_multiply

from aic_example_policies.ros.PerceptionInsertYOLO import (
    T_GRASP, PORT_FILTER_N, SWEEP_BOARD, SWEEP_ONLY,
    SPIRAL_PITCH, SPIRAL_MAX, GOLDEN, DITHER_SEQ,
)
from aic_example_policies.ros.PerceptionInsertSCFix import PerceptionInsertSCFix

F_ABS = float(os.environ.get("AIC_F_ABS", 4.0))            # |dF| from baseline -> contact (sign-insensitive)
STALL_CMD = float(os.environ.get("AIC_STALL_CMD", 0.010))  # commanded progress since ref...
STALL_ACT = float(os.environ.get("AIC_STALL_ACT", 0.004))  # ...with less actual tip progress => blocked
LAT_STIFF = float(os.environ.get("AIC_LAT_STIFF", 40.0))
AX_STIFF = float(os.environ.get("AIC_AX_STIFF", 90.0))
STIFF_FREE = [90.0, 90.0, 90.0, 50.0, 50.0, 50.0]

SEAT_TIP = 0.005          # tip within 5mm of (or below) the port datum   -- DEPTH gate
SEAT_LAT = 0.006          # ...AND the plug's ACTUAL xy within 6mm of the port -- LATERAL gate
ENTRANCE_H = {"sfp": 0.0458, "sc": 0.0156}   # CAD funnel depth (GT-free). SC's is 3x shallower.
RETRACT = 0.004
CLEAR_WAIT = 12
COOLDOWN = 20             # after a probe, IGNORE contact for this many steps so the plug can RE-DESCEND.
                          # THE BUG THIS FIXES: a plug RESTING on the cage face gives a PERSISTENT unload
                          # (dF stuck at -5..-8N), so |dF|>F_ABS is true on EVERY step -> every step
                          # retracted z by 4mm while the loop only descends 0.5mm -> z PINNED at the 0.02
                          # cap for 30 straight probes. SC hovered at 18.4mm above its port, 0.2mm laterally
                          # centred on the right hole, and was never pushed in. Same class of bug as the
                          # original: a trigger reading a PERSISTENT STATE as a REPEATING EVENT.
Z_FLOOR = -0.015
STEP = 0.0005
DT = 0.05
LOG = "/home/skr/aic_data/full_log.jsonl"


class PerceptionInsertFull(PerceptionInsertSCFix):

    def _stiff_contact(self):
        return [LAT_STIFF, LAT_STIFF, AX_STIFF, 50.0, 50.0, 50.0]

    def _rec(self, **kw):
        try:
            with open(LOG, "a") as f:
                f.write(json.dumps(kw) + "\n")
        except Exception:
            pass

    def _dither(self, port_tf, dyaw):
        if dyaw != 0.0:
            qz = (math.cos(dyaw / 2), 0.0, 0.0, math.sin(dyaw / 2))
            qp = (port_tf.rotation.w, port_tf.rotation.x, port_tf.rotation.y, port_tf.rotation.z)
            qn = quaternion_multiply(qz, qp)
            (port_tf.rotation.w, port_tf.rotation.x,
             port_tf.rotation.y, port_tf.rotation.z) = (float(qn[0]), float(qn[1]), float(qn[2]), float(qn[3]))
        return port_tf

    def insert_cable(self, task, get_observation, move_robot, send_feedback):
        typ = (getattr(task, "plug_type", "") or ("sc" if "sc" in task.port_name.lower() else "sfp")).lower()
        self.get_logger().info(f"FULL [{typ.upper()}] F_ABS {F_ABS} · RCC {[LAT_STIFF, LAT_STIFF, AX_STIFF]} "
                               f"· seat gate {SEAT_TIP*1000:.0f}mm deep / {SEAT_LAT*1000:.0f}mm lateral · task: {task}")
        self._task = task; self._trial_idx += 1
        self._plug_type = typ if typ in T_GRASP else ("sc" if "sc" in task.port_name.lower() else "sfp")
        self._tid = torch.tensor([self._vocab.get(f"{task.target_module_name}|{task.port_name}", 0)],
                                 dtype=torch.long, device=self._device)
        self._port_hist = []; self._gate_count = 0
        self._tip_x_error_integrator = 0.0; self._tip_y_error_integrator = 0.0

        if SWEEP_BOARD:
            self.get_logger().info("board sweep: rastering wrist to map the board footprint...")
            self._sweep_board(get_observation, move_robot)   # SC-gated sweep + reset + SAFE-CLEAR
            if SWEEP_ONLY:
                return True
        if getattr(self, "_sweep_port", None) is None:
            self.get_logger().warn("no port lock -> abort trial")
            return True

        samples = []
        for _ in range(PORT_FILTER_N):
            try:
                _, _, info = self._perceive(get_observation); samples.append(info["force"])
            except TransformException:
                pass
            self.sleep_for(DT)

        z = 0.2                                             # ---- approach: UNCHANGED ----
        for t in range(100):
            frac = t / 100.0
            try:
                port_tf, plug_tf, _ = self._perceive(get_observation)
                self.set_pose_target(move_robot=move_robot,
                                     pose=self.calc_gripper_pose(port_tf, plug_tf, slerp_fraction=frac,
                                                                 position_fraction=frac, z_offset=z,
                                                                 reset_xy_integrator=True),
                                     stiffness=STIFF_FREE)
            except TransformException:
                pass
            self.sleep_for(DT)

        try:
            _, plug0, i0 = self._perceive(get_observation)
            f_base = i0["force"]; tip0 = float(plug0.translation.z)
        except TransformException:
            f_base = float(np.median(samples)) if samples else 20.0; tip0 = 0.0
        port_x, port_y, port_z = (float(v) for v in self._locked_xyz)
        ent = ENTRANCE_H.get(self._plug_type, 0.046)
        self.get_logger().info(f"SEAT[{self._plug_type}]: f_base {f_base:.1f}N · tip {(tip0-port_z)*1000:.0f}mm "
                               f"above port · funnel {ent*1000:.0f}mm")

        # ================= DESCENT: contact-driven, for BOTH types =================
        spiral_k, sx, sy, dyaw = 0, 0.0, 0.0, 0.0
        seated = False
        z_ref, tip_ref = z, tip0
        fired, first_fire, step = 0, None, 0
        cooldown = 0                     # probe cooldown -> lets the descent actually happen
        while z >= Z_FLOOR:
            z -= STEP; step += 1
            try:
                port_tf, plug_tf, info = self._perceive(get_observation)
                fn = info["force"]; d_signed = fn - f_base; d_abs = abs(d_signed)
                tip_z = float(plug_tf.translation.z)
                tip_lat = math.hypot(float(plug_tf.translation.x) - port_x,
                                     float(plug_tf.translation.y) - port_y)
                tip_above = tip_z - port_z

                if (tip_ref - tip_z) >= STALL_ACT:            # re-arm while the tip is genuinely moving
                    z_ref, tip_ref = z, tip_z
                cmd_prog = z_ref - z; act_prog = tip_ref - tip_z
                stalled = cmd_prog > STALL_CMD and act_prog < STALL_ACT
                force_hit = d_abs > F_ABS
                contact = (force_hit or stalled) and cooldown == 0
                if cooldown > 0:
                    cooldown -= 1        # persistent unload is NOT a new contact -> keep descending
                reason = ("both" if (force_hit and stalled) else
                          "force_abs" if force_hit else "stall" if stalled else None)
                self._vk, self._vreason = spiral_k, reason      # published for the viz HUD (free)

                if step % 4 == 1 or contact:
                    self._rec(trial=self._trial_idx, port=task.port_name, type=self._plug_type, step=step,
                              z_offset=round(z, 5), force=round(fn, 2), d_signed=round(d_signed, 2),
                              tip_above_port_mm=round(tip_above * 1000, 1), tip_lat_mm=round(tip_lat * 1000, 1),
                              cmd_prog_mm=round(cmd_prog * 1000, 1), act_prog_mm=round(act_prog * 1000, 1),
                              contact=bool(contact), reason=reason, spiral_k=spiral_k,
                              sx_mm=round(sx * 1000, 2), sy_mm=round(sy * 1000, 2))

                # ---- SEAT EXIT: measured DEPTH *and* LATERAL (depth alone => false positives at 30mm) ----
                if tip_above < SEAT_TIP and tip_lat < SEAT_LAT:
                    seated = True
                    self.get_logger().info(f"*** SEATED [{self._plug_type}] tip {tip_above*1000:+.1f}mm deep, "
                                           f"{tip_lat*1000:.1f}mm lateral · spiral {spiral_k} · STOPPING ***")
                    break

                if contact:
                    fired += 1
                    if first_fire is None:
                        first_fire = reason
                        self.get_logger().info(f"*** CONTACT [{reason}] step {step} dF {d_signed:+.1f}N | "
                                               f"tip {tip_above*1000:.0f}mm above, {tip_lat*1000:.0f}mm lateral ***")
                    if spiral_k >= SPIRAL_MAX:
                        self.get_logger().warn("spiral exhausted")
                        break
                    spiral_k += 1
                    r = SPIRAL_PITCH * spiral_k; a = GOLDEN * spiral_k
                    sx, sy = r * math.cos(a), r * math.sin(a)
                    dyaw = DITHER_SEQ[spiral_k % len(DITHER_SEQ)]
                    z = min(z + RETRACT, 0.02)
                    z_ref, tip_ref = z, tip_z
                    cooldown = COOLDOWN          # ← allow the re-descent before contact may fire again
                    for _ in range(CLEAR_WAIT):              # let the contact force settle after the retract
                        pt, pl, inf = self._perceive(get_observation)
                        pt.translation.x += sx; pt.translation.y += sy
                        self.set_pose_target(move_robot=move_robot,
                                             pose=self.calc_gripper_pose(self._dither(pt, dyaw), pl, z_offset=z),
                                             stiffness=self._stiff_contact())
                        self.sleep_for(DT)
                        if abs(inf["force"] - f_base) < F_ABS * 0.5:
                            break

                port_tf.translation.x += sx; port_tf.translation.y += sy
                self.set_pose_target(move_robot=move_robot,
                                     pose=self.calc_gripper_pose(self._dither(port_tf, dyaw), plug_tf, z_offset=z),
                                     stiffness=self._stiff_contact() if (contact or spiral_k > 0) else STIFF_FREE)
            except TransformException:
                pass
            self.sleep_for(DT)

        self.get_logger().info(f"FULL[{self._plug_type}] DONE: seated={seated} · contact {fired}x "
                               f"(first {first_fire}) · spiral_k {spiral_k} · off ({sx*1000:+.1f},{sy*1000:+.1f})mm")
        try:                                                 # never hold a press / never drag a seated plug
            from geometry_msgs.msg import Point, Pose, Quaternion
            from aic_example_policies.ros.PerceptionInsertYOLO import _quat
            T = self._T_base_frame("gripper/tcp"); q = _quat(T[:3, :3])
            for _ in range(20):
                self.set_pose_target(
                    move_robot=move_robot,
                    pose=Pose(position=Point(x=float(T[0, 3]), y=float(T[1, 3]), z=float(T[2, 3])),
                              orientation=Quaternion(w=float(q[0]), x=float(q[1]), y=float(q[2]), z=float(q[3]))),
                    stiffness=self._stiff_contact())
                self.sleep_for(DT)
        except Exception:
            pass
        self.sleep_for(2.0)
        return True
