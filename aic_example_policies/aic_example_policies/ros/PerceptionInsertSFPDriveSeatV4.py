#
#  PerceptionInsertSFPDriveSeatV4 — fix the CONTACT PREDICATE, then instrument. Nothing else changes.
#
#  DIAGNOSIS (measured, not assumed): the deployed "force stack" is DEAD. Its predicate is
#      if d_force > F_STOP:            # F_STOP = 8.0, d_force = force - baseline
#  but when the plug lands on the SFP cage FACE the board takes its weight and the wrist UNLOADS:
#  force goes 21 N -> 8 N, i.e. d_force ~ -13 (NEGATIVE). The branch never fires. spiral_k stays 0,
#  so sx=sy=0 and dyaw=0, and no stiffness is ever passed => the default STIFF [90,90,90].
#  At runtime the whole force stack therefore reduces to a plain stiff straight-down position ramp
#  (functionally CheatCode's descent). Proof: ZERO force-stack log lines in every SFP run, ever.
#
#  THE FIX (this file):
#    1. INSTRUMENT   — per-step JSONL: z, force, f_base, d_signed, d_abs, tcp/tip progress (commanded vs
#                      actual), contact_triggered, trigger_reason, spiral_k, sx, sy, dyaw, stiffness.
#    2. TRIGGER      — contact = |force - f_base| > F_ABS  OR  STALL.  Sign-insensitive: an UNLOAD is
#                      contact too.
#    3. STALL        — RE-ARMING WATCHDOG (this is the bug I made last time: I anchored the reference at
#                      the DESCENT START, 200 mm up, so actual_progress was ~190 mm and the "<4 mm" test
#                      could never be true). Here the reference ADVANCES whenever the tip actually moves,
#                      so it stays LOCAL: stalled = commanded >10 mm since ref AND actual <4 mm since ref.
#    4. COMPLIANCE   — [90,90,90] while descending free; [40,40,90] (RCC: lateral soft, axial firm) once
#                      contact/stall fires, so the plug can self-align instead of jamming rigidly.
#
#  NOT CHANGED (deliberately — one variable at a time): detection, frozen target, SAFE-CLEAR, approach.
#  The SPIRAL PATTERN is left exactly as-is (golden, pitch 1.2 mm) — pattern quality is IRRELEVANT until
#  the search actually runs, and every earlier pattern test was invalid because the branch never fired.
#  SC is delegated to the untouched path.
#
#  PASS CRITERION FOR RUN 1 IS *NOT* SCORE. It is:
#      contact_triggered = True   ·   spiral_k > 0   ·   stiffness switches to [40,40,90]
#      ·   the plug stops parking inertly on the cage face
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
    SPIRAL_PITCH, SPIRAL_MAX, GOLDEN, DITHER_SEQ, Z_SEATED,
)
from aic_example_policies.ros.PerceptionInsertSFPDrive import PerceptionInsertSFPDrive

# ---- the only tuned knobs (step 4 of the plan sweeps THESE, once the branch fires) ----
F_ABS = float(os.environ.get("AIC_F_ABS", 4.0))            # |force - baseline| N  -> contact
STALL_CMD = float(os.environ.get("AIC_STALL_CMD", 0.010))  # commanded progress since ref
STALL_ACT = float(os.environ.get("AIC_STALL_ACT", 0.004))  # ...with less actual tip progress => stalled
LAT_STIFF = float(os.environ.get("AIC_LAT_STIFF", 40.0))   # lateral stiffness during contact/search
AX_STIFF = float(os.environ.get("AIC_AX_STIFF", 90.0))     # axial stiffness (keep pushing)

STIFF_FREE = [90.0, 90.0, 90.0, 50.0, 50.0, 50.0]
RETRACT = 0.004
CLEAR_WAIT = 12            # after a retract, wait for the contact force to settle before descending
SEAT_TIP = 0.005      # tip within 5mm of (or below) the port datum  -- DEPTH gate
SEAT_LAT = 0.006      # ...AND the plug tip's ACTUAL xy within 6mm of the port -- LATERAL gate.
                      # DEPTH ALONE IS NOT A SEAT: V3 fired seated=True at a 30mm offset (plug had
                      # descended BESIDE the cage, not into it). 3 of 6 seated=True claims were FALSE.
MAX_PROBES = 30       # GLOBAL probe budget across ALL recenters. Without it the recenter cycle loops
                      # forever: spiral 1..9 -> cap -> recenter -> spiral 1..9 -> ...  seat_v4_1 hit
                      # 45 recenters / 5281 steps / 264 s and timed out with the plug still 41mm up.
MAX_RECENTER = 2      # after this many, ABANDON the search and just descend on the lock
R_CAP = 0.010         # ABANDON the spiral past 10mm. r = 1.2mm*k reached 36mm -- WIDER THAN THE PORT;
                      # it was leaving the neighbourhood, not searching it. MEASURED: every real seat
                      # had spiral_k<=8 (r<=9.6mm); every runaway (k 20-30, 24-36mm) failed.      # plug tip within 5mm of (or below) the port datum => IT IS IN.
                      # On the cage face the tip sits at +42..46mm, so this is unambiguous.
Z_FLOOR = -0.015
STEP = 0.0005
DT = 0.05
LOG = "/home/skr/aic_data/seat_v4_log.jsonl"


class PerceptionInsertSFPDriveSeatV4(PerceptionInsertSFPDrive):

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
        if typ != "sfp":
            self.get_logger().info("SC -> delegating to the UNCHANGED path (isolating the SFP seat)")
            return super().insert_cable(task, get_observation, move_robot, send_feedback)

        self.get_logger().info(f"SeatV2 [F_ABS {F_ABS} · stall {STALL_CMD*1000:.0f}/{STALL_ACT*1000:.0f}mm · "
                               f"RCC {[LAT_STIFF, LAT_STIFF, AX_STIFF]}] task: {task}")
        self._task = task; self._trial_idx += 1
        self._plug_type = typ if typ in T_GRASP else "sfp"
        self._tid = torch.tensor([self._vocab.get(f"{task.target_module_name}|{task.port_name}", 0)],
                                 dtype=torch.long, device=self._device)
        self._port_hist = []; self._gate_count = 0
        self._tip_x_error_integrator = 0.0; self._tip_y_error_integrator = 0.0

        if SWEEP_BOARD:
            self.get_logger().info("board sweep: rastering wrist to map the board footprint...")
            self._sweep_board(get_observation, move_robot)   # + per-trial reset + SAFE-CLEAR (inherited)
            if SWEEP_ONLY:
                return True

        samples = []
        for _ in range(PORT_FILTER_N):
            try:
                _, _, info = self._perceive(get_observation); samples.append(info["force"])
            except TransformException:
                pass
            self.sleep_for(DT)

        # ---- approach: UNCHANGED ----
        z = 0.2
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
        port_z = float(self._locked_xyz[2])
        self.get_logger().info(f"SeatV2 descent: f_base {f_base:.1f}N  tip {tip0:.4f}  port_z {port_z:.4f} "
                               f"({(tip0-port_z)*1000:.0f}mm above)")

        # ================== DESCENT — the ONLY thing that changes ==================
        spiral_k, sx, sy, dyaw = 0, 0.0, 0.0, 0.0
        seated = False
        total_probes = 0; recenters = 0     # GLOBAL bounds -- the recenter cannot loop forever
        z_ref, tip_ref = z, tip0            # RE-ARMING watchdog reference (stays LOCAL)
        fired, first_fire = 0, None
        step = 0
        while z >= Z_FLOOR:
            z -= STEP; step += 1
            try:
                port_tf, plug_tf, info = self._perceive(get_observation)
                fn = info["force"]
                d_signed = fn - f_base
                d_abs = abs(d_signed)
                tip_z = float(plug_tf.translation.z)
                tip_x = float(plug_tf.translation.x); tip_y = float(plug_tf.translation.y)
                tip_lat = math.hypot(tip_x - float(self._locked_xyz[0]),
                                     tip_y - float(self._locked_xyz[1]))   # ACTUAL plug xy vs port
                tcp_z = float(self._T_base_frame("gripper/tcp")[2, 3])

                # --- RE-ARM: if the tip IS making progress, advance the reference (keeps it LOCAL) ---
                if (tip_ref - tip_z) >= STALL_ACT:
                    z_ref, tip_ref = z, tip_z

                cmd_prog = z_ref - z                     # commanded progress since the reference
                act_prog = tip_ref - tip_z               # actual tip progress since the reference
                stalled = cmd_prog > STALL_CMD and act_prog < STALL_ACT
                force_hit = d_abs > F_ABS
                contact = force_hit or stalled
                reason = ("both" if (force_hit and stalled) else
                          "force_abs" if force_hit else "stall" if stalled else None)

                stiff = self._stiff_contact() if (contact or spiral_k > 0) else STIFF_FREE
                self._vk, self._vreason = spiral_k, reason      # published for the viz HUD (no cost)

                if step % 4 == 1 or contact:
                    self._rec(trial=self._trial_idx, port=self._task.port_name, step=step,
                              z_offset=round(z, 5), force=round(fn, 2), f_base=round(f_base, 2),
                              d_signed=round(d_signed, 2), d_abs=round(d_abs, 2),
                              tcp_z=round(tcp_z, 5), tip_z=round(tip_z, 5),
                              tip_lat_mm=round(tip_lat * 1000, 1),
                              tip_above_port_mm=round((tip_z - port_z) * 1000, 1),
                              cmd_prog_mm=round(cmd_prog * 1000, 1), act_prog_mm=round(act_prog * 1000, 1),
                              contact_triggered=bool(contact), trigger_reason=reason,
                              spiral_k=spiral_k, sx_mm=round(sx * 1000, 2), sy_mm=round(sy * 1000, 2),
                              dyaw=round(dyaw, 4), lat_stiff=stiff[0], ax_stiff=stiff[2])

                # ================== THE FIX: SEAT TEST ON *MEASURED* TIP DEPTH ==================
                # V2's exit test was `if z <= Z_SEATED` — but z is the COMMANDED z_offset, and every
                # contact RETRACTS it by 4 mm, so it never gets down to -0.004. The plug would enter the
                # port (tip -1..-4 mm BELOW the port datum) and the loop would keep spiralling for 6-28
                # MORE probes, dragging the SEATED plug out to a 36 mm offset at 11-13 N.
                # MEASURED ACROSS 5 RUNS: 9/10 SFP trials ENTERED, but only 6/10 were KEPT — 3 were
                # dragged back OUT and scored as failures (one had entered at k=2 / 2.4 mm — a perfect
                # find — then 28 probes tore it loose). It also caused the 94-104 N peaks and the -12
                # force penalties.  Exit on what the PLUG is doing, not on what we commanded.
                tip_above = tip_z - port_z
                if tip_above < SEAT_TIP and tip_lat < SEAT_LAT:
                    self.get_logger().info(
                        f"*** SEATED (tip {tip_above*1000:+.1f}mm deep, {tip_lat*1000:.1f}mm lateral) spiral {spiral_k}, "
                        f"offset ({sx*1000:+.1f},{sy*1000:+.1f})mm — STOPPING SEARCH ***")
                    self._rec(trial=self._trial_idx, port=self._task.port_name, step=step, event="SEATED",
                              tip_above_port_mm=round(tip_above * 1000, 1),
                              tip_lat_mm=round(tip_lat * 1000, 1), spiral_k=spiral_k,
                              sx_mm=round(sx * 1000, 2), sy_mm=round(sy * 1000, 2), z_offset=round(z, 5))
                    seated = True
                    break

                if contact:
                    fired += 1
                    if first_fire is None:
                        first_fire = reason
                        self.get_logger().info(
                            f"*** CONTACT BRANCH FIRED [{reason}] step {step} z {z:+.4f} | dF {d_signed:+.1f}N "
                            f"(|dF| {d_abs:.1f}) | cmd {cmd_prog*1000:.1f}mm act {act_prog*1000:.1f}mm | "
                            f"tip {((tip_z-port_z)*1000):.0f}mm above port ***")
                    total_probes += 1
                    if total_probes >= MAX_PROBES:
                        self.get_logger().warn(f"probe budget exhausted ({MAX_PROBES}) -> ABANDON, descend on lock")
                        sx = sy = dyaw = 0.0
                        break
                    spiral_k += 1
                    r = SPIRAL_PITCH * spiral_k; a = GOLDEN * spiral_k
                    if r > R_CAP:                      # left the neighbourhood
                        recenters += 1
                        if recenters > MAX_RECENTER:
                            self.get_logger().warn(f"{MAX_RECENTER} recenters used -> ABANDON, descend on lock")
                            sx = sy = dyaw = 0.0
                            break
                        self.get_logger().warn(f"spiral hit {R_CAP*1000:.0f}mm cap at k={spiral_k} -> "
                                               f"RECENTER {recenters}/{MAX_RECENTER} on the lock")
                        sx = sy = 0.0; dyaw = 0.0; spiral_k = 0
                        z_ref, tip_ref = z, tip_z
                        continue
                    sx, sy = r * math.cos(a), r * math.sin(a)
                    dyaw = DITHER_SEQ[spiral_k % len(DITHER_SEQ)]
                    z = min(z + RETRACT, 0.02)
                    z_ref, tip_ref = z, tip_z                       # re-arm after retract
                    self.get_logger().info(f"  -> spiral {spiral_k} off ({sx*1000:+.1f},{sy*1000:+.1f})mm "
                                           f"dyaw {math.degrees(dyaw):+.0f}deg, COMPLIANT {self._stiff_contact()[:3]}")
                    for _ in range(CLEAR_WAIT):                     # let the contact force settle
                        pt, pl, inf = self._perceive(get_observation)
                        pt.translation.x += sx; pt.translation.y += sy
                        self.set_pose_target(move_robot=move_robot,
                                             pose=self.calc_gripper_pose(self._dither(pt, dyaw), pl, z_offset=z),
                                             stiffness=self._stiff_contact())
                        self.sleep_for(DT)
                        if abs(inf["force"] - f_base) < F_ABS * 0.5:
                            break

                port_tf.translation.x += sx
                port_tf.translation.y += sy
                self.set_pose_target(move_robot=move_robot,
                                     pose=self.calc_gripper_pose(self._dither(port_tf, dyaw), plug_tf, z_offset=z),
                                     stiffness=stiff)
            except TransformException:
                pass
            self.sleep_for(DT)

        self.get_logger().info(f"SeatV3 DONE: seated={seated} · contact fired {fired}x (first: {first_fire}) "
                               f"· spiral_k {spiral_k} · final off ({sx*1000:+.1f},{sy*1000:+.1f})mm")
        # NEVER hold a press / never drag a seated plug: re-command the CURRENT TCP pose and settle.
        try:
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
