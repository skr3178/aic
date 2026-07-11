#
#  PerceptionInsertSFPDrive — Level-2 MOTION fix (see board_pose.md / LAB_LOG "Execution stack").
#
#  SAME detection as PerceptionInsertSFP (fixed board pose + slide-select port lock), but the
#  APPROACH + DESCENT drive a STATIC FROZEN target — the sweep slide-select lock — instead of
#  re-perceiving every step.  WHY: the base _perceive re-triangulates RAW YOLO each step (line ~398),
#  and raw YOLO is instance-unaware for SFP, so the per-step target DRIFTS off the slide-selected
#  opening toward the sibling / between the two 21.8 mm openings.  The arm then chases a moving target
#  -> folded IK / self-collision (trial0 jam) or a bad descent target (SC stall).
#
#  This mirrors CheatCode's motion shape (ONE static port lookup + smooth interp + slow descent), but
#  fed OUR detection instead of GT, and it KEEPS our force stack (spiral + yaw dither) to absorb the
#  residual (~mm + orientation) error of a perceived target — CheatCode skips the force stack because
#  its GT target is exact; ours is not.
#
#  Only _perceive is overridden -> the inherited insert_cable / approach / descent / force stack are
#  UNCHANGED.  Phase 1 of 2:  Phase 2 (later) = have _perceive return a LIVE re-perceived pose
#  (continuous vision, stamp-synced TF) instead of the frozen one — same loop, one-method change.
#
import math

import numpy as np
from geometry_msgs.msg import Transform
from aic_control_interfaces.msg import JointMotionUpdate, TrajectoryGenerationMode

from aic_example_policies.ros.PerceptionInsertYOLO import T_GRASP, R_BOARD_OFFSET, _R, _T, _quat
from aic_example_policies.ros.PerceptionInsertSFP import PerceptionInsertSFP

# canonical known-safe joint pose (shipped example policies GentleGiant/SpeedDemon use it as "home").
# The SWEEP-END posture is part of the trial0 problem: reaching port_0 from it folds the upper arm into
# the NIC card. Returning to this clean branch BEFORE the Cartesian approach = CheatCode's start condition.
SAFE_HOME = [-0.16, -1.35, -1.66, -1.69, 1.57, 1.41]
SAFE_STIFF = [50.0, 50.0, 50.0, 20.0, 20.0, 20.0]     # GentleGiant: low stiffness + high damping
SAFE_DAMP = [40.0, 40.0, 40.0, 20.0, 20.0, 20.0]      # -> slow, smooth, low-jerk (jerk is scored)
SAFE_STEPS = 100                                       # x0.05 s = 5 s to settle into the branch


class PerceptionInsertSFPDrive(PerceptionInsertSFP):

    def _safe_clear(self, move_robot):
        """Drive to the known-safe HOME joint pose in JOINT space (resets the IK branch + clears the board).
        Cartesian alone can't do this: the same TCP pose is reachable in a folded branch."""
        jm = JointMotionUpdate(
            target_stiffness=SAFE_STIFF, target_damping=SAFE_DAMP,
            trajectory_generation_mode=TrajectoryGenerationMode(mode=TrajectoryGenerationMode.MODE_POSITION))
        jm.target_state.positions = list(SAFE_HOME)
        self.get_logger().info(f"SAFE-CLEAR: joint-space -> HOME {SAFE_HOME} (reset IK branch before approach)")
        for _ in range(SAFE_STEPS):
            move_robot(joint_motion_update=jm)
            self.sleep_for(0.05)
        try:
            t = self._T_base_frame("gripper/tcp")[:3, 3]
            self.get_logger().info(f"SAFE-CLEAR: done, TCP now {np.round(t, 3)}")
        except Exception:
            pass

    def _sweep_board(self, get_observation, move_robot):
        """sweep (inherited: fixed board pose + slide-select lock) -> RESET the frozen lock (per-trial!)
        -> SAFE-CLEAR to the home joint branch, so the Cartesian approach starts from a clean posture."""
        pts = super()._sweep_board(get_observation, move_robot)
        self._locked_xyz = None                      # BUG FIX: was persisting across trials ->
        self._locked_quat = None                     # t1/SC reused t0's target. Reset every trial.
        self._safe_clear(move_robot)
        return pts

    def _freeze_target(self):
        """Build the STATIC port pose ONCE: position = sweep slide-select lock, orientation = board-yaw
        (⊗ CAD port offset), identical to the orientation the base _perceive computes.
        AIC_GT_FREEZE=1 (needs ground_truth:=true) freezes from the GT port pose instead — the ablation
        that separates 'our perceived orientation is off' from 'the arm can't reach this pose'."""
        import os
        if os.environ.get("AIC_GT_FREEZE"):
            pf = f"task_board/{self._task.target_module_name}/{self._task.port_name}_link"
            g = self._gt_transform(pf)
            if g:
                self._locked_xyz = np.array(g[:3], float)
                self._locked_quat = tuple(float(v) for v in g[3:])
                self.get_logger().info(
                    f"DRIVE: FROZEN target from GT xyz {np.round(self._locked_xyz, 3)} (GT ABLATION; motion isolation)")
                return
            self.get_logger().warn("DRIVE: AIC_GT_FREEZE set but no GT transform (run with ground_truth:=true) -> perceived")
        byaw = float(self._board_yaw)
        cz, sz = math.cos(byaw), math.sin(byaw)
        R_flat = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])   # pure yaw (board flat)
        R_port = R_flat @ _R(R_BOARD_OFFSET[self._plug_type])
        self._locked_xyz = np.array(self._sweep_port, float)
        self._locked_quat = tuple(float(v) for v in _quat(R_port))            # wxyz
        self.get_logger().info(
            f"DRIVE: FROZEN target xyz {np.round(self._locked_xyz, 3)} yaw {math.degrees(byaw):.1f} deg "
            f"(static slide-select lock; per-step re-perception DISABLED)")

    def _perceive(self, get_observation):
        obs = get_observation()
        # no lock yet (shouldn't happen post-sweep) -> fall back to the base per-step perception
        if getattr(self, "_sweep_port", None) is None or getattr(self, "_board_yaw", None) is None:
            return super()._perceive(get_observation)
        if getattr(self, "_locked_xyz", None) is None:
            self._freeze_target()

        # FK plug (M14) — exact, cheap, recomputed each step (the plug DOES move with the gripper)
        gt_t, gt_q = T_GRASP[self._plug_type]
        plug_b = self._T_base_frame("gripper/tcp") @ _T(np.array(gt_t, float), np.array(gt_q, float))
        plug_tf = Transform()
        plug_tf.translation.x, plug_tf.translation.y, plug_tf.translation.z = map(float, plug_b[:3, 3])
        pw = _quat(plug_b[:3, :3])
        plug_tf.rotation.w, plug_tf.rotation.x, plug_tf.rotation.y, plug_tf.rotation.z = map(float, pw)

        # FRESH COPY of the frozen port each step -> the descent spiral/dither mutates the copy, never the lock
        port_tf = Transform()
        port_tf.translation.x, port_tf.translation.y, port_tf.translation.z = map(float, self._locked_xyz)
        w, x, y, z = self._locked_quat
        port_tf.rotation.w, port_tf.rotation.x, port_tf.rotation.y, port_tf.rotation.z = w, x, y, z

        f = obs.wrist_wrench.wrench.force
        fmag = float(np.linalg.norm([f.x, f.y, f.z]))
        # GUARD/telemetry: ACTUAL TCP + plug-tip vs the FROZEN target (is the arm tracking, or stuck/folded?)
        self._dstep = getattr(self, "_dstep", 0) + 1
        if self._dstep % 20 == 1:
            tcp = self._T_base_frame("gripper/tcp")[:3, 3]
            tip = plug_b[:3, 3]
            d_tip = float(np.linalg.norm(tip - self._locked_xyz))
            self.get_logger().info(
                f"TRACK[{self._dstep}] TCP {np.round(tcp, 3)} | plug_tip {np.round(tip, 3)} | "
                f"target {np.round(self._locked_xyz, 3)} | tip->target {d_tip*1000:.0f} mm | F {fmag:.1f} N")
        info = {"force": fmag, "gated": "frozen",
                "port_filt": [float(v) for v in self._locked_xyz],
                "plug": [float(v) for v in plug_b[:3, 3]], "n_views": 0}
        return port_tf, plug_tf, info
