#
#  InsertTuner — standalone GT-fed rig to tune the force-insertion endgame in isolation.
#
#  Does NOT touch CheatCode or PerceptionInsert. Torch-free (runs in the eval container).
#
#  Per trial:
#    1. GT port/entrance/plug from TF (requires ground_truth:=true).
#    2. Apply a DELIBERATE offset (dx, dy, dyaw) from a JSON schedule -> simulates a known
#       perception error.
#    3. Fly to entrance + START_ABOVE (default 10 mm), then run the v4 force stack:
#       force-stop + expanding spiral + yaw dither + proximity-scheduled compliance +
#       penetration cap + never-hold-a-press.
#    4. Verify: GT plug->port distance, then a TUG TEST (pull up softly; a latched connector
#       holds). Log everything to JSONL.
#
#  Env knobs (set by tuner_sweep.sh):
#    TUNER_OFFSETS    JSON file: [[dx,dy,dyaw], ...]      (m, m, rad)
#    TUNER_TRIAL_BASE index of this container's first trial into the schedule (default 0)
#    TUNER_LOG        output JSONL (default $AIC_RESULTS_DIR/tuner_log.jsonl)
#    TUNER_START_ABOVE start height above the entrance (default 0.010 m)
#

import json
import math
import os

import numpy as np

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import TransformException
from transforms3d._gohlketransforms import quaternion_multiply, quaternion_slerp

# ---- v4 force-stack constants (the parameters this rig exists to tune) ----
F_STOP = 4.0                 # N above baseline -> contact (impedance can only make ~5-6N at the mouth)
STALL_DZ_CMD = 0.012         # commanded depth advance that must move the tip...
STALL_DZ_TIP = 0.004         # ...by at least this, else we're mechanically blocked (stall = contact)
Z_SEATED = -0.004            # contact at/below this z_offset -> seated
Z_MIN = -0.010               # penetration cap (never command tip further below the port frame)
RETRACT = 0.006              # m back-off after early contact
CLEAR_WAIT = 20              # max steps to wait for force to clear after retract
SPIRAL_PITCH = 0.0012        # m radius growth per spiral step
SPIRAL_MAX = 30
GOLDEN = 2.399963
DITHER_SEQ = [0.0, 0.05, -0.05, 0.10, -0.10]          # rad, cycled with spiral_k
SOFT_MARGIN = 0.005          # switch to soft stiffness once z_offset < entrance_h + margin
STIFF_APPROACH = [90.0, 90.0, 90.0, 50.0, 50.0, 50.0]
STIFF_CONTACT = [40.0, 40.0, 90.0, 50.0, 50.0, 50.0]  # RCC: lateral soft (self-align), axial FIRM (push)
STEP = 0.0005                # z_offset decrement per control step
DT = 0.05                    # control period (s)
SEAT_DIST = {"sfp": 0.020, "sc": 0.008}   # per-type: SC entrance is only 15.6mm from the seat


class InsertTuner(Policy):
    def __init__(self, parent_node):
        self._task = None
        self._trial = 0
        self._tip_x_int = 0.0
        self._tip_y_int = 0.0
        base = int(os.environ.get("TUNER_TRIAL_BASE", "0"))
        sched_path = os.environ.get("TUNER_OFFSETS", "/collect/tuner_offsets.json")
        try:
            self._sched = json.load(open(sched_path))
        except Exception:
            self._sched = [[0.0, 0.0, 0.0]]
        self._base = base
        self._log_path = os.environ.get(
            "TUNER_LOG", os.path.join(os.environ.get("AIC_RESULTS_DIR", "/tmp"), "tuner_log.jsonl"))
        self._start_above = float(os.environ.get("TUNER_START_ABOVE", "0.010"))  # handover margin above entrance
        self._hover = float(os.environ.get("TUNER_HOVER", "0.2"))            # CheatCode-style hover above the PORT
        super().__init__(parent_node)
        self.get_logger().info(
            f"InsertTuner: {len(self._sched)} offsets from {sched_path}, base {base}, "
            f"start {self._start_above*1000:.0f}mm above entrance, log -> {self._log_path}")

    # ---------- helpers ----------
    @staticmethod
    def _zaxis(q):  # wxyz -> the frame's z axis in base coords
        w,x,y,z=q
        return np.array([2*(x*z+y*w), 2*(y*z-x*w), 1-2*(x*x+y*y)])

    @staticmethod
    def _qang(qa, qb):  # angle between two quats (deg)
        d=abs(sum(a*b for a,b in zip(qa,qb)))
        return math.degrees(2*math.acos(min(1.0,d)))

    def _tf(self, frame):
        return self._parent_node._tf_buffer.lookup_transform("base_link", frame, Time())

    def _wait_tf(self, frame, timeout_sec=15.0):
        start = self.time_now(); timeout = Duration(seconds=timeout_sec); i = 0
        while (self.time_now() - start) < timeout:
            try:
                self._tf(frame); return True
            except TransformException:
                if i % 20 == 0:
                    self.get_logger().info(f"waiting for TF {frame} (ground_truth on?)")
                i += 1; self.sleep_for(0.1)
        return False

    def _dist(self, fa, fb):
        a, b = self._tf(fa).transform.translation, self._tf(fb).transform.translation
        return float(np.linalg.norm([a.x - b.x, a.y - b.y, a.z - b.z]))

    def _force(self, get_observation):
        f = get_observation().wrist_wrench.wrench.force
        return float(np.linalg.norm([f.x, f.y, f.z]))

    def _log(self, rec):
        try:
            with open(self._log_path, "a") as fp:
                fp.write(json.dumps(rec) + "\n")
        except Exception as ex:
            self.get_logger().warn(f"tuner log failed: {ex}")

    # CheatCode's alignment math (verbatim behavior, standalone copy; plug from GT TF)
    def _gripper_pose(self, port_t, port_q, plug_frame, slerp_fraction=1.0,
                      position_fraction=1.0, z_offset=0.1, reset_int=False, dyaw=0.0):
        if dyaw != 0.0:
            qz = (math.cos(dyaw / 2), 0.0, 0.0, math.sin(dyaw / 2))
            port_q = quaternion_multiply(qz, port_q)
        plug = self._tf(plug_frame).transform
        q_plug = (plug.rotation.w, plug.rotation.x, plug.rotation.y, plug.rotation.z)
        q_diff = quaternion_multiply(port_q, (-q_plug[0], q_plug[1], q_plug[2], q_plug[3]))
        grip = self._tf("gripper/tcp").transform
        q_grip = (grip.rotation.w, grip.rotation.x, grip.rotation.y, grip.rotation.z)
        q_target = quaternion_multiply(q_diff, q_grip)
        q_cmd = quaternion_slerp(q_grip, q_target, slerp_fraction)
        self._last_cmd_q = tuple(q_cmd)

        gxyz = (grip.translation.x, grip.translation.y, grip.translation.z)
        pxyz = (plug.translation.x, plug.translation.y, plug.translation.z)
        tip_dx, tip_dy = port_t[0] - pxyz[0], port_t[1] - pxyz[1]
        if reset_int:
            self._tip_x_int = self._tip_y_int = 0.0
        else:
            self._tip_x_int = float(np.clip(self._tip_x_int + tip_dx, -0.05, 0.05))
            self._tip_y_int = float(np.clip(self._tip_y_int + tip_dy, -0.05, 0.05))
        i_gain = 0.15
        tx = port_t[0] + i_gain * self._tip_x_int
        ty = port_t[1] + i_gain * self._tip_y_int
        tz = port_t[2] + z_offset - (gxyz[2] - pxyz[2])
        blend = tuple(position_fraction * t + (1 - position_fraction) * g
                      for t, g in zip((tx, ty, tz), gxyz))
        return Pose(position=Point(x=blend[0], y=blend[1], z=blend[2]),
                    orientation=Quaternion(w=q_cmd[0], x=q_cmd[1], y=q_cmd[2], z=q_cmd[3]))

    def _hold_here(self, move_robot, lift=0.003):
        """Never hold a press: re-command a neutral target at/above the current pose."""
        try:
            g = self._tf("gripper/tcp").transform
            self.set_pose_target(
                move_robot=move_robot,
                pose=Pose(position=Point(x=g.translation.x, y=g.translation.y,
                                         z=g.translation.z + lift),
                          orientation=Quaternion(w=g.rotation.w, x=g.rotation.x,
                                                 y=g.rotation.y, z=g.rotation.z)),
                stiffness=STIFF_CONTACT)
        except TransformException:
            pass

    # ---------- the trial ----------
    def insert_cable(self, task: Task, get_observation: GetObservationCallback,
                     move_robot: MoveRobotCallback, send_feedback: SendFeedbackCallback):
        self._task = task
        k = self._base + self._trial
        dx, dy, dyaw = self._sched[k % len(self._sched)]
        self._trial += 1
        self._tip_x_int = self._tip_y_int = 0.0

        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        entrance_frame = f"{port_frame}_entrance"
        plug_frame = f"{task.cable_name}/{task.plug_name}_link"
        self.get_logger().info(
            f"TUNER trial {k}: target {task.port_name}@{task.target_module_name} "
            f"offset ({dx*1000:.1f},{dy*1000:.1f})mm yaw {math.degrees(dyaw):.1f}deg")

        for fr in (port_frame, plug_frame, entrance_frame):
            if not self._wait_tf(fr):
                self._log({"trial": k, "error": f"no TF {fr}"}); return False

        port = self._tf(port_frame).transform
        ent = self._tf(entrance_frame).transform
        entrance_h = ent.translation.z - port.translation.z
        # injected offset -> this is the "perceived" port the stack must recover from
        port_t = (port.translation.x + dx, port.translation.y + dy, port.translation.z)
        port_q = (port.rotation.w, port.rotation.x, port.rotation.y, port.rotation.z)

        # ---- Phase A: CheatCode-verbatim approach (hover 200mm above port) ----
        z0 = self._hover
        handover = entrance_h + self._start_above     # where the blind glide ends
        for t in range(100):
            fr = t / 100.0
            try:
                self.set_pose_target(move_robot=move_robot,
                                     pose=self._gripper_pose(port_t, port_q, plug_frame,
                                                             slerp_fraction=fr, position_fraction=fr,
                                                             z_offset=z0, reset_int=True, dyaw=dyaw),
                                     stiffness=STIFF_APPROACH)
            except TransformException as ex:
                self.get_logger().warn(f"TF during approach: {ex}")
            self.sleep_for(DT)

        # ---- Phase A (cont.): CheatCode's servoed glide, no contact logic, to handover ----
        z = z0
        while z > handover:
            z -= STEP
            try:
                self.set_pose_target(move_robot=move_robot,
                                     pose=self._gripper_pose(port_t, port_q, plug_frame,
                                                             z_offset=z, dyaw=dyaw),
                                     stiffness=STIFF_APPROACH)
            except TransformException as ex:
                self.get_logger().warn(f"TF during glide: {ex}")
            self.sleep_for(DT)

        # ---- Phase B/C: handover -> v4 force stack ----
        try:
            f_base = self._force(get_observation)
        except Exception:
            f_base = 20.0
        spiral_k = 0; sx = sy = 0.0
        peak_df = 0.0; min_d = 1e9; outcome = "floor"; snap = {}
        z_ref = z; tip_z_ref = self._tf(plug_frame).transform.translation.z
        while z >= Z_MIN:
            z -= STEP
            try:
                df = self._force(get_observation) - f_base
                peak_df = max(peak_df, df)
                d_now = self._dist(plug_frame, port_frame)
                if d_now < min_d:
                    min_d = d_now
                    plug_tf = self._tf(plug_frame).transform; grip_tf = self._tf("gripper/tcp").transform
                    qp = (plug_tf.rotation.w, plug_tf.rotation.x, plug_tf.rotation.y, plug_tf.rotation.z)
                    qg = (grip_tf.rotation.w, grip_tf.rotation.x, grip_tf.rotation.y, grip_tf.rotation.z)
                    za, zb = self._zaxis(qp), self._zaxis(port_q)
                    tilt = math.degrees(math.acos(min(1.0, abs(float(np.dot(za, zb))))))
                    rel = np.array([plug_tf.translation.x - port_t[0],
                                    plug_tf.translation.y - port_t[1]])
                    snap = {"tilt_deg": round(tilt, 1),
                            "lat_mm": round(float(np.linalg.norm(rel)) * 1000, 1),
                            "grip_track_deg": round(self._qang(qg, getattr(self, "_last_cmd_q", qg)), 1)}
                tip_z = self._tf(plug_frame).transform.translation.z
                stalled = (z_ref - z) > STALL_DZ_CMD and (tip_z_ref - tip_z) < STALL_DZ_TIP
                if df > F_STOP or stalled:
                    tip_below_entrance = (tip_z - port_t[2]) < entrance_h * 0.5
                    if z <= Z_SEATED and tip_below_entrance:
                        outcome = "seated_contact"; break
                    if spiral_k >= SPIRAL_MAX:
                        outcome = "spiral_exhausted"; break
                    spiral_k += 1
                    r = SPIRAL_PITCH * spiral_k; a = GOLDEN * spiral_k
                    sx, sy = r * math.cos(a), r * math.sin(a)
                    z = min(z + RETRACT, z0)
                    z_ref = z; tip_z_ref = tip_z
                    # retract first and wait until contact clears before descending again
                    tgt = (port_t[0] + sx, port_t[1] + sy, port_t[2])
                    for _ in range(CLEAR_WAIT):
                        self.set_pose_target(move_robot=move_robot,
                                             pose=self._gripper_pose(tgt, port_q, plug_frame,
                                                                     z_offset=z, dyaw=dyaw),
                                             stiffness=STIFF_CONTACT)
                        self.sleep_for(DT)
                        if self._force(get_observation) - f_base < F_STOP * 0.5:
                            break
                soft = z < entrance_h + SOFT_MARGIN
                dither = DITHER_SEQ[spiral_k % len(DITHER_SEQ)] if spiral_k else 0.0
                tgt = (port_t[0] + sx, port_t[1] + sy, port_t[2])
                self.set_pose_target(move_robot=move_robot,
                                     pose=self._gripper_pose(tgt, port_q, plug_frame,
                                                             z_offset=z, dyaw=dyaw + dither),
                                     stiffness=STIFF_CONTACT if soft else STIFF_APPROACH)
            except TransformException as ex:
                self.get_logger().warn(f"TF during descent: {ex}")
            self.sleep_for(DT)

        # ---- never hold a press, settle, measure ----
        self._hold_here(move_robot)
        self.sleep_for(2.0)
        try:
            d_final = self._dist(plug_frame, port_frame)
        except TransformException:
            d_final = float("nan")

        # ---- tug test: pull up softly; a latched connector holds ----
        tug_held = None
        try:
            g = self._tf("gripper/tcp").transform
            self.set_pose_target(move_robot=move_robot,
                                 pose=Pose(position=Point(x=g.translation.x, y=g.translation.y,
                                                          z=g.translation.z + 0.008),
                                           orientation=Quaternion(w=g.rotation.w, x=g.rotation.x,
                                                                  y=g.rotation.y, z=g.rotation.z)),
                                 stiffness=STIFF_CONTACT)
            self.sleep_for(1.5)
            d_tug = self._dist(plug_frame, port_frame)
            tug_held = bool(d_tug < SEAT_DIST.get(task.port_type, 0.010))
        except TransformException:
            d_tug = float("nan")

        rec = {"trial": k, "type": task.port_type, "module": task.target_module_name,
               "port": task.port_name,
               "dx_mm": round(dx * 1000, 2), "dy_mm": round(dy * 1000, 2),
               "dyaw_deg": round(math.degrees(dyaw), 2),
               "outcome": outcome, "spiral_k": spiral_k,
               "peak_dF": round(peak_df, 1),
               "min_d_mm": round(min_d * 1000, 1),
               "d_final_mm": round(d_final * 1000, 1),
               "d_tug_mm": round(d_tug * 1000, 1) if d_tug == d_tug else None,
               "seated": bool(d_final == d_final and d_final < SEAT_DIST.get(task.port_type, 0.010)),
               **snap,
               "tug_held": tug_held}
        self._log(rec)
        self.get_logger().info(f"TUNER trial {k} -> {rec}")
        self.sleep_for(1.0)
        return True
