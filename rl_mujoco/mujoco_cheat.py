"""MuJoCo CheatCode (Route B): reuse the CheatCode alignment logic against MuJoCo
ground truth (data.xpos / site_xpos) + a Jacobian-transpose Cartesian impedance
controller (what aic_controller does) writing torques into the committed motor actuators.

Stage 1 here: validate the impedance controller + GT reads by servoing the gripper TCP
to a target above the SC port, from the real AIC arm pose. Cable/insertion added next.
"""
import os, numpy as np, mujoco, imageio

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE = os.path.join(HERE, "aic_faithful_nocable/scene.xml")   # stable scene for controller validation

# real AIC operating arm pose (from the challenge UR5e config), MuJoCo joint order:
# [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]
Q_INIT = np.array([0.1597, -1.3542, -1.6648, -1.6933, 1.5710, 1.4110])

m = mujoco.MjModel.from_xml_path(SCENE)
d = mujoco.MjData(m)
TCP = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper_tcp")
PORT = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "sc_port_sensor")
arm_dofs = [m.jnt_dofadr[m.actuator_trnid[a, 0]] for a in range(6)]

def reset():
    mujoco.mj_resetData(m, d)
    d.qpos[:6] = Q_INIT
    mujoco.mj_forward(m, d)

def quat_err(q_cur, q_des):
    """rotation-vector error that rotates q_cur -> q_des (world frame)."""
    err = np.zeros(3)
    dq = np.zeros(4)
    mujoco.mju_negQuat(dq, q_cur)
    mujoco.mju_mulQuat(dq, q_des, dq)      # dq = q_des * q_cur^-1
    mujoco.mju_quat2Vel(err, dq, 1.0)
    return err

def impedance_ctrl(target_pos, target_quat, Kp=(400,400,400), Kr=(40,40,40), Kd=40.0):
    """Jacobian-transpose Cartesian impedance -> arm torques (+ gravity comp)."""
    jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jacp, jacr, TCP)
    Ja = np.vstack([jacp, jacr])[:, arm_dofs]          # 6x6
    pos_err = target_pos - d.site_xpos[TCP]
    tcp_quat = np.zeros(4); mujoco.mju_mat2Quat(tcp_quat, d.site_xmat[TCP])
    ori_err = quat_err(tcp_quat, target_quat)
    v = Ja @ d.qvel[arm_dofs]
    wrench = np.concatenate([np.array(Kp)*pos_err, np.array(Kr)*ori_err]) - Kd*v
    tau = Ja.T @ wrench + d.qfrc_bias[arm_dofs]         # + gravity/coriolis comp
    return tau

# --- Stage 1: servo TCP to 10 cm above the port, keeping current tool orientation ---
reset()
tcp0_quat = np.zeros(4); mujoco.mju_mat2Quat(tcp0_quat, d.site_xmat[TCP])
print("start: TCP", np.round(d.site_xpos[TCP],3), " PORT", np.round(d.site_xpos[PORT],3))
target = d.site_xpos[PORT].copy() + np.array([0,0,0.10])

opt = mujoco.MjvOption()
for g in range(len(opt.geomgroup)): opt.geomgroup[g] = 1
cam = mujoco.MjvCamera(); cam.lookat[:]=[0.3,0.15,0.25]; cam.distance=1.3; cam.azimuth=140; cam.elevation=-22
frames=[]
with mujoco.Renderer(m, height=480, width=640) as r:
    for t in range(1200):
        d.ctrl[:6] = impedance_ctrl(target, tcp0_quat)
        d.ctrl[6] = 0.0
        mujoco.mj_step(m, d)
        if t % 8 == 0:
            r.update_scene(d, camera=cam, scene_option=opt); frames.append(r.render())
    err = np.linalg.norm(d.site_xpos[TCP]-target)
print(f"end:   TCP {np.round(d.site_xpos[TCP],3)}  target {np.round(target,3)}  err={err*1000:.1f} mm  {'REACHED' if err<0.01 else 'OFF'}")
out=os.path.join(HERE,"mujoco_cheat_servo.mp4"); imageio.mimsave(out, frames, fps=30)
print("wrote", out, f"({len(frames)} frames)")
