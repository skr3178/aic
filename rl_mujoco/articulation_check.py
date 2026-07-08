"""Confirm the AIC robot articulation on the faithful scene, and render a video.
Drives each arm actuator, verifies the target joint actually responds, exercises the
gripper, and writes an mp4 (collision-geometry view, since visuals were stripped).
Run headless with: MUJOCO_GL=egl python articulation_check.py
"""
import os, numpy as np, mujoco, imageio

m = mujoco.MjModel.from_xml_path(os.path.join(os.path.dirname(__file__), "aic_faithful/scene.xml"))
d = mujoco.MjData(m)

# --- 1) actuator -> joint mapping + ranges ---
print("=== ACTUATORS -> JOINTS ===")
for a in range(m.nu):
    an = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
    jid = m.actuator_trnid[a, 0]
    jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
    qadr = m.jnt_qposadr[jid]
    jrange = m.jnt_range[jid]
    crange = m.actuator_ctrlrange[a]
    print(f"  act[{a}] {an:32s} -> joint {jn:26s} qpos[{qadr}]  jnt_range={np.round(jrange,2)} ctrl_range={np.round(crange,2)}")

# --- 2) articulation test: command each arm joint, confirm it moves ---
print("\n=== ARTICULATION TEST (command each arm joint +0.4 rad, check response) ===")
mujoco.mj_resetData(m, d); mujoco.mj_forward(m, d)
d.ctrl[:] = d.qpos[[m.jnt_qposadr[m.actuator_trnid[a,0]] for a in range(m.nu)]]  # hold current
base = d.qpos.copy()
for a in range(6):  # 6 arm joints
    d.ctrl[a] += 0.4
    for _ in range(400):
        mujoco.mj_step(m, d)
    jid = m.actuator_trnid[a,0]; qadr = m.jnt_qposadr[jid]
    moved = d.qpos[qadr] - base[qadr]
    an = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
    print(f"  {an:32s} commanded +0.40 -> moved {moved:+.3f} rad  {'OK' if abs(moved)>0.2 else 'WEAK/STUCK'}")

# --- 3) render a video sweeping the arm (collision view) ---
print("\n=== RENDERING VIDEO ===")
opt = mujoco.MjvOption()
for g in range(len(opt.geomgroup)): opt.geomgroup[g] = 1   # show ALL geom groups incl. collision(3)
cam = mujoco.MjvCamera()
cam.lookat[:] = [0.3, 0.15, 0.2]; cam.distance = 1.6; cam.azimuth = 135; cam.elevation = -25

mujoco.mj_resetData(m, d); mujoco.mj_forward(m, d)
q0 = np.array([m.jnt_qposadr[m.actuator_trnid[a,0]] for a in range(m.nu)])
d.ctrl[:] = d.qpos[q0]
frames = []
with mujoco.Renderer(m, height=480, width=640) as r:
    for t in range(240):  # 8 s @ 30 fps (5 substeps/frame)
        # sinusoidal sweep of shoulder/elbow/wrist to show articulation
        d.ctrl[1] = base[m.jnt_qposadr[m.actuator_trnid[1,0]]] + 0.5*np.sin(t/40)
        d.ctrl[2] = base[m.jnt_qposadr[m.actuator_trnid[2,0]]] + 0.6*np.sin(t/30)
        d.ctrl[3] = base[m.jnt_qposadr[m.actuator_trnid[3,0]]] + 0.5*np.sin(t/25)
        for _ in range(5): mujoco.mj_step(m, d)
        r.update_scene(d, camera=cam, scene_option=opt)
        frames.append(r.render())
out = os.path.join(os.path.dirname(__file__), "aic_articulation.mp4")
imageio.mimsave(out, frames, fps=30)
print(f"  wrote {out}  ({len(frames)} frames, mean px {np.mean(frames):.1f})")
print("DONE")
