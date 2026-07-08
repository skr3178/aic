"""Isolate and confirm the UR5e arm+gripper articulation on the faithful AIC scene.
Builds a cable-free variant (the cable blows up standalone), converts the torque
actuators to position servos, drives the arm to joint targets, verifies tracking,
and renders a video. Proves the arm articulation is sound independent of the cable.
"""
import os, numpy as np, xml.etree.ElementTree as ET, mujoco, imageio

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "aic_faithful")
OUT = os.path.join(HERE, "aic_faithful_nocable")
os.makedirs(OUT, exist_ok=True)
import shutil
for f in ("scene.xml", "aic_robot.xml", "aic_world.xml"):
    shutil.copy(os.path.join(SRC, f), os.path.join(OUT, f))
for stl in os.listdir(SRC):
    if stl.endswith(".stl"): shutil.copy(os.path.join(SRC, stl), os.path.join(OUT, stl))

# --- strip the cable + its plugin + the plug weld from the world (keep gripper mimic) ---
wp = os.path.join(OUT, "aic_world.xml")
tree = ET.parse(wp); root = tree.getroot()
parents = {c: p for p in root.iter() for c in p}
for ext in root.findall("extension"): root.remove(ext)          # cable plugin decl
removed = set()
for body in list(root.iter("body")):
    if body.get("name") == "cable_end_0":                       # whole cable chain
        removed = {b.get("name") for b in body.iter("body")}
        parents[body].remove(body)
for eq in root.findall("equality"):
    for w in eq.findall("weld"): eq.remove(w)                   # lc_plug->tool weld
# drop contact excludes that reference any removed cable body
for con in root.findall("contact"):
    for ex in list(con):
        if ex.get("body1") in removed or ex.get("body2") in removed:
            con.remove(ex)
tree.write(wp)

m = mujoco.MjModel.from_xml_path(os.path.join(OUT, "scene.xml"))
d = mujoco.MjData(m)
print(f"cable-free scene: nq={m.nq} nu={m.nu} nbody={m.nbody} nplugin={m.nplugin}")

# --- turn the 6 arm torque-motors into position servos: force = kp*(ctrl - q) - kv*qd ---
kp, kv = 300.0, 30.0
for a in range(6):
    m.actuator_gaintype[a] = mujoco.mjtGain.mjGAIN_FIXED
    m.actuator_biastype[a] = mujoco.mjtBias.mjBIAS_AFFINE
    m.actuator_gainprm[a, :3] = [kp, 0, 0]
    m.actuator_biasprm[a, :3] = [0, -kp, -kv]

# --- articulation test: drive to a sequence of joint targets, verify tracking ---
mujoco.mj_resetData(m, d); mujoco.mj_forward(m, d)
q_home = d.qpos[:6].copy()
targets = [
    q_home + np.array([ 0.5, 0, 0, 0, 0, 0]),
    q_home + np.array([ 0.5,-0.5, 0.6, 0, 0, 0]),
    q_home + np.array([-0.5,-0.3, 0.4, 0.5, 0.5, 1.0]),
    q_home,
]
print("\n=== ARM ARTICULATION (position servo, verify each joint tracks target) ===")
d.ctrl[:6] = q_home
for k, tgt in enumerate(targets):
    d.ctrl[:6] = tgt
    for _ in range(600): mujoco.mj_step(m, d)
    err = np.abs(d.qpos[:6] - tgt)
    print(f"  target {k}: max joint tracking error = {err.max():.4f} rad  {'OK' if err.max()<0.05 else 'OFF'}")

# --- render a video of the arm sweeping through the targets ---
opt = mujoco.MjvOption()
for g in range(len(opt.geomgroup)): opt.geomgroup[g] = 1
cam = mujoco.MjvCamera(); cam.lookat[:] = [0.3, 0.15, 0.3]; cam.distance = 1.7; cam.azimuth = 130; cam.elevation = -20
mujoco.mj_resetData(m, d); mujoco.mj_forward(m, d); d.ctrl[:6] = d.qpos[:6].copy()
frames = []
seq = [q_home] + targets
with mujoco.Renderer(m, height=480, width=640) as r:
    for i in range(len(seq)-1):
        a0, a1 = seq[i], seq[i+1]
        for s in range(60):
            d.ctrl[:6] = a0 + (a1-a0)*(s/60)
            for _ in range(4): mujoco.mj_step(m, d)
            r.update_scene(d, camera=cam, scene_option=opt); frames.append(r.render())
out = os.path.join(HERE, "aic_arm_articulation.mp4")
imageio.mimsave(out, frames, fps=30)
print(f"\nwrote {out}  ({len(frames)} frames, mean px {np.mean(frames):.1f})")
print("DONE")
