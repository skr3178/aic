#!/usr/bin/env python3
"""Play the recorded CheatCode trial as a looping kinematic animation in the MuJoCo viewer.
Each mocap body is driven to its recorded world pose per frame; loops at real-time * SPEED.
"""
import os, time, json
import numpy as np
import mujoco, mujoco.viewer

H = os.path.dirname(os.path.abspath(__file__))
SPEED = float(os.environ.get("SPEED", "1.0"))   # 1.0 = real time (~36s/loop)

model = mujoco.MjModel.from_xml_path(os.path.join(H, "replay_scene.xml"))
data = mujoco.MjData(model)
d = np.load(os.path.join(H, "replay_frames.npz"), allow_pickle=True)
times, poses, names = d["times"], d["poses"], list(d["names"])
mcid = [model.body_mocapid[model.body(n).id] for n in names]

# Trim to the FIRST insertion cycle: stop at the teleport-reset between trials
# (end-effector z jumps back up sharply after the seated hold).
ee_z = poses[:, names.index("mc_ati_tool_link"), 2]
reset = len(times)
lowered = False
for i in range(1, len(times)):
    if ee_z[i] < 1.55:
        lowered = True
    if lowered and ee_z[i] - ee_z[i-1] > 0.05:   # sharp jump up = trial reset
        reset = i
        break
times = times[:reset]; poses = poses[:reset]
N = len(times)

def set_frame(i):
    for j, mc in enumerate(mcid):
        data.mocap_pos[mc] = poses[i, j, :3]
        data.mocap_quat[mc] = poses[i, j, 3:]
    mujoco.mj_forward(model, data)

set_frame(0)
with mujoco.viewer.launch_passive(model, data) as v:
    v.cam.lookat[:] = [0.14, -0.11, 1.26]
    v.cam.distance = 1.35
    v.cam.azimuth = 140
    v.cam.elevation = -15
    print(f"playing {N} frames, {times[-1]:.1f}s/loop at {SPEED}x")
    while v.is_running():
        loop_start = time.time()
        for i in range(N):
            if not v.is_running():
                break
            set_frame(i)
            v.sync()
            # real-time pacing
            target = times[i] / SPEED
            ahead = target - (time.time() - loop_start)
            if ahead > 0:
                time.sleep(min(ahead, 0.1))
        time.sleep(0.4)  # brief pause at end before looping
