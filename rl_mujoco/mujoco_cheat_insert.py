"""CheatCode insertion in MuJoCo, targeting the MATCHING port.
The gripper holds an SFP-module assembly; its tip (sfp_tip_link) mates with the SFP
port (sfp_port_0) at only ~20deg -> real connector match. Ports CheatCode's align+descend
against MuJoCo ground truth, driven by Jacobian-transpose impedance. Records 3rd-person view.
"""
import os, numpy as np, mujoco, imageio

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE = os.path.join(HERE, "aic_faithful/scene.xml")
Q_INIT = np.array([0.1597, -1.3542, -1.6648, -1.6933, 1.5710, 1.4110])

m = mujoco.MjModel.from_xml_path(SCENE)
m.opt.timestep = 5e-4
d = mujoco.MjData(m)
TCP  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper_tcp")
PLUG = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "sfp_tip_link")     # held insertable tip
PORT = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "sfp_port_0_link")  # matching port
arm_dofs = [m.jnt_dofadr[m.actuator_trnid[a, 0]] for a in range(6)]

def qsite(sid):
    q=np.zeros(4); mujoco.mju_mat2Quat(q, d.site_xmat[sid]); return q
def mul(a,b):
    r=np.zeros(4); mujoco.mju_mulQuat(r,a,b); return r
def conj(a):
    r=np.zeros(4); mujoco.mju_negQuat(r,a); return r

def quat2mat(q):
    R=np.zeros(9); mujoco.mju_quat2Mat(R,q); return R.reshape(3,3)

def cheat_target(z_offset, ig, p_local):
    """Align held tip to port (q_diff) and place the tip at port + z_offset along the port's
    opening axis. Uses the FIXED tip-in-gripper offset (p_local) so rotating to align does
    not move the tip target."""
    q_port = d.xquat[PORT].copy(); q_plug = d.xquat[PLUG].copy(); q_grip = qsite(TCP)
    q_gt = mul(mul(q_port, conj(q_plug)), q_grip)                # gripper orientation that mates tip->port
    axis = d.xmat[PORT].reshape(3,3)[:, 2]                       # port opening axis (local +z) in world
    tgt_tip = d.xpos[PORT] + axis*z_offset + ig                  # desired tip position
    return tgt_tip - quat2mat(q_gt) @ p_local, q_gt             # gripper pos so the rigidly-held tip lands there

def impedance(tp, tq, ff=None, Kp=(700,700,900), Kr=(150,150,150), Kd=55.0):
    jp=np.zeros((3,m.nv)); jr=np.zeros((3,m.nv)); mujoco.mj_jacSite(m,d,jp,jr,TCP)
    Ja=np.vstack([jp,jr])[:,arm_dofs]
    pe=tp-d.site_xpos[TCP]
    dq=mul(tq, conj(qsite(TCP))); oe=np.zeros(3); mujoco.mju_quat2Vel(oe,dq,1.0)
    F=np.concatenate([np.array(Kp)*pe, np.array(Kr)*oe])
    if ff is not None: F[:3]+=ff                                 # feed-forward insertion force
    w=F - Kd*(Ja@d.qvel[arm_dofs])
    return Ja.T@w + d.qfrc_bias[arm_dofs]

def tip_port_dist(): return float(np.linalg.norm(d.xpos[PLUG]-d.xpos[PORT]))
def align_deg():
    dq=mul(d.xquat[PORT], conj(d.xquat[PLUG])); return float(np.degrees(2*np.arccos(np.clip(abs(dq[0]),-1,1))))
def penetration():  # how far tip is past the port origin along the port opening axis (mm, +=inserted)
    axis=d.xmat[PORT].reshape(3,3)[:,2]; return float(-np.dot(d.xpos[PLUG]-d.xpos[PORT], axis))*1000

mujoco.mj_resetData(m, d); d.qpos[:6]=Q_INIT; mujoco.mj_forward(m, d)
# fixed tip position in the gripper frame (rigid grasp) — computed once
P_LOCAL = quat2mat(qsite(TCP)).T @ (d.xpos[PLUG] - d.site_xpos[TCP])
print("init: tip", np.round(d.xpos[PLUG],3), "sfp_port", np.round(d.xpos[PORT],3), "| tip-port", round(tip_port_dist(),3))

for g in range(m.ngeom):
    if "enclosure" in (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[g]) or ""): m.geom_rgba[g,3]=0.0
opt=mujoco.MjvOption()
for g in range(len(opt.geomgroup)): opt.geomgroup[g]=1
cam=mujoco.MjvCamera(); cam.lookat[:]=[0.2,-0.05,1.3]; cam.distance=1.45; cam.azimuth=160; cam.elevation=-13

CTRL_HZ=20; sub=int((1.0/CTRL_HZ)/m.opt.timestep)
frames=[]; dmin=1e9; ncon_max=0; ig=np.zeros(3); nan=False; z=0.20; pen_max=-1e9
approach, descend = 140, 380     # longer approach so orientation fully aligns before descent
with mujoco.Renderer(m, height=540, width=720) as r:
    for tick in range(approach+descend):
        ff=None
        if tick < approach:
            z = 0.20; ig = np.zeros(3)                           # align & hover above port
        else:
            z = max(-0.03, z-0.0006)                             # descend into port
            err = d.xpos[PORT]-d.xpos[PLUG]; err[2]=0
            ig = np.clip(ig + 0.6*err, -0.05, 0.05)              # lateral integral trim to find hole
            axis = d.xmat[PORT].reshape(3,3)[:,2]
            ff = -axis*12.0                                      # 12 N feed-forward insertion push
        tp, tq = cheat_target(z, ig, P_LOCAL)
        for _ in range(sub):
            d.ctrl[:6]=impedance(tp,tq,ff=ff); d.ctrl[6]=0.0; mujoco.mj_step(m,d)
        if not np.isfinite(d.qpos).all(): nan=True; print("NaN tick",tick); break
        dmin=min(dmin,tip_port_dist()); ncon_max=max(ncon_max,d.ncon); pen_max=max(pen_max,penetration())
        if tick==approach: print(f"  end of approach: align={align_deg():.0f}deg  tip-port={tip_port_dist()*1000:.0f}mm")
        r.update_scene(d,camera=cam,scene_option=opt); frames.append(r.render())

out=os.path.join(HERE,"mujoco_cheatcode_3rdperson.mp4"); imageio.mimsave(out, frames, fps=30)
print(f"wrote {out} ({len(frames)} frames)")
print(f"=== PROXY ===  min tip-port={dmin*1000:.1f}mm  final={tip_port_dist()*1000:.1f}mm  "
      f"max_penetration={pen_max:.1f}mm  final_align={align_deg():.0f}deg  contacts={ncon_max}  cable_stable={not nan}")
print("SEATED (tip inside port)" if pen_max>2.0 else f"not seated; closest {dmin*1000:.0f}mm, penetration {pen_max:.0f}mm")
