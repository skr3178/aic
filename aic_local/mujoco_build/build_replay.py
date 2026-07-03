#!/usr/bin/env python3
"""Build a MuJoCo KINEMATIC REPLAY of a successful CheatCode trial.
Inputs:
  full_world.sdf     - static scene + per-link visual meshes & local offsets
  poses_trial1.txt   - gz dynamic_pose/info stream (per-link world poses over time)
  mesh_colors.json   - representative rgba per mesh (from glb PBR)
Outputs:
  replay_scene.xml   - MJCF: static board/enclosure + mocap bodies for moving links + cable beads
  replay_frames.npz  - times + per-mocap-body pose arrays
  mocap_map.json     - mocap body name -> row index in the pose arrays
  replay_check.png   - headless validation render at mid-trial
"""
import os, re, math, json
import numpy as np
import xml.etree.ElementTree as ET
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
MESHDIR = os.path.join(HERE, "meshes")
# prefer the world dumped DURING this recording (matches the trajectory's trial/board config)
SDF = os.path.join(HERE, "full_world_traj.sdf")
if not os.path.exists(SDF):
    SDF = os.path.join(HERE, "full_world.sdf")
POSES = os.path.join(HERE, "poses_trial1.txt")
COLORS = json.load(open(os.path.join(HERE, "mesh_colors.json")))

# UR5e arm meshes come from .dae (no glb color) -> UR signature colors
UR_COLORS = {
    "base": [0.25,0.28,0.32,1], "shoulder": [0.55,0.60,0.66,1], "upperarm": [0.60,0.64,0.70,1],
    "forearm": [0.60,0.64,0.70,1], "wrist1": [0.55,0.60,0.66,1], "wrist2": [0.55,0.60,0.66,1],
    "wrist3": [0.35,0.38,0.42,1],
}
def color_for(base):
    if base in COLORS:
        c = COLORS[base]; return [c[0],c[1],c[2],1.0]
    if base in UR_COLORS: return UR_COLORS[base]
    return [0.6,0.6,0.62,1.0]

def sn(t): return t.split('}')[-1]
def rpy_quat(r,p,y):
    cr,sr=math.cos(r/2),math.sin(r/2); cp,sp=math.cos(p/2),math.sin(p/2); cy,sy=math.cos(y/2),math.sin(y/2)
    return [cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy]
def pose_of(el):
    p=el.find('{*}pose')
    if p is None or not p.text: return [0,0,0],[1,0,0,0]
    v=[float(x) for x in p.text.split()]; return v[0:3], rpy_quat(v[3],v[4],v[5])
def q2R(q):
    w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def qmul(a,b):
    aw,ax,ay,az=a; bw,bx,by,bz=b
    return [aw*bw-ax*bx-ay*by-az*bz,aw*bx+ax*bw+ay*bz-az*by,aw*by-ax*bz+ay*bw+az*bx,aw*bz+ax*by-ay*bx+az*bw]
def compose(pp,pq,cp,cq):
    return (np.array(pp)+q2R(pq)@np.array(cp)).tolist(), qmul(pq,cq)
def obj_for(uri):
    b=os.path.splitext(os.path.basename(uri))[0]
    return b if os.path.exists(os.path.join(MESHDIR,b+".obj")) else None

# ---------- parse the dynamic_pose stream ----------
print("parsing pose stream...")
txt=open(POSES).read()
# split into messages on 'header {'
msgs=re.split(r'(?=^header \{)', txt, flags=re.M)
frames=[]   # (t, {name:(pos,quat)})
def num(block,key):
    if not block: return 0.0
    m=re.search(r'\b'+key+r':\s*(-?[0-9.eE+-]+)', block); return float(m.group(1)) if m else 0.0
for msg in msgs:
    if 'pose {' not in msg: continue
    sm=re.search(r'stamp \{(.*?)\}', msg, re.S)
    sec=num(sm.group(1) if sm else '', 'sec'); nsec=num(sm.group(1) if sm else '', 'nsec')
    t=sec+nsec*1e-9
    d={}
    for block in msg.split('pose {')[1:]:           # one block per pose entry
        nm=re.search(r'name: "([^"]+)"', block)
        if not nm: continue
        name=nm.group(1)
        pm=re.search(r'position \{(.*?)\}', block, re.S)
        om=re.search(r'orientation \{(.*?)\}', block, re.S)
        pb=pm.group(1) if pm else ''; ob=om.group(1) if om else ''
        pos=[num(pb,'x'),num(pb,'y'),num(pb,'z')]
        qx,qy,qz,qw=num(ob,'x'),num(ob,'y'),num(ob,'z'),num(ob,'w')
        n=math.sqrt(qx*qx+qy*qy+qz*qz+qw*qw)
        q=[qw/n,qx/n,qy/n,qz/n] if n>1e-9 else [1,0,0,0]
        d[name]=(pos,q)
    frames.append((t,d))
frames.sort(key=lambda f:f[0])
t0=frames[0][0]; times=np.array([f[0]-t0 for f in frames])
print(f"  {len(frames)} frames, {times[-1]:.1f}s, links seen: {len(set().union(*[set(f[1]) for f in frames]))}")
moving=set().union(*[set(f[1]) for f in frames])

# ---------- parse SDF: static geoms + moving-link local visuals + initial world poses ----------
root=ET.fromstring(open(SDF).read())
world=next((w for w in root if sn(w.tag)=='world'), root)
assets={}; static_geoms=[]; link_visuals={}; link_world0={}
def add_mesh(base):
    f=os.path.join(MESHDIR,base+".obj"); assets[base]=f; return base
def walk(el,wpos,wquat):
    lpos,lquat=pose_of(el); mypos,myquat=compose(wpos,wquat,lpos,lquat)
    name=el.get('name'); tag=sn(el.tag)
    if tag=='link':
        link_world0[name]=(mypos,myquat)
        vises=[]
        for vis in el.findall('{*}visual'):
            ue=vis.find('.//{*}mesh/{*}uri')
            if ue is None or not ue.text: continue
            base=obj_for(ue.text.strip())
            if base is None: continue
            vpos,vquat=pose_of(vis); add_mesh(base)
            vises.append((base,vpos,vquat))
            if name not in moving:  # static -> world-placed geom
                gpos,gquat=compose(mypos,myquat,vpos,vquat)
                static_geoms.append((base,gpos,gquat))
        if vises and name in moving:
            link_visuals[name]=vises
    for child in el:
        if sn(child.tag) in ('model','link'): walk(child,mypos,myquat)
for m in world:
    if sn(m.tag)=='model': walk(m,[0,0,0],[1,0,0,0])
print(f"  static geoms={len(static_geoms)}  moving links w/ meshes={len(link_visuals)}")

# cable rope segments (in stream, no mesh) -> beads
cable_names=sorted([n for n in moving if re.fullmatch(r'link_\d+',n) or n.startswith('cable_end') or n.startswith('cable_connection')])
print(f"  cable bead links={len(cable_names)}")

# ---------- emit MJCF ----------
def mat_name(b): return "m_"+b
asset_mesh="\n".join(f'    <mesh name="{b}" file="{f}" inertia="shell"/>' for b,f in assets.items())
asset_mat="\n".join(
    f'    <material name="{mat_name(b)}" rgba="{color_for(b)[0]:.3f} {color_for(b)[1]:.3f} {color_for(b)[2]:.3f} 1"/>'
    for b in assets)
static_xml="\n".join(
    f'    <geom type="mesh" mesh="{b}" material="{mat_name(b)}" pos="{p[0]:.5f} {p[1]:.5f} {p[2]:.5f}" '
    f'quat="{q[0]:.5f} {q[1]:.5f} {q[2]:.5f} {q[3]:.5f}" contype="0" conaffinity="0" group="1"/>'
    for (b,p,q) in static_geoms)

mocap_map={}; body_blocks=[]; midx=0
# moving mesh links
for name,vises in link_visuals.items():
    p0,q0=link_world0.get(name, ([0,0,0],[1,0,0,0]))
    bn="mc_"+re.sub(r'[^A-Za-z0-9_]','_',name)
    geoms="\n".join(
        f'      <geom type="mesh" mesh="{b}" material="{mat_name(b)}" pos="{vp[0]:.5f} {vp[1]:.5f} {vp[2]:.5f}" '
        f'quat="{vq[0]:.5f} {vq[1]:.5f} {vq[2]:.5f} {vq[3]:.5f}" contype="0" conaffinity="0" group="1"/>'
        for (b,vp,vq) in vises)
    body_blocks.append(
        f'    <body name="{bn}" mocap="true" pos="{p0[0]:.5f} {p0[1]:.5f} {p0[2]:.5f}" '
        f'quat="{q0[0]:.5f} {q0[1]:.5f} {q0[2]:.5f} {q0[3]:.5f}">\n{geoms}\n    </body>')
    mocap_map[bn]=name
# cable beads
for name in cable_names:
    p0,q0=link_world0.get(name, ([0,0,1.3],[1,0,0,0]))
    bn="mc_"+name
    body_blocks.append(
        f'    <body name="{bn}" mocap="true" pos="{p0[0]:.5f} {p0[1]:.5f} {p0[2]:.5f}">\n'
        f'      <geom type="sphere" size="0.007" rgba="0.95 0.55 0.1 1" contype="0" conaffinity="0" group="1"/>\n    </body>')
    mocap_map[bn]=name

mjcf=f'''<mujoco model="aic_replay">
  <compiler meshdir="{MESHDIR}" angle="radian"/>
  <visual><global offwidth="1600" offheight="1200"/><headlight diffuse="0.5 0.5 0.5" ambient="0.4 0.4 0.4"/><quality shadowsize="4096"/></visual>
  <asset>
{asset_mesh}
{asset_mat}
    <texture type="skybox" builtin="gradient" rgb1="0.30 0.38 0.50" rgb2="0.08 0.10 0.14" width="256" height="256"/>
  </asset>
  <worldbody>
    <light pos="0.2 0.2 3.2" dir="0 0 -1" diffuse="0.6 0.6 0.6" castshadow="true"/>
    <light pos="-0.6 0.4 2.6" dir="0.3 -0.2 -1" diffuse="0.35 0.35 0.35"/>
    <geom type="plane" size="4 4 0.1" rgba="0.45 0.45 0.5 1"/>
{static_xml}
{chr(10).join(body_blocks)}
  </worldbody>
</mujoco>'''
open(os.path.join(HERE,"replay_scene.xml"),"w").write(mjcf)
print(f"wrote replay_scene.xml: {len(mocap_map)} mocap bodies, {len(static_geoms)} static geoms")

# ---------- pack frames aligned to mocap bodies (carry-forward missing) ----------
names=list(mocap_map.keys()); link_of=[mocap_map[n] for n in names]
N=len(frames); B=len(names)
arr=np.zeros((N,B,7))
# cable_0's child links (plugs + rope segments) stream their pose RELATIVE to the cable_0 model
# frame, not world -> compose with cable_0's world pose each frame. Arm links belong to ur5e
# (at the world origin) so they need no composition.
cable_children=set()
for n in names:
    ln=mocap_map[n]
    if re.fullmatch(r'link_\d+',ln) or ln.startswith('cable_end') or ln.startswith('cable_connection') \
       or ln in ('sfp_module_link','sc_plug_link','lc_plug_link','sfp_tip_link','sc_tip_link'):
        cable_children.add(ln)
last={n:(link_world0.get(mocap_map[n],([0,0,1],[1,0,0,0]))) for n in names}
for i,(t,d) in enumerate(frames):
    cab=d.get('cable_0')
    for j,n in enumerate(names):
        ln=link_of[j]
        if ln in d:
            p,q=d[ln]
            if ln in cable_children and cab is not None:
                p,q=compose(cab[0],cab[1],p,q)   # relative -> world
            last[n]=(p,q)
        p,q=last[n]; arr[i,j,:3]=p; arr[i,j,3:]=q
np.savez(os.path.join(HERE,"replay_frames.npz"), times=times, poses=arr, names=np.array(names))
json.dump(mocap_map, open(os.path.join(HERE,"mocap_map.json"),"w"), indent=1)
print(f"wrote replay_frames.npz: poses {arr.shape}")

# ---------- headless validation render at mid-trial ----------
model=mujoco.MjModel.from_xml_path(os.path.join(HERE,"replay_scene.xml"))
data=mujoco.MjData(model)
mid=int(N*0.6)
for j,n in enumerate(names):
    bid=model.body(n).id; mid_=model.body_mocapid[bid]
    data.mocap_pos[mid_]=arr[mid,j,:3]; data.mocap_quat[mid_]=arr[mid,j,3:]
mujoco.mj_forward(model,data)
r=mujoco.Renderer(model,height=1200,width=1600)
cam=mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
cam.lookat[:]=[0.1,-0.1,1.2]; cam.distance=1.5; cam.azimuth=145; cam.elevation=-20
r.update_scene(data,cam); px=r.render()
from PIL import Image; Image.fromarray(px).save(os.path.join(HERE,"replay_check.png"))
print(f"wrote replay_check.png (frame {mid}/{N}, t={times[mid]:.1f}s)")
