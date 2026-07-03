#!/usr/bin/env python3
"""Build a MuJoCo scene from a FULL Gazebo world dump (generate_world_sdf), placing every
transcoded .obj mesh at its live world pose. Recurses into nested models (task_board ->
sc_port/nic_card_mount, cable_0 -> plugs). Static visual snapshot of a real CheatCode trial.

Usage: build_full_scene.py [full_world.sdf]
"""
import os, sys, math, xml.etree.ElementTree as ET
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
MESHDIR = os.path.join(HERE, "meshes")
SDF = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "full_world.sdf")
OUT_XML = os.path.join(HERE, "aic_full_scene.xml")
OUT_PNG = os.path.join(HERE, "aic_full_scene.png")

def sn(t): return t.split('}')[-1]

def obj_for(uri):
    base = os.path.splitext(os.path.basename(uri))[0]
    return base if os.path.exists(os.path.join(MESHDIR, base + ".obj")) else None

def rpy_quat(r, p, y):
    cr, sr = math.cos(r/2), math.sin(r/2)
    cp, sp = math.cos(p/2), math.sin(p/2)
    cy, sy = math.cos(y/2), math.sin(y/2)
    return [cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy,
            cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy]

def pose_of(el):
    p = el.find('{*}pose')
    if p is None or not p.text:
        return [0,0,0], [1,0,0,0]
    v = [float(x) for x in p.text.split()]
    return v[0:3], rpy_quat(v[3], v[4], v[5])

def q2R(q):
    w,x,y,z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
def qmul(a,b):
    aw,ax,ay,az=a; bw,bx,by,bz=b
    return [aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
            aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw]
def compose(pp, pq, cp, cq):
    return (np.array(pp) + q2R(pq) @ np.array(cp)).tolist(), qmul(pq, cq)

root = ET.fromstring(open(SDF).read())
world = next((w for w in root if sn(w.tag) == 'world'), root)

assets, geoms, skipped = {}, [], []

def walk(el, wpos, wquat):
    """el is a <model> or <link>; wpos/wquat is its parent's world frame."""
    lpos, lquat = pose_of(el)
    mypos, myquat = compose(wpos, wquat, lpos, lquat)   # this element's world frame
    for child in el:
        tag = sn(child.tag)
        if tag == 'model' or tag == 'link':
            walk(child, mypos, myquat)
        elif tag == 'visual':
            uri_el = child.find('.//{*}mesh/{*}uri')
            if uri_el is None or not uri_el.text:
                continue
            uri = uri_el.text.strip()
            base = obj_for(uri)
            vpos, vquat = pose_of(child)
            gpos, gquat = compose(mypos, myquat, vpos, vquat)
            if base is None:
                skipped.append(os.path.basename(uri)); continue
            key = base; i = 1
            f = os.path.join(MESHDIR, base + ".obj")
            while key in assets and assets[key] != f:
                key = f"{base}_{i}"; i += 1
            assets[key] = f
            geoms.append((key, gpos, gquat))

for m in world:
    if sn(m.tag) == 'model':
        walk(m, [0,0,0], [1,0,0,0])

asset_xml = "\n".join(f'    <mesh name="{k}" file="{v}" inertia="shell"/>' for k,v in assets.items())
geom_xml = "\n".join(
    f'    <geom type="mesh" mesh="{k}" pos="{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" '
    f'quat="{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}" contype="0" conaffinity="0" group="1"/>'
    for (k,p,q) in geoms)

mjcf = f'''<mujoco model="aic_full_trial">
  <compiler meshdir="{MESHDIR}" angle="radian"/>
  <visual><global offwidth="1600" offheight="1200"/><headlight diffuse="0.55 0.55 0.55" ambient="0.35 0.35 0.35"/></visual>
  <asset>
{asset_xml}
    <texture type="skybox" builtin="gradient" rgb1="0.32 0.40 0.52" rgb2="0.10 0.12 0.16" width="256" height="256"/>
  </asset>
  <worldbody>
    <light pos="0.2 0.2 3.2" dir="0 0 -1" diffuse="0.7 0.7 0.7"/>
    <light pos="-0.6 -0.6 2.6" dir="0.3 0.3 -1" diffuse="0.4 0.4 0.4"/>
    <geom type="plane" size="4 4 0.1" rgba="0.5 0.5 0.55 1"/>
{geom_xml}
  </worldbody>
</mujoco>'''
open(OUT_XML, "w").write(mjcf)
print(f"wrote {OUT_XML}: {len(geoms)} geoms, {len(assets)} meshes, skipped={len(skipped)}")
if skipped: print("  skipped:", sorted(set(skipped)))

model = mujoco.MjModel.from_xml_path(OUT_XML)
data = mujoco.MjData(model); mujoco.mj_forward(model, data)
print(f"compiled: ngeom={model.ngeom} nmesh={model.nmesh} nmeshvert={model.nmeshvert}")

r = mujoco.Renderer(model, height=1200, width=1600)
cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
# frame the task board / arm end-effector region (board ~ x0.16 y-0.21 z1.14)
cam.lookat[:] = [0.05, -0.05, 1.25]; cam.distance = 1.8; cam.azimuth = 140; cam.elevation = -18
r.update_scene(data, cam); px = r.render()
try:
    from PIL import Image; Image.fromarray(px).save(OUT_PNG)
except Exception:
    import imageio.v2 as imageio; imageio.imwrite(OUT_PNG, px)
print(f"rendered -> {OUT_PNG} {px.shape}")
