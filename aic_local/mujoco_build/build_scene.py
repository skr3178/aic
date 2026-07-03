#!/usr/bin/env python3
"""Assemble a static-visual AIC MuJoCo scene from the transcoded .obj meshes + aic.sdf poses.
Step 1: load-test every .obj individually (asset compile).
Step 2: parse world model/link/visual poses from aic.sdf, emit MJCF referencing the .obj we have,
        compile it, and offscreen-render a PNG. Meshes only available as .dae/.stl (UR5e arm links)
        are skipped and reported.
"""
import os, sys, math, xml.etree.ElementTree as ET
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
MESHDIR = os.path.join(HERE, "meshes")
SDF = os.path.join(HERE, "aic.sdf")

def obj_for(uri):
    """map an SDF mesh uri to a local .obj basename if we have it, else None."""
    base = os.path.splitext(os.path.basename(uri))[0]  # e.g. enclosure_visual
    cand = os.path.join(MESHDIR, base + ".obj")
    return base if os.path.exists(cand) else None

# ---------- Step 1: load-test each obj ----------
print("=== Step 1: load-test each .obj in MuJoCo ===")
objs = sorted(f for f in os.listdir(MESHDIR) if f.endswith(".obj"))
ok, bad = [], []
for f in objs:
    name = os.path.splitext(f)[0]
    xml = f'''<mujoco><asset><mesh name="m" file="{os.path.join(MESHDIR,f)}"/></asset>
    <worldbody><geom type="mesh" mesh="m"/></worldbody></mujoco>'''
    try:
        m = mujoco.MjModel.from_xml_string(xml)
        ok.append((name, m.nmeshvert, m.nmeshface))
    except Exception as e:
        bad.append((name, str(e).splitlines()[-1][:80]))
print(f"  loaded OK: {len(ok)}/{len(objs)}")
for n, nv, nf in ok:
    print(f"    ✓ {n:32s} verts={nv:7d} faces={nf:7d}")
for n, e in bad:
    print(f"    ✗ {n:32s} {e}")

# ---------- Step 2: parse SDF world poses, build MJCF ----------
print("\n=== Step 2: assemble scene from aic.sdf poses ===")

def pose_to_posquat(txt):
    v = [float(x) for x in txt.split()]
    pos = v[0:3]
    r, p, y = v[3:6]
    # rpy -> quat (w,x,y,z), MuJoCo order
    cr, sr = math.cos(r/2), math.sin(r/2)
    cp, sp = math.cos(p/2), math.sin(p/2)
    cy, sy = math.cos(y/2), math.sin(y/2)
    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    yq = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return pos, [w, x, yq, z]

def compose(p_pos, p_quat, c_pos, c_quat):
    """world = parent * child (pos+quat)."""
    def q2R(q):
        w, x, y, z = q
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    def qmul(a, b):
        aw,ax,ay,az=a; bw,bx,by,bz=b
        return [aw*bw-ax*bx-ay*by-az*bz,
                aw*bx+ax*bw+ay*bz-az*by,
                aw*by-ax*bz+ay*bw+az*bx,
                aw*bz+ax*by-ay*bx+az*bw]
    Rp = q2R(p_quat)
    wpos = (np.array(p_pos) + Rp @ np.array(c_pos)).tolist()
    wquat = qmul(p_quat, c_quat)
    return wpos, wquat

tree = ET.parse(SDF)
root = tree.getroot()
def strip_ns(t): return t.split('}')[-1]

world = None
for el in root.iter():
    if strip_ns(el.tag) == 'world':
        world = el; break
if world is None:
    world = root

used, skipped = [], []
assets = {}   # meshname -> file
geoms = []    # (meshname, wpos, wquat, modelname/linkname)

for model in world:
    if strip_ns(model.tag) != 'model':
        continue
    mname = model.get('name')
    # model pose
    mpose_el = model.find('{*}pose')
    mpos, mquat = ([0,0,0],[1,0,0,0])
    if mpose_el is not None and mpose_el.text:
        mpos, mquat = pose_to_posquat(mpose_el.text)
    for link in model.findall('{*}link'):
        lpose_el = link.find('{*}pose')
        lpos, lquat = ([0,0,0],[1,0,0,0])
        if lpose_el is not None and lpose_el.text:
            lpos, lquat = pose_to_posquat(lpose_el.text)
        # world pose of link = model * link
        w1pos, w1quat = compose(mpos, mquat, lpos, lquat)
        for vis in link.findall('{*}visual'):
            mesh_el = vis.find('.//{*}mesh/{*}uri')
            if mesh_el is None or not mesh_el.text:
                continue
            uri = mesh_el.text.strip()
            base = obj_for(uri)
            vpose_el = vis.find('{*}pose')
            vpos, vquat = ([0,0,0],[1,0,0,0])
            if vpose_el is not None and vpose_el.text:
                vpos, vquat = pose_to_posquat(vpose_el.text)
            wpos, wquat = compose(w1pos, w1quat, vpos, vquat)
            if base is None:
                skipped.append((mname, os.path.basename(uri)))
                continue
            key = base
            i = 1
            while key in assets and assets[key] != os.path.join(MESHDIR, base+".obj"):
                key = f"{base}_{i}"; i += 1
            assets[key] = os.path.join(MESHDIR, base+".obj")
            geoms.append((key, wpos, wquat, f"{mname}/{vis.get('name')}"))
            used.append((mname, base))

# emit MJCF
asset_xml = "\n".join(
    f'    <mesh name="{k}" file="{v}" inertia="shell"/>' for k, v in assets.items())
geom_xml = "\n".join(
    f'    <geom type="mesh" mesh="{k}" pos="{wp[0]:.6f} {wp[1]:.6f} {wp[2]:.6f}" '
    f'quat="{wq[0]:.6f} {wq[1]:.6f} {wq[2]:.6f} {wq[3]:.6f}" '
    f'contype="0" conaffinity="0" group="1"/>'   # visual-only: no collision -> no qhull
    for (k, wp, wq, _) in geoms)

mjcf = f'''<mujoco model="aic_static">
  <compiler meshdir="{MESHDIR}" angle="radian"/>
  <visual><global offwidth="1600" offheight="1200"/><headlight diffuse="0.6 0.6 0.6"/></visual>
  <asset>
{asset_xml}
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.4 0.55" rgb2="0.1 0.12 0.16" width="256" height="256"/>
  </asset>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom type="plane" size="3 3 0.1" rgba="0.5 0.5 0.55 1"/>
{geom_xml}
  </worldbody>
</mujoco>'''

out_xml = os.path.join(HERE, "aic_scene.xml")
with open(out_xml, "w") as fh:
    fh.write(mjcf)
print(f"  wrote {out_xml}")
print(f"  placed geoms: {len(geoms)}   distinct meshes: {len(assets)}")
print(f"  SKIPPED (no .obj — .dae/.stl only): {len(skipped)}")
for mn, u in skipped:
    print(f"    - {mn:20s} {u}")

# compile + offscreen render
print("\n=== Step 3: compile + offscreen render ===")
model = mujoco.MjModel.from_xml_path(out_xml)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
print(f"  ✓ compiled: nbody={model.nbody} ngeom={model.ngeom} nmesh={model.nmesh} "
      f"nmeshvert={model.nmeshvert}")

try:
    renderer = mujoco.Renderer(model, height=1200, width=1600)
    # aim a camera at the enclosure center (~z=1.3)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = [0.0, 0.0, 1.3]
    cam.distance = 3.2
    cam.azimuth = 135
    cam.elevation = -20
    renderer.update_scene(data, cam)
    px = renderer.render()
    out_png = os.path.join(HERE, "aic_scene.png")
    try:
        from PIL import Image
        Image.fromarray(px).save(out_png)
    except Exception:
        import imageio.v2 as imageio
        imageio.imwrite(out_png, px)
    print(f"  ✓ rendered -> {out_png}  ({px.shape})")
except Exception as e:
    print(f"  ! offscreen render failed (asset compile still OK): {e}")

print("\nDONE.")
