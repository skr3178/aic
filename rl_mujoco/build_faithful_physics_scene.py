"""Make the COMMITTED articulated AIC MuJoCo scene loadable for (state-based) RL,
without the sdformat_mjcf converter and without touching any original/Gazebo files.

Insight: for physics, the committed mjcf/ needs almost no meshes:
  - robot collision = 7 standard UR5e link STLs (available in ur_description)
  - world collision = 100% primitives (box/cylinder/plane) -> no mesh files
  - all .obj meshes are VISUAL (contype=0) and physically inert -> strippable

So: copy the 3 committed XMLs + the 7 real collision STLs into a fresh dir, strip the
inert visual geoms/assets, and load. Keeps the real UR5e + gripper + FT + cable(plugin)
+ task board/port geometry. No visuals (fine for state RL).
"""
import os, shutil, xml.etree.ElementTree as ET

SRC = "/media/skr/storage/aic/aic_utils/aic_mujoco/mjcf"
UR = "/home/skr/ws_aic/src/Universal_Robots_ROS2_Description/meshes/ur5e/collision"
OUT = "/media/skr/storage/aic/rl_mujoco/aic_faithful"
os.makedirs(OUT, exist_ok=True)

# committed collision-mesh filenames -> canonical UR5e collision STL (same link)
COLLISION_MAP = {
    "base-d994a04ff52ddcdf6d91bb6d4fcfff9b1425c18e.stl": "base.stl",
    "shoulder-a83506202923a888b5b5d3c371edb9e13f236336.stl": "shoulder.stl",
    "upperarm-03c2bdced333d049d7ff8fbd721d7d8230d808bc.stl": "upperarm.stl",
    "forearm-95860729b3e4915567db264330b9ab276f0c8308.stl": "forearm.stl",
    "wrist1-d0aff1dbf858639fbad7a22318ecfa5fba436f30.stl": "wrist1.stl",
    "wrist2-777eba24b0996c38308995e2e7740d22a77ee8a8.stl": "wrist2.stl",
    "wrist3-f7f56f92b9aad1eab55349b28bbe751f72de21da.stl": "wrist3.stl",
}

# 1) copy the 3 committed XMLs verbatim, then strip visuals from the two included ones
for f in ("scene.xml", "aic_robot.xml", "aic_world.xml"):
    shutil.copy(os.path.join(SRC, f), os.path.join(OUT, f))

# 2) supply the 7 real collision STLs under the exact names the committed XML expects
for want, have in COLLISION_MAP.items():
    shutil.copy(os.path.join(UR, have), os.path.join(OUT, want))


def strip_visuals(path):
    tree = ET.parse(path)
    root = tree.getroot()
    parents = {c: p for p in root.iter() for c in p}
    removed_meshes = set()

    # assets: drop every non-collision mesh (.obj), and all textures/materials
    for asset in root.findall("asset"):
        for el in list(asset):
            tag = el.tag
            if tag == "mesh":
                fn = el.get("file", "")
                if fn not in COLLISION_MAP:        # keep only the 7 collision STLs
                    removed_meshes.add(el.get("name"))
                    asset.remove(el)
            elif tag in ("texture", "material"):    # only used by visual geoms; files missing
                asset.remove(el)

    # geoms: drop visual geoms (contype=0) and any referencing a removed mesh;
    # strip material= from survivors so they don't reference deleted materials
    n_geom_removed = 0
    for geom in list(root.iter("geom")):
        vis = geom.get("contype") == "0"
        refs_removed = geom.get("mesh") in removed_meshes
        if vis or refs_removed:
            parents[geom].remove(geom)
            n_geom_removed += 1
        else:
            if "material" in geom.attrib:
                del geom.attrib["material"]
    tree.write(path)
    return len(removed_meshes), n_geom_removed


for f in ("aic_robot.xml", "aic_world.xml"):
    nm, ng = strip_visuals(os.path.join(OUT, f))
    print(f"{f}: removed {nm} visual meshes, {ng} visual geoms")

# 3) try to load
import mujoco
m = mujoco.MjModel.from_xml_path(os.path.join(OUT, "scene.xml"))
print("\n=== LOADED OK ===")
print(f"nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody} ngeom={m.ngeom} "
      f"nsensor={m.nsensor} nplugin={m.nplugin} neq={m.neq}")
acts = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
print("actuators:", acts)
d = mujoco.MjData(m)
mujoco.mj_step(m, d)
print("stepped OK; qpos[:6]=", [round(float(x), 3) for x in d.qpos[:6]])
