#!/usr/bin/env python3
"""gen_configs.py — generate diverse CheatCode scene configs for perception + IL data collection.

Source of truth: SAMPLE_DIVERSITY.md (co-located).

Method:
  * Perturb-from-known-good template: SFP = trial_1, SC = trial_3 from sample_config.yaml.
    Only *pose* fields are jittered; module/rail/port/cable NAMES are copied verbatim.
  * Group-A (path-changing) continuous dims are spread with a Latin-hypercube pool +
    farthest-point selection -> the chosen episodes span the space, no two near-identical.
  * Group-B (pixels-only) distractor toggles/jitter come from a per-episode seed (index i).
  * Deterministic: fixed master seed for the LHS/FPS, index-seeded per-episode choices ->
    the whole 50-episode set is reproducible/regenerable.
  * CheatCode-success acceptance is applied later (at collection), not here.

Outputs (under this dir):
  configs/chunk_{k}.yaml   (5 files x 10 trials; bounds blast radius)
  manifest.json            (every episode: seed params + exact port/plug frame names, ordered)
"""
import copy, json, math, os, random
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "/home/skr/ws_aic/src/aic"
SAMPLE = os.path.join(REPO, "aic_engine", "config", "sample_config.yaml")
OUT_CFG = os.path.join(HERE, "configs")
OUT_MANIFEST = os.path.join(HERE, "manifest.json")

N_TOTAL = 50
PER_CHUNK = 10
N_SFP = 25
N_SC = 25
POOL = 400                 # LHS candidate pool per stratum
MASTER_SEED = 12345

# ---- jitter ranges (SAMPLE_DIVERSITY.md section 2) ----
BOARD_XY   = 0.03          # board.pose.x/y  +/- m
BOARD_YAW  = 0.15          # board.pose.yaw  +/- rad
PORT_YAW   = 0.15          # target module entity_pose.yaw +/- rad
GRASP_XYZ  = 0.008         # gripper_offset x/y/z +/- m  (0.005-0.01)
GRASP_RPY  = 0.15          # grasp roll/pitch/yaw +/- rad (0.10-0.20)
TRANS_DELTA = {"nic": 0.015, "sc": 0.03}   # target rail translation +/- around nominal

# continuous "path" dims spread by LHS/FPS (SAMPLE_DIVERSITY section 4.2)
CONT = ["board_dx", "board_dy", "board_dyaw", "trans_frac",
        "grasp_droll", "grasp_dpitch", "grasp_dyaw"]
HALF = np.array([BOARD_XY, BOARD_XY, BOARD_YAW, 1.0, GRASP_RPY, GRASP_RPY, GRASP_RPY])


def lhs(n, d, rng):
    """Latin-hypercube samples in [0,1]^d, shape (n,d)."""
    cut = np.linspace(0.0, 1.0, n + 1)
    pts = cut[:n, None] + rng.random((n, d)) * (1.0 / n)
    for j in range(d):
        rng.shuffle(pts[:, j])
    return pts


def fps(feat, k):
    """Farthest-point selection: indices of k rows of `feat` with maximal min-pairwise spread."""
    d2 = np.sum((feat - feat[0]) ** 2, axis=1)
    chosen = [0]
    while len(chosen) < k:
        i = int(np.argmax(d2))
        chosen.append(i)
        d2 = np.minimum(d2, np.sum((feat - feat[i]) ** 2, axis=1))
    return chosen


def make_stratum(kind, n, rng):
    """Return n parameter dicts for a connector stratum, spread via LHS pool + FPS."""
    pool = lhs(POOL, len(CONT), rng)                 # (POOL, 7) in [0,1]
    u = pool * 2.0 - 1.0                             # -> [-1,1]
    feat = u                                         # already normalized per-dim -> good spread metric
    pick = fps(feat, n)
    out = []
    for idx in pick:
        v = u[idx] * HALF                            # scale to physical ranges (trans_frac stays [-1,1])
        out.append({
            "board_dx": float(v[0]), "board_dy": float(v[1]), "board_dyaw": float(v[2]),
            "trans_frac": float(u[idx][3]),          # keep normalized; scaled to delta per kind below
            "grasp_droll": float(v[4]), "grasp_dpitch": float(v[5]), "grasp_dyaw": float(v[6]),
        })
    # report achieved spread
    fsel = feat[pick]
    dmin = min(np.sqrt(((fsel[i] - fsel[j]) ** 2).sum())
               for i in range(len(fsel)) for j in range(i + 1, len(fsel)))
    print(f"  [{kind}] selected {n}/{POOL}; min pairwise feature distance = {dmin:.3f}")
    return out


def jitter_distractors(scene, target_rail_key, ep_rng):
    """Group-B: toggle/jitter NON-target mounts already named in the template (no new names)."""
    tb = scene["task_board"]
    for key, node in tb.items():
        if key == "pose" or key == target_rail_key:
            continue
        if not isinstance(node, dict) or "entity_pose" not in node:
            continue
        if not node.get("entity_present", False):
            continue
        # ~30% chance to remove this distractor (changes pixels, never touches target/path)
        if ep_rng.random() < 0.30:
            tb[key] = {"entity_present": False}
            continue
        ep = node["entity_pose"]
        ep["translation"] = round(ep.get("translation", 0.0) + ep_rng.uniform(-0.02, 0.02), 5)
        ep["yaw"] = round(ep.get("yaw", 0.0) + ep_rng.uniform(-0.10, 0.10), 5)


def build_episode(gidx, kind, params, templates, sfp_port=None):
    """Deep-copy the right template and apply Group-A pose jitters + Group-B distractor variety."""
    ep_rng = random.Random(gidx)                     # per-episode seed = global index
    trial = copy.deepcopy(templates[kind])
    scene = trial["scene"]
    board = scene["task_board"]

    # --- Group A: board pose (z/roll/pitch kept flat) ---
    board["pose"]["x"] = round(board["pose"]["x"] + params["board_dx"], 5)
    board["pose"]["y"] = round(board["pose"]["y"] + params["board_dy"], 5)
    board["pose"]["yaw"] = round(board["pose"]["yaw"] + params["board_dyaw"], 5)

    # --- Group A: target rail translation (nominal +/- delta) + target port yaw ---
    if kind == "sfp":
        target_rail_key = "nic_rail_0"
        delta = TRANS_DELTA["nic"]
        cable_key = "cable_0"
    else:
        target_rail_key = "sc_rail_1"
        delta = TRANS_DELTA["sc"]
        cable_key = "cable_1"
    tr = board[target_rail_key]["entity_pose"]
    nominal_trans = tr["translation"]
    tr["translation"] = round(nominal_trans + params["trans_frac"] * delta, 5)
    tr["yaw"] = round(tr.get("yaw", 0.0) + ep_rng.uniform(-PORT_YAW, PORT_YAW), 5)

    # --- Group A: cable grasp offset + orientation (moves plug start + required alignment) ---
    cpose = scene["cables"][cable_key]["pose"]
    cpose["gripper_offset"]["x"] = round(cpose["gripper_offset"]["x"] + ep_rng.uniform(-GRASP_XYZ, GRASP_XYZ), 6)
    cpose["gripper_offset"]["y"] = round(cpose["gripper_offset"]["y"] + ep_rng.uniform(-GRASP_XYZ, GRASP_XYZ), 6)
    cpose["gripper_offset"]["z"] = round(cpose["gripper_offset"]["z"] + ep_rng.uniform(-GRASP_XYZ, GRASP_XYZ), 6)
    cpose["roll"]  = round(cpose["roll"]  + params["grasp_droll"], 5)
    cpose["pitch"] = round(cpose["pitch"] + params["grasp_dpitch"], 5)
    cpose["yaw"]   = round(cpose["yaw"]   + params["grasp_dyaw"], 5)

    # --- Group A (SFP only): which of the 2 ports on the card is the target ---
    task = trial["tasks"]["task_1"]
    if kind == "sfp":
        task["port_name"] = sfp_port                  # balanced 0/1 by SFP ordinal (set by caller)
    port_name = task["port_name"]

    # --- Group B: distractor variety (pixels only) ---
    jitter_distractors(scene, target_rail_key, ep_rng)

    # --- manifest frame names (match ScoringTier2 / CheatCode templates) ---
    tmn = task["target_module_name"]
    port_frame = f"task_board/{tmn}/{port_name}_link"
    entrance_frame = f"{port_frame}_entrance"
    plug_frame = f"{task['cable_name']}/{task['plug_name']}_link"
    meta = {
        "episode": gidx, "type": kind,
        "cable_name": task["cable_name"], "plug_name": task["plug_name"],
        "port_name": port_name, "target_module_name": tmn,
        "port_frame": port_frame, "port_entrance_frame": entrance_frame, "plug_frame": plug_frame,
        "params": {
            "board_pose": {k: board["pose"][k] for k in ("x", "y", "yaw")},
            "target_translation": tr["translation"], "target_yaw": tr["yaw"],
            "grasp_offset": dict(cpose["gripper_offset"]),
            "grasp_rpy": {"roll": cpose["roll"], "pitch": cpose["pitch"], "yaw": cpose["yaw"]},
        },
    }
    return trial, meta


def main():
    with open(SAMPLE) as f:
        sample = yaml.safe_load(f)
    templates = {"sfp": sample["trials"]["trial_1"], "sc": sample["trials"]["trial_3"]}
    common = {k: sample[k] for k in ("scoring", "task_board_limits", "robot")}

    rng = np.random.default_rng(MASTER_SEED)
    print("Generating spread parameter sets (LHS + farthest-point):")
    sfp_params = make_stratum("sfp", N_SFP, rng)
    sc_params = make_stratum("sc", N_SC, rng)

    # interleave SFP/SC so every chunk of 10 is ~5+5 balanced
    order = []
    si = ci = 0
    for _ in range(N_TOTAL):
        if (len(order) % 2 == 0 and si < N_SFP) or ci >= N_SC:
            order.append(("sfp", sfp_params[si])); si += 1
        else:
            order.append(("sc", sc_params[ci])); ci += 1

    os.makedirs(OUT_CFG, exist_ok=True)
    manifest = []
    chunks = {}
    sfp_ord = 0
    for gidx, (kind, params) in enumerate(order):
        sfp_port = None
        if kind == "sfp":
            sfp_port = "sfp_port_0" if sfp_ord % 2 == 0 else "sfp_port_1"   # balanced by SFP ordinal
            sfp_ord += 1
        trial, meta = build_episode(gidx, kind, params, templates, sfp_port)
        k = gidx // PER_CHUNK
        t = gidx % PER_CHUNK + 1
        meta["chunk"] = k
        meta["trial"] = f"trial_{t}"
        chunks.setdefault(k, {})[f"trial_{t}"] = trial
        manifest.append(meta)

    for k, trials in chunks.items():
        doc = {"scoring": common["scoring"],
               "task_board_limits": common["task_board_limits"],
               "trials": trials,
               "robot": common["robot"]}
        path = os.path.join(OUT_CFG, f"chunk_{k}.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=False)
        print(f"  wrote {path}  ({len(trials)} trials)")

    with open(OUT_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    n_sfp = sum(1 for m in manifest if m["type"] == "sfp")
    n_p0 = sum(1 for m in manifest if m["type"] == "sfp" and m["port_name"] == "sfp_port_0")
    print(f"\nmanifest.json: {len(manifest)} episodes  "
          f"(SFP {n_sfp} [port_0={n_p0}, port_1={n_sfp-n_p0}], SC {len(manifest)-n_sfp}) "
          f"across {len(chunks)} chunks x {PER_CHUNK} trials")


if __name__ == "__main__":
    main()
