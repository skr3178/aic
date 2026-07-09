# Trial reference — board geometry, CAD sources & insights for the next test pass

Working reference gathered 2026-07-09 from the upstream **`media` branch** (`aic-media/`) + the CAD
assets + engine config. Purpose: don't re-derive these next pass. Cross-ref: `LAB_LOG.md`,
`notes/stage2_port_perception_plan.md`, `notes/yolo_finetune_findings.md`.

---

## 1. Board frame & dimensions (authoritative)
- **Frame = `task_board_base`** (media `aic_board_axis_1.png`): origin at a board corner, **X/Y along the
  two edges, Z up**. Board is **flat** (roll=pitch=0; only yaw + xy vary) — confirms M16.
- **Footprint = 0.30 × 0.425 × 0.012 m** — from `aic_assets/models/Task Board Base/model.sdf`
  (`<box><size>0.3 0.425 0.012</size>`). Confirms the SAM known-size fit (0.425 × 0.30).

## 2. Where the geometry is SAVED (3 layers)
| Layer | What | Where |
|---|---|---|
| **1. Port-in-component** (static CAD) | exact port/feature poses within a mount/module | `aic_assets/models/<component>/model.sdf` link/collision `<pose>` |
| **2. Mount/rail-on-board** (procedural) | nominal rail centers + per-trial slide | `aic_engine/src/aic_engine.cpp` + `aic_engine/config/sample_config.yaml` (`task_board_limits`) |
| **3. Composed per-trial pose** (runtime GT) | `task_board/<mount>/<port>_link` vs `task_board_base` | GT TF tree (what we read; M16 → `R_BOARD_OFFSET`) |

- **Mechanical CAD (STL, 3D-print) + assembly guide:** `aic_assets/taskboard_cad/*.stl` + `GUIDE.pdf`
  (5.5 MB). **No Gerbers** — it's a printed mechanical fixture, not a PCB.
- Layer-1 examples: `SC Port/model.sdf` → two SC port faces at **±0.012047 m**; `NIC Card Mount/model.sdf`
  → `nic_card_link` at `(-0.002, -0.01785, 0.0899)`.
- Visual meshes: `.glb` per model; converted `.obj` in `~/ws_aic/aic_local/mujoco_build/meshes/`.

## 3. Rail-slide budget (the SC problem) — from `sample_config.yaml → task_board_limits`
```
nic_rail   ±0.0215 / 0.0234 m  ->  SFP ~±2.3 cm  in 3.6 cm spiral  -> absorbed, no search
sc_rail    ±0.06   / 0.055  m  ->  SC   ~±6 cm    exceeds spiral    -> needs 1-D rail-axis search
mount_rail ±0.09425 m          ->  pick-fixtures (zones 3/4)
```
Card orientation limit ±10°; fixture orientation ±60°. These are the *authoritative* randomization
ranges (M18 budget came from here).

## 4. What our eval (GT-diagnostic) trials targeted
From our own `sweep_dump*/gt.json` (diagnostic GT runs — **read target from Task metadata at runtime,
never hard-code; qualification port numbers differ**):
| Trial | Connector | Target | Port |
|---|---|---|---|
| 1 | SFP | `nic_card_mount_2` | `sfp_port_0` |
| 2 | SFP | `nic_card_mount_4` | `sfp_port_1` |
| 3 | SC  | `sc_port_1` | `sc_port_base` |

Mix = **2 SFP + 1 SC** (2×75 + 75 = /300). Names: `SFP_MODULE`/`SC_PLUG` = grasped plug; SFP =
`nic_card_mount_N/sfp_port_{0,1}`; SC = `sc_port_N/sc_port_base`.

## 5. Key insights from the media (actionable next pass)
- **Magenta pick-zone = GT-free disambiguation lever.** It's a fixed, high-contrast landmark at a
  **known corner in `task_board_base`, diagonally opposite the origin** (media axis + trial renders). We
  already detect it (SAM seed). Its position relative to the board center **resolves the 90°/180°
  quadrant without GT** — closes the open disambiguation gap (currently on a `DISAMBIG_GT` crutch).
- **Eval board is SPARSE.** `aic_board_trial_{1_sfp,3_sc}.png` (labeled Gazebo scenes) show mostly a bare
  dark plate + the *target* module (standing NIC card, or a couple SC ports) + magenta zone — **not** the
  full 5-card board. -> fewer distractors; board-first + magenta anchor is well-suited; the standing
  target module is a prominent cue. (Trial-3 SC is the M13 "wrong connector: green NIC vs blue SC" case.)
- **CAD-derived offsets available.** We can compose nominal port-in-board offsets from Layer 1 (SDF) +
  Layer 2 (engine rail centers) instead of GT-measuring — removes a GT dependency.

## 6. Open items carried into next pass (from LAB_LOG)
1. **YOLO closed-loop depth fix** (immediate): sweep-lock triangulates depth from a narrow baseline ->
   wrong z (`yolo_sweep_score` −51). Fix = back-project YOLO xy to the **flat-board z-plane** (z≈0) +
   robust median, instead of triangulating. -> re-run `PerceptionInsertYOLOSweep`.
2. **90° disambiguation** via the magenta anchor (§5) — replace the `DISAMBIG_GT` crutch.
3. **SC 1-D rail search** for the ±6 cm slide (§3).
4. **YOLO t1 class confusion** / SAM t0 spill -> ensemble (they fail different trials).

## Reference locations
- Board diagrams / renders / gifs: `aic-media/` (clone of the `media` branch; gitignored local ref).
  Key: `aic_board_axis_{1,2,3}.png`, `aic_board_zone_*_legend.png`, `aic_board_trial_{1_sfp,3_sc}.png`.
- CAD STL + guide: `aic_assets/taskboard_cad/`. Model SDFs: `aic_assets/models/<component>/model.sdf`.
- Engine config (limits + trials): `aic_engine/config/sample_config.yaml` (public sample; **never** read
  the vaulted `eval_config.yaml`). Board world: `aic_description/world/aic.sdf`.
- Our data/scripts: `~/aic_data/` (see its `MANIFEST.md`), policies in `~/ws_aic/.../ros/`.
