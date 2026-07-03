# Plan: CheatCode data-collection pipeline (perception + IL dataset)

## Context

We're competing in the Intrinsic **AI for Industry Challenge** (cable insertion, scored in
Gazebo). Analysis of the scoring code ([aic_scoring/src/ScoringTier2.cc](../../../media/skr/storage/YC/aic/aic_scoring/src/ScoringTier2.cc))
shows the task collapses to one hidden variable: **where is the port**. The provided
`CheatCode` policy already solves control to reference-level scores *given* the port/plug
pose from ground-truth TF ([CheatCode.py:187](../../../media/skr/storage/YC/aic/aic_example_policies/aic_example_policies/ros/CheatCode.py)),
but that TF is unavailable at eval time. The winning strategy (approach #7: **learned
perception + analytic control**) is therefore to learn only the port-pose estimate from the
wrist cameras and feed it into CheatCode.

This plan builds the **data engine** that makes #7 (and, as a bonus, #5 imitation learning)
possible: run CheatCode headless across many randomized scenes with `ground_truth:=true`,
and record synchronized **(3 wrist images + proprio) → (ground-truth port/plug pose)** pairs.
Target: **~50 episodes, all connector types (SFP + SC), capturing both perception and IL data.**

Ground-truth TF is free supervision, so this is a cheap, exactly-labeled supervised dataset.
CheatCode's own insertion success is used as the acceptance filter, auto-rejecting any
mis-generated/unreachable scene.

## Connector types (board spec + assets)

Three connector types are **physically present** on the board; only **two are insertion targets** at eval.

| Type | Full name | Where on board | Assets |
|------|-----------|----------------|--------|
| SFP | Small Form-factor Pluggable | Zone 1 — NIC cards, each with 2 SFP ports (up to 5 cards) | SFP Module, SFP Mount, NIC Card |
| SC | Subscriber Connector (fiber) | Zone 2 — optical patch panel, up to 5 SC ports across 2 rails | SC Plug, SC Port, SC Mount |
| LC | Lucent Connector (fiber) | Zones 3–4 — LC plugs staged on mounts (pick area) | LC Plug, LC Mount, `lc_cable`, `lc_sc_cable` |

Cable assets tying them together: **`sfp_sc_cable`** (SFP on one end, SC on the other — the one used in
the trials), plus `lc_sc_cable`, `sc_cable`, `lc_cable`.

**Evaluation targets only two insertions** (qualification spec is explicit):
`SFP_MODULE → SFP_PORT` and `SC_PLUG → SC_PORT`, both on the single `sfp_sc_cable`. LC exists on the
board and in the pick zones / cable variants but is **not an insertion target in qualification** — it is
board layout / clutter / likely later phases. The docs also promise **"no unseen plug or port types will
be presented"** at eval, so a policy only ever inserts SFP and SC.

→ This confirms the dataset scope: **"all connector types" = SFP and SC targets** (both ends of
`sfp_sc_cable`); **LC is captured only as scene clutter/distractor** for image robustness, never as a
target. No LC-cable asset check is needed to proceed.

## Key facts established during exploration

- **Runner already exists:** `~/ws_aic/aic_local/score.sh` runs the `ghcr.io/intrinsic-dev/aic/aic_eval`
  container headless (`gazebo_gui:=false`, `launch_rviz:=false`, `start_aic_engine:=true`,
  `shutdown_on_aic_engine_exit:=true`, custom `aic_engine_config_file:=` via bind-mount, NVIDIA
  EGL for offscreen rendering via `~/ws_aic/aic_local/10_nvidia.json`). The container runs
  sim+engine+zenoh-router; the **policy runs on the host** via `~/.pixi/bin/pixi run ros2 run aic_model ...`.
  Note: `~/ws_aic/install/` is NOT built — the docker + host-pixi path is the working one here.
- **Engine loops all trials in the config sequentially, then exits** ([aic_engine.cpp:567,584](../../../media/skr/storage/YC/aic/aic_engine/src/aic_engine.cpp)),
  resetting the sim between trials (deletes cable+board, re-homes arm). `shutdown_on_aic_engine_exit`
  tears down the whole stack on completion → `docker wait` returns. So **N trials in one config = one container run.**
- **Labels:** with `ground_truth:=true`, launch relays `/scoring/tf → /tf`. Frame templates:
  - Port: `task_board/{target_module_name}/{port_name}_link` (+ `..._link_entrance` for partial-insertion band)
  - Plug: `{cable_name}/{plug_name}_link`
  - Base frame: `base_link`; TCP: `gripper/tcp`. Host TF lookups work (CheatCode does them on host).
- **Sensors:** a single composite topic **`/observations`** (`aic_model_interfaces/msg/Observation`,
  published at 20 Hz only when all 3 cameras are time-aligned, [aic_adapter.cpp:135-158](../../../media/skr/storage/YC/aic/aic_adapter/src/aic_adapter.cpp))
  bundles `left/center/right_image` + `camera_info` + `wrist_wrench` + `joint_states` +
  `controller_state`. Native image res 1152×1024. It does **NOT** contain GT pose (that's TF-only).
- **IL actions** are published on `/aic_controller/pose_commands` (`MotionUpdate`) and already
  captured by the engine's per-trial scoring bag (`bag_<trialid>_<ts>` in `AIC_RESULTS_DIR`),
  alongside `/scoring/tf`, `/joint_states`, wrench, `controller_state`.
- **Nothing existing pairs images with GT pose.** LeRobot recorder = proprio+images only (no GT,
  no wrench); IsaacLab recorder = different stack. So the image↔pose pairing is what we build.
- **Config schema** ([sample_config.yaml](../../../media/skr/storage/YC/aic/aic_engine/config/sample_config.yaml)):
  per-trial `scene.task_board.pose`, per-rail module `entity_present`/`entity_pose.{translation,roll,pitch,yaw}`,
  `cables.<name>.pose.{gripper_offset,roll,pitch,yaw}`, and `tasks.task_1.{cable_name,plug_name,port_name,target_module_name,time_limit}`.
  Robot `home_joint_positions` is fixed → trajectory variety comes purely from **goal (port pose) + start (grasp)**.
- **Connector scope (see "Connector types" section):** eval tests only `SFP_MODULE → SFP_PORT` and
  `SC_PLUG → SC_PORT`, both on `sfp_sc_cable`. So "all types" = **SFP + SC targets**; **LC is
  distractor-only** (never a target). No LC-cable asset check needed.
- **Nominal rail translations exceed `task_board_limits`** (e.g. `nic_rail_0`=0.036 vs stated ±0.023),
  so limits are not directly the absolute bound. Resolution: sample around the **nominal values in
  sample_config ± a conservative delta**, and let CheatCode-success filtering discard any bad scene.

## Files to create (co-located with `score.sh`, under `~/ws_aic/aic_local/collect/`)

Dataset output → `/media/skr/storage/YC/aic_data/perception_v1/` (same storage volume; configurable).

1. **`gen_configs.py`** — randomized trial-config generator.
   - Loads `sample_config.yaml`, copies `scoring:` and `task_board_limits:` blocks verbatim.
   - For episode index `i` (deterministic seed = `i`): pick task type (stratified SFP/SC across the 50);
     jitter board `x,y` (±0.03 m) and `yaw` (±0.15 rad); place the **target module** on its rail at
     `nominal ± delta` translation + yaw jitter; enable a random subset of **distractor** modules at
     random positions (image variety, no path effect); jitter the cable `gripper_offset` (±0.5–1 cm)
     and grasp `roll/pitch/yaw` (±0.1–0.2 rad). Set the `tasks.task_1` block to match the chosen target.
   - **Latin-hypercube** sampling across the continuous "path" dims (board x/y/yaw, target rail
     translation, grasp orientation) + a **min-distance reject** on (port pose, grasp orientation) so
     no two episodes are near-identical.
   - Emits **chunked** configs (default 5 files × 10 trials) to `configs/chunk_{k}.yaml`, plus a
     `manifest.json` listing every episode's seed params + exact port/plug frame names (ordered).

2. **`collector_node.py`** — host rclpy node (run in pixi env, like CheatCode).
   - Subscribes `/observations` (throttle configurable, default 10 Hz) and buffers `/tf`+`/tf_static`
     into a `tf2_ros.Buffer`.
   - Detects the **active trial** as the one whose `{cable_name}/{plug_name}_link` frame is currently
     present in TF (cable is spawned per-trial and deleted on reset → clean episode boundaries);
     matches it to the manifest to know target frames.
   - Per kept `/observations` msg (using its header stamp for TF lookup): decode + downscale the 3
     images (default scale 0.25 → ~288×256), save PNG per camera; look up **port, plug, port-entrance
     pose in `base_link`** and **port pose in each camera optical frame**; record proprio
     (`joint_states`, `wrist_wrench`, `controller_state` TCP pose/vel/error). Reuse
     `CheatCode.calc_gripper_pose` math to also store the convenience "goal gripper pose" label.
   - Appends rows to `episode_{i:04d}/frames.parquet`; writes `episode_{i:04d}/meta.json`
     (intrinsics from `camera_info`, seed params, target frame names). Decode Image with numpy if
     `cv_bridge` is unavailable.

3. **`run_collection.sh`** — orchestrator (adapts the `score.sh` docker block).
   - For each `chunk_{k}.yaml`: `docker run` the aic_eval container headless with `ground_truth:=true`
     and `AIC_RESULTS_DIR=/results` bind-mounted to a **unique per-chunk dir** (preserves `scoring.yaml`
     + engine per-trial bags); on the host start the **CheatCode** model and **`collector_node.py`**;
     `docker wait` for exit; then kill the host model + collector. Repeat per chunk (bounds blast radius).

4. **`finalize.py`** — join + acceptance filter.
   - Parses each chunk's `scoring.yaml` → per-episode `tier_3` (CheatCode success/partial/fail).
   - Marks each episode's split; drops CheatCode failures from the perception training split.
   - Archives the engine per-trial bags (IL actions + proprio + GT tf) under
     `episode_{i:04d}/engine_bag/` for the future ACT/Track-B pipeline.
   - Emits `index.parquet` summarizing all episodes (seed params, target type, frame count, tier_3, split).

## Reused code / assets (do not reinvent)

- Runner + EGL headless setup: `~/ws_aic/aic_local/score.sh`, `~/ws_aic/aic_local/10_nvidia.json`.
- TF lookup + pose-diff + goal-pose math: `CheatCode.calc_gripper_pose` / `insert_cable`
  ([CheatCode.py:72,187](../../../media/skr/storage/YC/aic/aic_example_policies/aic_example_policies/ros/CheatCode.py)).
- Config template + scoring/limits blocks: [sample_config.yaml](../../../media/skr/storage/YC/aic/aic_engine/config/sample_config.yaml).
- Observation schema: [Observation.msg](../../../media/skr/storage/YC/aic/aic_interfaces/aic_model_interfaces/msg/Observation.msg).
- Frame-name templates match scoring: `ScoringTier2.hh` `PortTfName/PlugTfName/PortEntranceTfName`.

## Verification (end-to-end, before scaling to 50)

1. **Smoke test (2 episodes):** generate one 2-trial config (1 SFP + 1 SC), run `run_collection.sh`.
   Confirm: container renders headless (no EGL error, RTF near 1.0); CheatCode inserts (`scoring.yaml`
   `tier_3 > 0` for both); collector produced `episode_0000/` + `episode_0001/` with PNGs and a
   `frames.parquet` whose port-pose columns are non-null.
2. **Label-chain sanity (the make-or-break for #7):** for a few frames, project the port position into
   the image using the stored `camera_info` intrinsics + the recorded port-in-camera pose, overlay a
   dot, and visually confirm it lands on the port. Also verify port-in-`base_link` is ~constant across an
   episode while images change (viewpoint variety present, labels stable).
3. **Scale to 50:** run all chunks, then `finalize.py`; check `index.parquet` — expect ~50 episodes,
   balanced SFP/SC, most with `tier_3 > 0`, total frame count in the low thousands.

## Risks / mitigations

- **Blackwell (sm_120) headless rendering** is empirical; the `score.sh` EGL path is known-good here.
  Fallback if Gazebo rendering fails: disable GlobalIllumination in `aic_description/world/aic.sdf`
  (per `troubleshooting.md`) — note it alters appearance.
- **Disk:** throttle (10 Hz) + downscale (0.25) keeps the image set well under ~1 GB for 50 episodes.
- **LC is intentionally not a target** (per qualification spec) — include LC modules only as scene
  clutter for image robustness; the generator never sets an LC insertion task.
- **Bad/unreachable generated scenes** are caught by the CheatCode-success acceptance filter, not by
  trusting `task_board_limits`.

## Out of scope (later)

- Training the perception network (approach #7) and wiring its output into a CheatCode-derived
  controller. Force-guided search (#3) at contact. ACT/Track-B training from the archived IL bags.
