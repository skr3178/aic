# Workspace Map — where the functional pieces live

> **This folder (`/media/skr/storage/aic`) is the reference/source copy of the AIC toolkit.**
> The **built, working environment is intentionally NOT here** — it lives under `~/ws_aic`
> (a separate colcon/pixi workspace with the 13 GB pixi env already built). This was kept as-is
> after moving the repo, to avoid a slow env rebuild. Run everything against the `~/ws_aic` paths
> below.

## Functional bit locations (source of truth)

| Thing | Location | Status |
|---|---|---|
| Built **pixi env** + working repo copy | `~/ws_aic/src/aic` (`.pixi/envs/default`) | ✅ works; all commands target this |
| Collection scripts + generated configs | `~/ws_aic/aic_local/collect/` | ✅ |
| `score.sh`, `gen_configs.py`, `10_nvidia.json`, `aicrun` | `~/ws_aic/aic_local/` (`gen_configs.py` under `collect/`) | ✅ paths valid |
| Vaulted `eval_config.yaml` (real leaderboard scenes) | `/media/skr/storage/YC/vault/eval_config.yaml` | ✅ |
| Eval Docker image | `ghcr.io/intrinsic-dev/aic/aic_eval:latest` (pulled locally) | ✅ |
| pixi binary | `/home/skr/.pixi/bin/pixi` | ✅ |

> Note: `~/ws_aic/src/aic` and this folder are **independent copies**. Editing files *here* does
> **not** affect runs — runs read `~/ws_aic/src/aic`. Keep code edits in `~/ws_aic` (or copy over).

## Key files & what they do (all under `~/ws_aic/aic_local/`)

- **`score.sh`** — one-command self-scoring of any policy against the Gazebo eval container
  (headless, GPU-rendered via the NVIDIA EGL ICD, single-`aic_model`-node hardened). Writes
  `~/aic_results/<run>/scoring.yaml`.
- **`10_nvidia.json`** — NVIDIA EGL vendor ICD injected into the container so Gazebo renders on the
  GPU (without it: CPU/llvmpipe → RTF ≈ 0.03, ~30× slower).
- **`aicrun`** — runs a command in a clean Kilted pixi env (strips leaked host-ROS Humble vars).
- **`collect/gen_configs.py`** — generates 50 diverse CheatCode scene configs (25 SFP / 25 SC),
  perturb-from-template + Latin-hypercube spread + farthest-point de-dup. Output:
  `collect/configs/chunk_{0..4}.yaml` (5×10 trials, engine-validated: "parsed 10/10") + `collect/manifest.json`.
- **`collect/SAMPLE_DIVERSITY.md`** — the diversity methodology (source of truth for `gen_configs.py`).
- **`collect/PLAN.md`** — full data-collection pipeline plan.
- **`CHEATCODE_RUNBOOK.md`** (in this folder) — standalone runbook to run the CheatCode baseline
  (uses the `~/ws_aic` paths above).

## Quick commands

```bash
# self-score a policy (floor / ceiling baselines)
~/ws_aic/aic_local/score.sh aic_example_policies.ros.WaveArm  false eval wavearm
~/ws_aic/aic_local/score.sh aic_example_policies.ros.CheatCode true  eval cheatcode

# (re)generate the diverse training configs
cd ~/ws_aic/aic_local/collect && python3 gen_configs.py

# run a policy manually in the clean env
~/ws_aic/aic_local/aicrun ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true -p policy:=aic_example_policies.ros.CheatCode
```

## Project status (milestones)

> **Living stage tracker with ✅/⏳ per stage: `RUN_PLAN.md`** (kept current; this list below is the long-form history).

- **M0 ✅** env: pixi + ROS 2 Kilted; torch 2.12.1+cu130 on Blackwell (sm_120); lerobot ACT imports.
- **M1 ✅** self-scoring validated: WaveArm 37.48 (floor), CheatCode 279.37 (ceiling, inserts on all 3 trials).
- **M2a ✅** `gen_configs.py` → 50 diverse configs + manifest (engine-parse validated).
- **M2b ✅** collection pipeline → 49 kept episodes in `perception_v1/` (`frames.parquet` = images↔GT port/plug pose, `il_frames.parquet`, `index.parquet`).
- **M3 ✅** perception net trained: port anchor ~9 mm @ close range. `~/aic_data/m3_perception_run/best.pt`.
- **M3.5 ✅** dual-anchor net (port **and** plug from vision; plug labels derived, no re-collection):
  port ~9 mm / plug ~8 mm @ close range, relative plug→port ~12 mm. Details → `perception_results.md`.
  `~/aic_data/m35_dual_run/best.pt`.
- **M4 ✅ (first light)** `PerceptionInsert` policy wired & scored GT-free: **−37.6** (trial_1 reached
  0.07 m → first GT-free Tier-3 proximity credit; score dominated by removable collision penalties +
  eval-scene generalization gap on 2 trials). Policy at `aic_example_policies/.../ros/PerceptionInsert.py`
  (also copied into the pixi site-packages — copy-install, re-copy after edits!). See `baseline_scores.md`.
- **M5 ✅** v2 policy (force-stop + spiral search + pose logging + per-trial state reset) and GT-diagnostic
  run: **far misses = training-coverage gap, not a frame bug** (61 mm in-range vs 335–372 mm out-of-range;
  v1 configs never varied the target rail — all 50 eps target `nic_card_mount_0`/`sc_port_1`).
  See `baseline_scores.md` (M5 diagnostic) and `~/aic_data/pi_pose_log.jsonl`.
- **M6 ⏳** coverage-fix dataset `perception_v2`: 160 eps stratified over **all 5 nic rails × 2 SFP ports +
  2 sc rails**, full published rail limits, board ±0.12 m / ±0.6 rad; retrain (optionally port-conditioned);
  re-score. **Vaulted eval config is never read** (fair play). Full plan: `collect/M6_PLAN.md`. *Next.*

## Gotchas (learned the hard way)

- Exactly **one** `aic_model` node may exist per run, or the engine fails Tier 1
  ("More than one node with name 'aic_model'"). Kill stale policy procs before launching.
- GPU rendering requires `NVIDIA_DRIVER_CAPABILITIES=all` **and** mounting `10_nvidia.json` as the
  EGL ICD; otherwise it silently CPU-renders (RTF ≈ 0.03).
- For a collection runner, launch the host policy **fully detached** (put it in a `.sh` file +
  `setsid`/`nohup`) — the harness kills foreground (120 s) and managed-background jobs.
- Native build is not viable on this host (Ubuntu 22.04; ROS Kilted needs 24.04) → use the Docker path.
</content>
