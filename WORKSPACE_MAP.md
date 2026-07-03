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

- **M0 ✅** env: pixi + ROS 2 Kilted; torch 2.12.1+cu130 on Blackwell (sm_120); lerobot ACT imports.
- **M1 ✅** self-scoring validated: WaveArm 37.48 (floor), CheatCode 279.37 (ceiling, inserts on all 3 trials).
- **M2a ✅** `gen_configs.py` → 50 diverse configs + manifest (engine-parse validated).
- **M2b ⏳** collection pipeline: `run_collection.sh` + `collector_node.py` (pair 3 wrist images ↔ ground-truth port/plug pose) + `finalize.py` (CheatCode-success acceptance filter). *Next.*
- **M3 ⏳** train perception / ACT policy on the collected data.
- **M4 ⏳** wrap the checkpoint as a policy; self-score with `ground_truth:=false`.

## Gotchas (learned the hard way)

- Exactly **one** `aic_model` node may exist per run, or the engine fails Tier 1
  ("More than one node with name 'aic_model'"). Kill stale policy procs before launching.
- GPU rendering requires `NVIDIA_DRIVER_CAPABILITIES=all` **and** mounting `10_nvidia.json` as the
  EGL ICD; otherwise it silently CPU-renders (RTF ≈ 0.03).
- For a collection runner, launch the host policy **fully detached** (put it in a `.sh` file +
  `setsid`/`nohup`) — the harness kills foreground (120 s) and managed-background jobs.
- Native build is not viable on this host (Ubuntu 22.04; ROS Kilted needs 24.04) → use the Docker path.
</content>
