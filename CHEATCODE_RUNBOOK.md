# CheatCode Baseline — Run Guide

Runs the **CheatCode** ground-truth policy against the Gazebo eval container and self-scores it
locally (no submission portal). Expected result: **total ≈ 279 / 300** — cable inserted
successfully on all 3 trials (Tier 3 = 75 each).

CheatCode is a *training/debugging* baseline: it reads the port pose from ground-truth TF, so it
is **not** a valid competition submission — but it's the reference "successful policy" and the
expert used to generate imitation-learning data.

> Architecture: a **Docker** eval container runs Gazebo + ROS 2 + the scoring engine + a Zenoh
> router. The **policy** runs on the host in a pixi environment and connects to that router over
> host networking. (No distrobox — plain `docker run`.)

---

## Prerequisites (already set up on this machine)

- Docker + NVIDIA Container Toolkit configured; NVIDIA GPU + driver.
- Eval image pulled: `ghcr.io/intrinsic-dev/aic/aic_eval:latest`
- pixi workspace built at **`/home/skr/ws_aic/src/aic`** (`pixi install` already run). pixi binary: `/home/skr/.pixi/bin/pixi`.
- NVIDIA EGL ICD file present at **`/home/skr/ws_aic/aic_local/10_nvidia.json`** (needed for GPU rendering — see below).
- CheatCode needs **no torch / GPU-compute** (uses TF + numpy only).

If the EGL ICD file is missing, create it with:
```json
{ "file_format_version": "1.0.0", "ICD": { "library_path": "libEGL_nvidia.so.0" } }
```

---

## Step 1 — Start the eval container (ground truth ON, headless)

```bash
mkdir -p ~/aic_results/cheatcode
docker rm -f aic_eval_cheat 2>/dev/null || true

docker run -d --rm --gpus all --network host --name aic_eval_cheat \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
  -v /home/skr/ws_aic/aic_local/10_nvidia.json:/usr/share/glvnd/egl_vendor.d/10_nvidia.json:ro \
  -e AIC_RESULTS_DIR=/results -v ~/aic_results/cheatcode:/results \
  ghcr.io/intrinsic-dev/aic/aic_eval:latest \
  ground_truth:=true gazebo_gui:=false launch_rviz:=false \
  start_aic_engine:=true shutdown_on_aic_engine_exit:=true \
  model_discovery_timeout_seconds:=300
```

No scene-config argument is passed → the launch uses the toolkit's **built-in default config**.
`ground_truth:=true` is required (CheatCode reads ground-truth TF).

## Step 2 — Wait for the engine, then run the policy on the host

```bash
# wait until the engine is up and polling for the model
until docker logs aic_eval_cheat 2>&1 | grep -q "aic_model"; do sleep 2; done

# IMPORTANT: exactly ONE aic_model node may exist. Kill any leftover policy first:
pgrep -af "aic_model --ros-args"      # inspect; kill leftovers by PID if any

# launch CheatCode (pixi env auto-sets RMW_IMPLEMENTATION=rmw_zenoh_cpp)
cd /home/skr/ws_aic/src/aic
/home/skr/.pixi/bin/pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true -p policy:=aic_example_policies.ros.CheatCode
```

The container stops itself after 3 trials (`shutdown_on_aic_engine_exit`). When it exits, stop the
host policy (Ctrl-C, or kill it).

## Step 3 — Read the score

```bash
cat ~/aic_results/cheatcode/scoring.yaml     # top-level `total:` ≈ 279
```

---

## Critical env / flags & gotchas

| Item | Why it matters |
|---|---|
| `--network host` | Host policy ↔ container Zenoh router at `localhost:7447` |
| `NVIDIA_DRIVER_CAPABILITIES=all` + the `10_nvidia.json` EGL ICD mount | Forces **GPU** headless rendering. Without it Gazebo falls back to CPU/llvmpipe → RTF ≈ 0.03 (~30× slower). Verify with `nvidia-smi` (util should be >0 during a run). |
| `ground_truth:=true` | CheatCode inserts using ground-truth TF; fails without it |
| **Exactly one `aic_model` node** | Orphaned policy processes → engine errors `More than one node with name 'aic_model'` and fails Tier 1. Kill stale ones before launching. |
| Start **container before** policy | The container hosts the Zenoh router |
| `-e AIC_RESULTS_DIR=/results -v …:/results` | `scoring.yaml` is written inside the container; mount it out or it's lost on `--rm` |
| ROS 2 distro | **Kilted Kaiju** — provided by the pixi env; don't mix host ROS |

## Cleanup

```bash
docker rm -f aic_eval_cheat 2>/dev/null || true
# also kill the host policy process if still running (find via: pgrep -af "aic_model --ros-args")
```

## Optional — watch it (GUI)

Add `-e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix` to the `docker run`, set
`gazebo_gui:=true launch_rviz:=true`, drop `--rm`/`shutdown_on_aic_engine_exit`, and run
`xhost +local:` on the host first. A Gazebo window (the simulated world) and RViz (the robot's
camera/sensor view) will show the arm inserting the cable.
```
