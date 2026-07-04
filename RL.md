# RL Findings — which simulator for reinforcement learning

Summary of what each of the three provided simulators offers for **RL**, and where to start.
(Our primary strategy is **#7 learned-perception + analytic control**, which is *supervised, not RL*.
RL — approaches **#6 (RL)** / **#8 (residual RL)** — is a *later* option to sharpen the contact phase.)

## TL;DR
- **Isaac Lab = the intended parallel-RL path.** It ships a **working, GPU-vectorized PPO env**
  (`AIC-Task-v0`) — but the reward is **pose-reaching** and the **gripper + cable are disabled**, so
  it's a **stub you *extend* to insertion**, not build from scratch.
- **MuJoCo = a single-instance ROS mirror, NOT an RL env.** No reward, no gym wrapper, no
  vectorization — it runs the scene under `mujoco_ros2_control` (same controller as Gazebo). Parallel
  RL on MuJoCo is possible only by building it yourself with **MJX** (from the shipped MJCF).
- **Gazebo = the evaluation env** (serial, no RL layer). Everything ultimately scores here.
- **To start RL right away → Isaac Lab** (extend a running env) beats MuJoCo (build from zero).

## Per-simulator detail

| | What's shipped for RL | Ready to RL-train? | Effort → parallel insertion RL |
|---|---|---|---|
| **Gazebo** | eval env + CheatCode expert + scoring; single-instance | ❌ (serial, no RL layer) | n/a — it's the scorer |
| **MuJoCo** | single-instance `mujoco_ros2_control` mirror + MJCF assets + SDF→MJCF converter + sim-compare | ❌ no reward / no env / no vectorization | **High** — build MJX env + reward + loop from scratch |
| **Isaac Lab** | **GPU-vectorized RL scaffold**: `AIC-Task-v0`, `rsl_rl` PPO, domain-randomization events, reward, env registration | ✅ **as a pose-reaching stub** (gripper/cable off) | **Moderate** — *extend* the existing env to insertion |

### Isaac Lab (the RL path)
- **Scripts:** `aic_utils/aic_isaac/aic_isaaclab/scripts/rsl_rl/{train.py, play.py, cli_args.py}`
  (+ `list_envs.py, random_agent.py, zero_agent.py, record_demos.py, replay_demos.py, teleop.py`).
- **Env / reward:** `aic_utils/aic_isaac/aic_isaaclab/source/aic_task/aic_task/tasks/manager_based/aic_task/`
  - `aic_task_env_cfg.py` — obs/action spaces, terminations
  - `mdp/rewards.py` — reward (currently **EE pose-reaching** + 2 cm bonus + smoothness/safety penalties)
  - `mdp/observations.py`, `mdp/events.py` (domain randomization), `agents/rsl_rl_ppo_cfg.py` (PPO hp)
- **Runs under the `isaaclab` launcher, NOT our pixi env.** Needs a separate **Isaac Lab v2.3.2**
  install + the NVIDIA **`Intrinsic_assets.zip`** pack. (An `env_isaaclab` conda env already exists on
  this machine → partway there.)
- **Stub → insertion work:** re-enable the gripper + cable articulation, replace the pose-reaching
  reward with an insertion/contact reward. Real effort (the cable is a fiddly deformable in Isaac),
  but the **RL infrastructure — vectorized envs, PPO, domain randomization — is done.**

### MuJoCo (a mirror, not a trainer)
- `aic_utils/aic_mujoco/`: `scripts/{sim_comparison_test,view_scene,add_cable_plugin,load_aic_world}.py`,
  prebuilt `mjcf/{scene,aic_robot,aic_world}.xml`, `launch/aic_mujoco_bringup.launch.py`,
  `mujoco.repos`. Integration = **`mujoco_ros2_control`** (single instance).
- **No** `mjx / num_envs / reward / gymnasium / vmap`. It's for running the *same* ROS policies in
  MuJoCo, scene-level domain randomization, and Gazebo↔MuJoCo fidelity checks.
- **Parallel RL potential is real but DIY:** MuJoCo's engine (**MJX**, JAX/GPU, or
  `mujoco_playground`/Brax) runs thousands of envs on one GPU — but you'd load the shipped MJCF into
  MJX and write obs/action/reward/loop yourself.

## How to start RL (Isaac) — cheapest first move
1. Ensure Isaac Lab **v2.3.2** + `Intrinsic_assets.zip` installed; `pip install -e .../source/aic_task`.
2. **Smoke-test the loop (and the GPU):**
   `isaaclab -p .../scripts/rsl_rl/train.py --task AIC-Task-v0 --num_envs 1024 --enable_cameras`
   → confirms PPO trains the pose-reaching stub *on this machine*.
3. **Then** extend `mdp/rewards.py` + `aic_task_env_cfg.py` to actual insertion.

## ✅ EMPIRICALLY VALIDATED ON THIS MACHINE (2026-07-03)

I actually ran the stack. **The binding blocker is NOT the #4951 TiledCamera hang — it's that
Isaac Sim's RTX renderer won't initialize *at all* on this Blackwell GPU.** #4951 is moot here
because we never reach camera code.

**Machine:** RTX PRO 4000 Blackwell (sm_120), driver **595.71.05**, CUDA 12.8.
**Installed (NOT the AIC-required versions):** Isaac Sim **4.5.0** (`~/isaacsim`, symlinked as
`~/IsaacLab/_isaac_sim`) + Isaac Lab **0.47.7** (repo on `main`, ~v2.2.1+115). AIC wants Sim **5.1.0**
+ Lab **2.3.2**.

**Correcting earlier claims in this doc:**
- ❌ "Isaac Sim not installed / `import isaacsim` fails." **Wrong** — Sim 4.5.0 *is* installed and
  imports fine in `env_isaaclab`. (The earlier failure was a `importlib.metadata.version` lookup, not
  the import.)
- The right interpreter is the **`env_isaaclab` conda env**, NOT `./isaaclab.sh -p` (that uses Sim's
  bundled python, which has no `isaaclab` → `ModuleNotFoundError`).

**Test cascade (each a real, mundane blocker — none was #4951):**
1. Wrong interpreter (`isaaclab.sh` bundled python) → no `isaaclab`. Fix: use `env_isaaclab`.
2. Broken **user-site torch** (`~/.local/.../torch`, `undefined symbol: ncclCommWindowDeregister`)
   shadows the env → **segfault at `import torch`**. Fix: **`export PYTHONNOUSERSITE=1`** → env resolves
   the correct **torch 2.7.0+cu128**, `cuda.is_available()==True`, sees "NVIDIA RTX PRO 4000 Blackwell".
3. Missing **`h5py`** → `isaaclab_tasks` extension load segfaults. Fix: `pip install h5py`.
4. **RENDERER SEGFAULT (the real wall):** with `enable_cameras=True`, the app crashes during
   `SimulationApp.__init__` at
   `omni.kit.widget.viewport/impl/texture.py:377 __enable_hydra_engine` — **even for an empty stage
   with zero cameras.** Isolated cleanly:
   - `enable_cameras=True`  → **segfault** in Hydra/RTX engine init.
   - `enable_cameras=False` → **APP OK + SIM RESET OK + 20 physics steps OK.**

**Conclusion:** Isaac Sim 4.5.0 runs **physics** fine on this Blackwell box but its **RTX/Hydra
renderer does not initialize** (Blackwell support landed in Sim 5.x; 4.5.0 predates it). So **any
vision/camera workload — TiledCamera *or* standard Camera — is impossible on the current stack.**
Proprioception-only (no-render) Isaac RL *is* runnable today; the AIC `AIC-Task-v0` env can't be,
because its obs bake in 3 TiledCameras.

**What this means for the TiledCamera/#4951 risk specifically:** it's a *second* gate behind a *first*
one. To even get a renderer on this GPU you must upgrade to **Isaac Sim 5.1 + Isaac Lab 2.3.2** (the
tens-of-GB install). **Only then** do you hit #4951 — still **OPEN** (opened Mar 2026, root cause is
NVIDIA `omni.replicator`, no fix in any release, and **no evidence driver 595 resolves it**; the
report was on 590.48.01). The "driver 595 might dodge it" glimmer remains **untestable until the 5.1
upgrade**. Net: the Isaac parallel-vision-RL path now has **two sequential blockers**, strengthening
the recommendation below (perception-first; MuJoCo/MJX as the RL fallback that already runs here).

## Version requirements & Blackwell compatibility (checked 2026-07)

**Required versions**
- **Isaac Lab 2.3.2** (AIC repo's tested version). This machine's `~/IsaacLab` is **2.3.0** on `main`
  → `git checkout release/2.3.2`.
- **Isaac Sim 5.1.0** — Isaac Lab 2.3.x is built on Isaac Sim 5.1 (support for ≤4.2 dropped).
  **Not installed here** (`import isaacsim` fails; only the Isaac Lab *framework* is present). This is
  the big download (tens of GB).
- **Assets:** `Intrinsic_assets.zip` (latest, post-PR #491). Not downloaded.

**⚠️ Blackwell blocker — parallel vision-RL is broken right now**
- Isaac Lab **issue #4951 (OPEN):** `TiledCamera` **hangs indefinitely on Blackwell sm_120** with
  Isaac Sim 5.1.0 (100% CPU, no progress). Root cause is NVIDIA's `omni.replicator` tiled-rendering
  (not patchable from our side).
- `TiledCamera` renders all N parallel envs' cameras in one GPU pass — **the entire reason to use
  Isaac for a vision task.** The documented workaround (standard `Camera`) only matches it "for single
  environments" → **kills the parallelism.**
- **Glimmer:** the bug awaits "driver > 590.48.01"; this box has **595.71.05** (newer) → *might* be
  fixed here, but **unverified** — the only way to test is to install the full Isaac Sim 5.1 stack.

**If pursuing Isaac anyway — exact set + first test**
1. `git checkout release/2.3.2` in `~/IsaacLab`.
2. Install **Isaac Sim 5.1.0** (matches Isaac Lab 2.3.x).
3. Download **`Intrinsic_assets.zip`** → place in `.../aic_task/Intrinsic_assets/`.
4. **First test = a `TiledCamera` smoke-run.** If driver 595 resolves #4951 → parallel vision RL is
   viable; if it still hangs → Isaac parallel-vision-RL is off the table on this GPU until NVIDIA
   fixes it, and **MuJoCo/MJX becomes the better RL bet** (runs wherever cu128 works — confirmed here).

Sources: [Isaac Lab releases](https://github.com/isaac-sim/IsaacLab/releases) ·
[Isaac Sim 5.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html) ·
[Isaac Lab #4951 TiledCamera Blackwell hang](https://github.com/isaac-sim/IsaacLab/issues/4951)

## Caveats (eyes open)
- ⚠️ **Blackwell (sm_120) risk** — Isaac Sim on this GPU is the least-certain part; step 2 is really a
  "does Isaac even launch here" test. Do it *before* investing in the reward. (MuJoCo/MJX runs
  wherever JAX+cu128 works — which we've confirmed on this box.)
- **Sim-to-sim gap** — an Isaac-trained policy must transfer to **Gazebo** for scoring (issues
  #424/#434 territory, since fixed). Budget domain randomization + a reality-gap check.
- **Strategic** — RL layers *on top of* perception, not instead of it. The challenge itself hands
  qualified teams a *Vision Model* (perception), not an RL env — a strong hint that **#7 is the main
  lever**. Recommendation: finish **M3 (perception)** first (hours, on data we already have); treat
  Isaac RL as a later contact-phase sharpener.

## Recommendation
- **Don't invest in the Isaac download now** — the one reason to use Isaac (parallel *vision* RL)
  sits behind the OPEN Blackwell bug **#4951**, and confirming whether driver 595 dodges it needs the
  full Isaac Sim 5.1 install (tens of GB). High cost, uncertain payoff, for a non-primary path.
- **Do M3 (perception) first** — primary lever, data already collected, runs on the proven
  Gazebo/pixi stack on this Blackwell.
- **If/when RL is needed:** run the Isaac `TiledCamera` smoke-test (versions above); if #4951 still
  bites, go **MuJoCo/MJX** instead (build the parallel RL env from the shipped MJCF, which runs
  wherever cu128 works — confirmed here).
