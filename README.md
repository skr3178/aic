# 🏆 OUR SOLUTION — GT-free cable insertion

> **Best GT-free score: `92.2 / 300`** — policy **`PerceptionInsertSFPDrive`** (2.3× the previous best of 40.1).
> This section tracks **our own solution**; the challenge's upstream toolkit guide follows below.

```bash
# run the best GT-free policy (no ground truth anywhere)
~/ws_aic/aic_local/score.sh aic_example_policies.ros.PerceptionInsertSFPDrive false eval my_run
```

## Scoreboard

| run | score | ground truth? | note |
|---|---|---|---|
| CheatCode | 279.4 | ✅ GT | oracle ceiling (no force stack — pure position control) |
| InsertTuner (M9) | 277.7 | ✅ GT | our force rig, 3/3, tug-verified |
| `kp_gtport` | 227.8 | ✅ GT port | proven perception ceiling |
| **`sfp_reach_gt`** | **171.4** | ✅ GT port | **our motion + our force stack — both SFP FULL SEAT** ⇒ motion is solved |
| **`sfp_free`** | **92.2** | ❌ **none** | ⭐ **BEST GT-FREE — the number we claim** |
| `geo_v1` | 40.1 | ❌ none | previous best (superseded) |
| WaveArm | 37.5 | ❌ none | challenge floor (no insertion) |
| ACT (pretrained) | −21 | ❌ none | shipped learned baseline |

**Per-trial (`sfp_free`, GT-free):**

| trial | our lock vs GT | tier-3 | result |
|---|---|---|---|
| SFP `port_0` | **~2 mm** | 38.0 | partial 0.05 m (+ −36 approach penalty) |
| SFP `port_1` | **~3 mm** | 38.0 | partial 0.05 m · **zero contacts** |
| SC | 58 mm ❌ | 17.4 | no insert (perception bug) |

🎥 **GT-free insert videos** (green X = our perceived lock driving the arm; white ring = GT, reference only):
[`sfp_trial1_gtfree.mp4`](media/videos/sfp_trial1_gtfree.mp4) · [`sfp_trial2_gtfree.mp4`](media/videos/sfp_trial2_gtfree.mp4)

### 🔌 GT-free FULL SEAT — SFP `port_0` (branch `seat-v2`)

![GT-free SFP port_0 full seat](media/videos/seat_v2_sfp_port0_seated.gif)

*Fully GT-free control (2× speed).* The **green X** is our *perceived* lock — it is what drives the arm; the white ring is GT, **drawn for reference only, never read**. The HUD shows the seat happening: `lock err 3 mm` → `CONTACT: force_abs` → stiffness drops to `[40,40,90] RCC-SOFT` (lateral-soft, so the plug can self-align) → `tip −1.0 mm above port` = **below the port datum ⇒ seated**, scored `tier_3 = 75 "Cable insertion successful"`.

This is the **revived force stack** (branch `seat-v2`): the deployed contact predicate required a force *increase*, but landing on the SFP cage face **unloads** the wrist (21 N → 8 N), so the branch had **never fired in any SFP run, ever** — the "force stack" was silently a stiff straight-down ramp. A sign-insensitive trigger + re-arming stall watchdog + RCC compliance makes the search actually run. ⚠️ **Not yet banked as a score:** on an identical config the total swings **67.8 → 160.8**, so the headline above remains the reproducible **92.2**. See [LAB_LOG.md](LAB_LOG.md).

### 🔒 SC — perception FIXED (58 mm → 2 mm), but the plug is mechanically BLOCKED

🎥 [`sc_trial_gtfree_blocked.mp4`](media/videos/sc_trial_gtfree_blocked.mp4) — same HUD format as the SFP videos.

**The perception bug is solved.** The eval's SC trial spawns **three** SC modules, and the old lock took a robust median over *every* `sc` detection on the board — averaging across all three and landing **58 mm** from the true port. SFP had a target-selection gate; SC never got one. Adding the same idea — a **board-Y module gate** ([`PerceptionInsertSCFix.py`](aic_example_policies/aic_example_policies/ros/PerceptionInsertSCFix.py)) — takes the lock to **2 mm**.

**But SC still does not seat, and it is no longer a perception problem.** The plug tracks the port to within **0.1–0.4 mm laterally through the entire 150 mm descent**, then hits a hard stop **18.0 mm above the port datum** — at **0.1 mm lateral error**:

| step | z cmd | tip above port | tip lateral | ΔF | spiral k |
|---|---|---|---|---|---|
| 273 | 0.0635 | 37.1 mm | **0.1 mm** | −0.4 | 0 |
| 301 | 0.0495 | 20.5 mm | **0.1 mm** | −0.4 | 0 |
| **317** | 0.0155 | **18.5 mm** | **0.2 mm** | **−8.4** | 1 | ← **hard stop** |
| 421 | −0.0140 | 18.7 mm | 2.7 mm | −12.0 | 5 |

We command `z` **33 mm past the block** and the plug moves **0.4 mm**. `ΔF −12 N` means the wrist is being *unloaded* — the plug's weight is resting on something solid. This rules out perception, the trigger, *and* the search: the spiral only makes it worse, dragging a perfectly-centred plug from 0.1 mm out to 2.7 mm.

Two open candidates:
1. **Not enough axial force.** `AX_STIFF = 90 N/m` × 33 mm overtravel ≈ **3 N total**. InsertTuner's own note says *"impedance can only make ~5–6 N at the mouth"* — enough for SFP's 46 mm funnel, likely not for SC's 15.6 mm one.
2. **A ~2.4 mm z-datum error.** The plug stops at 18.0 mm; SC's funnel entrance is at 15.6 mm — it halts *2.4 mm short of even reaching the funnel mouth*, consistent with `CAD_Z["sc"]` or the plug-tip FK offset being off by that much.

---

## The stack — 3 levels

```
[0] SWEEP  13 wrist poses x 3 cams
     |
[1] DETECTION ....... board pose + port lock          ✅ = GT quality (<=3 mm, 0.00 deg)
     |
[2] MOTION .......... SAFE-CLEAR + approach + descend ✅ SOLVED (both SFP full-seat w/ GT)
     |
[3] SEAT ............ force stack                     ❌ BROKEN — has never run (see below)
```

### LEVEL 1 — DETECTION ✅
| piece | what it does | file |
|---|---|---|
| **Board pose** | full-scale mask → largest-CC (drop detached NIC blob) → **known-size box yaw** → **magenta-anchored center**. vs GT: **≤1.3° / ≤6.9 mm** on eval poses | `PerceptionInsertSFP.py` |
| **SFP slide-select** | the 2 SFP openings are 21.8 mm apart; both-visible frames estimate the rail slide → identity fixed → picks the **named** opening | `PerceptionInsertSFP.py` |
| **YOLO port detector** | candidate generator (fine-tuned; sharp close, 0–1 mm) | `PerceptionInsertYOLO.py` |
| **CAD-z back-projection** | port depth from the known board plane (replaced broken triangulation) | `PerceptionInsertGeo.py` |
| **FK plug** | plug pose = `FK(gripper) · T_grasp[type]` — **zero vision** | `PerceptionInsertYOLO.py` |
| **Magenta quadrant** | resolves the 90° board ambiguity — 12/12 | `PerceptionInsertGeo.py` |

### LEVEL 2 — MOTION ✅ (the big win)
| piece | what it does | file |
|---|---|---|
| **`_safe_clear`** ⭐ | **THE FIX.** Between sweep and approach, drive **joint-space** to `SAFE_HOME` to **reset the IK branch**. The sweep-end *posture* made IK fold the upper arm into the NIC card — Cartesian commands can't fix this (the bad posture *satisfies* the pose). **Also cured the long-standing SC ~53 cm stall.** | `PerceptionInsertSFPDrive.py` |
| **Frozen target** | perceive once during the sweep, then hold (no per-step re-perception drift) | `PerceptionInsertSFPDrive.py` |
| Approach + descent | 100-step slerp/position interp → hover → slow straight descent | `PerceptionInsertYOLO.py` |

### LEVEL 3 — SEAT ❌ **BROKEN — the #1 open item**
The deployed force stack is a **degraded port of `InsertTuner`** (our rig validated at **277.7**) and **has never actually run**:

| | InsertTuner (validated) | Deployed (broken) |
|---|---|---|
| `F_STOP` | **4.0 N** — *"impedance can only make ~5–6 N at the mouth"* | **8.0 N** ← **unreachable** |
| stall detect | `df > F_STOP **or stalled**` | **GONE** |
| RCC compliance | `[40, 40, 90]` lateral-soft / axial-firm | **GONE** (falls back to stiff `[90,90,90]`) |

**Consequence:** contact is only tested as a force *increase*, but the plug landing on the SFP cage face **unloads** the wrist (20.9 → 8 N). **Zero spiral log lines in any run, ever.** The "partial insertion 0.05 m" is really **the plug resting on the cage face**, never searching. GT seats because an exact target slides straight in and never touches the face.

---

## Open items (ranked)

| # | fix | gain | risk |
|---|---|---|---|
| **A** | **Two-segment approach** — hold altitude while translating, *then* straight down. (Today it descends **diagonally while translating** into the card: TCP z 0.328→0.271 during the lateral move.) **Rule: no simultaneous lateral translation + descent near the card.** | **+36** → ~128 | low |
| **B** | ⚠️ **TRIED — NET REGRESSION.** Restored force stack — `F_STOP` 4.0, stall detect, RCC compliance. *A restoration of validated code, not an invention.* Do **after A**: `F_STOP=4.0` is sensitive enough that trial-0's 93 N card scraping would false-trigger the search. | **+74** → ~166 | medium — capture basin at 2–3 mm **never measured** |
| **C** | **Fix the SC port lock** (58 mm off; offline was 6/6 ≤1 mm ⇒ findable bug) | +20…40 | low |
| ~~D~~ | sub-mm perception | +74 | slow, uncertain |
| ~~E~~ | continuous re-lock | 0 | solves a problem we don't have (target is static **and** correct) |

**Do NOT retry:** lateral probe-pattern search (fermat / raster / dense grid / sliding spiral). All 4 scored **worse** than 92.2 — searching *drags the plug off* the spot that at least scores a partial. → `force-insert-v2` branch, LAB_LOG.

---

## Code map (our files)

```
aic_example_policies/aic_example_policies/ros/
  PerceptionInsertSFPDrive.py   ⭐ BEST GT-FREE (92.2) — safe-clear + frozen target
  PerceptionInsertSFP.py           board-pose fix + SFP slide-select
  PerceptionInsertGeo.py           CAD-z depth + magenta quadrant
  PerceptionInsertYOLO.py          YOLO detector + FK plug + approach/descent + (broken) force stack
  PerceptionInsertKP.py            keypoint variant (earlier)
  SFPDriveViz.py                   GT-free insert video (X = our lock)
  SweepDump{,Full,Val}.py          frame/TF dumpers for offline eval
  CheatViz*/CheatLog*.py           diagnostics (YOLO-vs-range, slide-select videos)
  Geo{Home,RealPlug,ReplayA}.py    execution forensics (all superseded by safe-clear)
aic_engine/config/val_config.yaml  held-out 12-scene validation set
```

**Docs:** [LAB_LOG.md](LAB_LOG.md) (experiment index) · [pipeline.md](pipeline.md) (gates) · [board_pose.md](board_pose.md) (board fix) · [notes/](notes/)

---
---

# AI for Industry Challenge Toolkit

[![build](https://github.com/intrinsic-dev/aic/actions/workflows/build.yml/badge.svg)](https://github.com/intrinsic-dev/aic/actions/workflows/build.yml)
[![style](https://github.com/intrinsic-dev/aic/actions/workflows/style.yml/badge.svg)](https://github.com/intrinsic-dev/aic/actions/workflows/style.yml)

![](../media/aic_banner.png)

The **AI for Industry Challenge** is an open competition for developers and roboticists aimed at solving some of the hardest, high-impact problems in robotics and manufacturing.

This repository contains the official toolkit to help participants start developing their solutions. For registration details, official rules, and FAQs, please visit the [AI for Industry Challenge event page](https://www.intrinsic.ai/events/ai-for-industry-challenge).

---

## Toolkit Guide

Welcome to the AIC toolkit documentation. This guide walks you through the complete workflow for participating in the challenge — from understanding the requirements to submitting your solution.

Follow the sections below to navigate through each phase of the process.

1. **📖 Understand the Challenge**
   - Read the [Challenge Overview](./docs/overview.md) to understand the goals.
   - Review the [Qualification Phase](./docs/phases.md#qualification-phase-train-your-model) to understand what you'll be building.
   - Review the [Scoring Guide](./docs/scoring.md) to understand how you'll be scored.

2. **🔧 Set Up Your Environment**
   - Follow the [Getting Started](./docs/getting_started.md) guide to set up and validate your development environment.
   - Run the evaluation container and set up your local workspace with Pixi.

3. **💻 Develop Your Policy**
   - Explore the [Scene Description](./docs/scene_description.md) to learn how to customize and explore the environment.
   - Review [AIC Interfaces](./docs/aic_interfaces.md) to understand available interfaces to communicate with sensors and actuators.
   - Consult [AIC Controller](./docs/aic_controller.md) to learn about controlling the robot.
   - Consult the [Challenge Rules](./docs/challenge_rules.md) to ensure compliance.
   - Start with the [Policy Integration Guide](./docs/policy.md) to implement your solution.
   - See [Participant Utilities](./docs/participant_utilities.md) for a list of helpful tools.

4. **🧪 Test Your Solution**
   - Use the provided simulation environment to test your policy.
   - Run `aic_engine` with the `sample_config` in [`aic_engine/config/`](./aic_engine/config/) to test different scenarios. For more information on running the `aic_engine` with different configs, see the [aic_engine README file](./aic_engine/README.md).
   - Create your own test scenarios by following the configuration example in [`aic_engine/config/`](./aic_engine/config/) to run with `aic_engine`.
   - Refer to [Troubleshooting](./docs/troubleshooting.md) if you encounter issues.

5. **📦 Submit Your Entry**
   - Package your solution following the [Submission Guidelines](./docs/submission.md).
   - Test your container locally before submitting following [these instructions](./docs/submission.md#verify-locally).
   - Submit through the official portal following [these instructions](./docs/submission.md#2-upload-your-image-to-our-registry).

---

## Toolkit Architecture

![AIC Competition Components](../media/aic_competition_components.png)

The AI for Industry Challenge toolkit is divided into **two main components**:

### 1. Evaluation Component (Provided - Run by Organizers)

This component provides the complete evaluation infrastructure:
- **`aic_engine`** - Orchestrates trials and computes scores.
- **`aic_bringup`** - Launches simulation environment (Gazebo, robot, sensors).
- **`aic_controller`** - Low-level robot control with force management.
- **`aic_adapter`** - Sensor fusion and data synchronization.

**What you receive:** Standard ROS sensor topics providing camera images, joint states, force/torque measurements, and TF frames.

### 2. Participant Model Component (Your Implementation - What You Submit)

This is what you develop and submit:
- **A ROS 2 node** that follows the behavioral requirements defined in [Challenge Rules](./docs/challenge_rules.md).
- **Your custom logic** - Code to process sensor data and command the robot to insert cables.

**What you provide:** A container with a ROS 2 Lifecycle node named `aic_model` that responds to the `/insert_cable` action and outputs robot motion commands via standard ROS topics/services.

**Convenient Entry Point:** We provide an `aic_model` framework that handles all the ROS 2 boilerplate and lifecycle management. You simply implement a Python policy class that gets dynamically loaded at runtime. See the [Policy Integration Guide](./docs/policy.md) for details.

### Development and Submission Workflow

> [!IMPORTANT]
> **ROS 2 Distribution:** The official evaluation of all submissions will be conducted using **ROS 2 Kilted Kaiju**. If you choose to develop or test your policy using a different ROS 2 distribution (e.g., Humble or Jazzy), it is entirely your responsibility to ensure compatibility and support. Please note that **inter-distro communication is not guaranteed and not officially supported**.

**Development Options:**
- Develop inside a container (recommended - matches evaluation environment).
- OR develop in native Ubuntu 24.04 environment (requires all dependencies).

**Submission Requirements:**
- Package your solution using the provided `aic_model` Dockerfile.
- Submit your container - it must respond to standard ROS inputs and command the robot to insert cables.
- Your container interfaces with the evaluation component via ROS topics.

---
## Repository Structure

```
aic/
├── aic_adapter/          # Adapter for interfacing between model and controller
├── aic_assets/           # 3D models and simulation assets
├── aic_bringup/          # Launch files for starting the challenge environment
├── aic_controller/       # Robot controller implementation
├── aic_description/      # Robot and environment URDF/SDF descriptions
├── aic_engine/           # Trial orchestration and validation engine
├── aic_example_policies/ # Example policy implementations
├── aic_gazebo/           # Gazebo-specific plugins and configurations
├── aic_interfaces/       # ROS 2 message, service, and action definitions
├── aic_model/            # Template for participant policy implementation
├── aic_scoring/          # Scoring system implementation
├── aic_utils/            # Utility packages and tools
├── docker/               # Docker container definitions
└── docs/                 # Comprehensive documentation
```

---

## Key Packages for Participants

### `aic_model` - Convenient Policy Framework (Recommended)
This package provides a ready-to-use ROS 2 Lifecycle node that dynamically loads and executes your Python policy implementation. It handles all ROS 2 boilerplate, lifecycle management, and challenge rule compliance, allowing you to focus on implementing your policy logic.
- **Location**: `aic_model/`.
- **Documentation**: [Policy Integration Guide](./docs/policy.md).
- **Tutorial**: [Creating a New Policy Node](./docs/policy.md#tutorial-creating-a-new-policy-node).

> **Note:** While we recommend using this framework, you may implement your own ROS 2 node from scratch as long as it adheres to the [Challenge Rules](./docs/challenge_rules.md).

### `aic_interfaces` - Communication Protocols
Defines all ROS 2 messages, services, and actions used in the challenge.
- **Location**: `aic_interfaces/`.
- **Documentation**: [AIC Interfaces](./docs/aic_interfaces.md).

### `aic_example_policies` - Reference Implementations
Example policies demonstrating different approaches and techniques.
- **Location**: `aic_example_policies/`.
- **README**: [aic_example_policies/README.md](./aic_example_policies/README.md).

### `aic_bringup` - Launch the Environment
Launch files to start the simulation, robot, and scoring systems.
- **Location**: `aic_bringup/`.
- **README**: [aic_bringup/README.md](./aic_bringup/README.md).

### `aic_engine` - Trial Orchestrator
Manages trial execution, validates participant models, and collects scoring data.
- **Location**: `aic_engine/`.
- **README**: [aic_engine/README.md](./aic_engine/README.md).

---

## Additional Documentation

### Challenge Information

* **[Challenge Overview](./docs/overview.md):** High-level summary of the competition goals and structure.
* **[Competition Phases](./docs/phases.md):** Details on Qualification, Phase 1, and Phase 2.
* **[Qualification Phase](./docs/qualification_phase.md):** Detailed technical overview of the qualification phase trials and scoring.
* **[Challenge Rules](./docs/challenge_rules.md):** Required behavior for participant models.
* **[Scoring](./docs/scoring.md):** Metrics and methods used to evaluate performance.
* **[Scoring Test Examples](./docs/scoring_tests.md):** Reproducible examples exercising each scoring tier with exact commands.

### Technical Documentation

* **[Getting Started](./docs/getting_started.md):** How to set up your local development environment.
* **[Policy Integration](./docs/policy.md):** Guide to implementing your policy in the `aic_model` framework.
* **[AIC Interfaces](./docs/aic_interfaces.md):** ROS 2 topics, services, and actions available to your policy.
* **[AIC Controller](./docs/aic_controller.md):** Understanding the robot controller and motion commands.
* **[Scene Description](./docs/scene_description.md):** Technical details of the simulation environment.
* **[Task Board Description](./docs/task_board_description.md):** Physical layout and specifications of the task board.
* **[Troubleshooting](./docs/troubleshooting.md):** Common issues and debugging strategies.

### Reference Materials

* **[Glossary](./docs/glossary.md):** Terminology and definitions used throughout the AI for Industry Challenge

### Submission

* **[Submission Guidelines](./docs/submission.md):** How to package and submit your final model.

---


## Support and Resources

- **Discussions**: Engage in conversations and ask questions about the challenge on [Open Robotics Discourse](https://discourse.openrobotics.org/c/competitions/ai-for-industry-challenge/). The community is encouraged to participate in discussions and assist each other.
- **Issues**: Report any bugs or technical issues via [GitHub Issues](https://github.com/intrinsic-dev/aic/issues). Please refrain from using the Issue tracker for general questions about the challenge.
  - **Note:**: Review the list of [known issues](https://github.com/intrinsic-dev/aic/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22known%20issue%22) and [bugs](https://github.com/intrinsic-dev/aic/issues?q=is%3Aissue%20state%3Aopen%20label%3Abug) before opening a new ticket.
- **Event Page**: Visit the [AI for Industry Challenge](https://www.intrinsic.ai/events/ai-for-industry-challenge) for official updates.

---

## License

This project is licensed under the Apache License 2.0 - see the individual package files for details.
The [aic_isaac](./aic_utils/aic_isaac/) folder contains files licensed under BSD-3 - see [aic_isaac/LICENSE](./aic_utils/aic_isaac/LICENSE).
