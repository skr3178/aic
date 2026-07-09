# Problem Formulation — where the AIC insertion problem actually stands (2026-07-06)

## 1. Environment, robot, sensors, eval

**Simulator / harness.** Gazebo via `aic_engine` (ROS 2). Scoring is Gazebo-only. `ground_truth:=false`
for real scoring; `:=true` exposes GT TF frames (diagnostics / CheatCode only).

**Robot & task.** Universal Robots **UR5e** (6-DOF arm) + **Robotiq Hand-E** parallel gripper. Task =
**cable insertion**: the plug spawns **already grasped** in the gripper (the loose cable end lies on the
table) → the robot must **insert the plug into the target board port**. No reach-and-grab phase; the
grasp offset is **type-fixed** (measured, M14).

**Scene.** A task board at a **randomized pose (x, y, yaw)**, populated with NIC-card mounts (SFP ports)
and SC mounts sitting on **rails** (mounts slide ± a randomized amount → the "rail slop"), plus
distractor cards.

**Model inputs (the `Observation`, per control step):**
- **3 wrist RGB cameras** — left / center / right, native **1152×1024** (often used at 288×256).
  Wrist-mounted → they **move with the arm**.
- **26-dim proprioceptive state** — TCP pose (7), TCP linear+angular velocity (6), TCP error (6), joint
  positions (7).
- **Wrist F/T** — 6-DOF force/torque (`wrist_wrench`). **No tactile sensors** (force/torque only).
- **`Task` message (all free/known)** — `plug_type` (sfp/sc), `port_type`, `target_module_name`,
  `port_name`, `cable_name`, `plug_name`, `time_limit`.
- **Robot TF (FK)** — `base_link ← {left,center,right}_camera/optical`, `base_link ← gripper/tcp`.
  GT port/plug frames appear **only under `ground_truth:=true`**.

**Action / control.** Cartesian **TCP pose target in `base_link`** (position mode) → an
**admittance/impedance controller** (stiffness/damping + feedforward wrench). Compliant, force-yielding
(won't ram).

**Reference policies.** **CheatCode** — GT expert (reads true port/plug TF, aligns, descends), **279.4**
ceiling, *not a valid submission* (uses GT). **WaveArm** — trivial floor (37.5).

**Connector / plug types (what's inserted):**
- **SFP** — transceiver plug → SFP cage port on a NIC card. **2 of 3 eval trials.**
- **SC** — fiber connector → SC port. **1 of 3 eval trials** (lower on the board → off-frame more; keyed).

**Eval — what is tested.** `aic_engine` runs **3 fixed trials** on the **vaulted `eval_config.yaml`**
(leaderboard scenes: 2 SFP + 1 SC, distractor cards, off-axis board yaws), real scoring at
`ground_truth:=false`, writing `scoring.yaml`, **max 300** (3 × 100):
- **Tier 1 — validity** (0/1)
- **Tier 2 — motion quality** — off-limit contact **−24**, insertion force >20 N **−12**, + duration /
  efficiency / smoothness bonuses
- **Tier 3 — insertion** — full **+75**, partial 38–50, proximity 0–25, else 0

## 2. The pipeline the task requires
```
3 wrist images (+ FK poses, F/T, Task info)  →  PORT POSE (position + orientation)  →  aim plug  →  insert
```

## 3. What is SOLVED and proven (do not re-litigate)
| Piece | How | Evidence |
|---|---|---|
| **Plug pose** | `FK(gripper) · T_grasp[task.plug_type]` — grasp offset is type-fixed, measured. No vision. | M14 |
| **Hands / insertion** | reactive force-stop + spiral descent (compliant); seats when aimed well | 227.8 (ours) · 277.7 (InsertTuner), both with GT port |
| **Geometry chain** | pixel → ray (per cam) → **triangulate 3 cams** → 3D; camera↔gripper + plug↔gripper calibrated | self-check **0.0 mm** |
| **Ceiling** | GT port + our FK plug + our hands | **227.8** (2 full SFP + 1 SC partial), M15 |

### Best scores to date (and the exact method for each)
| Score | Conditions | Method (eyes → hands) | Submittable? |
|---|---|---|---|
| 279.4 | CheatCode | reads **GT** port+plug TF → align → descend | ❌ GT |
| 277.7 | InsertTuner | **GT** port → SC-tuned force endgame (3/3 inserts) | ❌ GT |
| **227.8** | **our best PROVEN** | **GT port** → **FK-plug (kinematics) + reactive descent** — 2 full SFP + 1 SC partial | ❌ GT port only (ceiling proof) |
| **−9** | **our best GT-FREE (real)** | classical-**sweep orientation** + **KPNet position** + FK-plug + reactive descent | ✅ real (but negative) |
| +1.4 | M7b baseline | DualPoseNet-regression eyes + reactive descent | ✅ real |
| −21.3 | old KP | KPNet eyes + DualPoseNet orient/plug + reactive | ✅ real |

**Reading:** **227.8 is our best score** and its method is **the entire solved stack (FK plug + compliant
reactive descent) aimed by a GT port** — i.e. only the *eyes* are GT; the plug and hands are ours. It is
the **proven ceiling**, not a submission. The best **honest GT-free** score is **−9** (sweep orientation
now works CV-only, but port *position* still fails on the eval domain) — up from −21.3, capped negative
purely by the unsolved image-localization (§5).

## 4. The remaining problem — stated formally
Find a function
```
f(3 images, camera poses (FK), task_name)  →  port pose (position, board yaw)
```
such that, **on the vaulted eval domain** (scenes unseen in training):
- **position error < 3.6 cm** (the spiral radius the hands forgive; SFP need). SC has ±6 cm rail slide → also needs a **1-D search along the known rail axis**.
- **board-yaw error < ~2°** (2–4° only partial-seats; the keyed connector needs tighter).
- **GT-free** and **domain-gap-immune**.

If `f` hits that bar, the already-proven plug + hands deliver **~227**. Nothing else is missing.

## 5. The core difficulty (the ONE unsolved sub-problem)
Everything downstream of a correct pixel is solved (triangulation → 3D → aim → insert). The entire gap
is a single question:

> **Which pixels are the target port?**  (localize the target in the image, GT-free, on the eval domain)

Three obstacles make it hard:
1. **Domain gap** — trained ML localizes on *our generated data* but collapses on the *vaulted eval* scenes
   (KPNet: 1.9 mm proxy → 300–430 mm SC eval).
2. **Clutter** — fixed-rule CV (threshold/edges) can't separate the target from **lookalike clutter**
   (heatsink, white floor, other ports, rails) in this busy scene.
3. **Visibility** — the **SC port is off-frame 78–84%** of the trial; you cannot localize what is not in
   the image → needs **active perception** (move the wrist to bring it into view; aim from board pose).

## 6. What has been tried — and why each failed (the evidence)
| Approach | Kind | Result on eval |
|---|---|---|
| DualPoseNet (port regression) | trained ML | fails — domain gap (69–220 mm) |
| KPNet (port keypoint + triangulation) | trained ML | fails — domain gap (SC 300–430 mm) |
| BoardNet (board-yaw regression) | trained ML | fails — **125°** off (domain gap + symmetry) |
| Dark-board rectangle | fixed-rule CV | fails — grabs the whole dark workspace |
| Rail-line Hough | fixed-rule CV | fails — clutter (board edge/cards/floor drown the rails) |
| Bright SFP-faceplate | fixed-rule CV | fails — **196 px** off, grabs heatsink/floor clutter |
| Board sweep + ground back-projection | CV + geometry | **2–24° inconsistent**; earlier "~2°" was a GT-anchored crutch |

**Pattern:** every *learned* method dies on the domain gap; every *fixed-rule CV* dies on clutter. That
dual failure is the crux — the target is neither reliably learnable (from our data) nor reliably
carve-out-able by a hand threshold.

## 7. The two open escape routes (the parallel bet)
Both escape the domain gap by construction; they fail **independently**, so run in parallel.
| Track | Escapes the gap via | First move | Status |
|---|---|---|---|
| **1 — better CV** | geometry has no appearance gap | clutter-robust primitive (slot-pair; not fixed-threshold) | 3 CV variants tried, all clutter-limited |
| **2 — foundation models** | robustness from *massive external* data, not our tiny set | **SAM** segment the SFP module / board → centroid → triangulate | **untested — highest-leverage hedge** |

**Shared substrate** (both tracks need, build once): **active perception** (wrist move for SC visibility,
aimed from board pose) · FK plug · reactive hands · **the offline eval-frame harness**.

## 8. The discipline (the M10 scar → the cure)
Never trust a proxy that isn't the eval domain (KPNet's 1.9 mm proxy lied). **Gate every method offline
on real vaulted-eval frames** before spending an eval run. **Pass bar:** target within 3.6 cm (position)
AND board-yaw within 2°, on the real eval frames. Prereq: one GT run dumping **clean eval frames + GT
port + FK extrinsics** = the shared test set (not yet captured; `kp_frames` are annotated).

## 9. One-line statement
**Everything is solved except localizing the target port in the image on the unseen eval domain** —
a problem that is *neither* reliably learnable from our generated data (domain gap) *nor* carve-out-able
by a fixed CV rule (clutter) — and the two open bets are **SAM-class foundation segmentation** (primary)
and a **clutter-robust CV primitive** (hedge), each gated offline on real eval frames at **3.6 cm / 2°**,
feeding the already-proven kinematic plug + compliant hands to the **227** ceiling.
