 STEP 0 ── SWEEP ────────────────────────────────────────────────
   Raster the wrist over ~13 poses × 3 cams.
   Collect two raw things per frame:
     • board pixels  (gray/SAM mask → back-project to z=0 plane)
     • YOLO port boxes (→ back-project to CAD z-plane, e.g. SFP z=0.1335)

 STEP 1 ── BOARD POSE (perceived) ───────────────────────────────
   From the accumulated board pixels:
     center (x, y)  = middle of the fitted board rectangle
     yaw            = orientation of the long edge
     +90° branch    = resolved by MAGENTA marker quadrant
   ── result:  BOARD FRAME = (center, yaw)          e.g. yaw = −2.4°

     [ GT CHECK (offline only) ]
        GT board pose from task_board TF:  yaw_GT = −2.4°
        error = |perceived − GT|           ← this is the scorecard
        (GT is NEVER used to drive the arm — only to grade the estimate)

 STEP 2 ── PICK THE PORT (predict plug target on the board) ─────
   YOLO gives candidate openings in base-xy.
   Put them in the BOARD FRAME (uses STEP-1 yaw).
   SFP has two openings 21.8 mm apart → use CAD + slide to say
   which one is port_0 vs port_1 → keep the target opening.
   ── result:  TARGET PORT position  (x, y, z)      in base frame

 STEP 3 ── BUILD THE GRIPPER POSE ───────────────────────────────
   target position  (from STEP 2)
   + approach orientation (aligns plug yaw to STEP-1 board yaw)
   + FK plug offset (T_grasp: where the plug tip sits vs the gripper)
   ── result:  desired GRIPPER POSE  =  position + quaternion

 STEP 4 ── DRIVE INVERSE KINEMATICS ─────────────────────────────
   set_pose_target(gripper_pose)  →  IK solves joint angles → arm moves
   GATED-LOCK: perceive once, then HOLD this target + FK-track it in
   (don't re-perceive up close where YOLO is noisy).

 STEP 5 ── SEAT ─────────────────────────────────────────────────
   FORCE STACK: spiral search + yaw dither + tug → compliance mates
   the plug into the socket. → score /100 for this trial.

---

# STEP 1 (BOARD POSE) — THE BUG AND THE FIX  (2026-07-10)

The live policy's board pose was the single blocker: one bad `(center, yaw)` poisons BOTH
the port selector (STEP 2) and the insertion orientation (STEP 3). It was overlooked because
the offline SAM solve was banked as ✓ but never shipped — the live policy silently ran a
weaker method, and **no evaluator was watching the live method**.

## What the live method actually was
`gray-threshold [35,135] → back-project every mid-gray pixel to z=0 → accumulate cloud →
2 cm density core (≥25% of max) → cv2.minAreaRect → center + yaw; magenta picks the 90° branch.`
Rests on "gray blob = board", which is false on SFP scenes.

## Diagnosis (measured, not guessed) — evaluators on 3 real (`sweep_dump`) + 12 val (`sweep_dump_val`)
- New evaluator that reproduces the DEPLOYED method: [live_minarea_eval.py](aic_data/track2_sam/live_minarea_eval.py)
- Result: **catastrophic on every SFP trial (~23° yaw, ~250 mm center), near-perfect on SC** —
  systematic per scene-type, not "run-dependent".
- Render of WHY: [live_board_viz_sweep_dump_val.png](aic_data/track2_sam/live_board_viz_sweep_dump_val.png)
  ([live_board_viz.py](aic_data/track2_sam/live_board_viz.py)) — a **detached NIC/base blob**
  leaks into the cloud and drags the `minAreaRect` sideways; **magenta stays dead-on**.
- Head-to-head vs PnP / SAM ([board_pose_pnp.py](aic_data/track2_sam/board_pose_pnp.py),
  [sweep_sam_knownsize2.py](aic_data/track2_sam/sweep_sam_knownsize2.py)): "just port SAM" would
  NOT have worked — SAM yaw ok on SFP (~4°) but its **center is ~100 mm off** (asymmetric coverage).

## The four fixes applied → [magenta_board_eval.py](aic_data/track2_sam/magenta_board_eval.py)
1. **Full native scale** — stop the 0.25 downscale; the blur was merging board+NIC.
   (alone: yaw 7.4→1.8° med, center 80→7.3 mm med on val)
2. **Largest connected component** — drop the *detached* base blob before fitting.
3. **Known-size fixed-box yaw** — a fixed 0.425×0.30 box can't stretch onto the *attached* NIC
   lobe (max-coverage placement, ±20° search around the `minAreaRect` seed) → snaps to true yaw.
   `minAreaRect` alone could not (attached clutter tilts it).
4. **Center ← magenta anchor** — `center = magenta_base − R(yaw)·MAG_BOARD`, immune to the
   cloud-centroid bias that the clutter causes. (magenta is 12/12 reliable even when the rect is 250 mm wrong.)

Self-contained: **cv2 + numpy only** — no SAM, no YOLO, no torch → drops into the live sweep cheaply.

## Result — vs GT
| method | yaw med / max | center med / max | pass |
|---|---|---|---|
| LIVE gray+minAreaRect (val 12) | 7.4 / **33.9°** | 80 / **263 mm** | yaw 4/12, ctr 6/12 |
| **FIX** magenta + known-size (val 12) | **1.3 / 8.2°** | **6.3 / 31.2 mm** | yaw 10/12, **ctr 12/12** |
| **FIX** (3 real eval poses) | **0.3 / 1.3°** | **5.3 / 6.9 mm** | **3/3 both** |

Before/after render: [fixed_board_viz_sweep_dump_val.png](aic_data/track2_sam/fixed_board_viz_sweep_dump_val.png)
([fixed_board_viz.py](aic_data/track2_sam/fixed_board_viz.py)) — RED=old live (flies to clutter),
BLUE=new fix (snaps to GT green). Per-trial SFP: trial0 23°/247mm→1.0°/6mm · trial2 25°/262→1.9°/2 ·
trial4 33°/244→2.5°/8 · trial8 19°/217→0.4°/14 · trial10 26°/260→8.2°/31 (straggler).

## Status
- Board pose is now SOLVED offline on the deployed method's data (center 12/12 <36 mm; yaw ≤1.3°
  on the 3 real eval poses). SC board pose was always fine — SC's failure is EXECUTION, not pose.
- NEXT: port fixes 1–4 into `PerceptionInsertSFP._sweep_board` (replace the gray+minAreaRect block),
  then re-run the SFP closed loop (trials 0/1).
