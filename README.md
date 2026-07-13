# 🏆 OUR SOLUTION — GT-free cable insertion

A UR5e inserts a cable plug into a randomised task-board port, using **only** the RGB cameras and the
wrist force/torque sensor. **No ground truth anywhere in the control path.**

![GT-free SFP port_0 full seat](media/videos/seat_v2_sfp_port0_seated.gif)

*A fully GT-free full seat (2× speed).* The **green X** is our *perceived* port lock — it is what drives the
arm. The white ring is ground truth, **drawn for reference only, never read**.

## What worked

**1. Board pose — anchor on the magenta marker, not the point cloud**
- Segment the board at **full native resolution** (the deployed code downscaled first and lost the edges).
- Keep only the **largest connected component**, which drops the detached NIC-card blob that was dragging the fit.
- Fit yaw with a **known-size fixed box** (0.425 × 0.30 m) — a correctly-sized box *cannot* stretch onto an
  attached NIC lobe, whereas `minAreaRect` happily does.
- Take the centre from the **magenta marker**, not the cloud centroid: `center = M − R(yaw) · MAG_BOARD`.
  The centroid is biased by whatever the segmenter happened to include; the marker is not.
- → yaw ≤ **1.3°**, centre ≤ **6.9 mm** on all eval scenes.

**2. Port lock — gate to the *target* module before averaging**
- Back-project YOLO port detections onto the CAD z-plane of the board.
- **SFP:** gate to the target NIC card by board-Y, split the card's two 21.8 mm-apart openings by board-X,
  then estimate the rail slide and take a robust median. → **2–3 mm**.
- **SC:** the eval scene spawns **three** SC modules, and the original code took a median across *all* of
  them — landing **58 mm** from the true port. The same board-Y **module gate** fixes it. → **2 mm**.
- The lesson both times: *never average across candidates until you have selected the right module.*

**3. Motion — reset the IK branch in joint space before approaching**
- The camera sweep ends in a posture that leaves IK in a **folded branch**, which then swings the upper arm
  through the NIC card on the way to the port.
- No Cartesian command can fix this — the bad posture *satisfies* the requested pose.
- Fix: a joint-space move to a known-good `SAFE_HOME` between the sweep and the approach (`_safe_clear`).
- This one change **more than doubled** the GT-free result and cured a long-standing SC stall.

**4. Seat — make the contact test sign-insensitive**
- The deployed contact predicate required the wrist force to **increase**. But a plug landing on the SFP cage
  face **unloads** the wrist (21 N → 8 N). So the branch had **never fired in any run, ever** — the "force
  stack" was silently just a stiff straight-down ramp, and the "partial insertion" it scored was the plug
  *resting on the cage face*.
- Fix: trigger on **|ΔF|** (sign-insensitive) **or** a **re-arming stall watchdog**, then drop to **RCC
  compliance** `[40, 40, 90]` — laterally soft so the plug can self-align, axially firm so it still advances.
- The search now actually runs, and produces real full seats (above).

## Status

- **SFP** — seats GT-free, but *stochastically*: on an identical config the same policy swings between
  seating both trials and seating none. Not yet reliable enough to bank.
- **SC** — perception is **solved** (58 mm → 2 mm), but the plug is **mechanically blocked**: it tracks the
  port to within **0.1 mm laterally** for the whole descent, then hard-stops **18 mm above the port datum**
  and will not move even when commanded 33 mm deeper. Not a perception, trigger, or search problem.
  🎥 [`sc_trial_gtfree_blocked.mp4`](media/videos/sc_trial_gtfree_blocked.mp4)

Full history, failed approaches, and the things **not** to retry: [LAB_LOG.md](LAB_LOG.md).

## Code map

```
aic_example_policies/aic_example_policies/ros/
  PerceptionInsertSFPDrive.py   ⭐ best GT-free — safe-clear + frozen target
  PerceptionInsertSFP.py           board-pose fix + SFP slide-select
  PerceptionInsertSCFix.py         SC module gate
  PerceptionInsertGeo.py           CAD-z depth + magenta quadrant
  PerceptionInsertYOLO.py          YOLO detector + FK plug + approach/descent
```

```bash
# run it (no ground truth anywhere)
~/ws_aic/aic_local/score.sh aic_example_policies.ros.PerceptionInsertSFPDrive false eval my_run
```
