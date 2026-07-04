# Baseline Scores (local self-scoring, Gazebo eval_config, ground_truth:=false unless noted)

Scored via `~/ws_aic/aic_local/score.sh <policy> <ground_truth> eval <name>` → `~/aic_results/<name>/scoring.yaml`.
Max = 300 (3 trials × 100). Tier 1 validity (0/1) · Tier 2 motion quality (smoothness/duration/efficiency;
penalties: force −12, off-limit contact −24) · Tier 3 insertion (75 correct / −12 wrong / partial 38–50 / proximity 0–25).

| Policy | ground_truth | Total /300 | Notes |
|---|---|---:|---|
| **CheatCode** (ceiling) | true | **279.37** | inserts on all 3 trials (Tier-3 = 75 each). Ground-truth expert — not a valid submission. |
| **WaveArm** (floor) | false | **37.48** | valid; proximity/partial credit on 1 trial; no insertions. |
| **RunACT — pretrained ACT** (`grkw/aic_act_policy`) | false | **−21** | collided with the NIC-card mount (−24 off-limit contact on trial_1); plug ended 0.15–0.22 m from port on all 3 → Tier-3 = 0. Only Tier-1 (+3). **Trained on the author's small teleop set (velocity actions), NOT our data.** |
| **PerceptionInsert v1 — M4 first light** (native-1024 dual anchor + CheatCode descent, no force search) | false | **−37.6** | trial_1 reached **0.07 m** → Tier-3 proximity **+18.3** (first GT-free proximity credit), but −24 finger↔board contact; trials 2–3 missed grossly (0.34/0.41 m) + contact/force penalties. Pipeline proven; score dominated by *removable* collision penalties + an eval-scene generalization gap. |
| **InsertTuner force stack** (GT diagnostic; CheatCode glide + contact-driven endgame) | true | **277.7** | **3/3 insertions on the official eval** (75/75/75, 0 contacts, 0 force penalties, tug-verified latches). 1.9 pts below CheatCode's fresh 279.6 — all duration bonus. Execution solved on eval geometry; see `force_insertion.md`. |
| **PerceptionInsert v3 — M7** (M6-coverage checkpoint `best_epoch6_18mm` + spiral + yaw dither + sanity gate) | false | **+1.4** | **+39 over v1.** All trials now in proximity range (0.07/0.12/0.27 m vs 0.07/0.34/0.41); trial_2 fully clean (0 contacts, t3 17.9). Still 0 insertions — M7a diagnostic: port err **69–220 mm on eval scenes** (>30 mm criterion failed) despite 18 mm val → perception remains the bottleneck; reactive stack validated. |

## Reading of the ACT baseline (−21)
- **Below the WaveArm floor** — end-to-end ACT here is net *negative* because it **rams the board and never localizes the port** (0.15–0.22 m away). This is the exact end-to-end-IL failure mode on out-of-distribution scenes.
- Caveat: worst-case ACT (someone else's checkpoint, wrong action space for our expert). ACT trained on **our** 50 episodes would improve but remain a fragile end-to-end baseline (50 episodes is tiny for vision→action generalization).
- **Empirically supports the plan:** the winning path is **#7 learned perception (images → port pose) + force-guided analytic insertion**, not end-to-end IL. See `ranked.md`.

## Reading of PerceptionInsert v1 (−37.6) — M4 first light, 2026-07-03
First fully **GT-free** run of the `#7+#3` pipeline: 3 wrist images → dual-anchor net
(`~/aic_data/m35_native_1024/best.pt`, port+plug in `center_camera/optical`) → `base_link` via robot TF →
CheatCode alignment math + port median filter. Policy: `aic_example_policies/.../ros/PerceptionInsert.py`
(standalone; CheatCode untouched). v1 deliberately ships **without** the force-stop/spiral-search layer.

Per-trial: t1 = **0.07 m** final distance, Tier-3 **+18.3** *(closest any GT-free policy has gotten)* but
−24 finger↔board contact; t2 = 0.34 m, −24 arm↔cable contact −12 force; t3 = 0.41 m, −12 force.

- **What's proven:** the perception→TF→control wiring works end-to-end in the eval loop (net loads on host
  pixi env, eval publishes native 1152×1024, ~20 Hz inference keeps up, one trial genuinely close).
- **What's killing the score (both believed fixable):**
  1. **Off-limit contacts** (−24 ×2, −12 force ×2): with pose error, the CheatCode-style **open-loop descent
     rams** the board/cable. This is exactly the missing **#3 reactive layer** (stop-on-force + small spiral
     search) — not a perception failure.
  2. **Gross misses on 2/3 trials** (0.34/0.41 m ≫ the ~12 mm val error): the eval uses **leaderboard scenes
     outside the 50 training configs** → a real perception generalization gap, and/or a frame/convention bug
     that only bites certain poses. Needs per-trial logging of perceived-vs-actual pose to separate the two.
- **Not yet a win** — below the −21 ACT on raw total. But the failure modes are *diagnosed and separable*,
  unlike the ACT's (which never localized anything). Trial_1 minus the contact penalty ≈ **+32** — above the
  WaveArm floor — showing the ceiling once the reactive layer lands.
- **Pose error ≠ success (confirmed):** 5.7 mm relative val error still produced 0 insertions. The deciding
  metric is tier-3 success rate, as anticipated; resolution/accuracy work must be judged by score, not mm.

**Next (M5):** (1) force-stop + spiral search on descent (removes −24s; biggest lever); (2) log perceived vs
GT pose on eval scenes to split generalization-gap vs frame-bug; (3) re-score; then judge 224-vs-1024 by
success rate.

## M5 diagnostic (2026-07-03): far misses = perception coverage gap, NOT a frame bug
v2 policy added: force-stop + golden-angle spiral search on descent, per-trial state reset, and JSONL pose
logging (`/home/skr/aic_data/pi_pose_log.jsonl`). Diagnostic run with `ground_truth:=true` (GT frames
published **for logging only** — control stays perception-only): `~/aic_results/pi_diag/`.

Perceived-vs-GT port error on the eval scenes (val error was ~11 mm):

| Trial | GT port in training range? | Port err (med) | Plug err (med) | Final dist |
|---|---|---:|---:|---:|
| 1 | inside | **61 mm** | 49 mm | 0.11 m |
| 2 | **outside** (y=0.119 < train-min 0.179) | **372 mm** | 82 mm | 0.37 m |
| 3 | at edge (x,z at extremes) | **335 mm** | 78 mm | 0.48 m |

- **Verdict:** error scales with how far the eval board/port pose sits outside the 50 training configs →
  **textbook coverage/generalization gap**. The frame chain is fine (in-range trial 1 lands within striking
  distance; a frame-convention bug would corrupt all trials equally).
- Plug error also degrades 6–10× vs val → close-up appearance shift too, not just geometry.
- **Force-stop/spiral works mechanically** (contact detected → retract → spiral; no sustained-force penalty
  on trial 1) but can't rescue a 300+ mm target miss; the remaining −24 contacts (finger/upper-arm) are
  downstream of the perception error.
- Suspect kept open: net is **not conditioned on which port is the target** — on multi-port boards the
  regressor may lock onto the wrong port (0.3–0.4 m ≈ inter-port distances). Coverage fix + target-type
  conditioning both address it.

**Root fix (M6):** widen the data distribution — read the board-pose/port ranges from the vaulted
`eval_config.yaml`, regenerate configs to **span those ranges** (match ranges, don't copy scenes),
recollect (free in sim), retrain, re-score.

## Raw results
- `~/aic_results/{cheatcode,wavearm,runact,perceptioninsert}/scoring.yaml` (per-trial tiers + messages).
