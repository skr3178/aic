# AIC — Lab Log (experiment tracking index)

**Single entry point for the whole project.** Terse, chronological, links to the detail docs — which
live in [`notes/`](notes/) (project docs) and [`aic_data/`](aic_data/) (external, ~90 GB — see its
[MANIFEST](/home/skr/aic_data/MANIFEST.md)). *Details are NOT duplicated here — follow the → link.*

Task in one line: insert a cable plug into a randomized task-board port (UR5e, /300 over 3 vaulted
trials). Plug pose + insertion "hands" are solved; **GT-free port localization on the eval domain is
the sole remaining bottleneck.**

---

## Status snapshot (2026-07-11)
- **Ceiling** 279.4 (CheatCode, GT, seats ALL 3 incl. SC) · **proven perception ceiling** 227.8 (GT port + FK plug + our hands, `kp_gtport`).
- **Best real GT-free (BANKED)** score: **92.2** (`sfp_free`, 2026-07-10 — **2.3× the old 40.1**): board-pose fix + SFP slide-select + **SAFE-CLEAR motion fix**. Both SFP lock the CORRECT opening within **2–3 mm of GT** and PARTIAL-seat (38 each); SC clean but aims 58 mm off (17.4). → see "MOTION SOLVED" below.
- **★ FORCE STACK WAS DEAD — REVIVED (2026-07-11, branch `seat-v2`, `5dbe781`). REAL FULL SEATS, but NOT BANKED.** The contact predicate `d_force > F_STOP` required a force **INCREASE**, but landing on the SFP cage face **UNLOADS** the wrist (21 N → 8 N ⇒ `d_force ≈ −13`) — **the branch never fired in ANY SFP run, ever**; the "force stack" was silently a stiff straight-down ramp. Sign-insensitive trigger + re-arming stall watchdog + RCC compliance ⇒ the search **finally runs** and yields **`tier_3 = 75` "Cable insertion successful"** (peaks **167.8 / 160.8** > 92.2). ⚠️ **NOT a result yet:** identical config spans **67.8 → 160.8**, and `seated=True` is a **false positive** (claimed in 5/6 trials, only 3 inserted; once at **40 mm** from the port). Seats happen iff `spiral_k ≤ 8` / offset < 10 mm; failures have the spiral wandering to **36 mm > port width**. → see the 07-11 ledger entry.
- **★ CheatCode has NO spiral, NO wiggle, NO force sensing at all** (read the source): GT lookup → interpolate to 0.2 m above → **monotonic 0.5 mm/step descent** → stop; its only feedback is an xy integrator on the **GT plug TF**. It seats **3/3 with ZERO search.** ⇒ **the spiral is not part of the task — it is compensation for perception error.** "CheatCode seats it" is evidence that a *good target needs no spiral*, NOT that a spiral rescues a bad one.
- **Motion ceiling (GT target, `sfp_reach_gt`) = 171.4** — with a perfect target our arm now **FULL-SEATS both SFP (75+75)** + SC partial 0.01 m. Motion is PROVEN; the 92.2→171.4 gap is *perception precision*, not motion.
- **~~Open bottleneck #1~~ SC EXECUTION STALL — SOLVED (2026-07-10).** The ~53 cm SC stall (which survived every prior fix, *even with a perfect GT port*) was the **sweep-end arm POSTURE poisoning the IK branch**. One fix (SAFE-CLEAR) removed it: SC now descends cleanly, **zero contacts**, 0.01 m partial w/ GT. SC's *remaining* miss is a **58 mm PERCEPTION error** on the SC port lock — no longer execution.
- **SFP two-port — SOLVED offline** (candidate→select→lock: CAD pair + both-visible frames estimate slide → fixed identity → classify): **0 % wrong-port, 0.4 mm, video-confirmed** (`yolo_viz_slide/`). Wired into live policy + a good perceived board pose → **both SFP trials now LOCK the correct opening** (see board-pose fix below).
- **BOARD POSE — deployed method was the real blocker, now FIXED (2026-07-10)** → [board_pose.md](board_pose.md). Live `gray+minAreaRect` was catastrophic on SFP (~23°/~250 mm; NIC-blob leak) — the banked SAM "≤1.5°" was never shipped and nothing measured the live method. New self-contained fix (full-scale + largest-CC + **known-size yaw** + **magenta-anchored center**, cv2+numpy) → vs GT **12/12 <36 mm center, yaw ≤1.3° on the 3 real eval poses**; ported into `PerceptionInsertSFP` and confirmed live (trial0 yaw −0.0°, trial1 2.0°, SC 75.1°, both SFP lock correct port). **Closed-loop `sfp_fix` = 0.57** — perception now correct on all 3, but correct locks don't seat (trial1 partial, trial0 misses 0.20 m w/ path penalty, SC stalls) ⇒ **EXECUTION is now the binding constraint**, not perception.
- **Corrected belief:** YOLO is **NOT** blind up close — it's blind FAR, sharp CLOSE (100 %, 0–1 mm down to ~24 cm; measured via CheatCode-to-contact). Earlier "blind up close" was a stall/viewpoint/desync confound.
- **Solved & banked:** insertion/force stack (M9) · plug = FK·T_grasp (M14) · **board pose — DEPLOYED** (full-scale + largest-CC + known-size yaw + magenta-anchored center; live in `PerceptionInsertSFP`) · **MOTION — DEPLOYED** (`_safe_clear`: joint-space IK-branch reset before the approach; `PerceptionInsertSFPDrive`) · magenta yaw disambiguation · CAD-z depth · gated-lock (A5a, 3 mm).
- **NEXT (biggest levers, from `sfp_free` 92.2):** (1) **A — two-segment approach (+36, LOW RISK)**: hold altitude while translating, *then* straight down. t0 currently descends **diagonally while translating** and drags the upper arm across the NIC card (−24 contact, −12 force). Deterministic path fix, no variance gamble. (2) **SC port lock 58 mm off ≈ +40** — SC perception bug (offline was 6/6 ≤1 mm). (3) **Force stack** — ⚠️ **TRIED (`seat-restore-v1`) → NET REGRESSION** (seat rate 1/4, mean 85.9 < 92.2) *and the spiral still never ran* (my stall reference was anchored at the descent start). Needs a rolling-window stall detector / InsertTuner's near-mouth glide, then **repetitions** — not a single run.
  ⚠️ **CORRECTED:** the seat failure is **NOT** an orientation problem — our port orientation measures **0.00° vs GT** on all 3 eval trials.
- **Active thread (PARKED):** real-time re-lock (continuous target updating) — the SC approach blocker is now GONE, but the target is static+correct, so re-lock buys nothing until precision (above) is fixed; still needs stamp-synced TF (R4 desync).

---

## Execution stack — 3 levels · options · deployed · DEFAULT LINE (2026-07-10)
The pipeline splits into three independent levels. **Default line = CheatCode (GT-driven, seats 3/3, 279)** — the only reference path that seats, so we compare against it. (Challenge floor = WaveArm 37.5 no-insert; shipped learned baseline = ACT −21.)

**LEVEL 1 — DETECTION (produce the port target)** — ✅ **SOLVED (= GT quality on all 3)**
- GT port TF — oracle — **← DEFAULT LINE (CheatCode)**
- DualPoseNet (M3.5) ✗ didn't transfer · KPNet (M10) ⚠ 5 mm close, didn't transfer · SAM board+CAD ⚠ good offline, clutter-degrades
- **YOLO candidate→select→lock + fixed board pose** — ★ **DEPLOYED** (full-scale board magenta+known-size → YOLO candidates → slide-select) → correct all 3

**LEVEL 2 — MOTION (approach: sweep-end → target → descend)** — ✅ **SOLVED 2026-07-10 (was THE bottleneck)**
- CheatCode `calc_gripper_pose` → straight-in from a clean start — ✓ 3/3 — **← DEFAULT LINE** (never sweeps ⇒ never enters the bad IK branch)
- **★ FIX — `_safe_clear`:** sweep-end → **JOINT-SPACE to `SAFE_HOME`** (resets the folded IK branch) → Cartesian interp → descend → force stack. **DEPLOYED** (`PerceptionInsertSFPDrive`) → GT target **171.4** (both SFP FULL SEAT), GT-free **92.2**.
- ROOT CAUSE was the **sweep-end POSTURE** (same TCP pose reachable in a folded branch → upper arm into the NIC card). Cartesian commands can't fix it; only joint-space can.
- Superseded/failed attempts: GeoHome ✗ · GeoReplayA ✗ (SC stalled even w/ GT) · frozen-target-only (`sfp_drive`) ✗ (refuted target-drift) · continuous re-lock ◻ still PARKED (Phase 2)
- **Residual (open):** t0 −36 (contact+force) — approach still descends *diagonally while translating* → needs two-segment path (hold altitude → hover over port → straight down).

**LEVEL 3 — FORCE STACK (seat once at target)** — ❌ **BROKEN — it has NEVER RUN in the closed loop** *(corrected 2026-07-10; previously mislabelled ✅ SOLVED)*
- CheatCode "seat" — ✓ 3/3 — **← DEFAULT LINE** — but it has **no force stack at all**: pure position control, exact GT target
- **InsertTuner (M9)** — ✓ **277.7 / 3/3 with GT** — *but it is a standalone TUNING RIG, never wired into the perception policy* (`force_insertion.md` open item #3, never done)
- **DEPLOYED (`PerceptionInsertYOLO`)** — ❌ a **DEGRADED PORT** of InsertTuner: `F_STOP` 8.0 (vs 4.0 = **unreachable**; ctrl makes only ~5–6 N at the mouth) · stall detect **dropped** · RCC compliance **dropped** (falls back to stiff `[90,90,90]` = CheatCode-grade). **ZERO spiral log lines in any run, ever.**
- **RESTORATION ATTEMPT (`seat-restore-v1`)** — ❌ **NET REGRESSION** (seat rate 1/4, mean **85.9 < 92.2**); *and the spiral **still** never ran* (my stall reference anchored at the descent start). One full seat did occur ⇒ **compliance CAN absorb 2–3 mm**, but it is a coin-flip.

**Takeaway (corrected 2026-07-10):** **Levels 1 (detection) and 2 (motion) work. Level 3 (seat) does NOT — and never has.** The "partial insertion 0.05 m" we bank on every SFP trial is really **the plug resting on the SFP cage face**, having never searched. Our best GT-free score (92.2) is therefore built on *two working levels and one inert one*. Remaining gap to the 171.4 GT-target ceiling: (a) **the seat mechanism** (fix the stall reference so the search can run; then repetitions), (b) **t0's −36 approach contact** (two-segment path), (c) **SC lock 58 mm off**. → [pipeline.md](pipeline.md), [board_pose.md](board_pose.md)

---

## Experiment ledger
`id · what · outcome · → detail doc`. Milestone chronology from [RUN_PLAN](notes/RUN_PLAN.md).

### Baselines & data pipeline (M0–M2)
- **M0** env (pixi + ROS Kilted, torch on Blackwell) — ✅ → [RUN_PLAN](notes/RUN_PLAN.md)
- **M1** self-scoring baselines — ✅ WaveArm **37.5** floor · CheatCode **279.4** ceiling · pretrained ACT −21 → [baseline_scores](notes/baseline_scores.md), [CHEATCODE_RUNBOOK](notes/CHEATCODE_RUNBOOK.md)
- **M2** config-gen + collection (`perception_v1`, 49 eps) — ✅ images↔GT labels → [collect/PLAN](collect/PLAN.md)

### Perception nets — first attempt (M3–M8)
- **M3 / M3.5** port-only, then dual-anchor (port+plug from vision) — ✅ ~9 mm close; rel ~12 mm → [perception_results](notes/perception_results.md), [arch](notes/arch.md)
- *(res ablation)* native 1024² beats 288×256 — ✅ rel **5.7 mm (−52%)**
- **M4** `PerceptionInsert` wired, first GT-free score — ✅ **−37.6** → [baseline_scores](notes/baseline_scores.md)
- **M5 / M6a** far misses = **coverage gap** → `perception_v2` (169 eps, 35% seat) — ✅ → [collect/M6_PLAN](collect/M6_PLAN.md), [perception_v2_episodes](notes/perception_v2_episodes.md)
- **M6c–e** v2 retrain @1024 (⏸ ep6, 18 mm) · conditioned retrain (219 eps) · v3 policy (yaw dither + sanity gate) ✅
- **M7a** GT diag — ✅ ran / ❌ criterion: port **69–220 mm** on eval → perception is the bottleneck
- **M7b** real score (gt:=false) — ✅ **+1.4** (best real GT-free ever); still 0 insertions
- **M8** iterate — ⏳ still perception-limited

### Force-insertion "hands" (M9)
- **M9** InsertTuner force rig — ✅ **eval parity 277.7 vs CheatCode 279.6, both 3/3** → [force_insertion](notes/force_insertion.md)

### Domain-gap diagnosis (M10–M14)
- **M10** keypoint + triangulation rebuild — ⚠️ proxy **1.9 mm** on v1 but **did NOT transfer** → [perception_results](notes/perception_results.md), [keypoint_pnp_perception_plan](notes/keypoint_pnp_perception_plan.md)
- **M11** eyes+hands live (`PerceptionInsertKP`) — ❌ **−43.3**; SFP 8–40 mm, **SC 300–430 mm catastrophic**
- **M12** ablation — ✅ cause = **scene-content/geometry OOD** (ruled out scale/photometric/architecture)
- **M13** 3rd-person diagnosis — ✅ 3 failure modes: (A) wrong connector, (B) SC off-frame 78–84%, (C) plug twist
- **M14** plug grasp is **type-fixed** — ✅ plug pose = FK(gripper)·T_grasp[type], zero vision → solves mode C

### Board geometry & Stage-2 (M15–M18)
- **M15** ceiling validation (GT port + FK plug) — ✅ **227.8**; port perception = sole gap → [problem_formulation](notes/problem_formulation.md)
- **M16** board→port geometry — ✅ port orientation = **1 scalar (board yaw)**; board verified flat
- **M17** Stage-2 port perception plan — 📋 → [stage2_port_perception_plan](notes/stage2_port_perception_plan.md)
- **M18** error budget + board-extractor **"A1"** — ⚠️ **A1 FAILED** (dark-pixel board fit, yaw err 24–42°). Rail slide: SFP ±2.3 cm ⊂ spiral 3.6 cm (airtight); **SC ±6.0 cm ⊄ spiral → needs 1-D rail search** → [stage2 §10](notes/stage2_port_perception_plan.md)
  - ⚠️ **Two different "A1"s:** Stage-2 §10 "A1" = board-size-prior fit (FAILED, above) ≠ action-plan "A1" = eval-frame harness (below).

### Action-plan fix + YOLO detector (T0 / A-series, 2026-07-07/08)
- **T0** FK-reproject label diagnostic — ✅ **PASS**: mount recovered 0.8 mm; the **85 px moving-frame label corruption** explained & fixable → [action_plan_perception_fix](notes/action_plan_perception_fix.md)
- **A2** clean labels (FK-reprojection + SAM-mask QC) → `yolo_ds` (11.5k, 46 scenes) — ✅ build
- **YOLO-World zero-shot** (pre-finetune probe) — ❌ open-vocab can't recognize the niche synthetic port: top-1 box **20–62 % of image**, conf ≤0.39 (SC "100 % contains GT" was a 62 %-of-frame blob). Class-agnostic **SAM** segments it (0.4 % tight mask) but a *semantic* detector doesn't → motivated **fine-tuning** over zero-shot.
- **A3a** fine-tune YOLOv8m on clean labels — ✅ mAP50-95 **0.847** (our-data)
- **A3b** THE GATE (detector on real eval frames) — ✅ **2/3, SC 3.9 mm** (256 res 0/3 → **resolution decisive**); lone fail t1 = class confusion. First learned detector through the gate; "domain gap" was confounded by 85 px labels + low res → [yolo_finetune_findings](notes/yolo_finetune_findings.md)
- **A5a** gated-lock **EXTRAPOLATE** (not interpolate; causal + exact since port is static) + FK-track — ✅ **3.0 mm median (p90 5.7), 100 % within 3.6 cm basin (49/49)**. Naive per-frame jitter **152 mm** (p90 331) → speed + jump-reject gate **8.8 mm** (~17×). Collapses deployment to *getting one good lock*. *Insight: moving-camera desync is the shared root of the 85 px label corruption (T0) AND the earlier jagged KP trajectory — synced-FK labels + gated-lock fix both.*
- **Test #2** at-rest GT-free lock probe (SAM-auto + heuristic) — ⚠️ **oracle 43 % / central-heuristic 27 %** (SFP 17 %, SC 36 %), centroid err 108 px. SAM *segments* but **disambiguation is the wall** — no simple GT-free rule picks the port among ~16 module-sized masks ⇒ confirms a **trained detector (A3) is required**; SAM-auto + rule is not a no-training shortcut.

### SAM board-geometry track (parallel, `aic_data/track2_sam/`)
- SAM segments the board cleanly on eval frames (vs CV gray-threshold total fail) — ✅ GO → [track2_sam/FINDINGS](/home/skr/aic_data/track2_sam/FINDINGS.md)
- Sweep test — board pose → `board·offset` → port: **2/3 within 36 mm** (t1 3.2, t2 SC 16.7; t0 fail 102–155). Known-size CAD fit **solved yaw ≤1.5°**. t0 fail diagnosed = near-side density spill (not coverage).
- Complementary with YOLO: **SAM fails t0, YOLO fails t1** → ensemble path to 3/3.

### YOLO closed-loop scored eval (2026-07-08)
- `yolo_score_diag` — drop-in per-step YOLO → **lock never formed** (YOLO blind at close-up insertion viewpoint); no score.
- `yolo_sweep_score` — **sweep-lock** design (`PerceptionInsertYOLOSweep`): lock the port during the sweep, hold through insertion → lock forms & holds (insufficient-TF 371→0), but **total −51.48, 0/3** — triangulated **depth** wrong (narrow baseline). **Next fix:** back-project YOLO xy to the flat-board z-plane instead of triangulating.

### GT-free closed-loop + SC-execution forensic + YOLO-vs-range (2026-07-09/10)
- **DEPTH FIX lands** — `PerceptionInsertGeo` (CAD-z back-projection replaces the −51 triangulation, + magenta yaw + CAD offsets, fully GT-free): closed-loop **geo_v1 = 40.1/300** — beats the +1.4 floor, reverses the −51.
- **Held-out perception (offline, 12 wide-slide scenes, `ensemble_val_gtfree2.py`)** — GT-free, **on the correct opening**: SC 6/6 ≤1 mm; SFP 4/6 (2 flagged = two-port ambiguity, sibling SFP openings **21.8 mm** apart). Dropping FUSE-averaging (trust confident YOLO) fixed SC from 8–15 → ≤1 mm.
- **SC trial failure = EXECUTION, not perception.** Forensic (`kp_pose_log.jsonl` + replays): with a PERFECT GT port (`GeoReplayA`) the SC arm STILL stalls ~53 cm short, z never descends, force baseline (no contact). Ruled out: perception, sweep end-pose (`GeoHome`), plug model (`GeoRealPlug` real plug frame), orientation target (validated sane, insertion axis correct). `calc_gripper_pose` is IDENTICAL to CheatCode. **CheatCode seats ALL 3 (75/75, ~279)** → the SC pose IS reachable; the stall is a bug in *our* approach path — **still open** (sweep-presence vs per-step-perception timing).
- **Score is timing-fragile** — the SAME lock scored **+40 / −45 / −21** by system/recording load.
- **YOLO is NOT blind up close — OVERTURNS the earlier `yolo_score_diag` belief.** Experiment: CheatCode drives the plug to contact (seats all 3 = valid) while a background thread PASSIVELY logs YOLO vs true camera→port range (`CheatLogYolo` / `CheatVizYolo`). Result: **YOLO is blind FAR, sharp CLOSE** — 100 % detection, **0–1 mm error down to ~24 cm** (closest the wrist cam reaches); error SHRINKS on approach (54 → 1 mm). The old "3 % / blind" was CONFOUNDED (our policy stalled/plateaued → bad viewpoints, measured on fast-motion desynced frames). Videos: `aic_data/yolo_viz/trial{0,1,2}_yolo.mp4` (red = YOLO, green = GT).
- **Real-time re-lock REOPENED.** Phase 0 (YOLO on OUR stalled frames) had shelved it; this clean CheatCode measurement shows YOLO **and** KPNet (5 mm close) both localize on approach → continuous target updating is viable *given a clean approach*. CAVEAT: R4 label desync (~85 px on moving frames) → re-lock needs **STAMP-SYNCED TF** (camera pose at image timestamp); SC re-lock gated on the execution stall. Plan: `notes/dynamic-relock-plan.md`.
- **SFP TWO-PORT DISAMBIGUATION — SOLVED** (candidate→select→lock, `sfp_slide_eval.py` / `CheatVizSlide`). Raw YOLO is a candidate generator only: sharp (0-1mm) but picks the WRONG SFP instance (~46% SC, ~11% SFP-1) and FLICKERS between the two 21.8 mm-apart openings; a lone opening is fundamentally ambiguous (slide ±23 mm ≈ spacing). FIX = CAD/board trick: CAD gives the nominal pair → the ~25 % **both-openings-visible** frames ESTIMATE the rail slide (median) → target/sibling identity FIXED → classify every detection in board frame → temporal-smooth only the target. OFFLINE (CheatCode-to-contact candidate log): **trial0 (port_0) 100 % on-target, trial1 (port_1) 97 %, BOTH 0 % wrong-port, 0 % flicker, 0.4 mm median**; est slide within ~1 mm of GT. VIDEO confirms (the metric alone had hidden a fragile-init bug the video caught): `aic_data/yolo_viz_slide/trial{0,1}_slide.mp4` — "acquiring identity" until slide committed, then GREEN lock dead-on the correct opening (err 0 mm), while raw red candidates still flicker. YOLO is VERY accurate once CheatCode drives the EE and the identity is fixed. CAVEAT: offline used GT board pose (deploy = perceived ~1 cm) + not yet closed-loop. Next: wire selector into live policy → SFP closed-loop (trials 0/1).
- **BOARD POSE — the LIVE method WAS the blocker; measured, fixed, wired (2026-07-10).** → [board_pose.md](board_pose.md). The banked "board yaw ≤1.5° (SAM known-size)" was a YAW-only, clean-scene result that **was never shipped** — the deployed `_sweep_board` silently ran `gray-threshold + minAreaRect`, and **no evaluator watched the deployed method**. New evaluator reproducing it (`track2_sam/live_minarea_eval.py`) on 3 real (`sweep_dump`) + 12 val (`sweep_dump_val`) vs GT: **catastrophic on EVERY SFP trial (~23° yaw, ~250 mm center), near-perfect on SC** — systematic per scene-type, not "run-dependent". Render (`live_board_viz*.png`) shows why: a **detached NIC/base blob** leaks into the gray cloud and drags the rectangle; **magenta stays dead-on**. Head-to-head vs PnP/SAM: **"just port SAM" would NOT have fixed it** (SAM yaw ok ~4° but SFP **center ~100 mm off** via asymmetric coverage; PnP abstains 5/12). FIX (`track2_sam/magenta_board_eval.py`, **cv2+numpy only — no SAM/YOLO/torch**): (1) **full native scale** — stop the 0.25 downscale (blur merged board+NIC; alone: yaw 7.4→1.8° med, ctr 80→7 mm med); (2) **largest connected component** drops the detached blob; (3) **known-size fixed-box yaw** — a fixed 0.425×0.30 box can't stretch onto an *attached* NIC lobe (minAreaRect could); (4) **center ← magenta anchor** `= M − R(yaw)·MAG_BOARD` (immune to the cloud-centroid bias). Offline vs GT: **val 12 → yaw 1.3/8.2° med/max, center 6.3/31 mm (ctr 12/12 <36 mm); 3 real eval poses ≤1.3°/≤6.9 mm**. Ported into `PerceptionInsertSFP._sweep_board` (replaced the gray+minAreaRect block; +`_largest_cc`/`_best_box_yaw`). **Live closed-loop confirms**: trial0 `board FIX yaw −0.0°` → **LOCK sfp_port_0** (was −26.4°/AMBIGUOUS), trial1 `yaw 2.0°` → **LOCK sfp_port_1**. Both SFP openings now locked on the correct instance. **Closed-loop `sfp_fix` = 0.57/300** — all 3 board poses now GOOD (trial0 yaw −0.0°, trial1 2.0°, **SC trial2 75.1° vs GT 76.8°** — SC pose was already fine) and both SFP LOCK the correct opening, **BUT the correct locks did NOT convert to seats**: trial1 partial (0.05 m, tier_3 38) — same as before; **trial0 correct-lock port_0 yet final plug-port 0.20 m, tier_3 0 + tier_2 −36 path penalty** (regressed vs geo_v1's trial0 partial); SC stall 0.58 m (unchanged bottleneck #1). ⇒ **Perception (board pose + port selection) is now CORRECT on all trials; EXECUTION is the binding constraint** — the corrected yaw did not fix the trial0 approach (needs a forensic: why does trial1's correct lock reach+partial-seat but trial0's identical-looking lock miss by 0.20 m with a large path penalty).
- **MOTION SOLVED — the SWEEP-END POSTURE was the bug. `SAFE-CLEAR` fix → GT-free 40.1 → 92.2 (2.3×) (2026-07-10).** `PerceptionInsertSFPDrive`.
  - **Diagnosis chain.** (a) Froze the target (`sfp_drive`, kill per-step re-perception) → **t0 STILL jammed** ⇒ REFUTED the "target drift" hypothesis. (b) **GT ABLATION** (`sfp_drive_gt`): froze a **PERFECT GT target** — `[-0.419 0.305 0.133]`, which turned out to be **within 1 mm of our perceived lock** — and t0 **jammed identically** (upper_arm↔NIC-card, 95 N/30 s, 0.20 m). ⇒ **NOT perception, NOT orientation, NOT target-source → PURE REACHABILITY.** Perception was exonerated *against GT*.
  - **ROOT CAUSE.** The **arm's posture at sweep-end** put IK in a *folded branch*: the same TCP pose is reachable in many postures, and from the sweep-end posture the solver picked one that swings the **upper arm into the NIC card**. Cartesian commands CANNOT fix this (the bad posture *satisfies* the pose). CheatCode never sweeps → never enters the bad branch → that's the whole reason it seats.
  - **THE FIX (`_safe_clear`).** Between sweep and approach, command **JOINT-SPACE** to the canonical safe home `SAFE_HOME=[-0.16,-1.35,-1.66,-1.69,1.57,1.41]` (from the shipped GentleGiant/SpeedDemon) with GentleGiant's low-stiffness/high-damping gains (smooth; jerk is scored) → **resets the IK branch** → then the existing Cartesian interp + descent + force stack. Also fixed a bug of mine: `_locked_xyz` wasn't reset per trial (t1/SC had been reusing t0's target).
  - **RESULT — GT target (`sfp_reach_gt`) = 171.4:** both SFP **FULL SEAT (75+75, "Cable insertion successful")**, SC **partial 0.01 m** (39.4). ⇒ **motion PROVEN.**
  - **RESULT — GT-FREE (`sfp_free`) = 92.2** (prev best 40.1): SFP port_0 **13.8** (t3 38.0 partial 0.05 m, but −24 contact −12 force), SFP port_1 **48.5** (t3 38.0, **zero contacts** — cleanest trial), SC **29.9** (t3 17.4, clean but 0.06 m). Both SFP locked the **correct opening**, within **2–3 mm of GT**.
  - **REMAINING GAPS (92.2 → ~240 potential).** (1) **SFP partial→FULL seat ≈ +74** — with GT these seat fully, so the blocker is *orientation* (our board yaw 1–3° off vs GT's exact quat) binding the plug. (2) **t0 contact/force −36** — the approach still descends *diagonally while translating* (TCP z 0.328→0.271 during the lateral move) ⇒ fix with a two-segment path (hold altitude → hover over port → straight down). (3) **SC port lock is 58 mm off GT** ⇒ SC's miss is now **PERCEPTION**, not motion (offline SC was 6/6 ≤1 mm → live SC lock has a bug).
  - **🎥 VIDEO — GT-FREE insert, both SFP trials** (`SFPDriveViz`, control 100 % GT-free; `gt:=true` used ONLY to draw the reference ring):
    [sfp_trial1_gtfree.mp4](aic_data/sfp_free_viz/sfp_trial1_gtfree.mp4) (**sfp_port_0**) · [sfp_trial2_gtfree.mp4](aic_data/sfp_free_viz/sfp_trial2_gtfree.mp4) (**sfp_port_1**).
    **GREEN X = our PERCEIVED frozen lock (drives the arm)** · white ring = GT (reference only) · yellow = plug tip (FK).
    HUD reads e.g. `GT-FREE sfp_port_1 yaw +2.8deg tip->X 43mm **lock err 2mm** F 8N` — the X sits *on top of* the GT ring,
    on the **correct** one of the two SFP openings, and holds rock-steady through the approach (frozen ⇒ zero flicker).
  - **Credit:** the parts were already in the codebase (`JointMotionUpdate` API, `SAFE_HOME`, gentle gains) — **the insight that the sweep-end posture poisons the IK branch, and must be reset in JOINT space, was the missing piece.**

- **THE DEPLOYED FORCE STACK IS A DEGRADED PORT OF InsertTuner — AND HAS NEVER RUN (2026-07-10).** ⭐ *biggest finding of the session; all 4 probe experiments FAILED but produced it.* → branch `force-insert-v2` (NOT merged).
  - **THE BUG.** `PerceptionInsertYOLO`'s descent detects contact ONLY as a force **INCREASE** (`if d_force > F_STOP`, **F_STOP = 8.0 N**). But when the plug (2–3 mm off) lands on the **SFP cage FACE**, the board takes its weight and the wrist **UNLOADS**: force **DROPS 20.9 → 8 N**. The trigger can never fire. **grep proves it: ZERO spiral/contact log lines across EVERY SFP run, ever.** The "Partial insertion 0.05 m" was never a partial — it is **the plug resting on the cage face**, frozen (TRACK: `tip_z` stuck at 0.179, `tip→target` stuck at 46 mm for 300 steps) while we uselessly command it lower. GT seats because an exact target slides straight in and never touches the face.
  - **WHY IT'S UNREACHABLE BY CONSTRUCTION.** The VALIDATED rig `InsertTuner` (M9, **277.7 / 3-3 w/ GT**) uses **F_STOP = 4.0** with the comment *"impedance can only make **~5–6 N at the mouth**"*. The deployed copy **raised it to 8.0 — above what the controller can even produce.**
  - **WHAT ELSE THE PORT DROPPED** (deployed vs InsertTuner): **stall detection** (`stalled = commanded ↓12 mm but tip moved <4 mm` — `if df > F_STOP or stalled`) → **GONE**; **RCC compliance** (`STIFF_CONTACT=[40,40,90,…]` lateral-SOFT/axial-FIRM + `SOFT_MARGIN` proximity switch) → **GONE** (deployed passes NO stiffness ⇒ falls back to the stiff `[90,90,90]` defaults = *identical to CheatCode*); also `RETRACT` 6→4 mm, `Z_MIN` cap, per-type `SEAT_DIST`. ⇒ **the deployed policy silently reverted to CheatCode-grade STIFF position control.** A rigid plug jams on the face instead of complying into it.
  - **This is `force_insertion.md` OPEN ITEM #3 ("wire the tuned endgame into the perception policy") — never done.** And **OPEN ITEM #1 ("offset sweep on eval-like poses → capture-basin radius") — never done**, so the stack's tolerance at our 2–3 mm offset is **UNMEASURED**.
  - **4 PROBE-PATTERN EXPERIMENTS — ALL WORSE THAN BASELINE 92.2:** slide **79.2** · fermat **71.1** · raster **58.6** · golden-no-dither **57.1**. **Lateral search is the WRONG LEVER**: every method *lost* tier-3 by dragging the plug OFF the spot that at least scores a partial (SFP tier-3: baseline **76.0** → 62.9 / 49.4 / 49.7 / 44.4). SC delegated & unchanged throughout (17.4–17.5) — isolation verified.
  - **MY OWN BUGS (fermat/raster/golden runs are INVALID):** (a) tip-z stall detector **false-fires right after each retract** (tip moves UP ⇒ "didn't descend" ⇒ blocked) → they burned 114/169/31 probes in ~3 control steps each and never tested one; (b) force-guided bias sign **inverted** (`atan2(-fy,-fx)`), sending the slide search 180° off — its "drop-in" was the plug **falling off the cage edge**.
  - **GENUINE FINDINGS KEPT:** (1) **tip-z stall is the correct contact signal** (it fired at `dF −2.4 N`, far below any force threshold) — *and it was already in InsertTuner, dropped in the port*; (2) **the lateral force vector points AT the hole** — within **12–20°** of the true correction on both SFP trials (`(-0.2,-6.9)N` vs needed `(-0.7,-1.8)mm`; `(+0.7,-5.9)N` vs `(+1.0,-3.0)mm`); (3) **our port ORIENTATION is EXACT (0.00° vs GT)** on all 3 eval trials → the old ±2.9/5.7° yaw dither was *injecting* an error we don't have.
  - **NEXT:** restore the REAL force stack (F_STOP 4.0 + stall + RCC compliance) — a *restoration of validated code*, not an invention; it simultaneously closes open items #1 and #3. Caution: on hard wide-pose scenes at **zero** offset InsertTuner seated only **1–2/10** SFP, failing **at the mouth (45.8–46.3 mm — our exact 46 mm)**; SFP entrance→seat funnel is **45.8 mm** (SC's is 15.6 mm). Those failures were blamed on *"commanded-pose/wrist-configuration geometry"* — **the same posture bug SAFE-CLEAR just fixed**, so the stack may have been handicapped by it.
- **SEAT RESTORATION (InsertTuner force stack) — NET REGRESSION; and the spiral STILL never ran (2026-07-10).** → branch `seat-restore-v1` (**NOT merged**). `PerceptionInsertSFPSeat`.
  - **WHAT WAS RESTORED:** `F_STOP` **8.0 → 4.0** · **stall detection** re-added · **RCC compliance `STIFF_CONTACT=[40,40,90]`** (lateral SOFT / axial FIRM) + `SOFT_MARGIN` proximity switch · `RETRACT` 6 mm + `CLEAR_WAIT` · `Z_MIN` cap · entrance-aware seat test. `entrance_h` from **CAD (SFP 45.8 mm — GT-free)**; InsertTuner had read it from a GT frame.
  - **⚠️ RESULT — WORSE THAN BASELINE ON AVERAGE. Seat rate 1/4.**

    | run | SFP port_0 | SFP port_1 | total |
    |---|---|---|---|
    | `sfp_seat` | Partial 0.04 m | ✅ **"Cable insertion successful"** (tier_3 **75**) | **132.2** ← the ONE seat |
    | `sfp_seat_v1` | No insertion | No insertion | 68.0 |
    | `sfp_seat_v2` | No insertion | No insertion | 88.6 |
    | `sfp_seat_viz` | Partial 0.04 m | No insertion | 54.6 |
    | | | **MEAN** | **85.9** |
    | **baseline `sfp_free`** (stiff) | Partial **38** | Partial **38** | **92.2** *(reliable, never seats)* |

  - **WHY IT REGRESSES:** compliance trades a **guaranteed partial (38)** for a **~25 % shot at a full seat (75)**. When it misses, the laterally-soft plug **drifts off the port** and scores *"No insertion" (~25)* instead of resting on it for the partial. EV ≈ 41/trial vs the baseline's reliable 38 — **not worth the variance.**
  - **⚠️ MY BUG — THE SPIRAL STILL NEVER RAN.** Every run logs `SEAT outcome: floor (spiral 0)`. `z_ref`/`tip_z_ref` are anchored at the **DESCENT START (200 mm up)**, so `(tip_z_ref − tip_z)` ≈ 190 mm and the `< 4 mm` stall test **can never be true**. **InsertTuner escapes this because its Phase C begins only 10 mm above the entrance** (after a servoed glide) — its reference is **LOCAL**. Starting the same loop from 200 mm invalidates it. ⇒ **the one seat was PURE COMPLIANCE LUCK, not the force stack working. The search has STILL never executed in the closed loop.**
  - **FIRST GT-FREE FULL SEAT did happen** (`sfp_seat` trial_2, tier_3 **75**) ⇒ **compliance CAN absorb a 2–3 mm error** — it is just a coin-flip at that offset. **No video of it exists:** the only recorded run (`sfp_seat_viz`) is one where port_1 FAILED — and the recording *itself* (PNG encode+write inside the 50 ms control loop) stalls the descent and costs the seat (**132.2 → 54.6 on identical code**). Viz since fixed to JPEG→RAM + post-trial flush.
  - **WHY port_0 FAILS AND port_1 SEATS** — approach GEOMETRY, not perception (both locks 2–3 mm from GT) and not the seat code (identical): from the same SAFE-CLEAR pose, **port_0 needs +113 mm in y (OUTWARD) and 519 mm reach; port_1 needs −72 mm (INWARD) and 464 mm.** The approach **descends while translating**, so port_0's longer outward sweep drags the upper arm across the NIC card (−24 contact, 86 N grind). ⇒ **option A (two-segment path) is the fix.**
  - **CORRECTIONS to earlier claims in this log:** (1) *"lateral search is the wrong lever"* — established only under **STIFF** gains; **untested under compliance** (all 4 probe variants ran at `[90,90,90]`, i.e. dragging a RIGID plug). (2) *"the intended mechanism is a feed-forward downward push"* — **WRONG**: `set_pose_target` hardcodes `feedforward_wrench_at_tip = 0`, and **InsertTuner never used one** — its validated 277.7 config is **compliance + position descent**, no feed-forward.
  - **NEXT:** fix the stall detector (**rolling window**, or replicate InsertTuner's glide so contact detection begins near the mouth) → the spiral finally gets a chance → **re-test WITH REPETITIONS**. *(M9 already warned "run-to-run variance is real; sweeps need repetitions" — single runs were reported as results twice today and were wrong both times.)* Prefer **option A first**: a deterministic path fix with no variance gamble.

### SEAT V2/V3 — the force stack was DEAD; revived it; got real full seats (branch `seat-v2`, 2026-07-11)
Commit `5dbe781`. **NOT BANKED** — see variance below. Files: `PerceptionInsertSFPDriveSeatV2/V3.py`, `SeatV2Viz.py`, `SFPDriveViz.py`.

- **ROOT CAUSE — the contact predicate had the WRONG SIGN, so the force stack never ran, in ANY SFP run, ever.** The test was `if d_force > F_STOP` (F_STOP=8.0, `d_force = force − baseline`) — it required force to **INCREASE**. But when the plug lands on the SFP cage FACE, the board takes its weight and the wrist **UNLOADS**: force goes **21 N → 8 N**, i.e. `d_force ≈ −13`, **NEGATIVE**. The branch never fired → `spiral_k` stayed 0 → `sx=sy=dyaw=0`, no stiffness ever passed ⇒ default `[90,90,90]`. **At runtime the entire "force stack" reduced to a stiff straight-down position ramp — functionally CheatCode's descent.** Proof: **ZERO force-stack log lines in every SFP run in project history.** ⇒ every prior spiral-PATTERN experiment (golden/fermat/raster/slide) was testing a search **that never executed**.
- **THE FIX (V2 → V3).** (1) **TRIGGER** sign-insensitive: `|force − f_base| > F_ABS`, **OR** a **STALL** watchdog — an UNLOAD is contact too. (2) **STALL = RE-ARMING**: the reference **advances whenever the tip actually moves**, so the test stays LOCAL. *(This was my repeat bug: V2's first cut anchored the reference at the DESCENT START, 200 mm up, so `actual_progress ≈ 190 mm` and the "<4 mm" test could never be true — the exact error already logged above.)* (3) **COMPLIANCE**: `[90,90,90]` free descent → `[40,40,90]` (RCC: lateral soft, axial firm) once contact/stall fires. (4) **INSTRUMENT**: per-step JSONL (`z, force, f_base, d_signed/abs, commanded-vs-actual progress, contact_triggered, trigger_reason, spiral_k, sx/sy, dyaw, stiffness`).
- **SECOND BUG — the search DRAGGED THE SEATED PLUG BACK OUT.** V2's exit test was `if z <= Z_SEATED`, but `z` is the **COMMANDED** z_offset and **every contact RETRACTS it by 4 mm** → it never reaches −0.004 → the plug seats, then the loop keeps probing and **drags the seated plug out to a 36 mm offset at 11–13 N.** **V3 fix:** exit on the **MEASURED tip height vs the port datum**, not commanded `z`; and at the end **never hold a press** — re-command the CURRENT TCP pose and settle.
- **RESULT — THE SEARCH NOW RUNS, AND IT PRODUCES REAL FULL SEATS.** Contact fires (310 `contact_triggered` lines; `first_fire` usually `stall`), `spiral_k` up to 30, stiffness switches. **`tier_3 = 75 "Cable insertion successful"`** in `seat_v2` (×2), `seat_v3_3` (×2), `seat_v3_4` (×1), `seat_v2_r1` (×1). Peak totals **167.8 / 160.8** vs banked best 92.2.
- **"Does it insert then get pulled OUT?" — NO.** The scorer has no such state: every trial is either `75 "Cable insertion successful"` or `"No insertion detected. Final plug port distance: X m."` **When it scores 75, the plug IS still seated at final scoring** ⇒ V3's anti-drag fix works *in the runs that succeed*.
- **⚠️ THE REAL REMAINING BUG — `seated=True` IS A FALSE POSITIVE.** The policy claimed `seated=True` in **5 of 6** V3 trials; only **3** actually inserted. In `seat_v3_2 t2` it announced **SEATED while the plug was 40 mm from the port**. The measured-tip seat test fires on things that are not seats. **This is an unvalidated claim with no measurement attached — structurally the SAME trap as the board-pose ✓.**
- **⚠️ THE VARIANCE HAS A CLEAN SIGNATURE — the spiral WANDERS OUT OF THE NEIGHBOURHOOD.** Every real seat had `spiral_k ≤ 8` and **final offset < 10 mm** (v3_3: k=3 off (+2.2,+2.9); k=6 off (−1.9,+7.0); v3_4: k=8 off (+9.0,+3.3)). **Every failure had the spiral running away**: `spiral_k` 20–30, offsets **15–36 mm** (v3_4 t2: k=20 off (−15.4,−18.4); v3_1 t2: k=25; v3_2 t1: k=30 off (−34.8,+9.2)). **36 mm = `SPIRAL_PITCH(1.2) × k(30)` ⇒ the search radius EXCEEDS the port width.** It isn't finding the hole; it's leaving the area. ⇒ **the 67.8 → 160.8 score lottery is simply WHETHER THE SPIRAL TERMINATED EARLY.**
- **CHEATCODE REALITY CHECK (read the source, `CheatCode.py`): it has NO wiggle, NO spiral, NO force sensing AT ALL.** It looks up GT port + GT plug, interpolates 100 steps to 0.2 m above, then descends **monotonically 0.5 mm/step** until `z_offset < −0.015`, and stops. Its only closed-loop element is an **xy error integrator** (`i_gain 0.15`, ±0.05 windup) on the **GT plug TF**. **It seats 3/3 with ZERO search.** ⇒ **The spiral is NOT part of the task — it is COMPENSATION FOR PERCEPTION ERROR.** With an exact target no search is needed, which is exactly what the run data says (seats happen when the spiral is barely used). **Corollary: "CheatCode seats it" is NOT evidence that a spiral would rescue a bad target — it is evidence that a good target needs no spiral.**
- **STILL BROKEN:** trial 1 still eats **−24 contacts** (`upper_arm_link ↔ nic_card pcb_collider` — the SAME collision `_safe_clear` was meant to kill) and **−12 insertion force** (max **89 N**, >20 N for 4.8 s). **SC (trial 3) NEVER inserts** — flat 17.4–17.5, "no insertion, 0.06 m"; its port lock is ~58 mm off ⇒ **SC is now a PERCEPTION bug, not motion.**
- **NEXT (in order):** (1) **make the seat predicate agree with the scorer** (it currently lies at 40 mm); (2) **ABANDON, don't wander** — cap the spiral at the port half-width instead of drifting to 36 mm; (3) **N repeat runs of ONE frozen config** for mean±spread — *no single run here is a result* (see 07-10's own warning, violated again); (4) two-segment approach for the t0 contact penalty.
- **META-PATTERN (3 for 3):** board fit, dead force branch, and `seated=True` all survived because **a CLAIM existed with NO MEASUREMENT attached to it.** The board-pose one hid for weeks behind a ✓. Caught this one early — *because we finally looked at what the deployed code does, not what the doc says it does.*

---

## Score ledger (`~/aic_results`, self-scored /300)
GT ceiling ~279 → GT-port perception 227.8 → every fully-GT-free perception run ≤ floor. Dumper runs
(`sweepdump`, `yolo_native_dump`, `kp_sweep3`) show −33 (they don't insert). Regenerate:
`for d in ~/aic_results/*/; do echo "$d $(grep -h '^total:' $d/scoring.yaml)"; done`

| total | date | run | note |
|---|---|---|---|
| 279.6 | 07-04 | cheat_recheck | CheatCode GT ceiling |
| 277.7 | 07-04 | tuner_eval | InsertTuner force stack (GT) |
| 249.7 | 07-03 | smoke_v2 | |
| **227.8** | 07-05 | **kp_gtport** | **proven perception ceiling (GT port + FK plug + hands)** |
| 209.3 | 07-06 | scenecam_gtport | |
| 120.7 | 07-06 | kp_sweeporient | |
| 47.1 | 07-06 | kp_boardorient | |
| **171.4** | 07-10 | **sfp_reach_gt** | **MOTION CEILING (GT target)** — SAFE-CLEAR fix; both SFP **FULL SEAT** 75+75, SC partial 0.01 m. Motion proven. |
| **92.2** | 07-10 | **sfp_free** | ⭐ **BEST REAL GT-FREE** (2.3× old 40.1) — board-pose fix + slide-select + SAFE-CLEAR. 2 SFP partials (correct opening, 2–3 mm vs GT) + SC clean |
| **40.1** | 07-09 | **geo_v1** | first GT-free closed-loop (CAD-z depth fix; beat +1.4 floor, reversed −51) — *superseded by sfp_free* |
| 37.5 | 07-02 | wavearm | floor |
| — | — | *— force-insert probe experiments (branch `force-insert-v2`, ALL WORSE than 92.2, not merged) —* | |
| — | — | *— seat restoration (branch `seat-restore-v1`, InsertTuner force stack; MEAN 85.9 < baseline 92.2, NOT merged) —* | |
| — | — | *— SEAT V2/V3 (branch `seat-v2`, `5dbe781`) — force stack REVIVED (sign-wrong contact predicate). **REAL full seats (tier_3 75), peaks 167.8/160.8 > banked 92.2 — but NOT BANKED: identical config spans 67.8–160.8.** Needs N repeats. —* | |
| **167.8** | 07-11 | **seat_v2** | ⚠️ **peak, not a result** — 2× `tier_3 75` full seat. Same config elsewhere gives 100.3. |
| **160.8** | 07-11 | **seat_v3_3** | ⚠️ 2× `tier_3 75`. Both seats had `spiral_k ≤ 6`, final offset < 10 mm. |
| 122.5 | 07-11 | seat_v3_4 | 1× 75 (k=8, off 9.6 mm); t2 `seated=True` but **no insertion** (k=20, off 22 mm) |
| 116.9 / 115.7 | 07-11 | seat_v2_r2 / r3 | repeats |
| 100.3 | 07-11 | seat_v2_r1 | 1× 75 — same code as the 167.8 |
| 71.2 | 07-11 | seat_v3_2 | no seat; t2 claimed `seated=True` at **40 mm from the port** (false positive) |
| 67.8 | 07-11 | seat_v3_1 | no seat; spiral ran to k=30, wandered to **−34.8 mm** (> port width) |
| 132.2 | 07-10 | **sfp_seat** | ⚠️ **OUTLIER** — the ONE run that seated (SFP port_1 tier_3 **75**, first GT-free full seat). Does NOT reproduce. |
| 88.6 | 07-10 | sfp_seat_v2 | repeat — no seat |
| 68.0 | 07-10 | sfp_seat_v1 | repeat — no seat |
| 54.6 | 07-10 | sfp_seat_viz | recording run — PNG writes in the 50 ms control loop stall the descent, killing the seat |
| 79.2 | 07-10 | sfp_m_slide | compliant sliding spiral (valid; sign-inverted bias) |
| 71.1 | 07-10 | sfp_m_fermat | Fermat r∝√k probes — **INVALID** (stall false-fire burned the budget) |
| 58.6 | 07-10 | sfp_m_raster | dense grid probes — **INVALID** (same bug) |
| 57.1 | 07-10 | sfp_m_golden | original pattern, no dither — **INVALID** (same bug) |
| 33.7 | 07-04 | pi_diag_v3 | |
| 1.4 | 07-04 | pi_v3_score | **best real GT-free (M7b)** |
| 0.57 | 07-10 | **sfp_fix** | board pose FIXED (all 3 poses good, both SFP lock correct port) — but locks don't seat → **execution-bound** |
| 0.07 | 07-04 | pi_diag_m7 | |
| −18.8 | 07-04 | m7_eval | |
| −43.3 | 07-05 | kp_ab_false | PerceptionInsertKP live (M11) |
| **−51.5** | 07-08 | **yolo_sweep_score** | YOLO sweep-lock closed-loop (depth bug) |
| −105 | 07-05 | kp_boardgeo | |

*(full 37-run table lives in `~/aic_results/`; above are the milestones — best-per-thread + floor/ceiling.)*

---

## Doc map (`notes/`)
| Doc | Purpose |
|---|---|
| [RUN_PLAN.md](notes/RUN_PLAN.md) | Living M0–M18 stage tracker + CoStream map + standing rules |
| [problem_formulation.md](notes/problem_formulation.md) | Formal task statement, solved-vs-open, best-scores table |
| [ranked.md](notes/ranked.md) | 8 candidate approaches ranked; strategy "#7 perception ⊕ #3 force" |
| [baseline_scores.md](notes/baseline_scores.md) | Per-policy local self-scoring table |
| [perception_results.md](notes/perception_results.md) | Detailed M3–M16 perception experiments/ablations (largest) |
| [perception_v2_episodes.md](notes/perception_v2_episodes.md) | Per-episode plug→port gap for the 169 v2 demos |
| [arch.md](notes/arch.md) | Dual-anchor perception trainer pipeline walkthrough |
| [keypoint_pnp_perception_plan.md](notes/keypoint_pnp_perception_plan.md) | Keypoint + PnP/triangulation "eyes" plan |
| [force_insertion.md](notes/force_insertion.md) | InsertTuner force-insertion endgame (M9) |
| [stage2_port_perception_plan.md](notes/stage2_port_perception_plan.md) | Stage-2 board-decomposition plan; §10 error budget + A1 fail |
| [action_plan_perception_fix.md](notes/action_plan_perception_fix.md) | T0 + A1–A6 ordered fix plan (labels/detector/jitter) |
| [yolo_finetune_findings.md](notes/yolo_finetune_findings.md) | The A3b YOLO gate result (2/3, SC 3.9 mm) |
| [RL.md](notes/RL.md) | Where RL fits; Blackwell Isaac blocker; MuJoCo/MJX fallback |
| [WORKSPACE_MAP.md](notes/WORKSPACE_MAP.md) | Where functional pieces live (repo vs ~/ws_aic) |
| [CHEATCODE_RUNBOOK.md](notes/CHEATCODE_RUNBOOK.md) | Run guide for the CheatCode GT baseline |
| [collect/PLAN.md](collect/PLAN.md), [collect/M6_PLAN.md](collect/M6_PLAN.md), [collect/COLLECTION_FIXES.md](collect/COLLECTION_FIXES.md), [collect/SAMPLE_DIVERSITY.md](collect/SAMPLE_DIVERSITY.md) | Data-collection pipeline docs (kept with the pipeline) |
| `/home/skr/aic_data/track2_sam/FINDINGS.md` | SAM board-geometry track findings |

Upstream toolkit docs (`docs/*`, package `README.md`s) are challenge-provided background — not project history.

---

## Where things live (3 roots)
| Root | In git? | Holds |
|---|---|---|
| `/media/skr/storage/aic` (this repo) | ✅ | Source: ROS policies, `aic_engine/config`, this LAB_LOG, `notes/` (docs), `media/` (videos+frames; overlays in `vision_overlays/`, diagrams in `diag_frames/`), `rl_mujoco/`. Reference copy; built env is `~/ws_aic`. |
| `~/aic_data` | ❌ (~90 GB) | Datasets (`perception_*`), checkpoints/runs (`m3/m6/m7/m35/kp/boardyaw*`, `yolo_runs`), dumps, foundation weights (`sam_vit_b.pth`), `fm_venv`. → [MANIFEST](/home/skr/aic_data/MANIFEST.md) |
| `~/aic_results` | ❌ | 47 scored-eval run dirs (`scoring.yaml` + `policy.log` + rosbags) — the score record above. |
| `~/ws_aic/src/aic` | (workspace) | The **built/run** copy of the policies + `aic_local/` harness (`score.sh`, `collect/`, `mujoco_build/`). Policies: CheatCode · PerceptionInsert · PerceptionInsertKP · PerceptionInsertYOLO · PerceptionInsertYOLOSweep · SweepDump(Full) · InsertTuner. |

Run a policy: `~/ws_aic/aic_local/score.sh <module.Class> [ground_truth] [config] [run_name]`.
