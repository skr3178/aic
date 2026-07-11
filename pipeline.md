Board-first geometric perception pipeline (GT-free).  ★ = new/fixed this pass.  (*) = KNOWN GAP → fix/fine-tune later.
Per-stage GATE STATUS at the bottom.   ✓ pass · ⚠ partial · ✗ fail · ◻ not yet run

   3 wrist cams (native 1024)
          │
          ▼
   ┌───────────────────────────────────────────────────────────┐
   │ [0] SWEEP   raster wrist ~13 poses × 3 cams (= 39 frames) │  board fully in view here
   └───────────────────────────────────────────────────────────┘
          │
          ├──────────────────────────────────┐   TWO INDEPENDENT estimators
          ▼                                  ▼    (they FAIL on DIFFERENT trials → cover each other)
 ┌───────────────────────┐         ┌───────────────────────┐
 │ [1a] SAM BOARD        │         │ [1b] YOLO PORT        │
 │  mask → back-proj z=0 │         │  detect port box      │
 │ → fit KNOWN 0.30×.425 │         │  → CAD z-plane bp     │ ★depth-fix (not triangulate)
 │ → board (x,y,yaw≤1.5°)│         │  → port xyz (direct)  │   SFP z=.1335 SC z=.0145
 └──────────┬────────────┘         └──────────┬────────────┘   (*)SFP: 2 ports/card 21.8mm apart →
    [2] MAGENTA → disambiguate 90° quadrant ★ │       select by board-x ORDER (port_0=+x);
    [3] port = board · CAD_offset[target]     │       1-opening detected = LOW-conf (see note)
        anchor (-.070,+.121) · xacro nominal  │   GT-free
            ▼                                 ▼
       port estimate A                    port estimate B
            └───────────────┬─────────────────┘
                            ▼
   ┌───────────────────────────────────────────────────────── ──┐
   │ [1.5] CONSISTENCY + ENSEMBLE   ← the fail-proofing ★       │
   │   self-checks : known-size coverage · multi-view variance  │
   │   cross-check A vs B :                                     │
   │      agree     → high confidence, use fused pose           │
   │      disagree  → ONE failed → keep the geometrically-      │
   │                  plausible one (in-board / higher quality) │
   │   OUT: fused PORT pose  +  CONFIDENCE / FAIL flag          │
   └────────────────────────────────────────────────────────── ─┘
                            │
                            ▼
   ┌────────────────────────────────────────────────────────── ─┐
   │ [4] ACTUAL PORT = nominal ± SLIDE   (PERCEPTION, BOTH types)│
   │   DIRECT detection [1b] sees the ACTUAL slid port →        │
   │   resolves the slide for SFP *and* SC — SAME mechanism     │
   │   board·CAD [1a] = NOMINAL only (a prior / cross-check)    │
   │   SFP: small slide → nominal already ~ok (forgiving)       │
   │   SC : large slide → DIRECT detection REQUIRED             │
   │   ── insertion spiral / 1-D rail search = FALLBACK only    │
   │      (force stack / InsertTuner, tested SEPARATELY)        │
   └───────────────────────────────────────────────────────── ──┘
                            │
                            ▼
   ┌─────────────────────────────────────────────────────────── ┐
   │ [5] LOCK   perceive ONCE at a clean frame → hold + FK-track│  (A5: 3mm, 100% basin)
   │            + envelope / jump-reject gate (drop implausible)│
   └─────────────────────────────────────────────────────────── ┘
                            │
                            ▼
   FK plug  +  solved force stack (spiral + yaw dither)  +  TUG test  →  seat


NOTE: the top-down "heatmap" is a VISUALIZATION only — the pipeline works on the
point cloud / density grid, and OUTPUTS a pose, not an image.

CAD CONSTANTS KEPT (the board is a KNOWN object → perceive only pose + slide):
   PERCEIVED per-trial : board pose (x,y,yaw)  +  port slide along its rail (1 scalar, via [1b])
   CAD constants (never perceived): size 0.425×0.30 · flat top at base z=0 · port z (SFP .1335 SC .0145)
     · magenta anchor (-.070,+.121) · nominal offsets (nic_0 -.081,-.188; sc_1 -.075,+.0705; cards 40mm)
     · rail axis = board-x · legal slide nic ±.023 sc ±.06.   GT (port_gt, board_tf) = SCORING ONLY.

CANDIDATE → SELECT → LOCK   (2026-07-10 — CORRECTS the [1b] YOLO belief) ★
   YOLO is a CANDIDATE GENERATOR, not a target lock. MEASURED to contact (CheatCode + passive log, yolo_viz/):
   YOLO is SHARP CLOSE — 100% detect, 0-1mm error <=300mm; NOT blind up close (old belief = stall/viewpoint/
   desync confound). BUT raw YOLO is NOT target-conditioned → picks the WRONG instance often (SC 46%, SFP-1
   11% wrong-port) and FLICKERS between sibling ports (21.8mm hops, up to 59mm). So DO NOT trust confidence.
   ARCHITECTURE:  YOLO detects ALL candidates → BOARD GEOMETRY selects the TARGET → TEMPORAL GATE locks:
     1. back-project every YOLO box → CAD-z base-frame candidate
     2. express in board frame (board pose from [1a]+[2])
     3. filter to the target CARD/rail (board-Y = -.188 + .04*k) ; split the 2 openings by the board-x gap
     4. pick the opening NAMED by Task.port_name (port_0 = +board-x side) — geometry, NOT YOLO confidence
     5. lock + temporal gate: accept updates <12mm, REJECT ~22mm sibling hops, median over N
   STRATEGY — SFP FIRST: 2/3 eval trials are SFP, slide (±2.3cm) ⊂ spiral basin (3.6cm), geometry-selectable.
   SC DEFERRED: SC fails on EXECUTION not perception — with a PERFECT GT port the SC arm STILL stalls ~53cm
   short (CheatCode reaches + seats it) → fix the APPROACH path (sweep/timing/controller) before SC perception.

WHY THE ENSEMBLE WORKS — complementary failures (measured):
   trial      SAM board (via CAD)   YOLO port (direct)   → [1.5] ensemble
   t0 (SFP)   ✗ 116 mm (spill)      ✓ 22 mm              use YOLO
   t1 (SFP)   ✓ 3 mm                ✗ 58 mm (class-conf) use SAM
   t2 (SC)    ✓ 13 mm               ✓ 3.9 mm             agree
   → every trial has ≥1 good estimator; cross-check picks it → 3/3 + a confidence signal.

HELD-OUT VALIDATION — 12 NEW scenes (6 SFP + 6 SC), CLUTTERED + FULL eval-legal WIDE slides,
FROZEN thresholds, GT-free (CAD-z + CAD-offset + magenta yaw), sweep_dump_val/ :
   HONEST metric = ON THE CORRECT OPENING within 8 mm (insertion-relevant; the loose 36 mm over-counts):
     SC   : 6/6 on-target, all <=1 mm   (★ dropped FUSE-averaging that had degraded these to 8-15 mm)
     SFP  : 4/6 on-target (1-5 mm) ; 2/6 NOT on a real port — t0 on the SIBLING, t8 BETWEEN the 2 openings
   → the 2 SFP misses are EXACTLY the 2 LOW-confidence cases (only ONE of the 2 openings detected).
     Confidence flag is trustworthy: every 'ok' est = on-target (8/8); every fail = flagged LOW (2/2).
   routing generalizes (kills N=3 overfit); GT crutches removed at ~no cost (board z=0; magenta 12/12 quadrant).
   CAVEAT: SAM board degrades on clutter (70-128 mm) → YOLO carries. Offline accuracy, not closed-loop /300.

(*) SFP TWO-PORT DISAMBIGUATION — NEEDS FIX / FINE-TUNE LATER  ───────────────────────────────
   A NIC card has TWO SFP ports 21.8 mm apart along the RAIL axis (port_0 = +board-x side, from CAD).
   Eval NAMES the target (Task.port_name) and scores THAT port only → the wrong opening = a FAIL.
   YOLO's class is generic 'sfp' (both openings look alike), so:
     • BOTH openings detected → select target by board-x ORDER (port_0 = larger board-x) → CORRECT (t4,t6,t10).
     • ONLY ONE opening detected → AMBIGUOUS (a lone cluster fits target OR sibling within ±slide) → FLAG LOW.
   (*) TO CLOSE: (a) ±22 mm 1-D RAIL SEARCH at insertion on any LOW-conf SFP — sibling is one step away, the
       force stack seats on the real port;  OR (b) retrain YOLO with 2 classes sfp_port_0 / sfp_port_1 so a lone
       detection self-labels.  Until then, SFP single-opening scenes are NOT guaranteed on the named port.

FAILURE HANDLING by layer (there is NO single fail-proof estimator — robustness is layered):
   • [1.5] both disagree & implausible → LOW confidence → re-sweep / active view (or safe abort)
   • [5]   envelope + jump-reject → never chase an outlier estimate
   • [4]   PERCEPTION resolves the ACTUAL port (slide included) for BOTH types via direct detection [1b];
           board·CAD [1a] is only the nominal prior.  SFP small-slide → forgiving; SC large-slide → direct req'd.
           Insertion spiral / 1-D rail search = FALLBACK only (SC-heavy; force stack / InsertTuner, tested separately).
   • the force stack is NOT a fix for a confident 116 mm board error → that is exactly why [1.5] exists

──────────────────────────────────────────────────────────────────────────────────────────────
GATE STATUS   (each stage's pass criterion, latest result, and did it CLEAR that gate)
──────────────────────────────────────────────────────────────────────────────────────────────
 stage           GATE CRITERION (pass bar)              RESULT (latest)                STATUS
 [0] SWEEP       board in view across the raster        39 frames · eval 3/3 · val 12/12 ✓ PASS
 [1a] BOARD POSE  yaw < 2°  AND  center < 36 mm         ★FIX: eval 3/3 ≤1.3°/≤6.9mm; val 12/12 ctr<36mm ✓ PASS (DEPLOYED)
 [1b] YOLO port  detect port (CANDIDATE generator)      sharp close 0-1mm; wrong-inst raw ⚠ candidates only (select→lock)
 [SEL] SFP TARGET geometry selects named opening + gate   LIVE: both SFP lock correct port (sfp_fix) ✓ PASS (wired+live)
 [2] DISAMBIG    correct 90° quadrant, GT-free          magenta anchor: 12/12 correct  ✓ PASS  (GT-free)
 [3] PORT (CAD)  CAD offset reproduces GT port < few mm xacro offsets consistent /GT   ✓ PASS  (verified)
 [1.5] ENSEMBLE  on-correct-port, GT-free, held-out     val 10/12 on-port; 2 flagged LOW ✓ trust-YOLO; ⚠ (*)SFP
 [4] PORT+SLIDE  ACTUAL slid port (both types)          SC wide-slide 6/6 via [1b]     ✓ SC ; ⚠ (*)SFP 2-port
 [5] LOCK        ≤ 3 mm median, 100% within basin       A5: 3.0 mm, 49/49 in basin     ✓ PASS  (validated)
 [M] MOTION      reach the target without folding IK    ★SAFE-CLEAR (joint-space HOME) → GT 171.4: both SFP FULL SEAT ✓ SOLVED
 — INSERTION     tug-verified seat (given the port)     GT-fed 227.8 ✓ ; GT-free: 2 SFP partials + SC clean ⚠ seat-depth (orientation)
 ==> CLOSED-LOOP real /300 beats the +1.4 floor         GT-FREE sfp_free = 92.2/300    ★ BEST (2.3× old 40.1); gap = tier_3 seat depth
──────────────────────────────────────────────────────────────────────────────────────────────
 Read: CURRENT PLAN = candidate→select→lock (above).  YOLO = candidate generator (sharp close, but wrong-
 instance + flicker on raw) → board-geometry SELECTS the named SFP opening → temporal gate LOCKS.
 SFP FIRST (2/3 trials, in spiral basin).  SC DEFERRED = EXECUTION stall (GT port still stalls ~53cm; CheatCode
 seats it) — NOT perception.  Closed-loop GT-free sfp_fix = 0.57/300.  CheatCode ceiling 279.

──────────────────────────────────────────────────────────────────────────────────────────────
★ BOARD POSE FIXED + EXECUTION IS NOW THE BINDING CONSTRAINT   (2026-07-10, sfp_fix) → board_pose.md
──────────────────────────────────────────────────────────────────────────────────────────────
 [1a] the DEPLOYED board pose was the blocker (live gray+minAreaRect, NOT the banked-but-never-shipped SAM):
   catastrophic on SFP (~23° yaw, ~250mm ctr) — a detached NIC/base blob dragged minAreaRect; nothing measured it.
   FIX (cv2+numpy, in PerceptionInsertSFP._sweep_board): full-scale mask · largest-CC (drop detached blob) ·
   known-size fixed-box yaw (ignores ATTACHED NIC lobe) · center ← MAGENTA anchor.  vs GT: 3 eval poses ≤1.3°/≤6.9mm.
   LIVE sfp_fix: yaw −0.0°/2.0°/75.1°, both SFP LOCK the correct opening.  PERCEPTION [0]-[5] now all ✓.

 WHERE IT FAILED (sfp_fix) — ⚠ SUPERSEDED by "MOTION SOLVED" below; kept for the diagnosis trail.
 INSERTION APPROACH (perception exonerated: target correct, seat mechanism works on trial1):
   trial1 SFP port_1 : ✓ WORKS — clean path (0 contacts), reached, 40N/1.3s, PARTIAL seat 0.05m (+38). seat stack OK.
   trial0 SFP port_0 : ✗ JAM  — upper_arm ↔ NIC-card collision (−24) + force-stack ground ~95N/30s (−12), 0.20m off.
                        ★ CONFIRMED REACHABILITY (GT ablation sfp_drive_gt): frozen a PERFECT GT target
                        ([-0.419 0.305 0.133] = our lock within 1mm) → STILL JAMS identically. NOT perception,
                        NOT orientation, NOT target-source → the arm folds its upper arm into the card reaching
                        port_0's pose. CheatCode seats t0 → difference is the APPROACH (clean start + straight-in)
                        vs ours (enters from sweep-end → bad IK branch).
   SC   sc_port_base : ✗ STALL — NO sustained contact (force <20N), 0.58m short. Arm never reaches/descends
                        (bottleneck #1, unchanged; GT port still stalls, CheatCode seats).
   ⇒ TWO distinct MOTION bugs (reach/IK fold on t0, never-reach stall on SC); seat/force stack is NOT the bug.
   ⚠ BUG (mine, PerceptionInsertSFPDrive): self._locked_xyz not reset per trial → t1 & SC reused t0's frozen
     target in BOTH drive runs → their numbers are INVALID artifacts (drove to the wrong port).
   NEXT: (1) reset _locked_xyz per trial → valid t1/SC; (2) pre-approach RETRACT (go high above board before the
   interp) so IK doesn't fold upper arm into card = mimic CheatCode's clean start; (3) SC descent stall (deferred).
   Continuous re-lock (yolo_viz_slide) PARKED behind a working approach.

──────────────────────────────────────────────────────────────────────────────────────────────
FINDINGS & SUMMARY DIAGRAM   (2026-07-10)
──────────────────────────────────────────────────────────────────────────────────────────────
 PIPELINE  ── green to [5], red at the motion ──
   [0]SWEEP ─►[1a]BOARD POSE─►[SEL]PORT─►[5]LOCK─► ⟦MOTION⟧ ─►SEAT
      ✓          ✓ FIXED         ✓ correct  ✓ <1mm    ✗ HERE     ✓ (t1)
              magenta+knownsize  slide-sel  vs GT!   reach/IK

 3-LEVEL STACK        OUR DEPLOYED                   vs DEFAULT LINE (CheatCode/GT)
   1 DETECTION  ✓     YOLO→select + fixed board pose  = GT quality (proven <1mm)
   2 MOTION     ✗     sweep→approach→descend          CheatCode clean straight-in (3/3 GT)
                      └► arm folds into card on t0
   3 FORCE STK  ✓     InsertTuner spiral+dither+tug    = 277.7 vs 279.6 (GT)

 TODAY'S RUNS          target source          t0    t1       SC     total
   sfp_fix             perceived (per-step)   JAM   partial  stall   0.57
   sfp_drive           perceived (FROZEN)     JAM   [bug]    stall  -45
   sfp_drive_gt <ABL   GT (FROZEN)            JAM*  [bug]    [bug]  -45

 * DECISIVE (t0, only valid ablation trial):
     GT target [-0.419 0.305 0.133] ~= perceived [-0.419 0.306 0.134]  (<1mm) -> perception OK
     feed PERFECT GT -> STILL JAMS (upper_arm<->card -24, 95N/30s, 0.20m)
     => PURE REACHABILITY (arm geometry), not perception/orientation/target.

 ! BUG: self._locked_xyz not reset per trial -> t1/SC reused t0's target -> those numbers INVALID.

 PROVEN  board pose . port select . perception <1mm vs GT . force stack . seat-when-arrives(t1)
 OPEN    MOTION reach/IK (upper-arm<->card on t0) . SC descent stall . slide-select fragile (few both-vis)

 NEXT  1.reset _locked_xyz/trial -> valid t1/SC   2.pre-approach retract (clean IK start, mimic CheatCode)
       3.SC descent stall (deferred)   |   continuous re-lock PARKED behind a working approach

──────────────────────────────────────────────────────────────────────────────────────────────
*** MOTION SOLVED — SAFE-CLEAR fix.  GT-FREE 40.1 -> 92.2 (2.3x)   (2026-07-10, PerceptionInsertSFPDrive)
──────────────────────────────────────────────────────────────────────────────────────────────
 ROOT CAUSE (confirmed by GT ablation): the arm's POSTURE at sweep-end put IK in a FOLDED BRANCH.
   The same TCP pose is reachable in many postures; from the sweep-end posture the solver picked one that
   swings the UPPER ARM into the NIC card.  Cartesian commands CANNOT fix this (the bad posture satisfies
   the pose).  CheatCode never sweeps -> never enters the bad branch -> that is the whole reason it seats.

 THE FIX  `_safe_clear`:   sweep_end -> [JOINT-SPACE to SAFE_HOME] -> Cartesian interp -> descend -> force stack
   SAFE_HOME=[-0.16,-1.35,-1.66,-1.69,1.57,1.41] (shipped GentleGiant/SpeedDemon) + GentleGiant gains
   (low stiffness / high damping -> smooth; jerk is scored).  Resets the IK branch before the approach.
   (Also fixed: _locked_xyz was not reset per trial -> t1/SC had been reusing t0's target.)

 RESULTS
   sfp_reach_gt (GT target)  = 171.4   both SFP **FULL SEAT** (75+75) · SC partial 0.01 m  -> MOTION PROVEN
   sfp_free     (GT-FREE)    =  92.2   SFP p0 13.8 (partial 0.05m, -36 contact/force) · SFP p1 48.5 (partial,
                                       ZERO contacts) · SC 29.9 (clean, 0.06m).  Both SFP lock the CORRECT
                                       opening, within 2-3 mm of GT.   [prev best GT-free: geo_v1 40.1]
   ** SC 53cm STALL (old bottleneck #1) IS GONE ** — it was the same bad posture, not a descent bug.

 VIDEO (GT-FREE control; gt:=true only to draw the reference ring) — SFPDriveViz:
   aic_data/sfp_free_viz/sfp_trial1_gtfree.mp4   (sfp_port_0)
   aic_data/sfp_free_viz/sfp_trial2_gtfree.mp4   (sfp_port_1)
   GREEN X = our PERCEIVED frozen lock (drives the arm) · white ring = GT (reference) · yellow = plug tip.
   HUD: "GT-FREE sfp_port_1 yaw +2.8deg tip->X 43mm  LOCK ERR 2mm  F 8N" — the X sits ON the GT ring, on the
   CORRECT one of the two SFP openings, rock-steady through the approach (frozen -> zero flicker).

 WHERE THE REMAINING POINTS ARE (92.2 -> ~240 potential).  The 92.2->171.4 gap is ENTIRELY tier_3 (seat depth):
   tier_2 actually IMPROVED (+16.8) GT-free -> our MOTION/FORCE did NOT degrade; only the TARGET precision did.
   (1) SFP partial->FULL seat  ~ +74   orientation precision (board yaw 1-3 deg vs GT's exact quat) binds the
                                        plug.  It is a CLIFF not a slope: 2-3mm/1-3deg flips 75 -> 38.
   (2) SC port lock 58 mm off  ~ +40   SC's miss is now PERCEPTION, not motion (offline SC was 6/6 <=1mm).
   (3) trial0 contact/force    ~ +36   approach still descends DIAGONALLY while translating (TCP z 0.328->0.271
                                        during the lateral move) -> two-segment path: hold altitude -> hover
                                        over port -> straight down.  NO lateral motion near the card.
