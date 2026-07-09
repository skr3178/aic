Board-first geometric perception pipeline (GT-free).  ★ = new/fixed this pass.
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
 │  → fit KNOWN 0.30×.425 │        │  → CAD z-plane bp     │ ★depth-fix (not triangulate)
 │  → board (x,y,yaw≤1.5°)│        │  → port xyz (direct)  │   SFP z=.1335 SC z=.0145
 └──────────┬────────────┘         └──────────┬────────────┘
    [2] MAGENTA → disambiguate 90° quadrant ★ │   anchor (-.070,+.121) board-frame
    [3] port = board · CAD_offset[target]     │   (xacro nominal, GT-free)
            ▼                                 ▼
       port estimate A                    port estimate B
            └───────────────┬─────────────────┘
                            ▼
   ┌───────────────────────────────────────────────────────────┐
   │ [1.5] CONSISTENCY + ENSEMBLE   ← the fail-proofing ★       │
   │   self-checks : known-size coverage · multi-view variance  │
   │   cross-check A vs B :                                      │
   │      agree     → high confidence, use fused pose           │
   │      disagree  → ONE failed → keep the geometrically-      │
   │                  plausible one (in-board / higher quality) │
   │   OUT: fused PORT pose  +  CONFIDENCE / FAIL flag          │
   └───────────────────────────────────────────────────────────┘
                            │
                            ▼
   ┌──────────────────────────────────────────────────────────  ─┐
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
   ┌───────────────────────────────────────────────────────────┐
   │ [5] LOCK   perceive ONCE at a clean frame → hold + FK-track│  (A5: 3mm, 100% basin)
   │            + envelope / jump-reject gate (drop implausible)│
   └───────────────────────────────────────────────────────────┘
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

WHY THE ENSEMBLE WORKS — complementary failures (measured):
   trial      SAM board (via CAD)   YOLO port (direct)   → [1.5] ensemble
   t0 (SFP)   ✗ 116 mm (spill)      ✓ 22 mm              use YOLO
   t1 (SFP)   ✓ 3 mm                ✗ 58 mm (class-conf) use SAM
   t2 (SC)    ✓ 13 mm               ✓ 3.9 mm             agree
   → every trial has ≥1 good estimator; cross-check picks it → 3/3 + a confidence signal.

HELD-OUT VALIDATION — 12 NEW scenes (6 SFP + 6 SC), CLUTTERED + FULL eval-legal WIDE slides,
FROZEN thresholds (confB≥.70, disagree>4cm, NO re-tuning), sweep_dump_val/ :
   GT-crutch run : 12/12 within 36 mm   (worst 23 mm)
   FULLY GT-FREE : 12/12 within 36 mm   (worst 23 mm)  ← CAD-z + CAD-offset + magenta yaw, crutches removed
   → routing generalizes (kills N=3 overfit worry) AND the GT crutches were removable at ~no cost
     (board sits at z=0 so CAD-z IS the true plane; magenta picked the right quadrant on all 12).
   CAVEAT: on the cluttered set YOLO CARRIES (8 YOLO + 4 FUSE, 0 SAM-only); SAM board degrades
     (70–128 mm) → keep stress-testing YOLO robustness. Still offline accuracy, not closed-loop /300.

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
 [1a] SAM board  yaw < 2°  AND  center < 36 mm          eval 2/3; val degrades on clutter ⚠ weak cross-check
 [1b] YOLO port  port box within 36 mm (CAD-z depth)    val 8/8 chosen 0–23 mm         ✓ PASS  (CAD-z, GT-free)
 [2] DISAMBIG    correct 90° quadrant, GT-free          magenta anchor: 12/12 correct  ✓ PASS  (GT-free)
 [3] PORT (CAD)  CAD offset reproduces GT port < few mm xacro offsets consistent /GT   ✓ PASS  (verified)
 [1.5] ENSEMBLE  3/3 within 36 mm after cross-check     eval 3/3 · val 12/12 GT-FREE   ✓ PASS  (held-out)
 [4] PORT+SLIDE  perception gets ACTUAL port (both types) val: SC wide-slide 12/12 via [1b] ✓ PASS  (direct detect)
 [5] LOCK        ≤ 3 mm median, 100% within basin       A5: 3.0 mm, 49/49 in basin     ✓ PASS  (validated)
 — INSERTION     tug-verified seat (given the port)     227.8 w/ GT port (2 full+part) ✓ PASS  (GT-fed)
 ==> CLOSED-LOOP real /300 beats the +1.4 floor         not yet run w/ GT-free stack   ◻ PENDING (wire into policy)
──────────────────────────────────────────────────────────────────────────────────────────────
 Read: perception is now ✓ GT-FREE offline (ensemble 12/12 held-out; depth+disambiguation+slide land);
 [1a] SAM board is a weak cross-check under clutter (YOLO carries).  Execution ([5] + insertion) ✓.
 ONLY the CLOSED-LOOP /300 remains: wire [1.5]+CAD-z+magenta into PerceptionInsertYOLOSweep and score.
