# Plan: Real-time re-lock during insertion (dynamic port tracking)

## Context
The GT-free perception works well **statically** (SC ≤1 mm; SFP on-target when both openings seen) and now
runs **closed-loop** — first GT-free score **+40/300**, a clean reversal of the earlier −51 (the CAD-z depth
fix). The single biggest remaining weakness is that the port lock is **perceived once from a distance and
then FROZEN** through the whole insertion. Any lock error is therefore never corrected, and the blind descent
drives the plug into the wrong spot:
- trial 2 (SC): 58 mm lock → 610 mm miss (execution amplified the error);
- trial 1 (SFP): 64 mm wrong-opening lock → miss.

The wrist-cam recording proved: (a) the port stays **visible for most of the approach** (it only occludes at
contact), and (b) a fixed world error projects to **more pixels as the camera nears** — a re-perception on
approach would be *more accurate*. User directive: **the prediction must update in real time as the robot
approaches**.

**Key enabler (confirmed by exploration):** the base `PerceptionInsertYOLO._perceive` ALREADY runs a per-step
**gated running median** (`_port_hist` + envelope + jump-reject). The sweep variants *freeze* it by returning
all-NaN from `_yolo_uv` once the sweep lock is set (the old up-close multi-view triangulation — the −51 bug —
was unreliable). **Single-view CAD-z back-projection** (already used in the sweep) works from ONE view at ANY
range. So real-time re-lock = feed per-step CAD-z estimates into the *existing* running median instead of
freezing it. No new filtering machinery.

## Decisions (from the user)
- Approach = **continuous per-step re-lock** (reuse the existing running median; truest real-time).
- Scope = **dynamic re-lock only** — NOT folding in the live-path SAM upgrade, SFP rail-search, or a timing
  overhaul.

## Phase 0 — VERIFY THE LOAD-BEARING ASSUMPTION FIRST (before building anything)
The entire approach hinges on **YOLO usefully detecting the port DURING the approach**. But the sweep-lock
exists *precisely because* YOLO "returns no box" up close (the policy's own docstring) — i.e. the original
design assumes the opposite. So verify cheaply, on data we ALREADY have, before writing the policy:
- On the recorded approach frames (`~/aic_data/geo_rec_xo/trial{0,1,2}/`, ~559 frames each, with per-frame
  center `Tbc` + GT in `xo.json`), run YOLO per frame and measure **vs camera range**:
  (1) target-class **detection rate**; (2) **CAD-z back-projected error vs GT**.
- **PASS → build the re-lock:** detection stays high through the approach AND error shrinks as range drops.
- **FAIL → pivot:** YOLO drops out / drifts up close → re-perception isn't available on approach → pivot to
  an **intermediate hover at the last known-good detection range**, or a higher-ROI lever.
~10 min of offline analysis, no sim. This gate decides whether the rest of the plan is worth building.

## Reality check (honest ROI — even if Phase 0 passes)
Re-lock addresses at most **trial-2's position**, not the other failures:
- trial 0 (good 3 mm lock, failed **seating**) → re-lock doesn't help (timing/seating).
- trial 1 (**two-port** wrong opening) → needs the deferred two-port fix.
- trial 2 (58 mm lock **+ arm swing** 610 mm) → re-lock helps the position, NOT the orientation/IK swing.
Plus: added per-step YOLO compute may **worsen the timing fragility** (same lock scored +40/−45/−21 by load).
Right long-term architecture, uncertain near-term ROI — Phase 0 is the cheap decider.

## The new pipeline stage (ASCII — to be APPENDED to pipeline.md, existing diagram kept)
```
   ┌───────────────────────────────────────────────────────────┐
   │ [6] REAL-TIME RE-LOCK   ← the dynamic fix ★                │
   │   problem: [5] locks ONCE from distance then FREEZES → a   │
   │   coarse/wrong lock is never corrected (SC 58mm→610mm miss)│
   │                                                            │
   │   APPROACH loop (100 steps, camera getting CLOSER):        │
   │     every step (throttled ~1/3-5):                         │
   │       YOLO detect TARGET port on the live frame            │
   │        → CAD z-plane back-proj (SINGLE view, works close)  │
   │        → GATE  envelope + jump-reject vs running median    │
   │        → append _port_hist → median = REFINED lock         │
   │     ⇒ lock error SHRINKS as it nears (more px per mm)      │
   │     SFP: keep the coarse-selected opening (tight gate      │
   │          <15mm) so it can't drift to the 22mm sibling      │
   │                                                            │
   │   OCCLUSION (final blind cm): YOLO sees nothing → HOLD     │
   │     the last good lock → force-search / spiral seats it    │
   └───────────────────────────────────────────────────────────┘
           (sits between [5] LOCK and the force-stack insert;
            reuses the base _port_hist + envelope + jump-reject)
```

## Approach
New **standalone** policy `PerceptionInsertGeoLive(PerceptionInsertGeo)` — does NOT modify any base policy.
Two overrides:

1. **Un-freeze `_yolo_uv`.** `PerceptionInsertYOLOSweep.py:29-33` returns `np.full((3,2), np.nan)` whenever
   `_sweep_port` is set — that is the freeze. New: when locked, still run live YOLO and return the
   **target-class** port pixels (keep the existing geometry-select that rejects wrong-module boxes). Return
   NaN ONLY when YOLO genuinely sees nothing (occlusion) → the existing "hold median" path freezes gracefully.

2. **Per-step single-view CAD-z port solve.** Override `_perceive` (copy from `PerceptionInsertYOLO`, change
   ONLY the port-position block ~lines 392-398): for each detecting cam ray `(o,d)`, intersect the CAD
   z-plane `s = (pz - o[2]) / d[2]; p = o + s*d`, `pz = CAD_Z[self._plug_type]`; robust-median over the
   detecting cams → `port_kp`. Feed into the EXISTING gate + `_port_hist` median (unchanged). A single view
   now suffices → works at close range where triangulation failed. (This is the block already in
   `PerceptionInsertGeo._sweep_board:~131-136`, lifted into `_perceive`.)

Everything downstream (gate ~402-406, append ~413, median ~418 → `port_tf`) is reused untouched.

## Key design details
- **SFP two-port drift guard (must-have).** `JUMP_REJECT=0.08 m` but the SFP sibling is only ~22 mm away
  (inside the gate). Mitigation: on SFP, apply the board-x-order two-port selection **per step** (reuse
  `PerceptionInsertGeo`'s logic, anchored to the coarse-selected opening) OR tighten jump-reject to ≲15 mm
  for SFP. SC needs none (one port/mount).
- **Occlusion → graceful freeze (no change).** NaN up close → existing `<2 views → hold median` path keeps
  the last good lock for the final blind cm.
- **Throttle to protect timing.** Closed-loop is timing-fragile (same lock scored +40/−45/−21 by compute
  load). Re-lock at a **throttled cadence** (~every 3-5 steps) and **only in the APPROACH phase** (100-step
  loop at `z_offset=0.2`), NOT the fine descent. Bounds added compute.
- **First-relock gate.** Fresh estimate vs a ~60 mm-off sweep median: 60 mm < 80 mm → accepted; a wrong-port
  485 mm jump → rejected (this gate would have prevented the earlier hover-refine blow-up). Optionally relax
  for the first few live samples so a genuinely-better estimate isn't held back by a bad sweep median.

## Files
- New: `~/ws_aic/src/aic/aic_example_policies/aic_example_policies/ros/PerceptionInsertGeoLive.py`
  **and** mirror to `.../.pixi/envs/default/lib/python3.12/site-packages/aic_example_policies/ros/` (the sim
  loads from site-packages — both must be updated, same pattern as the other Geo policies).
- Reuse: `CAD_Z`, `MAG_BOARD`, `_rot2`, `_robust_med`, SFP two-port selection from `PerceptionInsertGeo.py`;
  envelope / jump-reject / `_port_hist` / `_perceive` structure from `PerceptionInsertYOLO.py`.
- **pipeline.md**: APPEND the `[6] REAL-TIME RE-LOCK` ASCII block above (do NOT delete the existing diagram);
  add a `[6]` gate-table row. (`/media/skr/storage/aic/pipeline.md`.)
- No changes to any base policy.

## Verification
1. **Diagnostic — the payoff metric (first).** `score.sh PerceptionInsertGeoLive true eval` (gt:=true); log
   per approach step the lock-vs-GT error → confirm it **SHRINKS as `z_offset` decreases** (vs frozen flat
   line). Reuse the x/o recorder to render a video where the red ◯ **converges** to the green ✕ on approach.
   Direct evidence, independent of the noisy score.
2. **Closed-loop score.** `score.sh PerceptionInsertGeoLive false eval geolive_v1` → compare `/300` to
   geo_v1 (40.1). Target: SFP-1 and SC-2 improve. Run **2-3×** for variance.
3. **Safety check.** When YOLO fails up close, the lock holds (no worse than frozen) — the change can only
   help or match, never catastrophically drift (gate + SFP guard prevent the 485 mm-style jump).

## Explicitly NOT doing (scope = dynamic re-lock only)
- No live-path SAM upgrade (coarse lock stays; re-lock refines from it).
- No SFP two-port rail-search (static gap deferred).
- No execution-timing overhaul (noted as risk; only the throttle mitigation is in).
- No staged intermediate-hover / camera-aimed refine (`PerceptionInsertGeo2` shelved).

## Risks
- If live eval YOLO is too noisy up close (coarse sweep showed 87-345 mm spread), per-step estimates get
  gated out → re-lock degrades to the frozen baseline (safe, no gain) — would justify the deferred live-path
  (SAM) upgrade next.
- Added per-step YOLO compute may worsen timing variance; the throttle bounds it — measure across 2-3 runs
  and lower cadence if needed.

---

## PHASE 0 RESULT (2026-07-09) — FAIL → DO NOT BUILD THE RE-LOCK
Ran YOLO on the recorded approach frames (geo_rec_xo/trial{0,1,2}) vs camera->port range. The load-bearing
assumption (YOLO usefully detects the port during approach, error shrinking as it nears) FAILS on all 3:
- trial 0 SFP: detection rate **3%** — YOLO basically blind; camera never gets closer than ~324mm (the SFP
  port faces sideways on a vertical card, plug inserts horizontally, so the downward wrist cam can't approach
  the port face). No signal to re-lock with.
- trial 1 SFP: detection **89%** but error **plateaus at ~24mm** = the two-port floor (deferred fix). Re-lock
  would stall there.
- trial 2 SC: detection **98%** but error **259-311mm** — camera at 625-746mm (arm swung the WRONG way), so
  YOLO back-projects garbage; the gate would reject it. The failure is trajectory/IK, not the lock.

Conclusion: the original freeze-the-lock design was quantitatively correct; continuous re-lock cannot help on
these trials. ~15 min of offline analysis saved building a policy + sim runs for no gain.

### Pivot (real bottlenecks Phase 0 exposed)
1. SC trial 2 (biggest loss): arm swings to a bad pose (orientation/IK at 66deg board yaw) = TRAJECTORY, not lock.
2. SFP two-port (trial 1): the 24mm floor = sibling-opening ambiguity (+/-22mm rail search or 2-class detector).
3. Coarse-lock quality (live path weaker than offline) — better coarse locks help all, since re-lock can't refine.
