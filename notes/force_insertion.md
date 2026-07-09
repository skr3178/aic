# Force-Insertion Endgame — InsertTuner results

Standalone GT-fed rig (`collect/insert_tuner/`) that isolates the insertion endgame from perception:
GT port/plug + deliberate injected offset + our force stack. Runner `collect/tuner_sweep.sh`
(single isolated container per chunk, ~23 s/trial); analysis `collect/tuner_basin.py`.
Baselines untouched (CheatCode, PerceptionInsert v3).

## Controller (v4 stack, all analytic — no training samples)
Phase A: CheatCode-verbatim hover **200 mm above the port** + servoed glide (≈20 s alignment runway)
→ handover at **entrance + 10 mm** → Phase C: contact-driven insertion — force-stop (ΔF > 4 N) **or**
stall detection (12 mm commanded, tip moves < 4 mm), golden-angle spiral (≤36 mm), yaw dither (±3/6°),
RCC compliance (lateral 40 / axial 90), penetration cap (−10 mm), never-hold-a-press, **tug test**
(true seat = seated AND survives an 8 mm pull).

## Smoke-run iteration history (each ≈ 7 min, chunk 0 = 10 wide-pose scenes, zero injected offset)

| Run | Found / changed | Result |
|---|---|---|
| #1 | axial stiffness inverted (z=25 → ~1.3 N push, physically can't insert); SC false-seat threshold; spiral thrash 334 N | 0/10 |
| #2 | RCC-corrected (lateral soft / axial firm); per-type seat dist; retract+clear before re-descend | 0/10 — contact undetectable (impedance makes only ~5 N at the mouth, F_STOP was 8) |
| #3 | F_STOP → 4 N + **stall trigger** | spiral engages everywhere; still 0/10 — orientation suspected |
| #4 | **CheatCode 200 mm glide + handover** (alignment-runway hypothesis) | **1/10 TRUE seat** (t8: 1.0 mm, latched, 3.2 N); SC arrivals tightened to ±0.1 mm at the mouth |

## Head-to-head vs the GT expert — exactly on par, scene-for-scene (chunk 0)

| Scene | CheatCode (GT expert) | Our force stack (smoke #4) |
|---|---|---|
| ep8 / t8 (sfp rail1) | ✅ seated, 0.3 mm | ✅ **seated, 1.0 mm, tug-verified, 3.2 N** |
| ep7 / t7 (sc rail0) | 9.6 mm (best failure) | 9.3 mm (partial entry) |
| all other SFP | 45.8–46.3 mm | 44.6–46.0 mm |
| all other SC | 13.5–13.7 mm | 13.6–13.8 mm |
| **Total** | **1/10** | **1/10** |

Same scene seated. Same scenes failed. Same closest-approach **to within a millimetre** on every trial.
(Chunk 0 is a hard draw — mostly weak rails; CheatCode's overall v2 average is 35%.)

### Reading
1. **The force stack is no longer the deficit** — two completely different endgames (blind servoed plunge
   vs contact-driven search) hit the identical per-scene ceiling ⇒ the cap is a **shared upstream
   constraint** (commanded-pose/wrist-configuration geometry per scene), not either controller's logic.
2. **Where we already beat parity:** tug-verified gentle seat (3.2 N vs blind pressing), safe mouth-park
   failures, and search machinery for *offset* targets — CheatCode has zero error recovery; ours exists
   precisely for perception error. (The 94-trial offset sweep quantifies this.)
3. **When aligned, entry is nearly forceless** (3.2 N) — insertion is alignment-gated, not force-gated.
4. Next attack is shared with the expert: whatever blocks alignment on non-t8 scenes (tilt/wrist-limit
   hypothesis) blocks both; fixing it lifts our policy **and** collection data quality together.

## Per-type stats (wide poses)

| | SFP | SC |
|---|---|---|
| entrance→seat funnel | 45.8 mm | 15.6 mm |
| CheatCode, 169 v2 eps | 48% seated | 22% seated |
| failure min-gap (both stacks) | ~44–46 mm (mouth) | ~13.6–13.9 mm (mouth) |
| eval trial mix | 2 of 3 trials (150 pts) | 1 of 3 (75 pts) |

Priority: SFP-reliable + SC-at-mouth ≈ 170+ pts — fix SFP alignment first; SC needs per-type tuning
(smaller funnel, latch geometry).

## Official-eval head-to-head (2026-07-04): FULL PARITY — 3/3 insertions

`score.sh` on the real eval config, both GT (diagnostic): **CheatCode 279.6** (75/75/75) vs
**our force stack 277.7** (75/75/75, zero contact/force penalties, tug-verified latches).
Entire 1.9-pt gap = duration bonus (search + tug overhead, a few s/trial).

- On eval-difficulty geometry, **execution is solved** — the wide-pose chunk-0 struggles are the hard
  tail of the legal space, not the eval regime.
- Remaining scoring equation: **277.7 × (perception delivers pose into the capture basin?)** — the
  offset sweep on eval-like poses measures the basin radius = the exact perception spec.
- Runs: `~/aic_results/{cheat_recheck,tuner_eval}/scoring.yaml`.

## Smoke #5 verdict (error channels): tilt EXONERATED — lateral hang + keying suspects
Max tilt 0.4° across all 10 wide-pose trials. Failures: 6/8 lateral (5.4–13.8 mm > integrator's
7.5 mm ceiling — the xy law only integrates the plug-hang; z feedforwards it. Intentional design:
pendulum-safe low-pass. Fix = EMA-filtered hang feedforward + integrator). 2/8 aligned-but-blocked
(t4: 0.5 mm lat, 0.1° tilt, no entry) → **rotation about the insertion axis (keying)** — unmeasured
by tilt; next instrumentation. 2/10 seated this run (t0 flipped vs smoke #4 → real run-to-run variance;
sweeps need repetitions).

## Open items (updated after eval parity)
1. **Offset sweep on EVAL-like poses** → capture-basin radius = the perception spec
   (`tuner_gen_offsets.py sweep` + score-aware objective in `tuner_basin.py`; repetitions per cell —
   run-to-run variance is real).
2. **EMA hang feedforward** (lateral fix, pendulum-safe) + **about-axis yaw metric** (keying suspect,
   t4-class) → smoke #6 on wide poses; lifts the hard-tail seat rate + collection-expert quality.
3. Wire the tuned endgame into the perception policy (as a NEW standalone policy version) once
   perception meets the basin spec.
- Camera capture path proven (`SNAP_OUT=… tuner_sweep.sh` → `aic_data/tuner_snaps/` + stitched MP4s);
  doubles as the DAgger capture path for perception.
