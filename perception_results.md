# Perception Results — M3 / M3.5 (images → port & plug pose)

The "semantic anchor" of the `#7 + #3` plan (see `ranked.md`): learn the hidden target pose from
the wrist cameras, then hand it to the compliant analytic insertion (CheatCode-style). GT labels are
the free simulator pose from TF, so this is **supervised regression, not end-to-end IL**.

## Data (no new collection)
- **`perception_v1/`** — 49 kept episodes (25 SFP / 24 SC), ~18.6k labeled trinocular frames, 288×256 RGB.
- **Split:** whole-episode holdout, type-stratified → 39 train / 10 val (`val = [0,1,12,13,15,28,29,35,36,46]`).
  Validation measures generalization to **unseen scenes**, not memorized frames.
- Input: 3 wrist RGB (left/center/right). Target frame: **`center_camera/optical`** (the frame the
  collector logged labels in, `collector_node.py:153`).

## Method
- **Model:** shared ImageNet ResNet-18 applied per camera → concat (3×512) → MLP trunk → regression heads.
  Position L2 (smooth-L1 on standardized targets) + quaternion geodesic loss. AMP, 15 epochs, ~4 min on the
  RTX PRO 4000 (24 GB).
- **M3** (`collect/train_perception.py`): port pose only.
- **M3.5** (`collect/train_perception_dual.py`): **port AND plug** pose. CheatCode uses GT for *both*
  (port line 207, plug tip line 87); the plug can't be assumed from the gripper (grasp varies 5–9 mm / up
  to 17°, and the deformable cable flexes 6–13 mm *within* an episode), so it must be **seen**.
  - Plug labels were **derived from existing data** (no re-collection):
    `T_base←optical = port_base_pose ∘ inv(port_center_pose)`, then `plug_optical = inv(T_base←optical) ∘ plug_base`.
  - **Validated:** derived plug projects **inside** the 288×256 image in **84.6%** of frames (using the
    stored intrinsics scaled ×0.25); camera-origin↔TCP ≈ 33 cm (consistent with the wrist mount).

## Results (held-out, unseen scenes)

### M3.5 dual anchor — position
| | whole-episode median | near insertion (<0.28 m) | closest 20% |
|---|---|---|---|
| **Port** | 12.6 mm | 9.8 mm | 9.0 mm |
| **Plug** | 19.7 mm | 9.0 mm | 7.7 mm |
| **Relative (plug→port)** | — | 11.6 mm | 11.9 mm |

### M3.5 dual anchor — orientation
| | whole-episode median | near insertion (<0.28 m) | closest 20% |
|---|---|---|---|
| **Port** | 9.5° | 9.0° | 13.7° |
| **Plug** | 9.1° | 11.4° | 13.1° |

(M3 port-only baseline was ~9 mm at close range — dual head did not hurt the port anchor.)

## Key findings
- **Both anchors sharpen on approach**: near contact the plug reaches **7.7 mm** (big and centred in view);
  the port ~9 mm.
- **Relative plug→port error ≈ 12 mm at insertion range.** Port/plug error *magnitudes* correlate 0.90, but
  the error *vectors* do not cancel — the relative term ≈ √(port² + plug²). Still, dual-vision (~12 mm) beats
  the alternative it replaced: a *nominal* plug offset would bake in ~15–20 mm (grasp slop + flex).
- **Orientation worsens at the very closest frames** (13° vs 9°) — anchors clip the frame edge / occlude just
  before contact. Temporal continuity should recover this.
- **Overfitting caps accuracy**: train loss keeps falling while val plateaus after ~epoch 2 (39 episodes is
  small). **Scaling the free sim data is the main accuracy lever.**
- ⚠️ **Checkpoint-selection caveat:** `best.pt` is selected by lowest *position* sum → landed on **epoch 0**,
  which has the **worst orientation** of the run. Later epochs reach **~5.5°** whole-val orientation at a ~4 mm
  position cost. Re-selecting with an orientation-weighted metric (one line, no retrain) roughly halves the
  reported orientation error.

## Resolution ablation — 288×256 vs native 1024×1024

**Question:** does feeding more pixels-on-target sharpen the anchors, or is 288×256 already enough?
**Setup:** same dual-anchor trainer, **same 10 held-out episodes**, only the input changes. New dataset
`~/aic_data/perception_native/` — the *same 50 scene configs* re-collected at native 1152×1024 (fresh
rollouts, not frame-identical). Native run uses aspect-preserving center-crop (`--keep_aspect`), input
1024²; batch 12 / workers 6 (see RAM note). Knobs added to the trainer: `--img_size --keep_aspect --val_eps`.

**Result (held-out, closest-20% frames = insertion moment):**

| Metric (closest 20%) | 224 (288×256) | **1024 (native)** | Δ |
|---|---|---|---|
| Port position | 9.0 mm | **6.3 mm** | −30% |
| Plug position | 7.7 mm | **7.4 mm** | ~same |
| **Relative (plug→port)** | 11.9 mm | **5.7 mm** | **−52%** |
| Port / Plug orientation | 13.7° / 13.1° | 5.0° / 5.3° | better* |

Whole-episode medians: port 12.6→**11.1 mm**, plug 19.7→**16.5 mm**.

**Verdict:** resolution helps exactly where predicted — the **small, hard-to-localize port** (−30%) and the
**relative plug→port alignment** (**halved to 5.7 mm**). The **plug barely changes** (already big/centred in
view → extra pixels add little). 5.7 mm relative + ~5° is materially better going into insertion than 12 mm.
Inference at 1024² is a single cheap forward pass/step — no training-memory cost — so **M4 should use the
1024 checkpoint** (policy must preprocess eval images with the same 1024 aspect-crop).

\* **Confound (be honest):** the two `best.pt` were selected by lowest *position* sum → 1024 landed on
**epoch 5** (good orientation), 224 on **epoch 0** (worst orientation of its run). So the *orientation* gap is
mostly the epoch, not the resolution (224's later epochs also reach ~5.5°). The *position/relative* gains are
real (close-range medians, resolution effect dominates). For a fully clean number, re-select both with an
orientation-aware metric before quoting the orientation delta.

**RAM note (native only):** 1024² images make the dataloader ~16× heavier per pixel. `batch 24 / workers 10`
spiked CPU RAM past 61 GB → silent OS OOM-kill (no traceback). Fix = shrink the dataloader footprint:
**`batch 12 / workers 6`** → peak 21.5 GB, stable (also halves GPU to ~12 GB). Nothing wrong with the trainer;
just scale batch/workers down in proportion to image size. Long detached runs must launch under the
harness-managed background (not bare `nohup`), or they get reaped mid-epoch.

## Artifacts
| Thing | Path |
|---|---|
| M3 trainer (port only) | `collect/train_perception.py` |
| M3.5 / ablation trainer (port + plug; `--img_size/--keep_aspect/--val_eps`) | `collect/train_perception_dual.py` |
| M3 checkpoint | `~/aic_data/m3_perception_run/best.pt` (+ `history.json`) |
| **M3.5 checkpoint** — 224, port+plug (weights + pose norm stats + val split) | `~/aic_data/m35_dual_run/best.pt` |
| **Native-1024 checkpoint** — best anchor accuracy (use for M4) | `~/aic_data/m35_native_1024/best.pt` |
| Native dataset (1152×1024, same 50 configs) | `~/aic_data/perception_native/` |

## Next
- **M4 wiring:** fork CheatCode → `PerceptionInsert`. net → port & plug in `center_camera/optical`
  → `base_link` via robot TF (published without `ground_truth`) → CheatCode alignment math, plus
  **temporal median filtering** of the estimates and a light **spiral/force search** on final descent.
  Then `score.sh <policy> false`. The score's gap to CheatCode's **279** ceiling = the real cost of the 12 mm.
- **Accuracy levers (after M4 tells us how far 12 mm gets us):** (a) re-select/orientation-aware checkpoint;
  (b) direct relative-pose head (`port − plug`); (c) scale the sim dataset; (d) temporal fusion across frames.

---

# Keypoint detector + multi-view triangulation (first pass) — 2026-07-05 — ⚠️ PROXY ONLY (see correction at end)

> **CORRECTION (M11, same day):** the 1.9 mm below is on `perception_v1`, a *generated* collection —
> NOT the vaulted eval domain. The live eyes+hands test (`PerceptionInsertKP`) measured real eval-scene
> perception at **8–40 mm (SFP) / 300–430 mm (SC, catastrophic)** and scored **−43.3 / −22.8** (below
> M7b +1.4). The v1 proxy did not transfer; the generated→vaulted domain gap is unsolved. Read this
> section as "the detector trains + triangulates correctly," not "perception solved." Full M11 at end.

The regression net above (DualPoseNet, global port/plug pose regression) looked fine in-distribution
(~12 mm val) but **blew up to 69–220 mm on the live eval scenes** (`RUN_PLAN.md` M7a) — it memorized scene
cues instead of localizing the port. Rebuilt perception as **local keypoint detection + geometry**: detect
the port-center pixel in each wrist view, back-project to a `base_link` ray, intersect the 3 rays.

## Method (all standalone; `PerceptionInsert.py` untouched, per constraint)
- **Detector** (`collect/kp_train.py`): ResNet-18 → upsample decoder → 1-channel heatmap → soft-argmax →
  (u,v). Trained on `perception_native_c256 + perception_v2_c256` (219 eps, 288×256), **episode-wise seeded
  split** (186/33, no frame leakage), stride 3 → 73k/13k image-views. GPU-bound ~81 s/epoch; converged
  epoch 8 (val 2D median **3.5 px ≈ mm**). `best.pt` → `~/aic_data/kp_v1_run/`.
- **Triangulation** (`collect/kp_eval3d.py`): per-cam pixel → ray in `base_link` via `T_base_cam`;
  least-squares 3-ray intersection → port xyz. Offline `T_base_cam` reconstructed from the GT port pose
  correspondence (`T_base_port · inv(T_cam_port)`); **self-check: feeding GT pixels recovers `port_base` to
  0.0 mm**, validating extrinsics + the wxyz convention. Live path uses TF for `T_base_cam`.
- **Deployable module** (`collect/kp_perceive.py`): `KeypointPerceiver` — images + K + T_base_cam → port
  xyz in `base_link`. The reusable "eyes."

## Results — 3D port position error (self-check 0.0 mm on both sets)
| Set | n | median | mean | p90 |
|---|---|---|---|---|
| In-dist val (native+v2 held-out episodes) | 1650 | 3.9 mm | 14.6 mm | 34.6 mm |
| **OOD — `perception_v1` (foreign collection, never trained on)** | 2348 | **1.9 mm** | 6.3 mm | **11.9 mm** |
| v1 depth near-half / far-half | — | 1.3 / 2.6 mm | — | — |

vs the regression it replaces: **~12 mm val → 69–220 mm eval**.

## Read
- **OOD gap eliminated** — keypoint+triangulation is *better* OOD (1.9 mm) than in-dist (3.9 mm), the
  opposite of the regression net. Local features transfer; global regression memorized.
- **Triangulation kills the 2D tail** — per-cam 2D p90 ~40 px on the hard val → 3D p90 12 mm on v1.
- **Consistency:** OOD p90 = 11.9 mm (90% of *single* frames < 12 mm); temporal median over the approach →
  ~1.9 mm effective — well inside any plausible insertion basin.
- **Trains hard → tests easy:** trained on v2's wide randomization (all rails, ±1.5 rad), so v1's narrow
  scenes are trivial (per-cam 1.6–2.2 px). Healthy direction of generalization.

## Honest gaps before "solved"
1. v1 is a held-out *collection* (strong OOD proxy), NOT the live `aic_engine` eval scenes where the
   69–220 mm was measured. Live eval-scene number is the last measurement.
2. GT-derived extrinsics offline; live uses TF/FK (+~1–2 mm). Self-check confirms residual is pure
   detector error, so live degradation is bounded by FK accuracy.

## Next
- Live eval-scene perception logger (`KeypointPerceiver` + TF) on one GT eval → the real OOD number.
- On confirmation, wire `KeypointPerceiver` into the standalone insertion re-score (InsertTuner hands +
  these eyes). Pass 2: mesh→port frame fix + 8 cage corners → 6-DoF orientation; 1024 high-res polish.
- Full technical record + pipeline diagram: `keypoint_pnp_perception_plan.md` §10.

---

# M11 — Live eyes+hands test on the REAL eval scenes (2026-07-05) — the proxy correction

`PerceptionInsertKP` (standalone): keypoint port position (our eyes) + DualPoseNet orientation/plug +
reactive CheatCode descent. Controlled A/B vs M7b (+1.4) — only the port-position source changed.

**Score:** −43.3 (gt:=false) / −22.8 (gt:=true, perception-logged). Both below the +1.4 baseline.

**Live perceived-port vs GT-port (the number that matters), per eval trial:**
| Trial | Connector | GT port (base_link) | kp error raw → filtered | scoring final dist |
|---|---|---|---|---|
| 1 | SFP | [-0.42, 0.31, 0.13] | 48 mm → **8 mm** | 0.11 m |
| 2 | SFP | [-0.45, 0.12, 0.13] | **37 mm** (0 gated) | 0.03 m (+25 proximity) |
| 3 | **SC** | [-0.50, 0.25, **0.01**] | **300–430 mm** (24/37 frames gated) | 0.54 m |

**Findings:**
- **NOT a scale/K bug** — live images arrive native `[1024,1152]`; corr(off-center pixel dist, 3D error)
  = 0.08. Triangulation + intrinsics are correct (self-check was 0.0 mm).
- **The v1 proxy (1.9 mm) did not transfer.** v1 is a *generated* collection sharing the training
  pipeline; the vaulted eval scenes are a different visual domain. Real SFP error is 8–40 mm (≈20× the
  proxy), and SC is broken (detector puts the low SC port — z=0.01 — 17 cm too high, 30–43 cm off,
  emitting image-edge u=0/279 garbage two-thirds of the time).
- **vs the regression it aimed to beat:** better on SFP (8–40 vs 69–112 mm) but **worse on SC**
  (430 vs 220 mm). The SC catastrophe drives the arm into the enclosure (−24 contact, −12 force) and is
  the primary score-killer.
- **Pipeline mechanically works:** trial 2 reached **0.03 m** (best GT-free proximity ever) — when the
  eyes lock on, eyes+hands compose correctly. Seating still blocked (orientation ~9° + plug + keying).

**Root cause:** the generated→vaulted **domain gap** (same gap the regression had), acute for SC; plus a
single un-conditioned "find-a-port" keypoint with no target identity. NOT geometry, NOT a bug.

**Candidate fixes (unvalidated):** (a) SC-specific data / debugging the low-z SC failure; (b) domain
randomization (lighting/texture) or eval-representative data to close generated→vaulted; (c) target
conditioning + confidence/peak-sharpness gating to kill edge-garbage; (d) an eval-domain proxy set so we
stop being fooled by v1. Judge every fix by the live eval-scene error + score, never by a generated proxy.

---

# M12 — Ablation study: what actually causes the eval perception gap (2026-07-05)

After M11 showed perception fails on the real eval scenes (SFP 8–40 mm, SC 300–430 mm) despite the
1.9 mm v1 proxy, we ran a systematic ablation to isolate the cause. Each hypothesis got a cheap,
targeted test on data where the detector is known-sharp (v1) or on the eval logs directly.

## The ledger

| # | Hypothesis | Test | Result | Verdict |
|---|---|---|---|---|
| 1 | Scale / K / preprocessing bug | live image dims + corr(off-center px, 3D err) | native `[1024,1152]`; corr **0.08** | ❌ ruled out |
| 2 | Corrupt / mis-loaded checkpoint | model prediction vs GT on the *same* episode as the label video (v2 ep0007) | **0.7 px** err, **0.2 px** frame-to-frame jitter | ❌ ruled out |
| 3 | SC connector intrinsically hard | v1-SC baseline error | **1.8 px** (SC is *easier* than v1-SFP 5.5 px) | ❌ ruled out |
| 4 | Photometric / lighting | brightness ×2.5, dark ×0.4, contrast washout, hue ±0.15, saturation 0.2/2.0, blur k5 on v1 | all **≤ 9 px**, peak-confidence 0.25–0.54 (never diffuse) | ❌ ruled out |
| 5 | Wrong-object lock-on (no target-conditioning) | distractor injection — paste a 2nd identical port into a clean v1 frame | **95% stayed** on the true port, **0% jumped** to the copy | ❌ ruled out |
| 6 | Architecture (keypoint vs regression) | same-frame KPNet vs DualPoseNet per eval scene | **both fail identically** (see below) | ❌ ruled out |
| 7 | **Scene-property OOD (content / geometry / pose), acute for SC** | #6 both nets on the eval scenes | SFP both marginal, **SC both catastrophic** | ✅ **confirmed** |

## #6 same-frame method comparison (both nets, identical eval frames)
| Trial | Scene | KPNet err | DualPoseNet err |
|---|---|---|---|
| 1 | SFP | 62 mm | 51 mm |
| 2 | SFP | 37 mm | 63 mm |
| 3 | **SC** | **322 mm** | **474 mm** |

Both architectures fail on the **same** SC scene (DualPoseNet even worse); on SFP they're comparable
and marginal (~40–60 mm). The method is not the differentiator — the **scene** is.

## Bonus finding — jitter = uncertainty (not a defect)
The "jagged eval trajectory" is the model's confidence signal, not a broken checkpoint. In-domain
frame-to-frame jitter is 0.2 px; on eval it is trial-1 26–42 px (low-confidence *wander*, rescued to
8 mm by the 15-frame median filter), trial-2 0.4 px (sharp → worked, 0.03 m), trial-3 5–8 px
(*confidently wrong* — glued to a distractor NIC card, GT port off-frame). Diffuse heatmap → soft-argmax
skates between candidate blobs.

## Conclusion
The eval gap is a **scene-content / geometry OOD**, acute for **SC**. The eval SC port sits at z≈0.01
(vs 0.13 for SFP) in a layout our generated SC data (only 2 SC rails) doesn't represent, so **no net
trained on the generated data recognizes it** — v1-SC was easy only because it matched training.
It is NOT lighting (ruled out), NOT target-conditioning (ruled out), NOT the architecture (ruled out).

## Implication (settles the "more diverse data?" fork)
Fix = **geometric / content data diversity spanning the legal eval space, SC prioritized** — regenerate
training scenes covering the SC pose/geometry (and more SFP pose diversity, since SFP is only ~40–60 mm
for both nets too). Photometric randomization, target-conditioning, and architecture swaps are **not**
the levers. Judge any fix on the live eval-scene error, never on a generated proxy.

## Artifacts
| Test | Script |
|---|---|
| in-domain prediction video (red=pred, green=GT) | `collect/make_pred_video.py` → `kp_pred_video_ep0007.mp4` |
| photometric sweep (#4) + heatmap confidence | `collect/probe_lighting.py` |
| distractor injection (#5) | `collect/distractor_test.py` |
| same-frame method comparison (#6) | `PerceptionInsertKP` logs `port_dual`; `scratchpad/analyze_method5.py` |
| red-dot eval videos | `kp_eval_t1/t2/t3.mp4` |

---

# M13 — Three confirmed failure modes from the 3rd-person view (2026-07-05)

A front-on scene camera (`~/aic_results/scenecam_kp/scene.mp4` = ours, `scenecam_cheat3/scene.mp4` =
CheatCode 279.4 reference) exposed **two failure modes the wrist-cam pixel analysis structurally could
not see**, which together with the visibility finding (M12/M13-B) give the complete picture.

## The three modes

| # | Failure | Trials | Root cause | Seen via |
|---|---|---|---|---|
| **A** | **Tracks the WRONG connector** — goes to a green NIC card instead of the blue SC target | 3 | KPNet is **not target-conditioned** — seeks "a port"; with the blue SC target at the frame edge + green NIC cards present (which our SC training scenes never contained), it grabs the familiar green card | 3rd-person: ours drives to the far-right green card while the blue SC connectors sit untouched center-board; CheatCode centers on the blue SC |
| **B** | **Target leaves the camera view** | 1, 3 | blue SC port is low (z=0.014) → **off-frame 78–84%** of the trial (100% at home → 25% approach → 0% descent) | wrist-cam GT-projection: `analyze_visibility.py` |
| **C** | **Plug twist not compensated** | **2** (position was fine at 0.03 m, still didn't seat) | plug is grasped at an angle; CheatCode reads `q_plug` from GT so `q_diff=q_port⊗q_plug⁻¹` tilts the gripper to align. DualPoseNet's plug/port orientation is too coarse → `q_diff≈identity` → **gripper stays straight** → plug enters off-axis → keyed connector rejects it | 3rd-person: CheatCode wrist visibly tilts to match; ours descends vertical/straight |

## How they interlock
- **A + B on trial 3:** can't *see* the blue connector (B) **and** doesn't *know* to insist on the blue
  one (A) → defaults to "the same old connector" (the habitual green NIC/SFP target) on the far right.
- **C on trial 2:** the only trial where *position* was solved (0.03 m proximity) — it failed purely on
  *angle*. Proof that fixing position alone will not seat the plug.

## Fix list (all perception/policy — no force work)
1. **Target-conditioning** — detector must seek the *specific* connector (blue SC), not any port.
   Attacks A. (Conditioning was dropped from KPNet; this is the case where it genuinely matters —
   unlike the same-type distractor test, which the un-conditioned net passed 95–97%.)
2. **Center-the-target behavior** — closed-loop: use the rough home-pose estimate to re-orient the wrist
   so the connector comes to frame center (where the detector is accurate), re-perceive, iterate until
   stable, *then* descend. Attacks B. Reuses FK extrinsics (moving the camera updates triangulation).
3. **Perceive-twist-once + FK-track** (or 6-DoF plug keypoints / direct relative-orientation head) —
   tilt the gripper to the grasp angle. Attacks C.

**Coupling:** 1+2 must go together (conditioning only helps once the connector is in view; centering only
helps once seeking the right connector). Even with both, trial 2 needs C to actually insert.

## Why GT-fed InsertTuner (277.7) sidesteps all three
GT gives the exact port identity+position (never wrong connector, never off-frame — A,B) and exact plug
orientation (correct tilt — C). Our perception must reconstruct all three; the eyes, not the hands, are
the gap. Videos: `scenecam_kp/scene.mp4`, `scenecam_cheat3/scene.mp4`, `scenecam_side_by_side.mp4`.

---

# M14 — Plug grasp offset is TYPE-FIXED → plug pose from kinematics, no vision (2026-07-05)

Follow-up to M13 mode C (plug twist uncompensated). The plug is rigidly held, so
`T_grasp = inv(gripper_FK) · plug` is the fixed rotation+translation from gripper to plug. Measured it
per frame across the eval trials (gt:=true, logging `gripper/tcp` FK + GT plug tip). Result: **the grasp
offset is constant within a trial AND identical across trials of the same connector type.**

## Measured grasp offset (gripper → plug)
| Connector | rotation (euler xyz, deg) | translation (m) | within-trial drift | same-type spread |
|---|---|---|---|---|
| **SFP** (trials 1,2) | **[20.5, 1.1, −3.0]** | [0.000, −0.021, 0.052] | ≤0.3° | **0.0°** (trial1 vs trial2) |
| **SC** (trial 3) | **[−1.6, −25.2, 75.5]** | [−0.004, −0.011, 0.012] | ≤0.1° | — |
| cross-type (SFP vs SC) | — | — | — | **80°** |

- **Rigid within a trial** (≤0.3° drift) → no cable-flex at the tip; a single measurement suffices.
- **Type-fixed** (SFP↔SFP = 0.0°, SFP↔SC = 80°) → one constant per connector type, not per episode.

## Consequence — the plug side becomes free
```
plug pose = FK(gripper) · T_grasp[type]        # exact, every frame, ZERO vision
```
- **Solves mode C:** the gripper tilts to the true grasp angle (20.5° SFP / 75° roll SC) instead of
  staying straight — the off-axis insertion that killed trial 2 (0.03 m, position was fine).
- **Eliminates DualPoseNet's plug entirely** — the coarse plug estimate is replaced by an exact
  FK-derived pose. Vision is now needed ONLY for the port, never the plug.
- Reconstructs exactly what CheatCode reads from GT, using FK + 2 baked constants.

## Fairness / generality
The eval uses a **deterministic** grasp per type (why it's fixed); our own generator randomizes grasp
roll/pitch/yaw (`gen_configs.py grasp_droll`), so our data wouldn't show this. The clean, unquestionably
fair source is the **public cable template's nominal grasp** — same constant, no reading eval GT.
TODO: verify the template's nominal grasp matches [20.5, 1.1, −3.0] (SFP) before baking in.

## What still needs vision (port side only)
`q_diff = q_port ⊗ q_plug⁻¹` — with `q_plug` now exact, the remaining error is **port orientation**
(coarse DualPoseNet) and **port position** (KPNet; the visibility/conditioning modes A/B). Plug is solved.

Script: `scratchpad/analyze_grasp.py`; logging added to `PerceptionInsertKP` (`gripper_tf`/`plug_gt_tf`/`port_gt_tf`).
