# AIC Run Plan — living tracker

> Strategy: **#7 learned perception ⊕ #3 force-compliant search**, composed CoStream-style
> (semantic anchor + reactive layer over a shared SE(3) interface; predictive/video-WM de-scoped — sim-only).
> Details: `ranked.md` · scores: `baseline_scores.md` · perception: `perception_results.md` · dataset: `collect/M6_PLAN.md`.

## Stages

| # | Stage | Status | Result / next action |
|---|---|---|---|
| M0 | Env (pixi + ROS Kilted, torch on Blackwell) | ✅ | works |
| M1 | Self-scoring baselines | ✅ | WaveArm **37.5** floor · CheatCode **279.4** ceiling · pretrained ACT **−21** |
| M2 | Config gen + collection pipeline (`perception_v1`) | ✅ | 49 eps, images ↔ GT port/plug labels |
| M3 | Port-only perception net | ✅ | ~9 mm @ close range (`m3_perception_run/best.pt`) |
| M3.5 | **Dual anchor** (port + plug from vision; plug labels derived, no re-collection) | ✅ | port ~9 / plug ~8 mm close; relative ~12 mm (`m35_dual_run/best.pt`) |
| — | Resolution ablation (288×256 vs native 1024²) | ✅ | native wins where it matters: port −30%, **relative 5.7 mm (−52%)** (`m35_native_1024/best.pt`) |
| M4 | `PerceptionInsert` policy wired, first GT-free score | ✅ | **−37.6** (trial_1 → 0.07 m, first GT-free proximity credit; failures diagnosed & separable) |
| M5 | v2 policy (force-stop + spiral + pose logging) + GT diagnostic | ✅ | far misses = **coverage gap, not frame bug** (61 mm in-range vs 335–372 mm OOD) |
| M6a | Coverage dataset `perception_v2` (169 eps, 5 nic rails × 2 ports + 2 sc rails, full public limits, native res) | ✅ | 35% expert seating — labels perfect regardless; failure mechanism fully diagnosed (below) |
| M6b | CheatCode failure forensics | ✅ | plug parks **exactly at the entrance plane** (45.8 SFP / ~15 SC): wide poses → wrist near limits → 3–14 mm lateral + yaw error > chamfer; offset-recomputing z-target self-cancels (hovers @ ~8 N). Axis verified vertical everywhere (≤0.6°) — axis-descent fix NOT needed. Missing ingredient = search-at-the-mouth (our spiral covers 3–14 mm; yaw dither still needed) |
| M6c | **v2 retrain — dual anchor @1024** | ⏸ **stopped at epoch 6** | port 18.2 mm / 7.6° · plug 33.4 mm on the *wider* val (`m6_dual_run/best_epoch6_18mm.pt`). Port still improving when stopped |
| M6d | **Conditioned retrain on merged v1-native + v2 (219 eps)** — trainer rewritten with target-identity conditioning (the ambiguity fix: 43 mm home-pose val / 69–220 mm eval), multi-root data, orientation-weighted checkpoint, FIRST/CLOSE gates. Policy updated to condition on the Task msg identity + auto-read img_size/vocab from checkpoint | ⏳ **launched by user** | @224 first (fast loop). **Gate 1: FIRST collapses (~<20 mm).** Then M7a (<30 mm) → 1024 polish → M7b. Expected ckpt: `~/aic_data/m7_cond224/best.pt` |
| M6e | v3 policy: **yaw dither** (keying failures) + **sanity gate** (reject implausible/jumping estimates) + checkpoint swap to `best_epoch6_18mm.pt` | ✅ | dither ±0/3/6° cycled with spiral; gate = base_link envelope + 80 mm jump-reject (outliers never enter the median filter; all-gated → hold, never chase). Installed to site-packages, import verified |
| M7a | GT diagnostic run (`ground_truth:=true`, control perception-only) | ✅ run, ❌ criterion | port err **112 / 69 / 220 mm** on eval trials (>30 mm) — perception still the bottleneck. Gate + spiral behaved (trial_3: 23/31 gated, no chase; spiral to k=30). Diag total 33.7 vs −51.5 (M5) |
| M7b | Real score (`ground_truth:=false`) | ✅ | **+1.4** (v1 was −37.6). Distances 0.07/0.12/0.27 m; trial_2 clean (0 contacts, t3 17.9); still 0 insertions. Reactive stack validated; floor not yet beaten (37.5) |
| M8 | Iterate by result | ⏳ **still perception** | M7a said perception (69–220 mm on eval) was the sole remaining bottleneck. M10 keypoint rebuild looked like the fix (1.9 mm on the v1 proxy) but **M11 live eval-scene test disproved it**: on the *actual* vaulted scenes the keypoint detector is 8–40 mm (SFP) / **300–430 mm (SC catastrophic)** — the v1 proxy was NOT the eval domain. Perception is STILL the bottleneck; the generated→vaulted domain gap is unsolved |
| M9 | **Force-insertion rig (InsertTuner)** — standalone GT-fed endgame tuner | ✅ **eval parity** | 5 smoke iterations → **official-eval head-to-head: our stack 277.7 vs CheatCode 279.6, both 3/3 insertions** (gap = duration bonus only). Smoke #5: tilt exonerated (≤0.4°); wide-pose failures = lateral hang (6/8, > integrator 7.5 mm) + keying suspects (2/8). Next: offset sweep on eval poses → basin = perception spec; EMA hang feedforward. Details: `force_insertion.md` |
| M10 | **Keypoint + triangulation perception rebuild** — port-center heatmap detector + 3-ray triangulation, replacing the DualPoseNet regression | ⚠️ **proxy only — did NOT transfer** | Held-out `perception_v1` **1.9 mm** median / 11.9 mm p90 (self-check 0.0 mm, detector val 3.5 px @ epoch 8) — but v1 is a *generated* collection, not the vaulted eval domain. **M11 showed the 1.9 mm does not hold on eval scenes.** Standalone `collect/kp_{train,eval3d,perceive}.py`; ckpt `~/aic_data/kp_v1_run/best.pt`. Lesson: never trust a proxy that isn't the eval domain |
| M11 | **Eyes+hands live test** (`PerceptionInsertKP`: keypoint port position + DualPoseNet orient/plug + reactive CheatCode descent) | ❌ **regressed** | Real score **−43.3** (gt:=false) / **−22.8** (gt:=true, perception-logged) vs M7b +1.4. Live perceived-vs-GT: SFP **8–40 mm** (better than regression's 69–112), **SC 300–430 mm (catastrophic)**. NOT a scale bug (native imHW, corr 0.08). One SFP trial hit **0.03 m** proximity (+25, best GT-free ever) — pipeline works when eyes lock on; SC failure drives the arm into the enclosure (−24/−12). Bottleneck = generated→vaulted **domain gap** + SC. Standalone `ros/PerceptionInsertKP.py`; log `~/aic_data/kp_pose_log.jsonl` |
| M12 | **Ablation study — isolate the eval-gap cause** | ✅ **cause = scene-content OOD** | Ruled OUT: scale bug (corr 0.08), bad checkpoint (in-domain 0.7 px), SC-intrinsic (v1-SC 1.8 px), **all photometric** (bright/dark/contrast/hue/sat/blur ≤9 px), **wrong-object lock-on** (distractor test 95% stayed), **architecture** (same-frame KPNet vs DualPoseNet both fail on SC: 322/474 mm). Confirmed: **scene-content/geometry OOD, acute for SC** (both nets marginal on SFP ~40–60 mm, catastrophic on SC). ⚠️ **Fix conclusion refined by M13** — the "content OOD" turned out to be *target off-frame + wrong connector*, not a data-diversity gap. Scripts: `collect/{probe_lighting,distractor_test,make_pred_video}.py`; details `perception_results.md` M12 |
| M13 | **3rd-person view diagnosis — 3 failure modes** | ✅ **root cause pinned** | Scene-camera (`scenecam_kp` vs `scenecam_cheat3` 279.4) + GT-visibility reveal: **(A)** tracks WRONG connector (green NIC card, not blue SC target) — KPNet **not target-conditioned**; **(B)** blue SC target **off-frame 78–84%** (low z=0.014 leaves wrist FOV; 100%→25%→0% home/approach/descent); **(C)** plug **twist not compensated** — DualPoseNet orientation coarse → `q_diff≈identity` → gripper straight → off-axis (trial 2 hit 0.03 m yet didn't seat). Fix = **conditioning + centering + twist** (all perception/policy, no force work); 1+2 coupled. Details `perception_results.md` M13 |
| M14 | **Plug grasp offset is TYPE-FIXED → mode C solved by kinematics** | ✅ **plug side free** | `T_grasp=inv(gripper_FK)·plug` measured constant within-trial (≤0.3° drift) AND identical same-type (SFP↔SFP **0.0°**, SFP↔SC **80°**). Values: SFP rot [20.5,1.1,−3.0]° / SC rot [−1.6,−25.2,75.5]°. → **plug pose = FK(gripper)·T_grasp[type]**, exact, ZERO vision; eliminates DualPoseNet plug; solves mode C. Fair source = public cable-template nominal grasp (TODO verify). Vision now needed ONLY for port. Script `scratchpad/analyze_grasp.py`; details `perception_results.md` M14 |

## CoStream component map (the "mix")

| Behavior | Ours | Status |
|---|---|---|
| Semantic anchor | **Keypoint detector + 3-ray triangulation** → port xyz in base_link (1.9 mm on v1 proxy but **8–430 mm on real eval scenes**, SC catastrophic) | ⚠️ **eval domain gap unsolved** |
| Reactive | force-stop + golden-angle spiral (≤36 mm) | ✅ · ⏳ + yaw dither |
| Predictive (video WM) | de-scoped (sim-only, no real data) | — deliberate |
| SE(3) composition | anchors → base_link via robot TF; port median-filtered, plug fresh per step | ✅ |
| Stage structure | CheatCode approach→descend, reused | ✅ |

## Parallel track (after M7)
- **IL baseline (#5)** — ACT/SmolVLA on `il_frames`, *seated episodes only* (~59 v2 + 49 v1 ≈ 108 clean demos;
  failed expert trajectories poison IL, harmless for perception). Comparison, not the primary path.
- **CheatCode v2.1** (spiral + yaw wiggle + integrator windup↑) if we top up close-range data on weak rails.

## Standing rules
- Vaulted eval config is **never read** (fair play — match the public legal space, not the draws).
- Keep ALL collected episodes for perception training; `seated` is a quality flag, not a filter.
- Checkpoints picked with orientation-weighted metric; per-epoch named saves.
- Judge changes by **task success rate / score**, not mm (pose error ≠ insertion — confirmed at M4).
- Policy edits: copy `src/` → pixi `site-packages/` (copy-install; edits invisible otherwise).
- Long orchestrators: `setsid nohup … </dev/null &` (harness background reaps them).
