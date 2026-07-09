# YOLO port detector — fine-tune + native-res eval gate (2026-07-08)

**The A3b gate from `action_plan_perception_fix.md`, finally run.** Result: with **clean labels
(85 px fix) + native resolution**, a fine-tuned YOLO **transfers to eval-domain frames — 2/3 trials
within the 3.6 cm budget, SC at 3.9 mm.** This is the **first learned detector to pass the eval gate**,
and it overturns the prior "domain gap kills all learned detectors" verdict: that verdict was
**confounded** by corrupted labels (R4) and by testing at 0.25× resolution.

---

## Question tested (A3b)
Does a detector **fine-tuned on our own data** localize the port on **unseen eval-config scenes**,
after the two prior confounds are removed —
1. **R4 label corruption** (recorded port labels were desynced up to ~85 px on moving frames), and
2. **resolution** (perception was run at 0.25× = 256×288, where the port is only ~10 px)?

Prior learned models (KPNet / DualPoseNet / BoardNet) were trained on the **corrupted** labels and
scored at low res, so their eval failure never cleanly isolated "domain gap" from "bad data."

## Setup
- **Dataset** `~/aic_data/yolo_ds/`: 11,452 train / 905 val frames, **native 1024×1152**, 2 classes
  `{0: sfp_port, 1: sc_port}`. Labels = **FK-reprojection** of the static port through each frame's
  synced `tcp_pose` (fixes R4; mount recovered to 0.8 mm) + **SAM-mask QC**. Balanced (5,618 / 5,834
  instances). **Episode-clean split** (46 train / 4 val episodes, **zero leakage**).
  - Caveat: 11k frames but only **46 distinct board scenes** (≈249 correlated frames/episode) — real
    diversity is thin; multi-view within a scene helps viewpoint robustness but not pose variety.
- **Model**: `yolov8m` (25.9 M params), fine-tuned from COCO. `imgsz=640, batch=32, patience=15`,
  50 epochs max. Standard aug only (lighting/domain-randomization ablation deliberately excluded —
  prior ablations showed no concrete effect).
- **Env**: `fm_venv` (torch 2.12.1+cu130, Blackwell sm_120), ultralytics 8.4.90.

## Training result (our-data val — the "proxy")
- **Early-stopped at epoch 30** (patience-15; **best = epoch 15**).
- **mAP50 ≈ 0.88, mAP50-95 = 0.847.** Learns the port fast; plateaus ~epoch 15 (mild overfit to the
  46 scenes). *This is the proxy metric that "repeatedly lied" before — not the verdict.*

## Eval gate — 256 vs native 1024 (the decisive comparison)
Gate = run detector on the 3 eval-config sweep scenes, project GT port → compare box → 3D error at the
GT port height-plane vs the **36 mm** spiral bar. (`eval_gate_full.py`.)

| Trial | Type | **256 (caveated)** | **native 1024 (decisive)** | det rate | right-class |
|---|---|---|---|---|---|
| 0 | SFP | 160 mm ❌ | **22.2 mm ✅** (67% within) | 0.86 | 0.57 |
| 1 | SFP | 113 mm ❌ | 58.2 mm ❌ (30% within) | 0.96 | **0.33** |
| 2 | SC | 56 mm ❌ | **3.9 mm ✅** (53% within) | 0.95 | 0.95 |
| — | median | 113 mm, **0/3** | **22.2 mm, 2/3** | | |

**Resolution was the dominant factor** — 0/3 → 2/3 by resolution alone. (256 showed high detection but
poor localization = the scale-problem signature, not a domain wall.)

## Key findings
1. **A learned detector DOES transfer to eval-domain frames** — 2/3 within budget, **SC at 3.9 mm**,
   detection rate 0.86–0.96. First time. The earlier "R1 domain gap" verdict was **confounded** by the
   85 px labels + low res; fix both and it works.
2. **The lone failure (t1) is class confusion, not blindness.** det rate 0.96 but **right-class 0.33** —
   it detects a box nearly every frame but calls the wrong connector (SFP↔SC), so the wrong box (a
   different port on the board) is picked → 58 mm. Fixable by better class discrimination **or**
   geometry-selecting among candidate boxes (use board pose to pick the box nearest the expected port).
3. **Complementary with the SAM board-geometry track** — they fail on *different* trials:
   - SAM board-geometry: **t0 FAIL** (155 mm), t1 pass (3.2 mm), t2 pass (9.5 mm)
   - YOLO detector:       **t1 FAIL** (58 mm), t0 pass (22 mm), t2 pass (3.9 mm)
   An ensemble / YOLO-box → board-geometry cross-check plausibly reaches **3/3**.

## Honest caveats
- Scenes are the **eval config in *our* simulator** (eval-difficulty, unseen poses) — a strong
  generalization test, but **not the true vaulted rendering domain**, which stays untestable locally.
  "Transfers to unseen eval scenes" ≠ "certified for the vault."
- Only **3 trials**; per-frame medians; low-diversity training set (46 scenes).
- Deployment implication: run the detector at **native 1024** (skip the 0.25× downscale) — cheap under
  the gated-lock (perceive-once) pattern; feeds SAM box-prompt → triangulate → 3D port.

## Next steps
1. **Geometry-select fix for t1** — pick YOLO's box nearest the board-derived expected port (kills the
   class-confusion failure). Highest-leverage; likely 3/3.
2. **Ensemble YOLO + SAM board-geometry** — they cover each other's failing trial.
3. Fold into the deployment chain: YOLO box → SAM mask → triangulate 3 cams → 3D port → gated-lock.
4. Diversity: 46 → several hundred board scenes would remove the training-diversity confound.

## Reproduction
- Train: `~/aic_data/track2_sam/train_yolo_port.py` (fm_venv) → `~/aic_data/yolo_runs/port_ft_m640/`.
  Best banked at `~/aic_data/yolo_runs/best_final.pt` (mAP50-95 0.847, epoch 15).
- Native eval dump (needs sim): `~/ws_aic/aic_local/score.sh aic_example_policies.ros.SweepDumpFull
  true eval yolo_native_dump` → `~/aic_data/sweep_dump_full/trial{0,1,2}` (native 1024 + Tbc + gt.json).
  Policy: `SweepDumpFull.py` (native-res clone of `SweepDump`, installed in src + site-packages).
- Gate: `eval_gate_full.py <weights.pt> ~/aic_data/sweep_dump_full` (fm_venv).
- Data/weights live under `~/aic_data/` (outside the git repo).

Cross-ref: `action_plan_perception_fix.md` (Phases 2–3, the A3b gate) · `aic_data/track2_sam/FINDINGS.md`
(SAM board-geometry track).
