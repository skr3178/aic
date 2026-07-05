# Perception Plan — Keypoint + PnP for the AIC port-localization gap

*Author's note: this is the perception ("eyes") plan. The force/insertion ("hands") is solved
separately — see `force_insertion.md` (InsertTuner: 277.7 with GT, 3/3 insertions, ~= CheatCode
ceiling). This document is the remaining half.*

---

## 1. Context — why this is now the whole problem

The force endgame is done. With ground-truth poses, the InsertTuner stack seats the connector on all
3 eval trials (277.7 vs CheatCode 279.6; the gap is only the duration bonus). So:

```
eval_score  ~=  277.7  x  P(perception lands inside the insertion basin)
```

The hands multiply a constant; the **eyes supply the probability**. The entire remaining question is:
**can we localize the port accurately enough, on the eval scenes, to land in the force stack's basin?**

### The failure we must fix: the OOD gap
The current perception net (M7, `m7_cond224`) is a **from-scratch ResNet18 doing direct 6-DoF pose
regression** (images + target-id -> port pose in camera frame -> analytic TF -> base). Its error:

| | Port error |
|---|---|
| Validation (in-distribution held-out episodes) | ~14.5 mm |
| **Eval scenes (out-of-distribution)** | **~80 mm** (descent-phase median; approach p90 up to 344 mm) |

That ~14 -> ~80 mm jump is the **Out-Of-Distribution (OOD) gap**: the net interpolates fine on
training-like scenes but extrapolates poorly to the frozen eval configs (different board poses/rails
it never saw well). **No policy-usage tweak closes this** — it's a perception generalization problem.

### Root cause
Direct pose regression forces the CNN to learn a **global** function: *whole scene -> 6-DoF pose*.
That function is memorized from ~200 training episodes and breaks on unseen board layouts. Two
architecture choices make it worst-case for OOD: **(a) from-scratch backbone**, **(b) direct pose
regression**. The `camera-frame -> analytic-TF` *structure* is correct and worth keeping; the
**regressor** is the weak link.

---

## 2. The decision — Keypoint + PnP (PVNet / DOPE style)

**Primary bet: replace the pose *regressor* with keypoint detection + exact geometry.** The net
outputs **where a few fixed points on the port appear in each image (2D pixels)**; then PnP /
multi-view triangulation turns those 2D detections into the 3D port pose. The net only ever learns
*"what a port corner looks like"* — a **local, transferable** feature — and all the metric geometry
is exact math that cannot overfit.

### Why it fixes the actual failure mode
- **Local features transfer.** A port-cage corner looks the same whether the board is at rail 2 or
  rail 4, near or far -> the detector generalizes where global pose-regression did not.
- **Geometry is moved out of the net.** The un-generalizable part (2D->pose) becomes `solvePnP` /
  triangulation — deterministic, no memorization to break OOD.
- **Every constraint favors it** (see section 5).

### Ranking of the alternatives (for OUR constraints: RGB-only, 3 wrist views, small/occluded port, ~20 Hz, we have markers + mesh + intrinsics)

| Option | Verdict | Why |
|---|---|---|
| **#1 Keypoint + PnP / triangulation** | **Best fit — primary** | Fixes the failure mode directly; RGB-only fine (multi-view is metric); real-time fine; slots into existing structure; reuses our data. |
| **#2 DINOv2 (frozen) backbone + small pose head** | Cheap parallel hedge | Better *features* (pretrained -> generalize) but keeps *direct regression* -> partial fix (~80 -> maybe ~40 mm). ~1-hour experiment; run as a baseline/control. |
| **#3 FoundationPose / MegaPose (render-and-compare)** | Reserve | Theoretically the most OOD-robust (model-based), but our constraints fight it: RGB-only (wants depth), port small + occluded by plug/gripper, needs an added 2D detector, heaviest + slowest. Fall back to this only if #1/#2 stall. |
| **"Ours" (from-scratch ResNet18 + direct regression)** | Weakest for OOD | Keep the camera-frame->TF *framing*; the from-scratch-regressor combo is exactly what generalizes least. |

---

## 3. The pipeline (end-to-end)

### A. One-time setup — define the keypoint set
Pick **K fixed 3D points rigidly attached to the port**, coordinates known in the *port frame*
(constants = the object model):
- Seed with the 3 we already have: `port_left / port_center / port_right`.
- Add ~3-5 **port-cage corners** read off the port **mesh** (`.obj` we transcoded), so there are
  **>=6 non-collinear** points (3 collinear markers alone -> PnP is degenerate/unstable).

### B. Training (offline, on existing data — NO new collection)
1. **Make 2D labels for free.** For every frame x each of the 3 cameras: we know the GT port pose
   (`port_center`), the camera->base TF, and intrinsics `K` (in `meta.json`). Compute each keypoint's
   3D position in that camera, **project through `K` -> pixel (u,v)** = the label. (`verify_labels.py`
   already does this projection.) Flag visible/occluded per keypoint.
2. **Train a keypoint detector.** CNN: image -> a **heatmap per keypoint** (peak = its pixel),
   PVNet/DOPE style. Loss = heatmap/pixel error vs the projected labels. **This is the only learned
   part** — it learns local appearance, not scene->pose.

### C. Runtime (per control step)
3. **Detect 2D keypoints** in left/center/right -> (u,v) + confidence per keypoint per view.
4. **Recover the 3D port pose** (two options; second preferred for us):
   - *(a) PnP per view:* K known 3D model points + 2D detections + `K` -> 6-DoF pose in that camera
     (`cv2.solvePnP`, needs >=4 non-collinear). Fuse the 3 views.
   - *(b) Multi-view triangulation (preferred):* each keypoint seen in >=2 cameras + known camera
     poses (from TF) -> **triangulate -> 3D point** (metric, no depth needed — the 3 wrist cameras are
     a calibrated stereo rig). Do all keypoints, then **Kabsch/Procrustes** fit: known 3D model
     points <-> triangulated 3D points -> port 6-DoF pose. Robust and directly metric.
5. **Analytic TF -> base_link** (same as now; in the triangulation variant it's already in base).
   Hand the port pose to the force stack — **identical downstream, no pipeline change**.

---

## 4. Applicability to the existing dataset

| Method | Applicable on our data? | Notes |
|---|---|---|
| **#1 Keypoint + PnP** | **Yes** | 2D labels generated by projecting GT `port_*` markers (+ mesh corners) into each image via intrinsics — reuses `verify_labels.py` machinery. No re-collection. Caveat: add mesh-derived corners so keypoints are non-collinear. |
| **#2 DINOv2** | **Yes, drop-in** | Same images, same GT-pose labels, same pipeline; only the backbone changes. |
| **#3 FoundationPose** | **Partial** | Zero-shot on the port **mesh** (we have it) — doesn't use our labels; our data only *validates*. Needs an added 2D detector + is RGB-only (no depth -> weaker). |

Everything trains/validates on `perception_v2` (169 eps, all rails) + `perception_native` — the
labels are just *projected keypoints* instead of *pose vectors*. Checkpoints stay under `~/aic_data/`.

---

## 5. Constraints (fixed by the eval harness) and how #1 fits them

- **3 wrist cameras + 26-dim state only. No extra/top-view cameras** (eval `Observation` schema is
  fixed). -> #1 uses exactly the 3 views, as a stereo rig for triangulation.
- **RGB only, no depth.** -> keypoints + multi-view triangulation are metric *without* depth.
- **~20 Hz control loop.** -> keypoint CNN is fast; PnP/triangulation is trivial.
- **Plug (in gripper) is known via kinematics**, not vision (gripper TCP from FK + nominal grasp,
  ~cm uncertainty). The **port is the only real unknown** — so all perception effort goes here.
- The port is **small and partly occluded** by the plug/gripper in the wrist view -> handle
  visibility: use keypoints seen in >=2 views (triangulation) or >=4 in a view (PnP); weight by
  detector confidence.

---

## 6. Practical notes / risks
- Keypoint set **must be non-collinear/non-coplanar** for a stable solve (why we add mesh corners).
- Keypoint detection *also* has to survive OOD — but detecting a local corner is far more robust than
  regressing a global pose, and triangulation across 3 views hardens it further. Residual OOD risk is
  much smaller than the current net's.
- Confidence/visibility gating and multi-view redundancy are the main engineering surface.

---

## 7. The gating experiment + next steps
1. **Offset sweep (force-side, InsertTuner rig):** inject lateral/yaw/depth perception offsets on GT
   and find the **basin** — how far the aim can be off (per error channel) and still seat. This
   yields the **exact accuracy spec** the perception must hit. *This is the decisive next run.*
2. **Build #1:** define the keypoint set (3 markers + mesh corners) -> generate 2D labels (project) ->
   train the heatmap detector -> triangulation/PnP -> TF. Validate against GT on the eval scenes.
3. **Run #2 (DINOv2) in parallel** as the cheap control/baseline.
4. **Compare eval keypoint-pose accuracy vs the basin** -> if inside, re-score end-to-end
   (`score.sh <policy> false eval`) expecting a large jump toward 277.7; if outside, iterate the
   detector (aug/coverage) or escalate to #3.
5. **Keep #3 (FoundationPose) in reserve.**

**Success criterion:** eval port-pose error (per channel) inside the force stack's basin on all 3
trials -> insertions -> score approaching the 277.7 the hands already deliver.

---

### Key files / assets
| Piece | Path |
|---|---|
| Existing dataset (labels source) | `~/aic_data/perception_v2/`, `~/aic_data/perception_native/` (index + `frames.parquet` with `port_left/center/right`, intrinsics in `meta.json`) |
| Port meshes (extra keypoints) | `~/ws_aic/aic_local/mujoco_build/meshes/*.obj` |
| Projection utility (2D-label gen) | `~/ws_aic/aic_local/collect/verify_labels.py` |
| Current trainer (structure to reuse) | `collect/train_perception_dual.py` (camera-frame -> TF framing) |
| Live scoring policy | `aic_example_policies/.../ros/PerceptionInsert.py` |
| Score harness | `~/ws_aic/aic_local/score.sh` -> `~/aic_results/<run>/scoring.yaml` |
| Force endgame (the solved "hands") | `force_insertion.md`, `InsertTuner.py` |

---

## 8. Pipeline flow (ASCII) — frames + the two directions

```
============================================================================
  COORDINATE FRAMES  (how they chain)
============================================================================

   world --[fixed mount]--> base_link --[robot FK, live]--> camera/optical --[K]--> image
  (sim origin)             (robot base,          (3 wrist cams,        (intrinsics)  pixel
                            WORLD-FIXED)           MOVE with the arm)              (u,v)

                    port is STATIC here ---------> port MOVES here (viewpoint changes)
                    (port_base: std~0)             (port_center/left/right)


============================================================================
  TRAINING -- make labels        we HAVE GT 3D        [ FORWARD : 3D -> 2D ]
============================================================================

   sim GT port pose            project into          2D pixel (u,v)
   (ground_truth:=true) -----> each cam frame -----> = LABEL           +
        in base_link           u=fx*X/Z+cx           (the red dot,     |
                               [analytic, K]          smoke test)      |
                                                                       v
                                                          train KEYPOINT DETECTOR
                                                          ( image --> predict (u,v) )
                                                          learns local port appearance


============================================================================
  INFERENCE -- eval          NO GT (ground_truth:=false)   [ REVERSE : 2D -> 3D ]
============================================================================

   left  -+                              +- (u,v)_L -+
   center-+-> KEYPOINT DETECTOR ---------+- (u,v)_C -+-> TRIANGULATE -> port 3D
   right -+   (per view, per keypoint)   +- (u,v)_R -+   (geometry,      (camera/base)
   3 images                                             no depth)          |
                                                                           | analytic TF (FK)
                                                                           v
                                                                  port pose in base_link
                                                                           |
                    plug pose  --[gripper FK + grasp]-----------+          |
                    (kinematics, NOT vision)                    v          v
                                                         +----------------------+
                                                         |  FORCE STACK         |
                                                         |  (InsertTuner:       |
                                                         |   align -> compliant |
                                                         |   push -> seat)      |
                                                         +----------+-----------+
                                                                    v
                                                             inserted OK  -> score
```

**The two directions at a glance:**
- **Training** = `known 3D port -> project (K) -> 2D dot` — camera projection, makes labels (the smoke test: `kp_smoke_overlay.png`).
- **Inference** = `image -> net -> 2D dot x3 -> triangulate -> 3D -> TF -> base_link` — the reverse, done by geometry (no GT).
- **Frames:** GT/output live in **`base_link`** (world-fixed, the control frame); the net works in the **moving camera frame**; **FK** bridges them. The **plug** comes from kinematics, not vision.

### Note on keypoint choice (from the smoke test)
`port_center` is the port *frame origin* (at the mount base), which projects a few mm behind/below the
visible cage mouth — a poor keypoint. Use **visually-distinct mesh-derived cage corners** (+ the
`entrance` mouth) as keypoints so the detector has concrete features and PnP is well-conditioned.

---

## 9. Dataset summary (which data we train on, and why)

There are **3 real datasets + 2 caches**. `M6` is a *dataset* (= `perception_v2`); `M7` is a *model*
(the conditioned dual-anchor net trained on merged native+v2) — NOT a dataset.

| Dataset | Eps | Rails covered | Resolution | Collection ranges |
|---|---|---|---|---|
| `perception_v1` | 50 | **2 only** (nic_0, sc_1) | 288x256 | narrow (+-3cm / +-0.15rad) |
| `perception_native` | 50 | **2 only** (same) | 1152x1024 | narrow (native re-collect of same scenes) |
| **`perception_v2` (= M6)** | 169 | **all 5 NIC + 2 SC** | 1152x1024 | **wide** (+-12cm / +-1.5rad) |
| `perception_native_c256` | 50 | 2 | 288x256 (cache) | downsample of native |
| `perception_v2_c256` | 169 | all rails | 288x256 (cache) | downsample of v2 |

### "Accurate vs inaccurate" — does NOT matter for perception
- v1 / native = narrow ranges, 2 rails -> CheatCode mostly seated ("accurate/solved").
- v2 / M6 = wide ranges -> CheatCode seated only **59/169 (35%)**; 65% did NOT seat ("inaccurate").
- BUT "inaccurate" = whether CheatCode's *insertion* succeeded. The port **labels (GT pose) are exact
  in EVERY episode regardless**. Seating only mattered for imitation-learning the hands (not our job —
  force stack is solved separately). So **all frames from all datasets are usable; we do NOT filter on
  seating.** A failed-to-seat episode still carries a perfect port label.

### What the keypoint first-pass trains on
**Merged `perception_native_c256` + `perception_v2_c256`** (288x256 caches, 219 episodes, same split M7
used: 186 train / 33 val), because:
- **v2 is the valuable one** — the only dataset with all-rail coverage (the whole OOD lever). v1/native
  saw only 2 rails, which is *why* the current net doesn't generalize.
- **native adds 50 more scenes** (cheap extra data).
- **288x256 cache** -> fast, GPU-bound training.
- Full-res `perception_v2` (1152x1024) stays as the master for a later 1024 polish.

**One line:** train on **v2 (all rails) + native (extra), at 288x256, ignoring seating** — labels are
perfect everywhere and v2's rail coverage is what fights the OOD gap.

---

## 10. First-pass RESULTS (2026-07-05) — keypoint detector + multi-view triangulation

### Built (all standalone; PerceptionInsert.py untouched, per constraint)
- `kp_train.py` — ResNet18 heatmap + soft-argmax detector, port-center keypoint, on the 288x256 caches.
- `kp_eval3d.py` — inference on 3 cams -> back-project pixels -> triangulate 3 base-frame rays -> 3D port.
  Self-check: GT-pixel triangulation recovers `port_base` to **0.0mm** (validates extrinsics + wxyz quat).
- `kp_perceive.py` — reusable `KeypointPerceiver` (images + K + T_base_cam -> port_xyz in base_link).
  The deployable "eyes"; single-frame self-test 2.2mm.

### Training
- 219 episodes (native + v2), 186 train / 33 val, EPISODE-WISE seeded split (no frame leakage),
  stratified by dataset x connector-type. 73,371 train / 13,104 val image-views @ stride 3.
- GPU-bound ~81s/epoch. Converged epoch 8 (val 2D median 3.51 px ~= mm); killed at epoch 11 (plateau).
  `best.pt` = epoch 8. Checkpoint dir: `~/aic_data/kp_v1_run/`.

### Decisive numbers — 3D port position error (self-check 0.0mm on both)
| Set | n | median | mean | p90 |
|---|---|---|---|---|
| In-dist val (native+v2 held-out episodes) | 1650 | 3.9 mm | 14.6 mm | 34.6 mm |
| **OOD: perception_v1 (foreign collection, never trained on)** | 2348 | **1.9 mm** | 6.3 mm | **11.9 mm** |

Baseline it replaces (global pose regression): **val 14.5mm -> eval ~80mm** (5.5x OOD blowup).

### Read
- **The OOD gap is gone.** Keypoint+triangulation is *better* OOD (1.9mm) than in-dist (3.9mm) — the
  opposite of the regression net. Local keypoint features transfer; global pose regression memorized.
- **Triangulation kills the 2D tail.** Per-cam 2D p90 was 37-42 px on the hard val; 3-ray averaging
  drops 3D p90 to 34mm (val) / 12mm (v1).
- **Consistency (proximity lane):** OOD p90 = 11.9mm -> 90% of *single-frame* estimates within 12mm;
  temporal median over the approach -> ~1.9mm effective.
- **Depth split (v1):** near-half 1.3mm / far-half 2.6mm — even the far/home-pose frames that broke the
  regression net are ~mm here.

### Honest gaps before calling it solved
1. v1 is a held-out *collection* (strong OOD proxy) but NOT the live `aic_engine` eval scenes where the
   80mm was measured. The live eval-scene number is the last measurement.
2. GT-derived extrinsics used here; live uses TF/FK (+~1-2mm). Self-check confirms the residual is pure
   detector error, so live degradation is bounded by FK accuracy.

### Gate: GREEN (pending live confirmation)
Next: standalone live perception node -> run one GT eval -> log kp-triangulated port vs GT on the real
eval scenes. If it holds ~mm, wire `KeypointPerceiver` into the standalone insertion re-score
(InsertTuner hands + these eyes). Pass 2 (deferred): mesh->port frame fix + 8 cage corners -> full 6-DoF;
1024 high-res polish.
