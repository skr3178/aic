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
