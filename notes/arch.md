# Training Pipeline Architecture — `collect/train_perception_dual.py`

ASCII walkthrough of the perception trainer (dual-anchor: port + plug pose from 3 wrist images).
Companion docs: `RUN_PLAN.md` (stage tracker) · `perception_results.md` (results) · `collect/M6_PLAN.md` (dataset).

```
                            DATA (on disk, ~/aic_data/<dataset>/)
 ┌──────────────────────────────────────────────────────────────────────┐
 │ index.parquet ──── episode list (drop split=="reject")               │
 │ episode_NNNN/                                                        │
 │   ├── left/ center/ right/   *.png   (wrist cameras, native)         │
 │   └── frames.parquet         GT poses per frame (from sim TF)        │
 └──────────────────────────────────────────────────────────────────────┘
                 │
                 ▼  episode-wise split (whole episodes, type-stratified,
                    seed=0 — or forced via --val_eps)
        ┌────────────────┐              ┌────────────────┐
        │ train episodes │              │  val episodes  │
        └────────┬───────┘              └────────┬───────┘
                 ▼                               ▼
     DualPoseDataset (one sample = one timestep, stride-subsampled)
 ┌──────────────────────────────────────────────────────────────────────┐
 │ labels:  port  = port_center_{pos,quat}          (in camera frame)   │
 │          plug  = DERIVED: inv(T_base←optical) ∘ plug_base            │
 │ images:  3 cams → resize (squash, or --keep_aspect crop) → S×S       │
 │          → ImageNet-normalize                                        │
 │ targets: positions standardized by TRAIN mean/std (pm,ps,gm,gs)      │
 └──────────────────────────────┬───────────────────────────────────────┘
                                ▼
                        DualPoseNet (12.1 M params)
 ┌──────────────────────────────────────────────────────────────────────┐
 │  left ──┐                                                            │
 │  center ├─► shared ResNet-18 (per cam) ─► 3×512 concat               │
 │  right ─┘                                   │                        │
 │                                             ▼                        │
 │                                  trunk MLP 1536→512→256              │
 │                            ┌─────────┬─────────┬─────────┬────────┐  │
 │                            ▼         ▼         ▼         ▼        │  │
 │                        port_pos  port_quat  plug_pos  plug_quat   │  │
 │                          (3)     (4,unit)     (3)     (4,unit)    │  │
 └──────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
        LOSS = smoothL1(port_pos) + smoothL1(plug_pos)
             + (1−|q·q̂|)_port  + (1−|q·q̂|)_plug        (all weight 1)
                                │
                                ▼
      AdamW (backbone lr 1e-4, heads lr 1e-3) + cosine LR + AMP
                                │
              ┌─────────────────┴──────────────────┐
              ▼ per epoch                          ▼ per epoch
        evaluate() on val                    checkpoints
        median mm + deg per anchor     ┌───────────────────────────────┐
        (whole-val only —              │ best.pt   if port_med+plug_med │
         NO close-range breakdown)     │           improves (weights +  │
                                       │           norm stats + val set)│
                                       │ last.pt   ALWAYS (full state:  │
                                       │           opt/sched/scaler/    │
                                       │           hist → --resume)     │
                                       └───────────────────────────────┘
                                │
                                ▼
                     history.json (end of run)
```

## Where it sits in the larger loop

```
 gen_configs.py ─► collect_parallel.sh ─► finalize ─► perception_v2/     (data)
                                                        │
                                                        ▼
                                            train_perception_dual.py    (this file)
                                                        │  best.pt
                                                        ▼
                                            PerceptionInsert.py (policy: net → base_link
                                                        │        → CheatCode math + spiral
                                                        │        + yaw dither + sanity gate)
                                            score.sh  ──►  scoring.yaml (judge)
```

## Known gaps — RESOLVED 2026-07-04 in the M7 trainer rewrite

1. ~~No target conditioning~~ ✅ `(target_module_name|port_name)` → 13-entry vocab → 16-d embedding
   concatenated into the trunk (`forward(x, tid)`). Identity read from `index.parquet` (v2) or derived
   from `meta.json`'s `port_frame` (v1-native); at eval from the Task msg (policy passes `self._tid`,
   `<unk>`=0 fallback). This was the 43 mm home-pose / 69–220 mm eval ambiguity fix.
2. ~~Position-only checkpoint metric~~ ✅ score = `port_mm + plug_mm + 1.0·(port_deg + plug_deg)`.
3. ~~Whole-val medians only~~ ✅ per-epoch **FIRST** (frame<5, the ambiguity gate) and **CLOSE**
   (nearest 20% by depth) columns in the training printout.
4. ~~Single data root~~ ✅ comma-separated `--data` roots, split stratified by (root, type);
   `--val_eps` format is now `rootIdx:ep,...`.
5. (unchanged guidance) Iterate at **224** (~1 min/epoch merged), reserve **1024** for final polish
   once the <30 mm diagnostic passes. Policy reads `img_size`/`keep_aspect`/`vocab` from the checkpoint,
   so the res switch is checkpoint-only — no policy edit.

## Review: higher-resolution capture (288×256 vs native 1152×1024)

Verdict: **collecting at native was unambiguously right; standardizing all *training* on 1024² was
premature** — it optimized the endgame while the opening (coverage, target ambiguity) was still broken.

### Pros (measured)
- Controlled ablation (same trainer, same 10 val episodes, only input res changed):
  close-range **port −30%** (9.0 → 6.3 mm), **relative plug→port −52%** (11.9 → **5.7 mm**).
  Gains concentrate exactly at insertion range — where the last millimetres decide seating.
- **Collection at native is strictly dominant**: same sim time, only ~7× disk (15 GB / 50 eps);
  down-res variants are derivable forever, up-res is impossible. Zero-regret data decision.
- Eval publishes native 1152×1024 frames → native training removes a train/test resolution mismatch.
- Plug anchor barely changes with res (big, centred in view) — the benefit is specifically the
  **small port** and the **relative** term, i.e. the insertion-critical quantities.

### Cons (paid)
- **~11× slower iteration**: ~18 s/epoch @224 → ~197 s/epoch @1024. Every retrain in the
  coverage/ambiguity debugging phase paid this tax while gaining nothing from it (eval errors were
  69–220 mm; resolution buys millimetres, the problems were decimetres).
- Memory pressure exists only at 1024: GPU 22.6 GB at batch 24 → forced batch 12 / workers 6;
  the CPU-RAM OOM-kill saga (silent SIGKILL of the dataloader-heavy run) was a 1024-only failure mode.
- **Score-justification still unproven** by our own standard (judge by insertion success, not mm):
  0 insertions at both resolutions so far — the deciding success-rate comparison never became runnable.

### Policy going forward
1. Keep **collecting** at native (unchanged).
2. **Iterate** coverage/ambiguity/conditioning fixes at **224** — 10× faster loop, sees decimetre
   errors fine.
3. Switch to **1024 only for the final polish** once the GT diagnostic passes <30 mm — that's when
  5.7 mm relative error converts to insertion probability inside the chamfer/search funnel.
4. Cost-optimal endpoint if needed: **foveation** — low-res full frame for approach + native-res crop
   around the port estimate at contact (most pixels-on-target per FLOP; designed, not yet built).

## Key file locations

| Piece | Path |
|---|---|
| Trainer (live) | `collect/train_perception_dual.py` (repo copy is the executed one) |
| Collection tooling (live) | `~/ws_aic/aic_local/collect/` (repo `collect/` snapshot has diverged) |
| Policy (source / executed) | `~/ws_aic/src/aic/.../ros/PerceptionInsert.py` → copy to pixi `site-packages/` |
| Checkpoints & datasets | `~/aic_data/` (`m6_dual_run/`, `perception_v2/`, `perception_native/`, …) |
| Scoring runs | `~/aic_results/<run>/` (symlinked as `aic_results/` in repo) |
