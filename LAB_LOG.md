# AIC — Lab Log (experiment tracking index)

**Single entry point for the whole project.** Terse, chronological, links to the detail docs — which
live in [`notes/`](notes/) (project docs) and [`aic_data/`](aic_data/) (external, ~90 GB — see its
[MANIFEST](/home/skr/aic_data/MANIFEST.md)). *Details are NOT duplicated here — follow the → link.*

Task in one line: insert a cable plug into a randomized task-board port (UR5e, /300 over 3 vaulted
trials). Plug pose + insertion "hands" are solved; **GT-free port localization on the eval domain is
the sole remaining bottleneck.**

---

## Status snapshot (2026-07-08)
- **Ceiling** 279.4 (CheatCode, GT) · **proven perception ceiling** 227.8 (GT port + FK plug + our hands, `kp_gtport`).
- **Best real GT-free** score: **+1.4** (M7b). Every fully GT-free perception run is still ≤ WaveArm floor (37.5), mostly negative.
- **Open bottleneck:** localize the target port GT-free on the *vaulted* eval domain (generated→vaulted domain gap; SC connector hardest).
- **Solved & banked:** insertion/force stack (M9, 277.7 vs 279.6, 3/3) · plug pose = FK·T_grasp (M14) · board yaw magnitude ≤1.5° (SAM known-size) · gated-lock jitter fix (A5a, 3 mm).
- **Active thread:** YOLO sweep-lock closed-loop (`yolo_sweep_score` **−51.48**, 0/3) — port lock now forms & holds, but the **triangulated depth is bad** (narrow sweep baseline); next = back-project to the flat-board z-plane instead of triangulating.

---

## Experiment ledger
`id · what · outcome · → detail doc`. Milestone chronology from [RUN_PLAN](notes/RUN_PLAN.md).

### Baselines & data pipeline (M0–M2)
- **M0** env (pixi + ROS Kilted, torch on Blackwell) — ✅ → [RUN_PLAN](notes/RUN_PLAN.md)
- **M1** self-scoring baselines — ✅ WaveArm **37.5** floor · CheatCode **279.4** ceiling · pretrained ACT −21 → [baseline_scores](notes/baseline_scores.md), [CHEATCODE_RUNBOOK](notes/CHEATCODE_RUNBOOK.md)
- **M2** config-gen + collection (`perception_v1`, 49 eps) — ✅ images↔GT labels → [collect/PLAN](collect/PLAN.md)

### Perception nets — first attempt (M3–M8)
- **M3 / M3.5** port-only, then dual-anchor (port+plug from vision) — ✅ ~9 mm close; rel ~12 mm → [perception_results](notes/perception_results.md), [arch](notes/arch.md)
- *(res ablation)* native 1024² beats 288×256 — ✅ rel **5.7 mm (−52%)**
- **M4** `PerceptionInsert` wired, first GT-free score — ✅ **−37.6** → [baseline_scores](notes/baseline_scores.md)
- **M5 / M6a** far misses = **coverage gap** → `perception_v2` (169 eps, 35% seat) — ✅ → [collect/M6_PLAN](collect/M6_PLAN.md), [perception_v2_episodes](notes/perception_v2_episodes.md)
- **M6c–e** v2 retrain @1024 (⏸ ep6, 18 mm) · conditioned retrain (219 eps) · v3 policy (yaw dither + sanity gate) ✅
- **M7a** GT diag — ✅ ran / ❌ criterion: port **69–220 mm** on eval → perception is the bottleneck
- **M7b** real score (gt:=false) — ✅ **+1.4** (best real GT-free ever); still 0 insertions
- **M8** iterate — ⏳ still perception-limited

### Force-insertion "hands" (M9)
- **M9** InsertTuner force rig — ✅ **eval parity 277.7 vs CheatCode 279.6, both 3/3** → [force_insertion](notes/force_insertion.md)

### Domain-gap diagnosis (M10–M14)
- **M10** keypoint + triangulation rebuild — ⚠️ proxy **1.9 mm** on v1 but **did NOT transfer** → [perception_results](notes/perception_results.md), [keypoint_pnp_perception_plan](notes/keypoint_pnp_perception_plan.md)
- **M11** eyes+hands live (`PerceptionInsertKP`) — ❌ **−43.3**; SFP 8–40 mm, **SC 300–430 mm catastrophic**
- **M12** ablation — ✅ cause = **scene-content/geometry OOD** (ruled out scale/photometric/architecture)
- **M13** 3rd-person diagnosis — ✅ 3 failure modes: (A) wrong connector, (B) SC off-frame 78–84%, (C) plug twist
- **M14** plug grasp is **type-fixed** — ✅ plug pose = FK(gripper)·T_grasp[type], zero vision → solves mode C

### Board geometry & Stage-2 (M15–M18)
- **M15** ceiling validation (GT port + FK plug) — ✅ **227.8**; port perception = sole gap → [problem_formulation](notes/problem_formulation.md)
- **M16** board→port geometry — ✅ port orientation = **1 scalar (board yaw)**; board verified flat
- **M17** Stage-2 port perception plan — 📋 → [stage2_port_perception_plan](notes/stage2_port_perception_plan.md)
- **M18** error budget + board-extractor **"A1"** — ⚠️ **A1 FAILED** (dark-pixel board fit, yaw err 24–42°). Rail slide: SFP ±2.3 cm ⊂ spiral 3.6 cm (airtight); **SC ±6.0 cm ⊄ spiral → needs 1-D rail search** → [stage2 §10](notes/stage2_port_perception_plan.md)
  - ⚠️ **Two different "A1"s:** Stage-2 §10 "A1" = board-size-prior fit (FAILED, above) ≠ action-plan "A1" = eval-frame harness (below).

### Action-plan fix + YOLO detector (T0 / A-series, 2026-07-07/08)
- **T0** FK-reproject label diagnostic — ✅ **PASS**: mount recovered 0.8 mm; the **85 px moving-frame label corruption** explained & fixable → [action_plan_perception_fix](notes/action_plan_perception_fix.md)
- **A2** clean labels (FK-reprojection + SAM-mask QC) → `yolo_ds` (11.5k, 46 scenes) — ✅ build
- **YOLO-World zero-shot** (pre-finetune probe) — ❌ open-vocab can't recognize the niche synthetic port: top-1 box **20–62 % of image**, conf ≤0.39 (SC "100 % contains GT" was a 62 %-of-frame blob). Class-agnostic **SAM** segments it (0.4 % tight mask) but a *semantic* detector doesn't → motivated **fine-tuning** over zero-shot.
- **A3a** fine-tune YOLOv8m on clean labels — ✅ mAP50-95 **0.847** (our-data)
- **A3b** THE GATE (detector on real eval frames) — ✅ **2/3, SC 3.9 mm** (256 res 0/3 → **resolution decisive**); lone fail t1 = class confusion. First learned detector through the gate; "domain gap" was confounded by 85 px labels + low res → [yolo_finetune_findings](notes/yolo_finetune_findings.md)
- **A5a** gated-lock **EXTRAPOLATE** (not interpolate; causal + exact since port is static) + FK-track — ✅ **3.0 mm median (p90 5.7), 100 % within 3.6 cm basin (49/49)**. Naive per-frame jitter **152 mm** (p90 331) → speed + jump-reject gate **8.8 mm** (~17×). Collapses deployment to *getting one good lock*. *Insight: moving-camera desync is the shared root of the 85 px label corruption (T0) AND the earlier jagged KP trajectory — synced-FK labels + gated-lock fix both.*
- **Test #2** at-rest GT-free lock probe (SAM-auto + heuristic) — ⚠️ **oracle 43 % / central-heuristic 27 %** (SFP 17 %, SC 36 %), centroid err 108 px. SAM *segments* but **disambiguation is the wall** — no simple GT-free rule picks the port among ~16 module-sized masks ⇒ confirms a **trained detector (A3) is required**; SAM-auto + rule is not a no-training shortcut.

### SAM board-geometry track (parallel, `aic_data/track2_sam/`)
- SAM segments the board cleanly on eval frames (vs CV gray-threshold total fail) — ✅ GO → [track2_sam/FINDINGS](/home/skr/aic_data/track2_sam/FINDINGS.md)
- Sweep test — board pose → `board·offset` → port: **2/3 within 36 mm** (t1 3.2, t2 SC 16.7; t0 fail 102–155). Known-size CAD fit **solved yaw ≤1.5°**. t0 fail diagnosed = near-side density spill (not coverage).
- Complementary with YOLO: **SAM fails t0, YOLO fails t1** → ensemble path to 3/3.

### YOLO closed-loop scored eval (2026-07-08)
- `yolo_score_diag` — drop-in per-step YOLO → **lock never formed** (YOLO blind at close-up insertion viewpoint); no score.
- `yolo_sweep_score` — **sweep-lock** design (`PerceptionInsertYOLOSweep`): lock the port during the sweep, hold through insertion → lock forms & holds (insufficient-TF 371→0), but **total −51.48, 0/3** — triangulated **depth** wrong (narrow baseline). **Next fix:** back-project YOLO xy to the flat-board z-plane instead of triangulating.

---

## Score ledger (`~/aic_results`, self-scored /300)
GT ceiling ~279 → GT-port perception 227.8 → every fully-GT-free perception run ≤ floor. Dumper runs
(`sweepdump`, `yolo_native_dump`, `kp_sweep3`) show −33 (they don't insert). Regenerate:
`for d in ~/aic_results/*/; do echo "$d $(grep -h '^total:' $d/scoring.yaml)"; done`

| total | date | run | note |
|---|---|---|---|
| 279.6 | 07-04 | cheat_recheck | CheatCode GT ceiling |
| 277.7 | 07-04 | tuner_eval | InsertTuner force stack (GT) |
| 249.7 | 07-03 | smoke_v2 | |
| **227.8** | 07-05 | **kp_gtport** | **proven perception ceiling (GT port + FK plug + hands)** |
| 209.3 | 07-06 | scenecam_gtport | |
| 120.7 | 07-06 | kp_sweeporient | |
| 47.1 | 07-06 | kp_boardorient | |
| 37.5 | 07-02 | wavearm | floor |
| 33.7 | 07-04 | pi_diag_v3 | |
| 1.4 | 07-04 | pi_v3_score | **best real GT-free (M7b)** |
| 0.07 | 07-04 | pi_diag_m7 | |
| −18.8 | 07-04 | m7_eval | |
| −43.3 | 07-05 | kp_ab_false | PerceptionInsertKP live (M11) |
| **−51.5** | 07-08 | **yolo_sweep_score** | YOLO sweep-lock closed-loop (depth bug) |
| −105 | 07-05 | kp_boardgeo | |

*(full 37-run table lives in `~/aic_results/`; above are the milestones — best-per-thread + floor/ceiling.)*

---

## Doc map (`notes/`)
| Doc | Purpose |
|---|---|
| [RUN_PLAN.md](notes/RUN_PLAN.md) | Living M0–M18 stage tracker + CoStream map + standing rules |
| [problem_formulation.md](notes/problem_formulation.md) | Formal task statement, solved-vs-open, best-scores table |
| [ranked.md](notes/ranked.md) | 8 candidate approaches ranked; strategy "#7 perception ⊕ #3 force" |
| [baseline_scores.md](notes/baseline_scores.md) | Per-policy local self-scoring table |
| [perception_results.md](notes/perception_results.md) | Detailed M3–M16 perception experiments/ablations (largest) |
| [perception_v2_episodes.md](notes/perception_v2_episodes.md) | Per-episode plug→port gap for the 169 v2 demos |
| [arch.md](notes/arch.md) | Dual-anchor perception trainer pipeline walkthrough |
| [keypoint_pnp_perception_plan.md](notes/keypoint_pnp_perception_plan.md) | Keypoint + PnP/triangulation "eyes" plan |
| [force_insertion.md](notes/force_insertion.md) | InsertTuner force-insertion endgame (M9) |
| [stage2_port_perception_plan.md](notes/stage2_port_perception_plan.md) | Stage-2 board-decomposition plan; §10 error budget + A1 fail |
| [action_plan_perception_fix.md](notes/action_plan_perception_fix.md) | T0 + A1–A6 ordered fix plan (labels/detector/jitter) |
| [yolo_finetune_findings.md](notes/yolo_finetune_findings.md) | The A3b YOLO gate result (2/3, SC 3.9 mm) |
| [RL.md](notes/RL.md) | Where RL fits; Blackwell Isaac blocker; MuJoCo/MJX fallback |
| [WORKSPACE_MAP.md](notes/WORKSPACE_MAP.md) | Where functional pieces live (repo vs ~/ws_aic) |
| [CHEATCODE_RUNBOOK.md](notes/CHEATCODE_RUNBOOK.md) | Run guide for the CheatCode GT baseline |
| [collect/PLAN.md](collect/PLAN.md), [collect/M6_PLAN.md](collect/M6_PLAN.md), [collect/COLLECTION_FIXES.md](collect/COLLECTION_FIXES.md), [collect/SAMPLE_DIVERSITY.md](collect/SAMPLE_DIVERSITY.md) | Data-collection pipeline docs (kept with the pipeline) |
| `/home/skr/aic_data/track2_sam/FINDINGS.md` | SAM board-geometry track findings |

Upstream toolkit docs (`docs/*`, package `README.md`s) are challenge-provided background — not project history.

---

## Where things live (3 roots)
| Root | In git? | Holds |
|---|---|---|
| `/media/skr/storage/aic` (this repo) | ✅ | Source: ROS policies, `aic_engine/config`, this LAB_LOG, `notes/` (docs), `media/` (videos+frames; overlays in `vision_overlays/`, diagrams in `diag_frames/`), `rl_mujoco/`. Reference copy; built env is `~/ws_aic`. |
| `~/aic_data` | ❌ (~90 GB) | Datasets (`perception_*`), checkpoints/runs (`m3/m6/m7/m35/kp/boardyaw*`, `yolo_runs`), dumps, foundation weights (`sam_vit_b.pth`), `fm_venv`. → [MANIFEST](/home/skr/aic_data/MANIFEST.md) |
| `~/aic_results` | ❌ | 47 scored-eval run dirs (`scoring.yaml` + `policy.log` + rosbags) — the score record above. |
| `~/ws_aic/src/aic` | (workspace) | The **built/run** copy of the policies + `aic_local/` harness (`score.sh`, `collect/`, `mujoco_build/`). Policies: CheatCode · PerceptionInsert · PerceptionInsertKP · PerceptionInsertYOLO · PerceptionInsertYOLOSweep · SweepDump(Full) · InsertTuner. |

Run a policy: `~/ws_aic/aic_local/score.sh <module.Class> [ground_truth] [config] [run_name]`.
