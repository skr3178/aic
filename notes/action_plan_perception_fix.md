# Action Plan — Fix perception labels, detector, and motion jitter (2026-07-07)

Concrete, ordered, **testable** actions to resolve the issues surfaced in the SAM/YOLO investigation.
Every learned/perception step is **gated offline on real eval-domain frames** before an eval run
(the M10 scar). Fair-play + standalone constraints hold: don't modify `PerceptionInsert.py`/`CheatCode.py`
(work in the standalone `PerceptionInsertKP.py`); never read the vaulted `eval_config.yaml`.

## Root causes (recap, one line each)
- **R1** learned-on-our-data → domain gap (KPNet/DualPoseNet/BoardNet fail on eval).
- **R2** SAM segments the port cleanly but gives ~29 masks — **no disambiguation**.
- **R3** YOLO-World zero-shot can't **semantically recognize** the port (huge low-conf boxes).
- **R4** projected-GT labels are **wrong on moving-camera frames** (timestamp desync, ~85px @ tcp_vel 0.335).
- **R5** per-frame perception + moving camera → **jagged deployment motion** (same root as R4).

Guardrail: **the eval-frame harness is the only trusted judge.** Pass bars = **3.6 cm position / 2° yaw**
on real eval frames.

---

## Phase 0 — One decisive diagnostic (do FIRST, ~30 min, cheap)
Determines which label-fix and which deployment path are valid.

| # | Action | Test | Pass → | Fail → |
|---|---|---|---|---|
| **T0** ✅ **PASS** | Re-project the **static** `port_base` through **FK(`tcp_pose`)** at the moving frame (ep0005 f0) | crop: does the cross now land on the connector? | ✅ **YES** — green FK-cross snapped onto the connector; recorded red-cross 85px off | — |

**RESULT (2026-07-07):** camera↔TCP mount recovered to **0.8 mm** (quaternion is **xyzw**). `tcp_pose` is
cleanly image-synced → **FK-reprojection recovers ALL frames exactly (labels) AND FK-track is valid
(deployment)**. Consequences: (a) **no velocity filter / no interpolation needed** — relabel every frame
by FK-reprojection; (b) the recorded `port_center_pos` (used by KPNet + all prior training) was desynced
up to ~85px on moving frames → **FK-reprojection is a free strict upgrade to every label**.

---

## Phase 1 — Shared eval-frame harness (unblocks all gating)
| # | Action | Pass bar |
|---|---|---|
| **A1** | One GT eval run dumping **clean full-res eval frames (3 cam) + GT port pixel + FK extrinsics** per frame → the common offline test set | ≥ N frames/trial saved with GT + extrinsics; SFP & SC covered |

Without A1 we cannot honestly score any detector on the eval domain (the R1 trap).

---

## Phase 2 — Clean label dataset (fixes R4)
| # | Action | Test | Pass bar |
|---|---|---|---|
| **A2a** | Label generator: **FK-reproject** static `port_base` (or `entrance`) through each frame's `tcp_pose` → box per cam. **T0 passed → all frames valid, no velocity filter needed.** | count boxes + spot-check on moving frames | all ~21k frames × 3 cams; labels correct on moving frames too |
| **A2b** | **SAM-mask labels**: clean point → SAM point-prompt → tight mask → box; **auto-QC** rejects a label if its point doesn't fall on a coherent SAM segment (catches residual bad labels) | visual check 30 random labels + QC reject-rate | ≥ 95% labels visibly on the connector |

Output: a clean, QC'd detection dataset with **zero manual annotation**.

---

## Phase 3 — Fine-tune the detector (fixes R3, tests R1)
| # | Action | Test | Pass bar |
|---|---|---|---|
| **A3a** | Fine-tune YOLO (or YOLO-World head) on the A2 dataset **+ domain randomization** (color/lighting/blur/crop); hold out episodes for val | our-data val mAP | sanity: high on held-out our-data |
| **A3b** | **THE REAL GATE** — run the fine-tuned detector on the **A1 harness eval frames** | top-1 box center vs GT port | **box on GT, center err < 3.6 cm-equiv, on unseen eval scenes** |

A3b is the domain-gap verdict. If it passes, we've beaten R1 with a detector for the first time.

---

## Phase 4 — Perception → 3D port (fixes R2)
| # | Action | Test | Pass bar |
|---|---|---|---|
| **A4a** | Pipeline: detector **box → SAM box-prompt** (reliable) → mask centroid → **triangulate 3 cams** → 3D port | vs GT on harness | 3D port err **< 3.6 cm (SFP)** |
| **A4b** | **SC rail-axis 1-D search** (board gives rail direction; ±6 cm slop > spiral) | vs GT SC | SC within search band |
| **A4c** | Orientation = board_yaw ⊗ offset[type] (already derived, M16) | vs GT yaw | **< 2°** |

---

## Phase 5 — Deployment jitter fix (fixes R5)  — **Test #1 ✅ PASSED (2026-07-07)**
**Strategy = gated-lock EXTRAPOLATE (not interpolate):** perceive the port once at a clean still frame,
express in base_link, **hold it (extrapolate) + FK-track**; reject any update that is high-speed OR jumps
too far from the running lock (speed + jump-reject gate). Because the port is a constant in base_link,
extrapolation is *exact*, and it's causal (online), unlike interpolation.

**Test #1 result (49 episodes, offline):** one still-frame lock + hold → target **3.0 mm median (p90 5.7)
from GT for the whole trajectory, 100% within the 3.6 cm basin**. Naive per-frame = 152 mm median max
jump (331 p90) — the jitter; the speed+jump-reject gate cuts it to **8.8 mm**. ⇒ deployment risk
collapses to *getting one good lock at one clean at-rest moment* (= Test #2 / perception).

| # | Action | Test | Pass bar |
|---|---|---|---|
| **A5a** ✅ | **Gated-lock extrapolate + FK-track** (speed + jump-reject gate) | Test #1 offline | **DONE: 3 mm, 100% basin** |
| **A5b** | **Stamp-synced TF** everywhere (camera pose at image timestamp) | re-proj consistency under motion | on-target during motion |
| **A5c** | (optional) depth-weighted Kalman if perceiving continuously | estimate variance vs depth | tightens on approach |

Test vs the earlier **jagged KP trajectory** — smoothness should visibly improve.

---

## Phase 6 — Integrate + self-score
| # | Action | Pass bar |
|---|---|---|
| **A6** | Wire A4 port + A5 tracking into standalone `PerceptionInsertKP.py`, compose with **FK-plug + reactive descent**; run **GT-free eval** | real GT-free score **≫ −9**, toward the **227** ceiling |

---

## Ordering / parallelism
```
T0 (diagnostic) ──┬─► A2 (labels) ─► A3 (fine-tune) ─► A3b GATE ─► A4 (3D) ─┐
                  │                                                          ├─► A6 (score)
A1 (harness) ─────┴─────────────────────────► [gates A3b, A4] ─► A5 (jitter)┘
```
- **A1 (harness) and T0 (diagnostic) first, in parallel** — both cheap, both unblock everything.
- A2→A3 is the main build; **A3b is the go/no-go gate** (domain gap).
- A5 (jitter) can be built in parallel with A2/A3 (it's control-side, independent of the detector).
- SAM already works (segmentation) — it appears in A2b (labels) and A4a (box→mask), no separate build.

## Fallbacks (if a gate fails)
- **A3b fails** (detector still domain-gaps) → (i) more domain randomization, (ii) train on SAM masks not boxes (coarser, more robust), (iii) fall back to **SAM-auto + board-plane geometry** disambiguation (no learned detector).
- **T0 fails** (joint_pos desynced) → velocity-filter labels + perceive-at-rest-only deployment (both still work, just less data / no in-motion perception).
- **SC still misses** → rail-axis search (A4b) + fold in InsertTuner's SC descent endgame (M15).

---

## Coverage of considerations raised in discussion
- ✅ YOLO / learned detector (Ph3) · fine-tune on our ports (Ph2–3) · high-speed filter (A2a) ·
  jitter from bad cross (Ph5) · interpolate low-speed points (A2a fallback).
- Foundation-model track (SAM) is embedded in A2b (labels) + A4a (box→mask), not a separate phase.
