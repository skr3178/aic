# Stage 2 — Port Perception Plan (board decomposition + static-target aggregation)

**Goal:** deliver the port pose (position + orientation) from the wrist cameras accurately enough to
insert, GT-free. This is the *sole remaining gap* — plug and hands are solved.

---

## 1. Where we start (what's already solved)

| Piece | Status | How |
|---|---|---|
| Plug pose | ✅ solved | `FK(gripper) · T_grasp[task.plug_type]` — grasp offset is type-fixed (M14) |
| Hands / insertion | ✅ works given a good port | reactive descent; GT-port + FK-plug = **227.8** (M15) |
| **Port pose** | ❌ **unsolved** | neither M6 regression nor FKV keypoint solves it: SFP ~40–60 mm, **SC 300–470 mm** |

**Proven ceiling: ~227** (with SC descent tuning, higher). Port perception is the whole ballgame.

---

## 2. The decomposition that makes it tractable (M16)

The port is fixed on the board; the board is a known rigid object **verified flat** (roll=pitch=0, z=0,
only yaw varies). So:
```
port pose(base) = board_pose(base) · offset[target]
port ORIENTATION = board_yaw ⊗ offset[type]     # offset[type] measured fixed: SFP[179,0,0] SC[-180,0,-90]
port POSITION    = board · offset[target]         # coarse: ± rail slop (2.3cm SFP / 6cm SC) → needs refine
```
**Consequence:** port *orientation* — the piece whose error caused the −81 regression — reduces to
estimating **one scalar, board yaw.** Port *position* still needs vision, but now with a known target
and a board-predicted region of interest.

---

## 3. Design principles (why this will work where M6/FKV didn't)

1. **The target is STATIC in base_link** → we estimate a *constant* → aggregate over time to kill jitter.
2. **The moving camera is an asset, not a liability** — many viewpoints of the same fixed board = free
   multi-view constraints (a mini bundle-adjustment).
3. **Closer = lower error** (`metric_err ≈ pixel_err · depth/focal`; halve depth → halve error) →
   depth-weighted coarse-to-fine: rough far, tight near.
4. **Target the board, not the tiny port** — the board is large, distinctive, always in view, so it
   dodges mode B (off-frame) that broke the port-center detector.
5. **Use the free `Task` info** — `plug_type` and `target_name` (type-aware → fixes mode A wrong-connector).
6. **Reuse the validated pipeline** — same `KPNet` conv arch + triangulation from `kp_perceive.py`.

---

## 4. Architecture (reuses KPNet, retargeted to board corners)

```
3 wrist RGB (3×256×288, ImageNet-norm)
  └─ per-camera CNN (shared weights):
       ResNet18 backbone (ImageNet)  conv1 7×7/2 → maxpool → layer1..4  → 512×8×9  (/32)
       decoder ×3 [Upsample×2 → Conv3×3 → BN → ReLU]  512→256→128→64     (/32 → /4)
       head Conv1×1 → K heatmaps (K board corners)  → spatial soft-argmax → K (u,v)/cam
  └─ GEOMETRY (no learning):
       FK extrinsics (stamp-synced) → back-project pixels → base_link rays
       triangulate 3 cams → K 3D board corners
       Kabsch fit vs known board-frame corners → board (x, y, YAW)   (z,roll,pitch=0)
  └─ port_orient = board_yaw ⊗ offset[type]     ;   port_pos ≈ board · offset[target] (+refine)
```
Only the head channel count (1 port → K corners) and the labels change vs KPNet.
**Training labels are free:** project known board corners (GT board pose + board geometry) into each image.

---

## 5. The jitter-killer: static-target aggregation (moving/coupled camera)

Because the board is a constant in base_link, and the camera moves *with* the arm:

1. **Stamp-synced TF** — look up `base_link ← camera` at the **image timestamp**, not "now." This removes
   the extrinsic error injected by arm motion between capture and lookup. *(This is THE coupling fix.)*
2. **Estimate at rest, then lock + FK-track** — do the board estimate while the arm is stationary (home),
   where there's no blur and the board is fully framed; the board doesn't move, so it's valid for the
   whole trajectory. Mirrors the plug solution (measure once, FK-track).
3. **Multi-frame accumulation** — accumulate rays across many frames (each with its own FK pose) and solve
   one least-squares for the static 3D corners (bundle-adjustment style), not per-frame.
4. **Depth-weighted Kalman filter** — measurement noise `R(depth)` shrinks as depth drops → the estimate
   **tightens monotonically as the arm approaches** (coarse far → tight near). Weight `w ∝ 1/σ²(depth)`.
5. **Robustness** — confidence gate (reject diffuse-heatmap frames), RANSAC/weighted Kabsch (one bad
   corner can't tilt the fit), jump-reject vs running estimate.

**Coarse-to-fine, split by quantity:**
| Quantity | Far (home) | Near (descent) |
|---|---|---|
| board **yaw** (orientation) | already good (big object) → light refine | stable |
| port **position** (seat-critical) | coarse aim / centering | **depth-weighted refine → mm** |

---

## 6. Control integration (fixes modes A + B, composes with Stage 1)

- **Centering behavior** — use the (coarse) board/port estimate to re-orient the wrist so the target
  comes to frame center, re-perceive, iterate until stable, *then* descend. Breaks the off-frame feedback
  loop (mode B). Coarse-aim-far → refine-near is the same loop as §5's coarse-to-fine.
- **Type-aware** — `task.plug_type` constrains the search to the correct connector (mode A): on an SC
  trial seek the SC mount on the known board, never a green NIC card.
- **FK plug (Stage 1)** + reactive descent (SC needs descent tuning per M15).

---

## 7. Milestones & validation

| # | Milestone | Validate against | Target |
|---|---|---|---|
| 2a | Board→port geometry + flat verification | GT TF | ✅ DONE (M16) |
| 2b-orient | Board-yaw estimator (KPNet corners → triangulate → Kabsch → yaw) | GT board yaw | yaw err < ~2° |
| — | Wire port-orient = board_yaw ⊗ offset[type] into policy (+FK plug) | eval re-score | reverse the −81; SFP seats |
| 2b-pos | Type-aware + centered port position (board-ROI + depth-weighted refine) | GT port pos | < basin (~mm–cm) |
| 2c | Full stack + static-target aggregation + Kalman | eval GT-free score | → ~227 |

**Order:** board-yaw **first** — one scalar, un-blocks the plug fix / the −81, and the SFP trials are
already near their ports → fastest path to a real GT-free insertion.

---

## 8. Risks / open items

- **Board yaw ambiguity** — a rectangular board may have 180° symmetry; disambiguate via an asymmetric
  feature (the magenta rectangle, connector layout) or the known approach side.
- **Rail slop on position** — board-pose gives ±2–6 cm; the type-aware close-range refine must cover it
  (bigger than the reactive spiral).
- **Fair source for `T_grasp` / offsets** — measured from GT for validation; a real submission derives
  them from the public board/cable models. (Learning build → measured values OK to validate.)
- **SC descent tuning** — even with a perfect port, SC only partial-seats on this policy's descent
  (M15); InsertTuner seats SC 3/3, so fold in its endgame for SC.

---

## 9. One-line summary
Estimate the **board** (big, static, always-in-view) instead of the tiny port: reuse the KPNet corner
detector + triangulation, exploit the static target with **stamp-synced, multi-frame, depth-weighted**
aggregation (tightens on approach), derive **port orientation from a single scalar (board yaw)** and
**port position from board·offset + a type-aware close-range refine** — then compose with the solved FK
plug and reactive hands. Ceiling proven at ~227.

---

## 10. Findings 2026-07-06 — error budget quantified + board-extractor A1 FAILED

**Error budget (why the type split is decisive).** The reactive force **spiral radius = PITCH×steps =
1.2 mm × 30 = 3.6 cm** — the max aim error the hands forgive. Public `task_board_limits` give the hidden
**rail slide** (how far the module is randomly slid along its rail): **SFP nic_rail ±2.3 cm, SC sc_rail
±6.0 cm**. So:
- **SFP: rail slide (±2.3) < spiral (3.6)** → board-nominal position + spiral is **airtight; no slide
  perception needed.**
- **SC: rail slide (±6.0) > spiral (3.6)** → board-nominal alone misses at the extremes → needs a **1-D
  search along the known rail axis** (board pose gives the rail direction), not blind 2-D.
- Reframes port position as **estimating ONE scalar along a known line**, not 3-D socket hunting.
- Diagrams: `diag_frames/rail_slide.png`, `diag_frames/spiral_radius.png`.

**A1 (board-size-prior fit) FAILED on the real saved sweep clouds** (`~/aic_data/sweep_pts_trial{1,2,3}.npy`).
Min-area / size-prior rectangle fit vs M16 GT yaw:

| trial | fit-yaw err | recovered aspect | recovered size |
|---|---|---|---|
| 1 (SFP) | **31.6°** | →~1.0 (square) when trimmed | ~1.3×1.0 m |
| 2 (SFP) | **41.1°** | →~1.0 | ~1.3×1.0 m |
| 3 (SC)  | **23.7°** | →~1.0 | ~1.3×1.0 m |

- Not <2° — it's 24–42° and unstable across trim levels. Footprint is a **square, table-sized blob 4× the
  board (0.30×0.425)**: the dark-gray back-projection grabs the **whole dark workspace** (board + table +
  shadows), not the board rectangle. No recoverable 1.4:1 board at any trim.
- **The earlier "~2° board yaw" leaned on the `DISAMBIG_GT` GT crutch** — remove GT and the fit is 24–42°.
  Board pose is **NOT yet reliably recoverable GT-free.** The plan's linchpin is unproven.

**Status honestly:** plug ✅ (FK), hands ✅ (277.7), **board pose ❌ (broken extractor)**. Phases A2/A3/B/C
all sit on board pose → blocked until a working board primitive exists.

**Next test (offline, cheap, before any eval run):** replace dark back-projection with a **rail-line
orientation** detector (Hough / dominant-angle on the saved sweep *images*, not the cloud). Rails are
high-contrast parallel slots at a fixed board angle → yaw directly, far more robust than a blob-rectangle.
Pass bar: dominant rail-angle <5° on all 3 trials. Fallbacks: board-outline edge/contour fit; colored-
feature (magenta zones + green cards) convex hull for extent/center. Also drop the "domain-gap-immune"
claim → **robust**: the gray threshold (35–135) is still an appearance assumption; guard with Otsu /
largest-planar-component.
