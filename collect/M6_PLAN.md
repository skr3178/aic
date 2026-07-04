# M6 — Coverage-Fix Dataset (`perception_v2`)

## Why (evidence)
M5 diagnostic (`baseline_scores.md`) proved the eval far-misses are a **training-coverage gap, not a frame
bug**: perceived-vs-GT port error was 61 mm on the one eval scene *inside* the v1 training range, vs
335–372 mm on the two scenes *outside/at the edge* of it. Root causes in the v1 configs (`gen_configs.py`):

1. **Board pose jitter too narrow:** ±3 cm / ±0.15 rad around the sample nominal — eval scenes sat ≥6 cm outside.
2. **Rail translations under-cover the published limits** (`task_board_limits` in the public
   `sample_config.yaml`): we used ±0.015/±0.03 vs legal nic −0.0215..+0.0234, sc −0.06..+0.055,
   mount −0.09425..+0.09425.
3. **Target module rail was FIXED** — all 50 v1 episodes target `nic_card_mount_0` (SFP) or `sc_port_1` (SC).
   The board supports the NIC card on **any of 5 nic rails** and the SC module on **2 sc rails**; the public
   sample's own trial_2 targets `nic_card_mount_1`, and the eval collided with `nic_card_mount_2` — rails we
   never trained. Inter-rail distances ≈ the observed 0.3–0.4 m misses.

## Fair-play constraint
**The vaulted `eval_config.yaml` is never read.** All ranges come from the *public* task spec
(`aic_engine/config/sample_config.yaml`, docs/) or are blind physical envelopes. We match the *legal space*,
not the leaderboard's draws. If a re-score still shows OOD misses, widen again — never peek.

## Ranges (v1 → v2)

| Parameter | v1 | **v2** | Basis |
|---|---|---|---|
| Target NIC rail | `nic_rail_0` only | **all 5 rails** (`nic_rail_0..4`) | template enumerates them; sample targets mount_1 |
| Target SC rail | module `sc_port_1` only (one rail) | **both rails** (`sc_rail_0..1`), port `sc_port_base` per module; enumerate entity/ports from template at gen time (multiple SC ports/rail per #582 if present) | template |
| Rail translation | ±0.015 (nic) / ±0.03 (sc) | **full published `task_board_limits`** | public spec |
| Board x/y | ±0.03 m, single SFP-centric window | **±0.12 m centered on each type's PUBLIC nominal**: SFP (0.15, −0.2), SC (0.17, 0.0) — per-type templates as in v1. A single window at (0.15,−0.2) would exclude SC's own public y=0.0 by 8 cm. | public per-type nominals + blind envelope |
| Board yaw | ±0.15 rad | **±1.5 rad** around each type's public nominal (π / 3.0). Yaw is the least-constrained DOF and where OOD hurts most — front-load the wide blind envelope now rather than spend a re-collection round later. Acceptance filter prunes unreachable extremes. | blind physical envelope |
| Module yaw | ±0.15 rad | **±0.3 rad** | spec: "randomized orientation offsets" |
| Grasp offset/rpy | ±0.15 rad etc. | unchanged | already realistic |
| Non-target clutter | toggle/jitter (Group B) | unchanged mechanism, jitter to full rail limits | public spec |

## Episode counts (160 total)

| Type | Strata | Episodes |
|---|---|---|
| **SFP** | 5 nic rails × 2 ports (`sfp_port_0/1`) = 10 strata | **8 each → 80** |
| **SC** | 2 sc rails × ports (template: one `sc_port_base` per `sc_port_N` module → **2 strata × 40**; if gen-time enumeration finds multiple SC ports/rail (#582), split evenly) | **80** |

Within each stratum: Latin-hypercube over the continuous dims + farthest-point de-dup (same machinery as v1,
`SAMPLE_DIVERSITY.md`), deterministic master seed. Acceptance filter unchanged (drop CheatCode failures —
expect a higher reject rate at extreme poses; over-generate configs ~10% to compensate).

## Pipeline (all existing tooling)
1. `gen_configs.py` (edited ranges + rail/port stratification) → `configs/chunk_*.yaml` + `manifest.json`
   (~16 chunks × 10 trials), engine-parse validated.
2. **`collect_parallel.sh` (NOT serial `run_collection.sh`)** — each chunk in its own isolated Docker
   network (own zenoh router; engine + CheatCode + collector all in-container, no host process), up to
   `MAX_PAR` chunks concurrent on the one GPU. Validated this session at native res: 3 workers = 33% VRAM /
   35% GPU; the box handles **MAX_PAR=5**. 160 eps serial ≈ ~2.5 h → **parallel ≈ ~1 h**.
   Run at `SCALE=1.0` (native 1152×1024; 160 eps ≈ 48 GB → `/home/skr/aic_data/perception_v2/`, 177 GB free).
   **Detachment (hard evidence this session):** launch the orchestrator with `setsid nohup … </dev/null &`
   — full detachment survives. The harness-managed background (`run_in_background`) **reaped the
   orchestrator (exit 144)**; bare `nohup` without `setsid` also gets reaped.
3. `finalize.py` → acceptance filter → `perception_v2/` (+ `index.parquet`).
4. Retrain dual-anchor at 1024² (`train_perception_dual.py`, batch 12 / workers 6). Optional in same pass:
   **condition the net on target port identity** (embed `port_name` + `target_module_name` from the Task
   msg — legitimately available at eval) to resolve multi-port/multi-rail ambiguity by design.
5. Re-score `PerceptionInsert` (`ground_truth:=false`) → update `baseline_scores.md`.

## Success criteria
- Perceived-vs-GT port error on eval scenes (one more diagnostic run) < **30 mm** on all 3 trials.
- No −24 off-limit contacts (spiral layer absorbs residual error once targeting is sane).
- Tier-3 > 0 on ≥2 trials; total above the WaveArm floor (37.5) minimum, aiming for partial/full insertions.

## Risks / notes
- Wider space with 160 eps = lower sample density per region. The true generalization limiter is **board-pose
  draws per stratum (8), not frame count** (~360 frames/ep is plenty). If eval error stays high *uniformly*
  (not just OOD), **add board-pose draws per stratum before adding strata** — scale to 300+ (data is free,
  collection time ≈ 1 h per 160 at MAX_PAR=5).
- CheatCode may fail more often at extreme board poses (reach limits) → acceptance filter handles it; watch
  the reject rate for a skewed final distribution.
- v1 datasets (`perception_v1`, `perception_native`) remain untouched for regression comparisons.
- Policy-side v3 guard (independent of dataset): sanity-gate the net output — reject/hold if the perceived
  port is outside the plausible board envelope or jumps between steps. Cuts −24s even under OOD perception.
