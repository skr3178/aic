# Diverse Sample-Collection Methodology (SFP + SC, perception + IL)

Source of truth for `gen_configs.py`. Goal: 50 CheatCode episodes whose **trajectories and
viewpoints genuinely span the space** (no two near-identical), each with free ground-truth pose
labels. Naive uniform sampling clusters and leaves gaps → a perception net that memorizes one pose;
this method prevents that.

## 0. Live-confirmed facts (from the sample run — use these exact names)

- **Base frame:** `base_link`. **TCP:** `gripper/tcp`.
- **Camera optical frames (perception label frames):** `left_camera/optical`, `center_camera/optical`,
  `right_camera/optical`.
- **Port frame:** `task_board/{target_module_name}/{port_name}_link`
  (e.g. `task_board/nic_card_mount_0/sfp_port_0_link`).
- **Port entrance frame:** `…/{port_name}_link_entrance` — sits ~**46 mm above** the port along
  base-z (scoring's partial-insertion band reference).
- **Plug frame:** `{cable_name}/{plug_name}_link` (e.g. `cable_0/sfp_tip_link`).
- **Each NIC card exposes 2 SFP ports:** `sfp_port_0_link` and `sfp_port_1_link` (~24 mm apart) →
  either can be the target, doubling SFP variety for free.
- Sensors: single composite topic `/observations` (3 images + camera_info + wrench + joint_states +
  controller_state); GT poses only on `/scoring/tf`→`/tf` (requires `ground_truth:=true`).

## 1. Core principle

CheatCode's trajectory is a deterministic function of exactly two things: the **goal**
(target port pose in `base_link`) and the **start** (how the plug sits in the gripper; robot home is
fixed). Therefore:

- To make **paths** differ → perturb goal and/or start (Group A).
- To make **pixels** differ without changing the path → perturb the scene (Group B).

Both matter: Group A gives pose/label variety, Group B gives image robustness.

## 2. Group A — variables that change the trajectory (MUST randomize)

| Variable | Config field | Range | Why |
|---|---|---|---|
| Board X / Y | `task_board.pose.x/y` | ±0.03 m | Shifts goal in base frame — biggest lever |
| Board yaw | `task_board.pose.yaw` | ±0.15 rad | Rotates approach direction |
| ⭐ Target rail translation | target module `entity_pose.translation` | nominal ± delta (nic ≈ ±0.015, sc ≈ ±0.03) | Slides the port along its rail — cleanest independent goal variation |
| Target port yaw | target module `entity_pose.yaw` | ±0.15 rad | Changes required orientation alignment |
| Target SFP port index | `tasks.task_1.port_name` | `sfp_port_0` / `sfp_port_1` | 2 ports/card (SFP only) |
| Cable grasp offset | `cables.<c>.pose.gripper_offset.x/y/z` | ±0.005–0.01 m | Moves plug tip vs TCP → different start + descent |
| ⭐ Cable grasp orientation | `cables.<c>.pose.roll/pitch/yaw` | ±0.10–0.20 rad | Changes the orientation the arm must correct — makes approach *rotation* vary |

The two ⭐ (rail translation, grasp orientation) are the workhorses: translation spreads the goal,
grasp orientation spreads the required alignment. Board X/Y alone gives 50 translated copies of the
same rotation — looks varied, teaches little.

## 3. Group B — variables that change pixels only (randomize for image robustness)

Same path, different images (so the net keys on the real port, not background):
- Which distractor modules are `entity_present` (incl. LC mounts — **clutter only, never a target**).
- Non-target module positions / yaws.
- Target connector type across the batch (SFP vs SC) — stratified ~50/50.
- `cable_type` (`sfp_sc_cable` vs `_reversed`) as dictated by the chosen target end.

## 4. Diversity mechanics (how to sample, not just what)

1. **Perturb-from-known-good template** (safety): copy a verified sample trial and jitter ONLY pose
   fields — never invent module/rail/port names. Templates: **SFP = `trial_1`**, **SC = `trial_3`**
   from `aic_engine/config/sample_config.yaml`. This sidesteps the rail→module→port wiring risk and
   the `task_board_limits`-vs-nominal ambiguity.
2. **Latin-hypercube (stratified) sampling** across the continuous Group-A dims (board x/y/yaw,
   target rail translation, grasp roll/pitch/yaw) so the 50 samples *span* the space — every band
   gets hit once — instead of clumping.
3. **Min-distance reject:** drop any candidate whose (port pose in base, grasp orientation) is within
   ε of an already-accepted one. Guarantees no two episodes are near-identical.
4. **Deterministic per-index seed:** episode `i` is seeded by `i` → the whole set is reproducible and
   regenerable. This is the correct notion of "seed" (there is no RNG seed in the stack).
5. **CheatCode-success acceptance filter:** since CheatCode runs anyway, only episodes it solves
   (Tier-3 > 0 in `scoring.yaml`) enter the training split — auto-rejects any out-of-range/unreachable
   generated scene. This is the real correctness guard, not `task_board_limits`.

## 5. Stratification target for the 50

- ~25 SFP (split across `sfp_port_0` / `sfp_port_1` and multiple rail positions) + ~25 SC.
- LHS applied within each connector-type stratum.
- Chunked into 5 configs × 10 trials (bounds blast radius; one Gazebo crash costs 10, not 50).

## 6. What is intentionally NOT varied (v1)

- Robot base / home joints (fixed → path variety comes purely from goal + grasp).
- Board z / roll / pitch (keep board flat).
- Target rail *slot* (kept on the template's slot; position on that slot is randomized). Expanding to
  other slots needs the rail→module→port mapping confirmed first — deferred.
