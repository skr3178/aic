# AIC Repo — Branch Tracker

Upstream: https://github.com/intrinsic-dev/aic/tree/main
Fork (mine): git@github.com:skr3178/aic.git

_Snapshot: 2026-07-09. Regenerate with the command at the bottom._

## Remotes
- `origin` → https://github.com/intrinsic-dev/aic.git
- `mine`   → git@github.com:skr3178/aic.git

## Core branches
| Branch | Last commit | Latest subject |
|---|---|---|
| main | 2026-06-24 | Add Task Board CAD and Assembly Guide (#585) |
| phase_1 | 2026-07-01 | Make cable moderator shared variables atomic, cleanup (#591) |
| phase_1_aic_gazebo | 2026-07-01 | Make cable moderator shared variables atomic, cleanup (#591) |
| phase_1_apr | 2026-04-23 | Merge branch 'origin/jt/agentbridge_ros' into phase_1_apr |
| media | 2026-03-02 | Update aic_banner.png |
| task | 2026-04-19 | Formatting |

## iche033 (cable / gazebo physics)
| Branch | Last commit | Latest subject |
|---|---|---|
| iche033/cable_damping | 2026-02-24 | updated script |
| iche033/cable_gripper_contact_opt | 2026-05-21 | revert changes |
| iche033/cable_phase_1_capsules | 2026-04-28 | update col length |
| iche033/cable_static | 2026-04-29 | add phase 1 world |
| iche033/cable_static_state | 2026-05-02 | Use new apis for toggling cable static state |
| iche033/flip_lc_sfp | 2026-05-11 | flip orientation |
| iche033/gz_sim_debug_statements | 2026-04-15 | Point to gz-sim commit that removes debug statement |
| iche033/jerk_test | 2026-02-20 | plot jerk |
| iche033/phase_1_aic_gazebo/gripper_contact | 2026-05-21 | lint |
| iche033/phase_1_aic_gazebo_sync | 2026-07-01 | style |
| iche033/pixi_fix | 2026-04-07 | updates |
| iche033/sc_end_collisions | 2026-06-29 | update collisions on sc end |
| iche033/weld_joint_component | 2026-06-10 | add cable activation pub |

## kaushik (taskboard / cheatcode / skills)
| Branch | Last commit | Latest subject |
|---|---|---|
| kaushik/add-lifecycle-node-transition-skill | 2026-05-30 | Add options block to lifecycle skill manifest |
| kaushik/add-transitions-to-insert-cable-skill | 2026-05-29 | Add generic lifecycle_transition skill |
| kaushik/manage-lifecycle-skill | 2026-05-05 | LIfecycle manager for aic_model |
| kaushik/mujoco-3-7-fix | 2026-04-19 | Use np array for joint damping values |
| kaushik/mujoco-gripper-debug | 2026-02-26 | isolate urdf from gz specific controller |
| kaushik/phase-1-cheatcode | 2026-04-30 | Working CheatCode with changes to frame naming convention |
| kaushik/phase-1-cheatcode-clean | 2026-05-07 | Add lifecycle_msgs dependency to insert_cable_skill |
| kaushik/support-five-cables | 2026-06-30 | Separate cable spawning logic from taskboard spawning |
| kaushik/taskboard-sc-port-spawning-update | 2026-04-19 | Formatting |
| kaushik/taskboard-sc-port-spawning-update-phase1 | 2026-05-03 | Add xacro expansion service (#428) |
| kaushik/taskboard-spawner | 2026-04-10 | Add warning for solution not saved |
| kaushik/taskboard-state-service | 2026-04-29 | Test & build |
| kaushik/taskboard-training-utils | 2026-05-28 | Add xacro expansion service (#428) |
| kaushik/trail-reset-fix | 2026-04-10 | Fix cleanup seqeunce in CablePlugin |
| kaushik/update-insert-cable-skill | 2026-06-04 | Fix lifecycle_transition_skill manifest options block |

## jt (TCP / FTS / skills)
| Branch | Last commit | Latest subject |
|---|---|---|
| jt/compute-tcp-pose-error | 2026-06-08 | add TCP pose error calculation |
| jt/fts-low-pass-filter | 2026-02-25 | undo unintended change |
| jt/update-build-insert-cable-skill | 2026-05-04 | update instructions |

## yadu (robot motion / tf)
| Branch | Last commit | Latest subject |
|---|---|---|
| yadu/acl | 2026-02-20 | Add zenoh config files |
| yadu/debug_tf | 2026-06-02 | Update gripper link names in sdf to match xacro used for quals |
| yadu/move_robot | 2026-02-24 | WIP |
| yadu/move_robot_callback | 2026-02-24 | Address some feedback |

## luca (scoring / pose)
| Branch | Last commit | Latest subject |
|---|---|---|
| luca/scoring_tuning | 2026-06-12 | More clear scoring messages |
| luca/task_board_pose_publisher | 2026-05-04 | Add topic |

## Others
| Branch | Last commit | Latest subject |
|---|---|---|
| b-corry/delay_spawners | 2026-02-27 | Delay spawners after gz spawn |
| koonpeng/lerobot-next | 2026-03-23 | support lerobot 0.5.0 |
| raanjali/submission | 2026-02-20 | Swap the symbolic text for the numeric one for podman compose |
| trushant/sphinx_docs | 2026-03-10 | Fix formatting |
| trushant/usd_asset_generator | 2026-03-03 | Add converter service in docker compose |
| xiyu/scoring_performance | 2026-02-23 | Merge remote-tracking branch 'origin/main' into xiyu/scoring_performance |

---

### Regenerate this table
```bash
git fetch --all --prune
for b in $(git branch -r | grep origin | grep -v HEAD | sed 's/ *origin\///'); do
  printf "| %s | %s | %s |\n" "$b" \
    "$(git log -1 --format='%ci' origin/$b | cut -d' ' -f1)" \
    "$(git log -1 --format='%s' origin/$b)"
done
```
