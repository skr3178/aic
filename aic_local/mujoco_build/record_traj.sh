#!/usr/bin/env bash
# Record a full successful CheatCode trial_1 trajectory from Gazebo for MuJoCo replay:
#  - launch CheatCode in-container (correct rmw_zenoh env)
#  - wait until trial objects spawn (task_board + cable_0)
#  - dump full_world.sdf (mesh map + initial poses)
#  - record /world/aic_world/dynamic_pose/info for DUR seconds (the insertion)
set -u
NAME=aic_worlddump
DUR=${DUR:-40}
RMWENV='source /opt/ros/kilted/setup.bash; source /ws_aic/install/setup.bash; export RMW_IMPLEMENTATION=rmw_zenoh_cpp; export ZENOH_ROUTER_CONFIG_URI=/aic_zenoh_config.json5; export GZ_CONFIG_PATH=/ws_aic/install/share/gz'

docker exec "$NAME" bash -lc "pkill -9 -f 'aic_model --ros-args' 2>/dev/null; true"
echo "[rec] launching CheatCode in-container..."
docker exec -d "$NAME" bash -lc "$RMWENV; ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.CheatCode >/results/cheat.log 2>&1"

echo "[rec] waiting for trial objects (task_board + cable_0)..."
for i in $(seq 1 60); do
  ml=$(docker exec "$NAME" bash -lc "$RMWENV; gz model --list 2>/dev/null")
  if echo "$ml" | grep -q task_board && echo "$ml" | grep -q cable_0; then echo "[rec] trial spawned at ~${i}s"; break; fi
  sleep 1
done

echo "[rec] dumping full_world.sdf (initial poses + mesh map)..."
docker exec "$NAME" bash -lc "$RMWENV; gz service -s /world/aic_world/generate_world_sdf --reqtype gz.msgs.SdfGeneratorConfig --reptype gz.msgs.StringMsg --timeout 20000 --req 'global_entity_gen_config: {expand_include_tags: {data: true}}' > /results/full_world_traj.txt 2>/dev/null"

echo "[rec] recording dynamic_pose/info for ${DUR}s (the insertion)..."
docker exec "$NAME" bash -lc "$RMWENV; timeout ${DUR} gz topic -e -t /world/aic_world/dynamic_pose/info > /results/poses_trial1.txt 2>/dev/null"
echo "[rec] done. sizes:"
docker exec "$NAME" bash -lc 'wc -c /results/poses_trial1.txt /results/full_world_traj.txt'
echo "[rec] cheat log tail:"; docker exec "$NAME" bash -lc 'tail -3 /results/cheat.log'
