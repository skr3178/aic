#!/usr/bin/env bash
# Orchestrate CheatCode data collection over generated chunk configs.
# For each chunk: run the eval container headless (ground_truth:=true), then on the host launch
# CheatCode (drives the arm) + collector_node.py (records image<->GT-pose), both FULLY DETACHED
# (setsid+nohup) so they survive harness reaping. Block on the container, then tear the hosts down.
#
# Usage:
#   run_collection.sh                 # all chunks 0..4
#   run_collection.sh 0 2             # only chunks 0 and 2
#   CONFIG_OVERRIDE=smoke_config.yaml run_collection.sh 0    # smoke (uses that yaml for chunk 0)
#
# This is a FILE (not an inline command) on purpose: its own argv does not contain the pkill
# patterns, so `pkill -f "aic_model --ros-args"` / "collector_node.py" never self-kills.
set -u   # NOT pipefail: `docker logs | grep -q` short-circuits grep -> SIGPIPE upstream ->
         # pipefail would mask the match and break engine-up detection.

COLLECT=/home/skr/ws_aic/aic_local/collect
AICLOCAL=/home/skr/ws_aic/aic_local
AICRUN=$COLLECT/aicrun
IMAGE=ghcr.io/intrinsic-dev/aic/aic_eval:latest
MANIFEST=$COLLECT/manifest.json
DATASET=${DATASET:-$HOME/aic_data/perception_v1}
HZ=${HZ:-10}
SCALE=${SCALE:-0.25}
CONFIG_OVERRIDE=${CONFIG_OVERRIDE:-}
CHUNKS=${*:-0 1 2 3 4}

mkdir -p "$DATASET" "$HOME/aic_results/collect"
echo $$ > "$HOME/aic_results/collect/run.pid"   # so a detached monitor can track liveness

kill_hosts() {
  pkill -9 -f "aic_model --ros-args" 2>/dev/null || true
  pkill -9 -f "collector_node.py"    2>/dev/null || true
}

for k in $CHUNKS; do
  if [ -n "$CONFIG_OVERRIDE" ]; then cfg="/aic_cfg/$(basename "$CONFIG_OVERRIDE")"; else cfg="/aic_cfg/chunk_${k}.yaml"; fi
  name="aic_collect_${k}"
  res="$HOME/aic_results/collect/chunk_${k}"
  echo "[collect] ===== chunk $k  cfg=$cfg  dataset=$DATASET ====="
  mkdir -p "$res"
  kill_hosts; sleep 1
  docker rm -f "$name" >/dev/null 2>&1 || true

  docker run -d --rm --gpus all --network host --name "$name" \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    -v "$AICLOCAL/10_nvidia.json:/usr/share/glvnd/egl_vendor.d/10_nvidia.json:ro" \
    -e AIC_RESULTS_DIR=/results -v "$res:/results" \
    -v "$COLLECT/configs:/aic_cfg:ro" \
    "$IMAGE" \
    aic_engine_config_file:="$cfg" ground_truth:=true \
    gazebo_gui:=false launch_rviz:=false start_aic_engine:=true \
    shutdown_on_aic_engine_exit:=true model_discovery_timeout_seconds:=600 >/dev/null

  # wait for the engine to be up and polling for the model
  up=0
  for i in $(seq 1 120); do
    if ! docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null | grep -q true; then
      echo "[collect] container died during boot:"; docker logs --tail 25 "$name" 2>&1; break
    fi
    if docker logs "$name" 2>&1 | grep -q "aic_model"; then up=1; echo "[collect] engine up after ~$((i*2))s"; break; fi
    sleep 2
  done
  [ "$up" = 1 ] || { echo "[collect] chunk $k: engine never came up, skipping"; docker rm -f "$name" >/dev/null 2>&1; continue; }

  # CheatCode (the single aic_model node) + collector, fully detached
  setsid nohup "$AICRUN" ros2 run aic_model aic_model \
    --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.CheatCode \
    >"$res/policy.log" 2>&1 </dev/null &
  setsid nohup "$AICRUN" python "$COLLECT/collector_node.py" \
    --chunk "$k" --manifest "$MANIFEST" --out "$DATASET" --hz "$HZ" --scale "$SCALE" \
    --ros-args -p use_sim_time:=true \
    >"$res/collector.log" 2>&1 </dev/null &

  # block until the engine finishes all trials (container self-exits), then tear the hosts down
  docker wait "$name" >/dev/null 2>&1
  echo "[collect] chunk $k finished; score -> $res/scoring.yaml"
  kill_hosts
  docker rm -f "$name" >/dev/null 2>&1 || true
  sleep 3
done
echo "[collect] ALL DONE (chunks: $CHUNKS)"
