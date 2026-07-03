#!/usr/bin/env bash
# Parallel CheatCode data collection at NATIVE resolution.
# Each chunk runs in its OWN isolated Docker network (own zenoh router in its own namespace
# -> zero cross-talk between workers), with engine + CheatCode + collector ALL in-container
# (no host process, no --network host). Up to MAX_PAR chunks run concurrently on the one GPU.
# Collector writes PNG+JSONL+meta (parquet skipped in-container; consolidated offline on host).
#
# Usage:  MAX_PAR=3 SCALE=1.0 collect_parallel.sh [chunk ...]      # default chunks 0..4
set -u
IMAGE=ghcr.io/intrinsic-dev/aic/aic_eval:latest
COLLECT=/home/skr/ws_aic/aic_local/collect
AICLOCAL=/home/skr/ws_aic/aic_local
DATASET=${DATASET:-/home/skr/aic_data/perception_native}
SCALE=${SCALE:-1.0}
HZ=${HZ:-10}
MAX_PAR=${MAX_PAR:-3}
CHUNKS="${*:-0 1 2 3 4}"
RES_ROOT=/home/skr/aic_results/parcollect
RMWENV='source /opt/ros/kilted/setup.bash; source /ws_aic/install/setup.bash; export RMW_IMPLEMENTATION=rmw_zenoh_cpp; export ZENOH_ROUTER_CONFIG_URI=/aic_zenoh_config.json5; export ZENOH_CONFIG_OVERRIDE=";transport/shared_memory/enabled=false"; export GZ_CONFIG_PATH=/ws_aic/install/share/gz'

mkdir -p "$DATASET" "$RES_ROOT"
echo "[par] dataset=$DATASET scale=$SCALE max_par=$MAX_PAR chunks=[$CHUNKS]"

run_chunk() {
  local k=$1
  local net=aicnet_$k name=aic_par_$k res="$RES_ROOT/chunk_$k"
  mkdir -p "$res"
  docker rm -f "$name" >/dev/null 2>&1 || true
  docker network rm "$net" >/dev/null 2>&1 || true
  docker network create "$net" >/dev/null 2>&1 || true
  local t0=$SECONDS
  docker run -d --gpus all --network "$net" --name "$name" \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    -v "$AICLOCAL/10_nvidia.json:/usr/share/glvnd/egl_vendor.d/10_nvidia.json:ro" \
    -e AIC_RESULTS_DIR=/results -v "$res:/results" \
    -v "$COLLECT:/collect:ro" -v "$DATASET:/data" \
    "$IMAGE" \
    aic_engine_config_file:=/collect/configs/chunk_${k}.yaml ground_truth:=true \
    gazebo_gui:=false launch_rviz:=false start_aic_engine:=true \
    shutdown_on_aic_engine_exit:=true model_discovery_timeout_seconds:=600 >/dev/null
  # wait engine up
  local up=0
  for i in $(seq 1 120); do
    docker logs "$name" 2>&1 | grep -q "aic_model" && { up=1; break; }
    docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null | grep -q true || { echo "[par] chunk $k container died on boot"; docker logs --tail 20 "$name"; break; }
    sleep 2
  done
  if [ "$up" != 1 ]; then docker rm -f "$name" >/dev/null 2>&1; docker network rm "$net" >/dev/null 2>&1; echo "[par] chunk $k FAILED (engine never up)"; return 1; fi
  # CheatCode + collector, both in-container, detached
  docker exec -d "$name" bash -lc "$RMWENV; ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.CheatCode >/results/cheat.log 2>&1"
  docker exec -d "$name" bash -lc "$RMWENV; python3 /collect/collector_node.py --chunk $k --manifest /collect/manifest.json --out /data --hz $HZ --scale $SCALE --ros-args -p use_sim_time:=true >/results/collector.log 2>&1"
  # block until engine finishes all trials (container self-exits)
  docker wait "$name" >/dev/null 2>&1
  docker rm -f "$name" >/dev/null 2>&1 || true
  docker network rm "$net" >/dev/null 2>&1 || true
  echo "[par] chunk $k DONE in $((SECONDS-t0))s"
}

# launch with concurrency cap MAX_PAR
running=0
for k in $CHUNKS; do
  run_chunk "$k" &
  running=$((running+1))
  if [ "$running" -ge "$MAX_PAR" ]; then wait -n 2>/dev/null || wait; running=$((running-1)); fi
done
wait
echo "[par] ALL CHUNKS DONE. dataset -> $DATASET"
ls "$DATASET" | grep -c episode_ | xargs echo "episodes collected:"