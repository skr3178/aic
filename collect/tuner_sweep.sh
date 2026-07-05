#!/usr/bin/env bash
# InsertTuner sweep — fork of collect_parallel.sh with the tuner policy instead of
# CheatCode+collector. Each chunk in its own isolated Docker network; MAX_PAR concurrent.
# Standalone: does not modify collect_parallel.sh or any policy.
#
# Usage:  MAX_PAR=5 tuner_sweep.sh [chunk ...]          # default chunks 0..4
# Results: /home/skr/aic_results/tuner/chunk_N/{tuner_log.jsonl, scoring.yaml, engine logs}
set -u
IMAGE=ghcr.io/intrinsic-dev/aic/aic_eval:latest
COLLECT=/home/skr/ws_aic/aic_local/collect
AICLOCAL=/home/skr/ws_aic/aic_local
MAX_PAR=${MAX_PAR:-5}
CHUNKS="${*:-0 1 2 3 4}"
RES_ROOT=${RES_ROOT:-/home/skr/aic_results/tuner}
TRIALS_PER_CHUNK=${TRIALS_PER_CHUNK:-10}
RMWENV='source /opt/ros/kilted/setup.bash; source /ws_aic/install/setup.bash; export RMW_IMPLEMENTATION=rmw_zenoh_cpp; export ZENOH_ROUTER_CONFIG_URI=/aic_zenoh_config.json5; export ZENOH_CONFIG_OVERRIDE=";transport/shared_memory/enabled=false"; export GZ_CONFIG_PATH=/ws_aic/install/share/gz'

mkdir -p "$RES_ROOT"
echo "[tuner] chunks=[$CHUNKS] max_par=$MAX_PAR results=$RES_ROOT"

run_chunk() {
  local k=$1
  local net=tunnet_$k name=aic_tuner_$k res="$RES_ROOT/chunk_$k"
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
    -v "$COLLECT:/collect:ro" ${SNAP_OUT:+-v "$SNAP_OUT:/data"} \
    "$IMAGE" \
    aic_engine_config_file:=/collect/configs/chunk_${k}.yaml ground_truth:=true \
    gazebo_gui:=false launch_rviz:=false start_aic_engine:=true \
    shutdown_on_aic_engine_exit:=true model_discovery_timeout_seconds:=600 >/dev/null
  local up=0
  for i in $(seq 1 120); do
    docker logs "$name" 2>&1 | grep -q "aic_model" && { up=1; break; }
    docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null | grep -q true || { echo "[tuner] chunk $k died on boot"; docker logs --tail 20 "$name"; break; }
    sleep 2
  done
  if [ "$up" != 1 ]; then docker rm -f "$name" >/dev/null 2>&1; docker network rm "$net" >/dev/null 2>&1; echo "[tuner] chunk $k FAILED (engine never up)"; return 1; fi
  docker exec -d "$name" bash -lc "$RMWENV; export PYTHONPATH=/collect:\$PYTHONPATH; export TUNER_OFFSETS=/collect/tuner_offsets.json; export TUNER_TRIAL_BASE=$((k*TRIALS_PER_CHUNK)); ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=insert_tuner.InsertTuner >/results/tuner_policy.log 2>&1"
  if [ -n "${SNAP_OUT:-}" ]; then
    docker exec -d "$name" bash -lc "$RMWENV; python3 /collect/collector_node.py --chunk $k --manifest /collect/manifest.json --out /data --hz 5 --scale 0.25 --ros-args -p use_sim_time:=true >/results/collector.log 2>&1"
  fi
  docker wait "$name" >/dev/null 2>&1
  docker rm -f "$name" >/dev/null 2>&1 || true
  docker network rm "$net" >/dev/null 2>&1 || true
  echo "[tuner] chunk $k DONE in $((SECONDS-t0))s ($(wc -l < "$res/tuner_log.jsonl" 2>/dev/null || echo 0) trials logged)"
}

running=0
for k in $CHUNKS; do
  run_chunk "$k" &
  running=$((running+1))
  if [ "$running" -ge "$MAX_PAR" ]; then wait -n 2>/dev/null || wait; running=$((running-1)); fi
done
wait
echo "[tuner] ALL DONE -> $RES_ROOT"
cat "$RES_ROOT"/chunk_*/tuner_log.jsonl 2>/dev/null | wc -l | xargs echo "total trials logged:"
