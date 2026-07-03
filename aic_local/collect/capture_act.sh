#!/usr/bin/env bash
# Capture the RunACT policy's camera frames on the 3 eval scenes, in ONE container run.
# Mirrors the CheatCode eval_scenes capture but with policy=RunACT, single 3-trial eval_config
# run (one boot). Collector splits trials via plug-TF spawn/despawn.
# ground_truth:=true is REQUIRED here: the collector detects/labels trials via the GT object TF
# frames (cable/port), which the engine only publishes on /tf when ground_truth:=true. RunACT
# consumes only /observations (images+state) and never reads TF, so its behavior is identical
# whether ground_truth is true or false -- the flag only re-exposes frames for the recorder.
# Output: $OUT/episode_000{0,1,2} -> renamed scene_{1,2,3}, mirroring eval_scenes/ layout.
#
# FILE (not inline): its argv lacks the pkill patterns, so kill_hosts never self-kills.
set -u   # NOT pipefail: `docker logs | grep -q` short-circuits grep -> SIGPIPE would mask the match.

COLLECT=/home/skr/ws_aic/aic_local/collect
AICLOCAL=/home/skr/ws_aic/aic_local
AICRUN=$COLLECT/aicrun
REPO=/home/skr/ws_aic/src/aic
PIXI=/home/skr/.pixi/bin/pixi
IMAGE=ghcr.io/intrinsic-dev/aic/aic_eval:latest
MANIFEST=$COLLECT/eval_manifest.json
OUT=${OUT:-/home/skr/aic_data/runact_scenes}
RES=/home/skr/aic_results/runact_capture      # engine bags land here (root-owned; cleaned after)
NAME=aic_runact_capture
HZ=${HZ:-10}
SCALE=${SCALE:-0.25}
POLICY=aic_example_policies.ros.RunACT

mkdir -p "$OUT" "$RES"
kill_hosts() {
  pkill -9 -f "aic_model --ros-args" 2>/dev/null || true
  pkill -9 -f "collector_node.py"    2>/dev/null || true
}
kill_hosts; sleep 1
docker rm -f "$NAME" >/dev/null 2>&1 || true

echo "[capture] launching eval container (eval_config, ground_truth:=true, 3 trials)"
docker run -d --rm --gpus all --network host --name "$NAME" \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
  -v "$AICLOCAL/10_nvidia.json:/usr/share/glvnd/egl_vendor.d/10_nvidia.json:ro" \
  -e AIC_RESULTS_DIR=/results -v "$RES:/results" \
  -v "$REPO/aic_engine/config:/aic_cfg:ro" \
  "$IMAGE" \
  aic_engine_config_file:=/aic_cfg/eval_config.yaml ground_truth:=true \
  gazebo_gui:=false launch_rviz:=false start_aic_engine:=true \
  shutdown_on_aic_engine_exit:=true model_discovery_timeout_seconds:=300 >/dev/null

# wait for the engine to come up and start polling for the model
up=0
for i in $(seq 1 120); do
  if ! docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null | grep -q true; then
    echo "[capture] container died during boot:"; docker logs --tail 30 "$NAME" 2>&1; break
  fi
  if docker logs "$NAME" 2>&1 | grep -q "aic_model"; then up=1; echo "[capture] engine up after ~$((i*2))s"; break; fi
  sleep 2
done
[ "$up" = 1 ] || { echo "[capture] engine never came up, aborting"; docker rm -f "$NAME" >/dev/null 2>&1; exit 1; }

pkill -9 -f "aic_model --ros-args" 2>/dev/null; sleep 1
echo "[capture] launching RunACT policy + collector on host..."
setsid nohup "$AICRUN" ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true -p policy:="$POLICY" \
  >"$RES/policy.log" 2>&1 </dev/null &
setsid nohup "$AICRUN" python "$COLLECT/collector_node.py" \
  --chunk 0 --manifest "$MANIFEST" --out "$OUT" --hz "$HZ" --scale "$SCALE" \
  --ros-args -p use_sim_time:=true \
  >"$RES/collector.log" 2>&1 </dev/null &

# block until the engine finishes all 3 trials (container self-exits), then tear hosts down
docker wait "$NAME" >/dev/null 2>&1
echo "[capture] engine finished; tearing down hosts"
kill_hosts
docker rm -f "$NAME" >/dev/null 2>&1 || true
sleep 2

# rename episode dirs -> scene_{1,2,3} to mirror eval_scenes/
for i in 0 1 2; do
  src="$OUT/episode_000${i}"; dst="$OUT/scene_$((i+1))"
  if [ -d "$src" ]; then rm -rf "$dst"; mv "$src" "$dst"; fi
done

echo "[capture] DONE. per-scene frame counts:"
for s in scene_1 scene_2 scene_3; do
  n=$(ls "$OUT/$s/center/"*.png 2>/dev/null | wc -l)
  echo "  $s: $n center frames"
done
echo "[capture] collector log tail:"; tail -6 "$RES/collector.log" 2>/dev/null
