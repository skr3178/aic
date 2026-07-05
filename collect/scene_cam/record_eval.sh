#!/usr/bin/env bash
# record_eval.sh — score.sh fork with a 3rd-person scene camera (DIAGNOSTIC RUNS ONLY).
# Standalone: score.sh, the image, and all baselines untouched. The extra camera exists only
# inside this container instance via bind-mount overlays of the world SDF + bridge config
# (same mechanism as the 10_nvidia.json EGL overlay).
#
# Usage: record_eval.sh <policy.module.Class> [ground_truth=false] [config=eval|sample|<path>] [run_name]
# Output: ~/aic_results/<run>/  + scene_frames/*.png + scene.mp4
set -u

POLICY="${1:?usage: record_eval.sh <policy.module.Class> [ground_truth] [config] [run_name]}"
GT="${2:-false}"
CFG="${3:-eval}"
RUN="${4:-scenecam_$(printf '%s' "$POLICY" | awk -F. '{print tolower($NF)}')}"

REPO="$HOME/ws_aic/src/aic"
IMAGE="ghcr.io/intrinsic-dev/aic/aic_eval:latest"
PIXI="$HOME/.pixi/bin/pixi"
KIT="$HOME/ws_aic/aic_local/collect/scene_cam"
RESULTS="$HOME/aic_results/$RUN"
CNAME="aic_eval_$RUN"

case "$CFG" in
  eval)   CFGP="/aic_cfg/eval_config.yaml" ;;
  sample) CFGP="/aic_cfg/sample_config.yaml" ;;
  *)      CFGP="$CFG" ;;
esac

mkdir -p "$RESULTS/scene_frames"; rm -f "$RESULTS/scoring.yaml"; rm -f "$RESULTS"/scene_frames/*.png
docker rm -f "$CNAME" >/dev/null 2>&1 || true
echo "[rec] policy=$POLICY gt=$GT config=$CFG -> $RESULTS  (scene cam ON — diagnostic run)"

docker run -d --rm --gpus all --network host --name "$CNAME" \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
  -e AIC_RESULTS_DIR=/results -v "$RESULTS:/results" \
  -v "$REPO/aic_engine/config:/aic_cfg:ro" \
  -v "$HOME/ws_aic/aic_local/10_nvidia.json:/usr/share/glvnd/egl_vendor.d/10_nvidia.json:ro" \
  -v "$KIT/aic_scene_cam.sdf:/ws_aic/install/share/aic_description/world/aic.sdf:ro" \
  -v "$KIT/bridge_config.yaml:/ws_aic/install/share/aic_bringup/config/ros_gz_bridge_config.yaml:ro" \
  "$IMAGE" \
  aic_engine_config_file:="$CFGP" ground_truth:="$GT" \
  gazebo_gui:=false launch_rviz:=false start_aic_engine:=true \
  shutdown_on_aic_engine_exit:=true model_discovery_timeout_seconds:=300 >/dev/null

for i in $(seq 1 120); do
  if ! docker inspect -f '{{.State.Running}}' "$CNAME" 2>/dev/null | grep -q true; then
    echo "[rec] ERROR: container exited during boot"; docker logs --tail 40 "$CNAME" 2>&1
    docker rm -f "$CNAME" >/dev/null 2>&1; exit 1
  fi
  docker logs "$CNAME" 2>&1 | grep -q "aic_model" && { echo "[rec] engine up after ~$((i*2))s"; break; }
  sleep 2
done

pkill -9 -f "aic_model --ros-args" 2>/dev/null; pkill -9 -f scene_recorder 2>/dev/null; sleep 1
echo "[rec] launching recorder + policy on host..."
( cd "$REPO" && "$PIXI" run python "$KIT/scene_recorder.py" --out "$RESULTS/scene_frames" \
    --ros-args -p use_sim_time:=true ) >"$RESULTS/scene_recorder.log" 2>&1 &
( cd "$REPO" && "$PIXI" run ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true -p policy:="$POLICY" ) >"$RESULTS/policy.log" 2>&1 &

docker wait "$CNAME" >/dev/null 2>&1
pkill -9 -f "aic_model --ros-args" 2>/dev/null
pkill -9 -f scene_recorder 2>/dev/null
docker rm -f "$CNAME" >/dev/null 2>&1 || true

N=$(ls "$RESULTS"/scene_frames/*.png 2>/dev/null | wc -l)
echo "[rec] $N scene frames captured"
if [ "$N" -gt 10 ]; then
  ffmpeg -y -loglevel error -framerate 10 -i "$RESULTS/scene_frames/%05d.png" \
    -c:v libx264 -pix_fmt yuv420p -crf 20 "$RESULTS/scene.mp4" && echo "[rec] wrote $RESULTS/scene.mp4"
fi
grep -E "^total:" "$RESULTS/scoring.yaml" 2>/dev/null || echo "(no scoring.yaml — check logs)"
