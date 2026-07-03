#!/usr/bin/env bash
# Self-score an AIC policy locally against the Gazebo eval container (no submission portal).
#
# Usage:
#   score.sh <policy.module.Class> [ground_truth=false] [config=eval|sample|<path>] [run_name]
#
# Examples:
#   score.sh aic_example_policies.ros.WaveArm                       # floor baseline, eval config
#   score.sh aic_example_policies.ros.CheatCode true               # ceiling (needs ground truth)
#   score.sh my_policy.ACTPolicy false eval act_v1                 # a trained policy
#
# Results (scoring.yaml, policy.log, rosbags) land in ~/aic_results/<run_name>/.
set -u   # NOT pipefail: `docker logs | grep -q` short-circuits grep -> SIGPIPE -> pipefail masks the match

POLICY="${1:?usage: score.sh <policy.module.Class> [ground_truth] [config] [run_name]}"
GT="${2:-false}"
CFG="${3:-eval}"
RUN="${4:-$(printf '%s' "$POLICY" | awk -F. '{print $NF}')}"

REPO="$HOME/ws_aic/src/aic"
IMAGE="ghcr.io/intrinsic-dev/aic/aic_eval:latest"
PIXI="$HOME/.pixi/bin/pixi"
RESULTS="$HOME/aic_results/$RUN"
CNAME="aic_eval_$RUN"

# The prebuilt aic_eval image (2026-04-16) only ships sample_config.yaml, so we
# bind-mount the repo's config dir (which has the newer eval_config.yaml) at /aic_cfg.
case "$CFG" in
  eval)   CFGP="/aic_cfg/eval_config.yaml" ;;
  sample) CFGP="/aic_cfg/sample_config.yaml" ;;
  *)      CFGP="$CFG" ;;
esac

mkdir -p "$RESULTS"; rm -f "$RESULTS/scoring.yaml"   # clear any stale score from a prior run
docker rm -f "$CNAME" >/dev/null 2>&1 || true
echo "[score] policy=$POLICY  ground_truth=$GT  config=$CFG  -> $RESULTS"

# 1) Start the eval container headless. With shutdown_on_aic_engine_exit it stops itself
#    once the engine has finished all trials.
docker run -d --rm --gpus all --network host --name "$CNAME" \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
  -e AIC_RESULTS_DIR=/results -v "$RESULTS:/results" \
  -v "$REPO/aic_engine/config:/aic_cfg:ro" \
  -v "$HOME/ws_aic/aic_local/10_nvidia.json:/usr/share/glvnd/egl_vendor.d/10_nvidia.json:ro" \
  "$IMAGE" \
  aic_engine_config_file:="$CFGP" ground_truth:="$GT" \
  gazebo_gui:=false launch_rviz:=false start_aic_engine:=true \
  shutdown_on_aic_engine_exit:=true model_discovery_timeout_seconds:=300 >/dev/null

# 2) Wait until the engine is up and polling for the model (or bail if the container dies).
for i in $(seq 1 120); do
  if ! docker inspect -f '{{.State.Running}}' "$CNAME" 2>/dev/null | grep -q true; then
    echo "[score] ERROR: eval container exited during boot"; docker logs --tail 40 "$CNAME" 2>&1
    docker rm -f "$CNAME" >/dev/null 2>&1; exit 1
  fi
  docker logs "$CNAME" 2>&1 | grep -q "aic_model" && { echo "[score] engine up after ~$((i*2))s"; break; }
  sleep 2
done

# 3) The engine requires EXACTLY ONE aic_model node, so kill any stale policy first
#    (orphans from interrupted runs cause "More than one node with name 'aic_model'").
#    The pattern matches all three policy processes (pixi wrapper, ros2 run, the node).
pkill -9 -f "aic_model --ros-args" 2>/dev/null; sleep 1
echo "[score] launching policy on host..."
( cd "$REPO" && "$PIXI" run ros2 run aic_model aic_model --ros-args \
    -p use_sim_time:=true -p policy:="$POLICY" ) >"$RESULTS/policy.log" 2>&1 &

# 4) Block until the engine finishes (container exits), then tear down the policy tree.
docker wait "$CNAME" >/dev/null 2>&1
pkill -9 -f "aic_model --ros-args" 2>/dev/null
docker rm -f "$CNAME" >/dev/null 2>&1 || true

# 5) Report the score.
echo "===== $RESULTS/scoring.yaml ====="
if [ -f "$RESULTS/scoring.yaml" ]; then
  cat "$RESULTS/scoring.yaml"
  echo "----------------------------------"
  grep -E "^total:" "$RESULTS/scoring.yaml" || echo "(no total: field found)"
else
  echo "NO scoring.yaml produced — tail of policy.log:"; tail -25 "$RESULTS/policy.log" 2>/dev/null
fi
