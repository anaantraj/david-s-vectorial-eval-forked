#!/usr/bin/env bash
# Continue each nested Study 4 scale arm after its full-corpus job finishes.
#
# Run on cthulhu2 with the raw and prior full-corpus training PIDs. Each arm
# independently verifies successful completion and then launches its matched
# 1k-per-audience control on the same GPU. It intentionally does not
# choose a winning base; that decision requires held-out frontier measurements.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 RAW_10K_PID PRIOR_10K_PID" >&2
  exit 2
fi

RAW_PID=$1
PRIOR_PID=$2
PROJECT_ROOT="${VECTORIAL_PROJECT_ROOT:-$HOME/Projects/vectorial}"
PYTHON_BIN="${VECTORIAL_PYTHON:-$HOME/micromamba/envs/vectorial/bin/python}"
cd "$PROJECT_ROOT"
mkdir -p runs/cluster_logs
FAILURE_MARKER="runs/cluster_logs/s4-stage-a.failed"
rm -f "$FAILURE_MARKER"
record_failure() {
  local status=$?
  if (( status != 0 )); then
    printf '%s\n' "$status" >"$FAILURE_MARKER"
  fi
}
trap record_failure EXIT

finished() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1]) / "metrics.json"
raise SystemExit(0 if path.exists() and json.load(path.open()).get("finished") else 1)
PY
}

export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/scripts/cluster"
launch_rung() {
  local gpu=$1
  local variant=$2
  local job=$3
  local pid_file="runs/cluster_logs/$job.pid"
  local safety_suffix=""
  if [[ "$variant" == "platform_prior_lm" ]]; then
    safety_suffix=".rich_safe"
  fi
  rm -f "$pid_file"
  CUDA_VISIBLE_DEVICES="$gpu" VECTORIAL_CKPT_DIR="runs/checkpoints/$job" \
    nohup setsid "$PYTHON_BIN" -u src/vectorial_eval/methods/lora/train.py \
      --objective unsupervised \
      --variant "$variant" \
      --config attn-r8 \
      --train-file "data/study4_priors/reddit_1000_per_audience${safety_suffix}.train.jsonl" \
      --dev-file "data/study4_priors/reddit_1000_per_audience${safety_suffix}.dev.jsonl" \
      --epochs 1 \
      --patience 1 \
      >"runs/cluster_logs/$job.log" 2>&1 < /dev/null &
  local job_pid=$!
  echo "$job_pid" >"$pid_file"
  echo "$job pid=$job_pid gpu=$gpu"
}

continue_rung() {
  local parent_pid=$1
  local parent_root=$2
  local gpu=$3
  local variant=$4
  local job=$5
  while kill -0 "$parent_pid" 2>/dev/null; do
    sleep 30
  done
  if ! finished "$parent_root"; then
    echo "parent run exited without a finished metrics artifact: $parent_root" >&2
    return 1
  fi
  launch_rung "$gpu" "$variant" "$job"
}

continue_rung "$RAW_PID" \
  runs/checkpoints/s4-reddit-rich-raw-10k \
  0 platform_lm s4-reddit-rich-raw-1k &
raw_waiter=$!
continue_rung "$PRIOR_PID" \
  runs/checkpoints/s4-reddit-rich-prior-10k \
  1 platform_prior_lm s4-reddit-rich-prior-1k &
prior_waiter=$!
wait "$raw_waiter"
wait "$prior_waiter"
