#!/usr/bin/env bash
# Launch all three conditional LoRA variants after both nested scale runs finish.
set -euo pipefail

PROJECT_ROOT="${VECTORIAL_PROJECT_ROOT:-$HOME/Projects/vectorial}"
PYTHON_BIN="${VECTORIAL_PYTHON:-$HOME/micromamba/envs/vectorial/bin/python}"
cd "$PROJECT_ROOT"
mkdir -p runs/cluster_logs
FAILURE_MARKER="runs/cluster_logs/s4-stage-b.failed"
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

wait_for_scale_run() {
  local run=$1
  local root="runs/checkpoints/$run"
  local pid_file="runs/cluster_logs/$run.pid"
  until finished "$root"; do
    if [[ -s runs/cluster_logs/s4-stage-a.failed ]]; then
      echo "Stage A failed before $run completed" >&2
      return 1
    fi
    if [[ -s "$pid_file" ]]; then
      local run_pid
      run_pid=$(<"$pid_file")
      if ! kill -0 "$run_pid" 2>/dev/null; then
        echo "$run died without a finished metrics artifact" >&2
        return 1
      fi
    fi
    sleep 30
  done
}

for run in s4-reddit-rich-raw-1k s4-reddit-rich-prior-1k; do
  wait_for_scale_run "$run"
done

INITIAL="runs/checkpoints/s4-reddit-rich-prior-10k/best"
if [[ ! -s "$INITIAL/adapter_config.json" ]]; then
  echo "frozen primary Stage A adapter is missing: $INITIAL" >&2
  exit 1
fi
"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

run = json.loads(Path("runs/checkpoints/s4-reddit-rich-prior-10k/run.json").read_text())
if run.get("prior_prompt_schema") != "study4-rich-priors-v3":
    raise SystemExit("primary adapter was not trained with Study 4 rich priors v3")
PY

export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/scripts/cluster"

# Matched coarse-to-fine ablation. The two training files are a deterministic,
# disjoint union of the primary corpus, so total target-completion exposure is
# unchanged rather than doubled by two full passes.
CUDA_VISIBLE_DEVICES=0 VECTORIAL_CKPT_DIR="runs/checkpoints/s4-reddit-coarse-platform-half" \
  "$PYTHON_BIN" -u src/vectorial_eval/methods/lora/train.py \
    --objective unsupervised \
    --variant platform_lm \
    --config attn-r8 \
    --train-file data/study4_priors/coarse_to_fine/reddit.platform_half.train.jsonl \
    --dev-file data/study4_priors/coarse_to_fine/reddit.shared.dev.jsonl \
    --epochs 1 \
    --patience 1 \
    >runs/cluster_logs/s4-reddit-coarse-platform-half.log 2>&1
CUDA_VISIBLE_DEVICES=0 VECTORIAL_CKPT_DIR="runs/checkpoints/s4-reddit-platform-then-audience" \
  "$PYTHON_BIN" -u src/vectorial_eval/methods/lora/train.py \
    --objective unsupervised \
    --variant platform_prior_lm \
    --config attn-r8 \
    --train-file data/study4_priors/coarse_to_fine/reddit.audience_half.rich_safe.train.jsonl \
    --dev-file data/study4_priors/coarse_to_fine/reddit.shared.rich_safe.dev.jsonl \
    --initial-adapter runs/checkpoints/s4-reddit-coarse-platform-half/best \
    --epochs 1 \
    --patience 1 \
    >runs/cluster_logs/s4-reddit-platform-then-audience.log 2>&1

# Validate that PEFT can reconstruct the nested frozen-LoRA plus soft-prompt
# stack in this exact cluster environment before committing the full grid.
CUDA_VISIBLE_DEVICES=0 VECTORIAL_CKPT_DIR="runs/checkpoints/s4-soft-nested-smoke" \
  "$PYTHON_BIN" -u src/vectorial_eval/methods/soft_prompt/train.py \
    --variant target_lm \
    --data-dir data/study3/training \
    --initial-adapter "$INITIAL" \
    --n-virtual-tokens 4 \
    --lr 1e-2 \
    --init text \
    --epochs 1 \
    --patience 1 \
    --limit 4 \
    >runs/cluster_logs/s4-soft-nested-smoke.log 2>&1
CUDA_VISIBLE_DEVICES=0 \
  VECTORIAL_SOFT_PROMPT_DIR="runs/checkpoints/s4-soft-nested-smoke/best" \
  "$PYTHON_BIN" -m vectorial_eval.cli \
    --data-dir data/study3 --run-dir runs/s4-soft-nested-smoke \
    transfer --fn soft_prompt --split val --limit 1 --n-samples 1 \
    >runs/cluster_logs/s4-soft-nested-smoke-generate.log 2>&1

launch_variant() {
  local gpu=$1
  local variant=$2
  local job="s4-prior10k-lora-${variant}"
  CUDA_VISIBLE_DEVICES="$gpu" VECTORIAL_CKPT_DIR="runs/checkpoints/$job" \
    "$PYTHON_BIN" -u src/vectorial_eval/methods/lora/train.py \
      --variant "$variant" \
      --config attn-r8 \
      --data-dir data/study3/training \
      --initial-adapter "$INITIAL" \
      --epochs 12 \
      --patience 3 \
      >"runs/cluster_logs/$job.log" 2>&1
}

launch_variant 0 target_lm &
PID_TARGET=$!
launch_variant 1 target_lm_paired &
PID_PAIRED=$!
wait "$PID_TARGET"
launch_variant 0 target_lm_aspect &
PID_ASPECT=$!
wait "$PID_PAIRED"
wait "$PID_ASPECT"

launch_soft_prompt() {
  local gpu=$1
  local variant=$2
  local job="s4-prior10k-soft-prompt-${variant}"
  CUDA_VISIBLE_DEVICES="$gpu" VECTORIAL_CKPT_DIR="runs/checkpoints/$job" \
    "$PYTHON_BIN" -u src/vectorial_eval/methods/soft_prompt/train.py \
      --variant "$variant" \
      --data-dir data/study3/training \
      --initial-adapter "$INITIAL" \
      --n-virtual-tokens 32 \
      --lr 1e-2 \
      --init text \
      --epochs 20 \
      --patience 4 \
      >"runs/cluster_logs/$job.log" 2>&1
}

launch_soft_prompt 0 target_lm &
PID_SOFT_TARGET=$!
launch_soft_prompt 1 target_lm_paired &
PID_SOFT_PAIRED=$!
wait "$PID_SOFT_TARGET"
launch_soft_prompt 0 target_lm_aspect &
PID_SOFT_ASPECT=$!
wait "$PID_SOFT_PAIRED"
wait "$PID_SOFT_ASPECT"

mkdir -p runs/study4_steering
fit_steering() {
  local gpu=$1
  local variant=$2
  CUDA_VISIBLE_DEVICES="$gpu" VECTORIAL_CKPT_DIR="runs/checkpoints/s4-steering-fit-${variant}" \
    "$PYTHON_BIN" -u src/vectorial_eval/methods/steering/fit_steering.py \
      --variant "$variant" \
      --data-dir data/study3 \
      --initial-adapter "$INITIAL" \
      --out "runs/study4_steering/steering.${variant}.pt" \
      >"runs/cluster_logs/s4-steering-fit-${variant}.log" 2>&1
}

fit_steering 0 target_lm &
PID_STEER_TARGET=$!
fit_steering 1 target_lm_paired &
PID_STEER_PAIRED=$!
wait "$PID_STEER_TARGET"
fit_steering 0 target_lm_aspect &
PID_STEER_ASPECT=$!
wait "$PID_STEER_PAIRED"
wait "$PID_STEER_ASPECT"

sweep_steering() {
  local gpu=$1
  local variant=$2
  CUDA_VISIBLE_DEVICES="$gpu" VECTORIAL_CKPT_DIR="runs/checkpoints/s4-steering-sweep-${variant}" \
    "$PYTHON_BIN" -u src/vectorial_eval/methods/steering/sweep_steering.py \
      --artifact "runs/study4_steering/steering.${variant}.pt" \
      --data-dir data/study3 \
      --limit 16 \
      --scope all \
      --out "runs/study4_steering/sweep.${variant}.json" \
      >"runs/cluster_logs/s4-steering-sweep-${variant}.log" 2>&1
}

sweep_steering 0 target_lm &
PID_SWEEP_TARGET=$!
sweep_steering 1 target_lm_paired &
PID_SWEEP_PAIRED=$!
wait "$PID_SWEEP_TARGET"
sweep_steering 0 target_lm_aspect &
PID_SWEEP_ASPECT=$!
wait "$PID_SWEEP_PAIRED"
wait "$PID_SWEEP_ASPECT"

# Secondary interaction: refit the activation contrasts in the final LoRA's
# parameterisation, then retain a non-zero setting only if it is Pareto
# non-inferior to that LoRA on both selection axes.
INTERACTION_ADAPTER="runs/checkpoints/s4-prior10k-lora-target_lm/best"
CUDA_VISIBLE_DEVICES=0 VECTORIAL_CKPT_DIR="runs/checkpoints/s4-lora-steering-interaction-fit" \
  "$PYTHON_BIN" -u src/vectorial_eval/methods/steering/fit_steering.py \
    --variant target_lm \
    --data-dir data/study3 \
    --initial-adapter "$INTERACTION_ADAPTER" \
    --out runs/study4_steering/steering.lora_interaction.pt \
    >runs/cluster_logs/s4-lora-steering-interaction-fit.log 2>&1
CUDA_VISIBLE_DEVICES=0 VECTORIAL_CKPT_DIR="runs/checkpoints/s4-lora-steering-interaction-sweep" \
  "$PYTHON_BIN" -u src/vectorial_eval/methods/steering/sweep_steering.py \
    --artifact runs/study4_steering/steering.lora_interaction.pt \
    --data-dir data/study3 \
    --limit 16 \
    --scope all \
    --out runs/study4_steering/sweep.lora_interaction.json \
    >runs/cluster_logs/s4-lora-steering-interaction-sweep.log 2>&1
echo "all LoRA, soft-prompt, and steering Study 4 variants finished"
