#!/usr/bin/env bash
# Run the complete Reddit-to-LinkedIn replication after the forward frontier exists.
set -euo pipefail

PROJECT_ROOT="${VECTORIAL_PROJECT_ROOT:-$HOME/Projects/vectorial}"
PYTHON_BIN="${VECTORIAL_PYTHON:-$HOME/micromamba/envs/vectorial/bin/python}"
# The two cards this run owns. These GPUs are shared and a 4-bit NF4 fit at
# length 768 peaks at 19.1 GB (docs/09-cluster-training-config.md), so both
# must be effectively free at launch. Defaults preserve the original 0 and 1.
GPU_A="${VECTORIAL_GPU_A:-0}"
GPU_B="${VECTORIAL_GPU_B:-1}"
cd "$PROJECT_ROOT"
mkdir -p runs/cluster_logs
FAILURE_MARKER="runs/cluster_logs/s4-reverse.failed"
rm -f "$FAILURE_MARKER"
record_failure() {
  local status=$?
  if (( status != 0 )); then
    printf '%s\n' "$status" >"$FAILURE_MARKER"
  fi
}
trap record_failure EXIT

until [[ -s runs/study4_forward/frontier.heldout.embeddinggemma-300m-d512.json ]]; do
  if [[ -s runs/cluster_logs/s4-generation.failed ]]; then
    echo "forward generation failed before the reverse replication" >&2
    exit 1
  fi
  sleep 30
done

export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/scripts/cluster"

finished() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1]) / "metrics.json"
raise SystemExit(0 if path.exists() and json.load(path.open()).get("finished") else 1)
PY
}

train_prior() {
  local gpu=$1
  local rung=$2
  local variant=$3
  local job="s4-linkedin-rich-${variant}-${rung}"
  if finished "runs/checkpoints/$job"; then
    echo "$job already finished; reusing the frozen checkpoint"
    return 0
  fi
  local safety_suffix=""
  if [[ "$variant" == "platform_prior_lm" ]]; then
    safety_suffix=".rich_safe"
  fi
  CUDA_VISIBLE_DEVICES="$gpu" VECTORIAL_CKPT_DIR="runs/checkpoints/$job" \
    "$PYTHON_BIN" -u src/vectorial_eval/methods/lora/train.py \
      --objective unsupervised \
      --variant "$variant" \
      --config attn-r8 \
      --train-file "data/study4_priors/linkedin_${rung}_per_audience${safety_suffix}.train.jsonl" \
      --dev-file "data/study4_priors/linkedin_${rung}_per_audience${safety_suffix}.dev.jsonl" \
      --epochs 1 \
      --patience 1 \
      >"runs/cluster_logs/$job.log" 2>&1
}

# The source corpus is named 10000_per_audience for rung provenance, but the
# manifest records the actual four-audience availability and every shortfall.
train_prior "$GPU_A" 10000 platform_lm &
PID_RAW=$!
train_prior "$GPU_B" 10000 platform_prior_lm &
PID_PRIOR=$!
wait "$PID_RAW"
wait "$PID_PRIOR"

PRIMARY="runs/checkpoints/s4-linkedin-rich-platform_prior_lm-10000/best"
if [[ ! -s "$PRIMARY/adapter_config.json" ]]; then
  echo "reverse primary prior did not improve on its base" >&2
  exit 1
fi
"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

run = json.loads(
    Path("runs/checkpoints/s4-linkedin-rich-platform_prior_lm-10000/run.json").read_text()
)
if run.get("prior_prompt_schema") != "study4-rich-priors-v3":
    raise SystemExit("reverse primary adapter was not trained with Study 4 rich priors v3")
PY

train_prior "$GPU_A" 1000 platform_lm &
PID_RAW_1K=$!
train_prior "$GPU_B" 1000 platform_prior_lm &
PID_PRIOR_1K=$!
wait "$PID_RAW_1K"
wait "$PID_PRIOR_1K"

CUDA_VISIBLE_DEVICES="$GPU_A" VECTORIAL_CKPT_DIR="runs/checkpoints/s4-linkedin-coarse-platform-half" \
  "$PYTHON_BIN" -u src/vectorial_eval/methods/lora/train.py \
    --objective unsupervised \
    --variant platform_lm \
    --config attn-r8 \
    --train-file data/study4_priors/coarse_to_fine/linkedin.platform_half.train.jsonl \
    --dev-file data/study4_priors/coarse_to_fine/linkedin.shared.dev.jsonl \
    --epochs 1 \
    --patience 1 \
    >runs/cluster_logs/s4-linkedin-coarse-platform-half.log 2>&1
CUDA_VISIBLE_DEVICES="$GPU_A" VECTORIAL_CKPT_DIR="runs/checkpoints/s4-linkedin-platform-then-audience" \
  "$PYTHON_BIN" -u src/vectorial_eval/methods/lora/train.py \
    --objective unsupervised \
    --variant platform_prior_lm \
    --config attn-r8 \
    --train-file data/study4_priors/coarse_to_fine/linkedin.audience_half.rich_safe.train.jsonl \
    --dev-file data/study4_priors/coarse_to_fine/linkedin.shared.rich_safe.dev.jsonl \
    --initial-adapter runs/checkpoints/s4-linkedin-coarse-platform-half/best \
    --epochs 1 \
    --patience 1 \
    >runs/cluster_logs/s4-linkedin-platform-then-audience.log 2>&1

train_variant() {
  local gpu=$1
  local variant=$2
  local job="s4-reverse-prior-lora-${variant}"
  CUDA_VISIBLE_DEVICES="$gpu" VECTORIAL_CKPT_DIR="runs/checkpoints/$job" \
    "$PYTHON_BIN" -u src/vectorial_eval/methods/lora/train.py \
      --variant "$variant" \
      --config attn-r8 \
      --data-dir data/study3_reverse/training \
      --initial-adapter "$PRIMARY" \
      --epochs 12 \
      --patience 3 \
      >"runs/cluster_logs/$job.log" 2>&1
}

train_variant "$GPU_A" target_lm &
PID_TARGET=$!
train_variant "$GPU_B" target_lm_paired &
PID_PAIRED=$!
wait "$PID_TARGET"
train_variant "$GPU_A" target_lm_aspect &
PID_ASPECT=$!
wait "$PID_PAIRED"
wait "$PID_ASPECT"

train_soft_prompt() {
  local gpu=$1
  local variant=$2
  local job="s4-reverse-prior-soft-prompt-${variant}"
  CUDA_VISIBLE_DEVICES="$gpu" VECTORIAL_CKPT_DIR="runs/checkpoints/$job" \
    "$PYTHON_BIN" -u src/vectorial_eval/methods/soft_prompt/train.py \
      --variant "$variant" \
      --data-dir data/study3_reverse/training \
      --initial-adapter "$PRIMARY" \
      --n-virtual-tokens 32 \
      --lr 1e-2 \
      --init text \
      --epochs 20 \
      --patience 4 \
      >"runs/cluster_logs/$job.log" 2>&1
}

train_soft_prompt "$GPU_A" target_lm &
PID_SOFT_TARGET=$!
train_soft_prompt "$GPU_B" target_lm_paired &
PID_SOFT_PAIRED=$!
wait "$PID_SOFT_TARGET"
train_soft_prompt "$GPU_A" target_lm_aspect &
PID_SOFT_ASPECT=$!
wait "$PID_SOFT_PAIRED"
wait "$PID_SOFT_ASPECT"

mkdir -p runs/study4_reverse_steering
fit_steering() {
  local gpu=$1
  local variant=$2
  CUDA_VISIBLE_DEVICES="$gpu" VECTORIAL_CKPT_DIR="runs/checkpoints/s4-reverse-steering-fit-${variant}" \
    "$PYTHON_BIN" -u src/vectorial_eval/methods/steering/fit_steering.py \
      --variant "$variant" \
      --data-dir data/study3_reverse \
      --initial-adapter "$PRIMARY" \
      --out "runs/study4_reverse_steering/steering.${variant}.pt" \
      >"runs/cluster_logs/s4-reverse-steering-fit-${variant}.log" 2>&1
}

fit_steering "$GPU_A" target_lm &
PID_STEER_TARGET=$!
fit_steering "$GPU_B" target_lm_paired &
PID_STEER_PAIRED=$!
wait "$PID_STEER_TARGET"
fit_steering "$GPU_A" target_lm_aspect &
PID_STEER_ASPECT=$!
wait "$PID_STEER_PAIRED"
wait "$PID_STEER_ASPECT"

sweep_steering() {
  local gpu=$1
  local variant=$2
  CUDA_VISIBLE_DEVICES="$gpu" VECTORIAL_CKPT_DIR="runs/checkpoints/s4-reverse-steering-sweep-${variant}" \
    "$PYTHON_BIN" -u src/vectorial_eval/methods/steering/sweep_steering.py \
      --artifact "runs/study4_reverse_steering/steering.${variant}.pt" \
      --data-dir data/study3_reverse \
      --limit 16 \
      --scope all \
      --out "runs/study4_reverse_steering/sweep.${variant}.json" \
      >"runs/cluster_logs/s4-reverse-steering-sweep-${variant}.log" 2>&1
}

sweep_steering "$GPU_A" target_lm &
PID_SWEEP_TARGET=$!
sweep_steering "$GPU_B" target_lm_paired &
PID_SWEEP_PAIRED=$!
wait "$PID_SWEEP_TARGET"
sweep_steering "$GPU_A" target_lm_aspect &
PID_SWEEP_ASPECT=$!
wait "$PID_SWEEP_PAIRED"
wait "$PID_SWEEP_ASPECT"

RUN_DIR="runs/study4_reverse"
DATA_DIR="data/study3_reverse"
mkdir -p "$RUN_DIR"
"$PYTHON_BIN" -m vectorial_eval.cli --data-dir "$DATA_DIR" --run-dir "$RUN_DIR" \
  transfer --fn identity target_sample shuffle_control --split heldout --n-samples 4 \
  >runs/cluster_logs/s4-reverse-references.log 2>&1

"$PYTHON_BIN" -m vectorial_eval.cli --data-dir "$DATA_DIR" --run-dir "$RUN_DIR" \
  transfer --fn llm_rewrite --tag s4_claude_reference --split heldout --n-samples 4 \
  >runs/cluster_logs/s4-reverse-claude-reference.log 2>&1

generate_lora() {
  local gpu=$1
  local tag=$2
  local adapter=$3
  local shuffled=${4:-no}
  if [[ ! -s "$adapter/adapter_config.json" ]]; then
    echo "null adaptation, no selected adapter: $adapter" \
      >>runs/cluster_logs/s4-reverse-null-adapters.log
    return 0
  fi
  local extra=()
  if [[ "$shuffled" == "yes" ]]; then extra+=(--audience-shuffle); fi
  CUDA_VISIBLE_DEVICES="$gpu" VECTORIAL_LORA_ADAPTER="$adapter" \
    "$PYTHON_BIN" -m vectorial_eval.cli --data-dir "$DATA_DIR" --run-dir "$RUN_DIR" \
      transfer --fn lora --tag "$tag" --split heldout --n-samples 4 "${extra[@]}" \
      >"runs/cluster_logs/s4-reverse-generate-${tag}-${shuffled}.log" 2>&1
}

generate_soft_prompt() {
  local gpu=$1
  local tag=$2
  local adapter=$3
  local shuffled=${4:-no}
  local extra=()
  if [[ "$shuffled" == "yes" ]]; then extra+=(--audience-shuffle); fi
  CUDA_VISIBLE_DEVICES="$gpu" VECTORIAL_SOFT_PROMPT_DIR="$adapter" \
    "$PYTHON_BIN" -m vectorial_eval.cli --data-dir "$DATA_DIR" --run-dir "$RUN_DIR" \
      transfer --fn soft_prompt --tag "$tag" --split heldout --n-samples 4 "${extra[@]}" \
      >"runs/cluster_logs/s4-reverse-generate-${tag}-${shuffled}.log" 2>&1
}

generate_steering() {
  local gpu=$1
  local tag=$2
  local variant=$3
  local artifact=$4
  local shuffled=${5:-no}
  local extra=()
  if [[ "$shuffled" == "yes" ]]; then extra+=(--audience-shuffle); fi
  CUDA_VISIBLE_DEVICES="$gpu" \
    VECTORIAL_STEERING_VARIANT="$variant" \
    VECTORIAL_STEERING_ARTIFACT="$artifact" \
    "$PYTHON_BIN" -m vectorial_eval.cli --data-dir "$DATA_DIR" --run-dir "$RUN_DIR" \
      transfer --fn steering --tag "$tag" --split heldout --n-samples 4 "${extra[@]}" \
      >"runs/cluster_logs/s4-reverse-generate-${tag}-${shuffled}.log" 2>&1
}

generate_local_plan() {
  local gpu=$1
  local tag=$2
  local adapter=$3
  local shuffled=${4:-no}
  local extra=()
  if [[ "$shuffled" == "yes" ]]; then extra+=(--audience-shuffle); fi
  CUDA_VISIBLE_DEVICES="$gpu" VECTORIAL_LORA_ADAPTER="$adapter" \
    "$PYTHON_BIN" -m vectorial_eval.cli --data-dir "$DATA_DIR" --run-dir "$RUN_DIR" \
      transfer --fn local_plan_then_transfer --tag "$tag" --split heldout --n-samples 4 "${extra[@]}" \
      >"runs/cluster_logs/s4-reverse-generate-${tag}-${shuffled}.log" 2>&1
}

chain_zero() {
  CUDA_VISIBLE_DEVICES="$GPU_A" "$PYTHON_BIN" -m vectorial_eval.cli \
    --data-dir "$DATA_DIR" --run-dir "$RUN_DIR" transfer \
    --fn llama_rewrite --tag s4_base_none --split heldout --n-samples 4 \
    >runs/cluster_logs/s4-reverse-generate-base-none.log 2>&1
  generate_lora "$GPU_A" s4_raw_full runs/checkpoints/s4-linkedin-rich-platform_lm-10000/best
  generate_lora "$GPU_A" s4_raw_full runs/checkpoints/s4-linkedin-rich-platform_lm-10000/best yes
  generate_lora "$GPU_A" s4_prior_full "$PRIMARY"
  generate_lora "$GPU_A" s4_prior_full "$PRIMARY" yes
  generate_local_plan "$GPU_A" s4_plan_prior_full "$PRIMARY"
  generate_local_plan "$GPU_A" s4_plan_prior_full "$PRIMARY" yes
  generate_lora "$GPU_A" s4_platform_then_audience runs/checkpoints/s4-linkedin-platform-then-audience/best
  generate_lora "$GPU_A" s4_platform_then_audience runs/checkpoints/s4-linkedin-platform-then-audience/best yes
  generate_lora "$GPU_A" s4_target_lm runs/checkpoints/s4-reverse-prior-lora-target_lm/best
  generate_lora "$GPU_A" s4_target_lm runs/checkpoints/s4-reverse-prior-lora-target_lm/best yes
  generate_lora "$GPU_A" s4_base_lora_target_lm runs/checkpoints/s3r_lora_target_lm/best
  generate_lora "$GPU_A" s4_base_lora_target_lm runs/checkpoints/s3r_lora_target_lm/best yes
  generate_lora "$GPU_A" s4_target_lm_aspect runs/checkpoints/s4-reverse-prior-lora-target_lm_aspect/best
  generate_lora "$GPU_A" s4_target_lm_aspect runs/checkpoints/s4-reverse-prior-lora-target_lm_aspect/best yes
  generate_lora "$GPU_A" s4_base_lora_target_lm_aspect runs/checkpoints/s3r_lora_target_lm_aspect/best
  generate_lora "$GPU_A" s4_base_lora_target_lm_aspect runs/checkpoints/s3r_lora_target_lm_aspect/best yes
  generate_soft_prompt "$GPU_A" s4_soft_target_lm runs/checkpoints/s4-reverse-prior-soft-prompt-target_lm/best
  generate_soft_prompt "$GPU_A" s4_soft_target_lm runs/checkpoints/s4-reverse-prior-soft-prompt-target_lm/best yes
  generate_soft_prompt "$GPU_A" s4_base_soft_target_lm runs/checkpoints/s3r_sp_target_lm/best
  generate_soft_prompt "$GPU_A" s4_base_soft_target_lm runs/checkpoints/s3r_sp_target_lm/best yes
  generate_soft_prompt "$GPU_A" s4_base_soft_target_lm_aspect runs/checkpoints/s3r_sp_target_lm_aspect/best
  generate_soft_prompt "$GPU_A" s4_base_soft_target_lm_aspect runs/checkpoints/s3r_sp_target_lm_aspect/best yes
  generate_steering "$GPU_A" s4_steer_base_target_lm target_lm runs/study3_reverse/steering/steering.target_lm.pt
  generate_steering "$GPU_A" s4_steer_prior_target_lm target_lm runs/study4_reverse_steering/steering.target_lm.pt
  generate_steering "$GPU_A" s4_steer_prior_target_lm target_lm runs/study4_reverse_steering/steering.target_lm.pt yes
  generate_steering "$GPU_A" s4_steer_base_target_lm_aspect target_lm_aspect runs/study3_reverse/steering/steering.target_lm_aspect.pt
  generate_steering "$GPU_A" s4_steer_prior_target_lm_aspect target_lm_aspect runs/study4_reverse_steering/steering.target_lm_aspect.pt
  generate_steering "$GPU_A" s4_steer_prior_target_lm_aspect target_lm_aspect runs/study4_reverse_steering/steering.target_lm_aspect.pt yes
}

chain_one() {
  generate_lora "$GPU_B" s4_raw_1k runs/checkpoints/s4-linkedin-rich-platform_lm-1000/best
  generate_lora "$GPU_B" s4_prior_1k runs/checkpoints/s4-linkedin-rich-platform_prior_lm-1000/best
  generate_lora "$GPU_B" s4_target_lm_paired runs/checkpoints/s4-reverse-prior-lora-target_lm_paired/best
  generate_lora "$GPU_B" s4_target_lm_paired runs/checkpoints/s4-reverse-prior-lora-target_lm_paired/best yes
  generate_lora "$GPU_B" s4_base_lora_target_lm_paired runs/checkpoints/s3r_lora_target_lm_paired/best
  generate_lora "$GPU_B" s4_base_lora_target_lm_paired runs/checkpoints/s3r_lora_target_lm_paired/best yes
  generate_soft_prompt "$GPU_B" s4_soft_target_lm_paired runs/checkpoints/s4-reverse-prior-soft-prompt-target_lm_paired/best
  generate_soft_prompt "$GPU_B" s4_soft_target_lm_paired runs/checkpoints/s4-reverse-prior-soft-prompt-target_lm_paired/best yes
  generate_soft_prompt "$GPU_B" s4_base_soft_target_lm_paired runs/checkpoints/s3r_sp_target_lm_paired/best
  generate_soft_prompt "$GPU_B" s4_base_soft_target_lm_paired runs/checkpoints/s3r_sp_target_lm_paired/best yes
  generate_soft_prompt "$GPU_B" s4_soft_target_lm_aspect runs/checkpoints/s4-reverse-prior-soft-prompt-target_lm_aspect/best
  generate_soft_prompt "$GPU_B" s4_soft_target_lm_aspect runs/checkpoints/s4-reverse-prior-soft-prompt-target_lm_aspect/best yes
  generate_steering "$GPU_B" s4_steer_base_target_lm_paired target_lm_paired runs/study3_reverse/steering/steering.target_lm_paired.pt
  generate_steering "$GPU_B" s4_steer_prior_target_lm_paired target_lm_paired runs/study4_reverse_steering/steering.target_lm_paired.pt
  generate_steering "$GPU_B" s4_steer_prior_target_lm_paired target_lm_paired runs/study4_reverse_steering/steering.target_lm_paired.pt yes
}

chain_zero &
PID_ZERO=$!
chain_one &
PID_ONE=$!
wait "$PID_ZERO"
wait "$PID_ONE"

REQUIRE_ARGS=()
while IFS= read -r system; do
  REQUIRE_ARGS+=(--require "$system")
done < <(
  "$PYTHON_BIN" - <<'PY'
from scripts.audit_study4_completion import REVERSE_SYSTEMS

print("\n".join(sorted(REVERSE_SYSTEMS)))
PY
)
"$PYTHON_BIN" scripts/audit_study4_outputs.py \
  "$RUN_DIR" "$DATA_DIR/tasks.heldout.jsonl" \
  --split heldout --n-samples 4 \
  "${REQUIRE_ARGS[@]}" \
  >runs/cluster_logs/s4-audit-reverse-outputs.log 2>&1

CUDA_VISIBLE_DEVICES="$GPU_A" "$PYTHON_BIN" -m vectorial_eval.cli \
  --data-dir "$DATA_DIR" --run-dir "$RUN_DIR" evaluate \
  --split heldout \
  --embedding-model google/embeddinggemma-300m \
  --embedding-dim 512 \
  --bootstrap-unit audience-cell \
  >runs/cluster_logs/s4-evaluate-reverse.log 2>&1

for dim in 128 256 768; do
  CUDA_VISIBLE_DEVICES="$GPU_A" "$PYTHON_BIN" -m vectorial_eval.cli \
    --data-dir "$DATA_DIR" --run-dir "$RUN_DIR" evaluate \
    --split heldout \
    --embedding-model google/embeddinggemma-300m \
    --embedding-dim "$dim" \
    --bootstrap-unit audience-cell \
    >"runs/cluster_logs/s4-evaluate-reverse-embeddinggemma-${dim}.log" 2>&1
done
CUDA_VISIBLE_DEVICES="$GPU_A" "$PYTHON_BIN" -m vectorial_eval.cli \
  --data-dir "$DATA_DIR" --run-dir "$RUN_DIR" evaluate \
  --split heldout \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --embedding-dim 384 \
  --bootstrap-unit audience-cell \
  >runs/cluster_logs/s4-evaluate-reverse-minilm.log 2>&1
"$PYTHON_BIN" -m vectorial_eval.cli \
  --data-dir "$DATA_DIR" --run-dir "$RUN_DIR" evaluate \
  --split heldout \
  --bootstrap-unit audience-cell \
  >runs/cluster_logs/s4-evaluate-reverse-tfidf.log 2>&1

"$PYTHON_BIN" scripts/analyze_study4_spaces.py \
  "$RUN_DIR" \
  "$DATA_DIR/cells.jsonl" \
  --split heldout \
  --comparison llama_rewrite_s4_base_none:lora_s4_prior_full \
  --comparison llm_rewrite_s4_claude_reference:llama_rewrite_s4_base_none \
  --comparison lora_s4_prior_full:local_plan_then_transfer_s4_plan_prior_full \
  --comparison lora_s4_raw_full:lora_s4_prior_full \
  --comparison lora_s4_raw_1k:lora_s4_prior_1k \
  --comparison lora_s4_raw_full:lora_s4_platform_then_audience \
  --comparison lora_s4_prior_full:lora_s4_platform_then_audience \
  --comparison lora_s4_prior_full_audience_shuffle:lora_s4_prior_full \
  --comparison local_plan_then_transfer_s4_plan_prior_full_audience_shuffle:local_plan_then_transfer_s4_plan_prior_full \
  --comparison lora_s4_prior_1k:lora_s4_prior_full \
  --comparison lora_s4_prior_full:lora_s4_target_lm \
  --comparison lora_s4_base_lora_target_lm:lora_s4_target_lm \
  --comparison lora_s4_base_lora_target_lm_paired:lora_s4_target_lm_paired \
  --comparison lora_s4_base_lora_target_lm_aspect:lora_s4_target_lm_aspect \
  --comparison lora_s4_target_lm_audience_shuffle:lora_s4_target_lm \
  --comparison lora_s4_target_lm_paired_audience_shuffle:lora_s4_target_lm_paired \
  --comparison lora_s4_target_lm_aspect_audience_shuffle:lora_s4_target_lm_aspect \
  --comparison lora_s4_prior_full:lora_s4_target_lm_paired \
  --comparison lora_s4_prior_full:lora_s4_target_lm_aspect \
  --comparison lora_s4_prior_full:soft_prompt_s4_soft_target_lm \
  --comparison soft_prompt_s4_base_soft_target_lm:soft_prompt_s4_soft_target_lm \
  --comparison soft_prompt_s4_base_soft_target_lm_paired:soft_prompt_s4_soft_target_lm_paired \
  --comparison soft_prompt_s4_base_soft_target_lm_aspect:soft_prompt_s4_soft_target_lm_aspect \
  --comparison soft_prompt_s4_soft_target_lm_audience_shuffle:soft_prompt_s4_soft_target_lm \
  --comparison soft_prompt_s4_soft_target_lm_paired_audience_shuffle:soft_prompt_s4_soft_target_lm_paired \
  --comparison soft_prompt_s4_soft_target_lm_aspect_audience_shuffle:soft_prompt_s4_soft_target_lm_aspect \
  --comparison lora_s4_prior_full:soft_prompt_s4_soft_target_lm_paired \
  --comparison lora_s4_prior_full:soft_prompt_s4_soft_target_lm_aspect \
  --comparison steering_s4_steer_base_target_lm:steering_s4_steer_prior_target_lm \
  --comparison steering_s4_steer_base_target_lm:lora_s4_target_lm \
  --comparison steering_s4_steer_base_target_lm_paired:steering_s4_steer_prior_target_lm_paired \
  --comparison steering_s4_steer_base_target_lm_aspect:steering_s4_steer_prior_target_lm_aspect \
  --comparison steering_s4_steer_prior_target_lm_audience_shuffle:steering_s4_steer_prior_target_lm \
  --comparison steering_s4_steer_prior_target_lm_paired_audience_shuffle:steering_s4_steer_prior_target_lm_paired \
  --comparison steering_s4_steer_prior_target_lm_aspect_audience_shuffle:steering_s4_steer_prior_target_lm_aspect

"$PYTHON_BIN" scripts/audit_study4_completion.py \
  >runs/cluster_logs/s4-completion-audit.log 2>&1

echo "Study 4 reverse generation and primary evaluation finished"
