#!/usr/bin/env bash
# Generate and score the frozen forward Study 4 case-study grid.
set -euo pipefail

PROJECT_ROOT="${VECTORIAL_PROJECT_ROOT:-$HOME/Projects/vectorial}"
PYTHON_BIN="${VECTORIAL_PYTHON:-$HOME/micromamba/envs/vectorial/bin/python}"
cd "$PROJECT_ROOT"
mkdir -p runs/cluster_logs
FAILURE_MARKER="runs/cluster_logs/s4-generation.failed"
rm -f "$FAILURE_MARKER"
record_failure() {
  local status=$?
  if (( status != 0 )); then
    printf '%s\n' "$status" >"$FAILURE_MARKER"
  fi
}
trap record_failure EXIT

stage_b_alive() {
  if [[ -s runs/cluster_logs/s4-stage-b.failed ]]; then
    echo "Stage B failed before generation prerequisites completed" >&2
    return 1
  fi
}

finished() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1]) / "metrics.json"
raise SystemExit(0 if path.exists() and json.load(path.open()).get("finished") else 1)
PY
}

soft_finished() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json, math, pathlib, sys

root = pathlib.Path(sys.argv[1])
path = root / "summary.json"
if not path.exists() or not (root / "best" / "adapter_config.json").exists():
    raise SystemExit(1)
summary = json.load(path.open())
# Soft-prompt training writes its summary only after the training loop has
# completed. Unlike LoRA metrics.json, this established artifact has no
# redundant `finished` field. Require the final selection fields and a finite
# held-out loss instead of waiting forever for a field the trainer never emits.
complete = (
    isinstance(summary.get("best_epoch"), int)
    and math.isfinite(summary.get("best_dev_loss", math.nan))
    and bool(summary.get("history"))
)
raise SystemExit(0 if complete else 1)
PY
}

REQUIRED_RUNS=(
  s4-reddit-rich-raw-1k
  s4-reddit-rich-prior-1k
  s4-reddit-platform-then-audience
  s4-prior10k-lora-target_lm
  s4-prior10k-lora-target_lm_paired
  s4-prior10k-lora-target_lm_aspect
)
for run in "${REQUIRED_RUNS[@]}"; do
  until finished "runs/checkpoints/$run"; do stage_b_alive; sleep 30; done
done
for run in \
  s4-prior10k-soft-prompt-target_lm \
  s4-prior10k-soft-prompt-target_lm_paired \
  s4-prior10k-soft-prompt-target_lm_aspect; do
  until soft_finished "runs/checkpoints/$run"; do stage_b_alive; sleep 30; done
done
for variant in target_lm target_lm_paired target_lm_aspect; do
  until [[ -s "runs/study4_steering/chosen.steering.${variant}.json" ]]; do
    stage_b_alive
    sleep 30
  done
done
until [[ -s "runs/study4_steering/chosen.steering.lora_interaction.json" ]]; do
  stage_b_alive
  sleep 30
done

export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/scripts/cluster"
RUN_DIR="runs/study4_forward"
DATA_DIR="data/study3"
mkdir -p "$RUN_DIR"

"$PYTHON_BIN" -m vectorial_eval.cli --data-dir "$DATA_DIR" --run-dir "$RUN_DIR" \
  transfer --fn identity target_sample shuffle_control --split heldout --n-samples 4 \
  >runs/cluster_logs/s4-generate-references.log 2>&1

"$PYTHON_BIN" -m vectorial_eval.cli --data-dir "$DATA_DIR" --run-dir "$RUN_DIR" \
  transfer --fn llm_rewrite --tag s4_claude_reference --split heldout --n-samples 4 \
  >runs/cluster_logs/s4-generate-claude-reference.log 2>&1

generate_lora() {
  local gpu=$1
  local tag=$2
  local adapter=$3
  local shuffled=${4:-no}
  if [[ ! -s "$adapter/adapter_config.json" ]]; then
    echo "null adaptation, no selected adapter: $adapter" \
      >>runs/cluster_logs/s4-generation-null-adapters.log
    return 0
  fi
  local extra=()
  if [[ "$shuffled" == "yes" ]]; then extra+=(--audience-shuffle); fi
  CUDA_VISIBLE_DEVICES="$gpu" VECTORIAL_LORA_ADAPTER="$adapter" \
    "$PYTHON_BIN" -m vectorial_eval.cli --data-dir "$DATA_DIR" --run-dir "$RUN_DIR" \
      transfer --fn lora --tag "$tag" --split heldout --n-samples 4 "${extra[@]}" \
      >"runs/cluster_logs/s4-generate-${tag}-${shuffled}.log" 2>&1
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
      >"runs/cluster_logs/s4-generate-${tag}-${shuffled}.log" 2>&1
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
      >"runs/cluster_logs/s4-generate-${tag}-${shuffled}.log" 2>&1
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
      >"runs/cluster_logs/s4-generate-${tag}-${shuffled}.log" 2>&1
}

chain_zero() {
  CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" -m vectorial_eval.cli \
    --data-dir "$DATA_DIR" --run-dir "$RUN_DIR" transfer \
    --fn llama_rewrite --tag s4_base_none --split heldout --n-samples 4 \
    >runs/cluster_logs/s4-generate-base-none.log 2>&1
  generate_lora 0 s4_raw_10k runs/checkpoints/s4-reddit-rich-raw-10k/best
  generate_lora 0 s4_raw_10k runs/checkpoints/s4-reddit-rich-raw-10k/best yes
  generate_lora 0 s4_prior_10k runs/checkpoints/s4-reddit-rich-prior-10k/best
  generate_lora 0 s4_prior_10k runs/checkpoints/s4-reddit-rich-prior-10k/best yes
  generate_local_plan 0 s4_plan_prior_10k runs/checkpoints/s4-reddit-rich-prior-10k/best
  generate_local_plan 0 s4_plan_prior_10k runs/checkpoints/s4-reddit-rich-prior-10k/best yes
  generate_lora 0 s4_platform_then_audience runs/checkpoints/s4-reddit-platform-then-audience/best
  generate_lora 0 s4_platform_then_audience runs/checkpoints/s4-reddit-platform-then-audience/best yes
  generate_lora 0 s4_target_lm runs/checkpoints/s4-prior10k-lora-target_lm/best
  generate_lora 0 s4_target_lm runs/checkpoints/s4-prior10k-lora-target_lm/best yes
  generate_lora 0 s4_base_lora_target_lm runs/checkpoints/s3_lora_target_lm/best
  generate_lora 0 s4_base_lora_target_lm runs/checkpoints/s3_lora_target_lm/best yes
  generate_lora 0 s4_target_lm_aspect runs/checkpoints/s4-prior10k-lora-target_lm_aspect/best
  generate_lora 0 s4_target_lm_aspect runs/checkpoints/s4-prior10k-lora-target_lm_aspect/best yes
  generate_lora 0 s4_base_lora_target_lm_aspect runs/checkpoints/s3_lora_target_lm_aspect/best
  generate_lora 0 s4_base_lora_target_lm_aspect runs/checkpoints/s3_lora_target_lm_aspect/best yes
  generate_soft_prompt 0 s4_soft_target_lm runs/checkpoints/s4-prior10k-soft-prompt-target_lm/best
  generate_soft_prompt 0 s4_soft_target_lm runs/checkpoints/s4-prior10k-soft-prompt-target_lm/best yes
  generate_soft_prompt 0 s4_base_soft_target_lm runs/checkpoints/s3_sp_target_lm/best
  generate_soft_prompt 0 s4_base_soft_target_lm runs/checkpoints/s3_sp_target_lm/best yes
  generate_soft_prompt 0 s4_base_soft_target_lm_aspect runs/checkpoints/s3_sp_target_lm_aspect/best
  generate_soft_prompt 0 s4_base_soft_target_lm_aspect runs/checkpoints/s3_sp_target_lm_aspect/best yes
  generate_steering 0 s4_steer_base_target_lm target_lm runs/study3/steering/steering.target_lm.pt
  generate_steering 0 s4_steer_prior_target_lm target_lm runs/study4_steering/steering.target_lm.pt
  generate_steering 0 s4_steer_prior_target_lm target_lm runs/study4_steering/steering.target_lm.pt yes
  generate_steering 0 s4_steer_base_target_lm_aspect target_lm_aspect runs/study3/steering/steering.target_lm_aspect.pt
  generate_steering 0 s4_steer_prior_target_lm_aspect target_lm_aspect runs/study4_steering/steering.target_lm_aspect.pt
  generate_steering 0 s4_steer_prior_target_lm_aspect target_lm_aspect runs/study4_steering/steering.target_lm_aspect.pt yes
  generate_steering 0 s4_lora_steer_interaction target_lm runs/study4_steering/steering.lora_interaction.pt
}

chain_one() {
  generate_lora 1 s4_raw_1k runs/checkpoints/s4-reddit-rich-raw-1k/best
  generate_lora 1 s4_prior_1k runs/checkpoints/s4-reddit-rich-prior-1k/best
  generate_lora 1 s4_target_lm_paired runs/checkpoints/s4-prior10k-lora-target_lm_paired/best
  generate_lora 1 s4_target_lm_paired runs/checkpoints/s4-prior10k-lora-target_lm_paired/best yes
  generate_lora 1 s4_base_lora_target_lm_paired runs/checkpoints/s3_lora_target_lm_paired/best
  generate_lora 1 s4_base_lora_target_lm_paired runs/checkpoints/s3_lora_target_lm_paired/best yes
  generate_soft_prompt 1 s4_soft_target_lm_paired runs/checkpoints/s4-prior10k-soft-prompt-target_lm_paired/best
  generate_soft_prompt 1 s4_soft_target_lm_paired runs/checkpoints/s4-prior10k-soft-prompt-target_lm_paired/best yes
  generate_soft_prompt 1 s4_base_soft_target_lm_paired runs/checkpoints/s3_sp_target_lm_paired/best
  generate_soft_prompt 1 s4_base_soft_target_lm_paired runs/checkpoints/s3_sp_target_lm_paired/best yes
  generate_soft_prompt 1 s4_soft_target_lm_aspect runs/checkpoints/s4-prior10k-soft-prompt-target_lm_aspect/best
  generate_soft_prompt 1 s4_soft_target_lm_aspect runs/checkpoints/s4-prior10k-soft-prompt-target_lm_aspect/best yes
  generate_steering 1 s4_steer_base_target_lm_paired target_lm_paired runs/study3/steering/steering.target_lm_paired.pt
  generate_steering 1 s4_steer_prior_target_lm_paired target_lm_paired runs/study4_steering/steering.target_lm_paired.pt
  generate_steering 1 s4_steer_prior_target_lm_paired target_lm_paired runs/study4_steering/steering.target_lm_paired.pt yes
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
from scripts.audit_study4_completion import FORWARD_SYSTEMS

print("\n".join(sorted(FORWARD_SYSTEMS)))
PY
)
"$PYTHON_BIN" scripts/audit_study4_outputs.py \
  "$RUN_DIR" "$DATA_DIR/tasks.heldout.jsonl" \
  --split heldout --n-samples 4 \
  "${REQUIRE_ARGS[@]}" \
  >runs/cluster_logs/s4-audit-forward-outputs.log 2>&1

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" -m vectorial_eval.cli \
  --data-dir "$DATA_DIR" --run-dir "$RUN_DIR" evaluate \
  --split heldout \
  --embedding-model google/embeddinggemma-300m \
  --embedding-dim 512 \
  --bootstrap-unit audience-cell \
  >runs/cluster_logs/s4-evaluate-forward.log 2>&1

for dim in 128 256 768; do
  CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" -m vectorial_eval.cli \
    --data-dir "$DATA_DIR" --run-dir "$RUN_DIR" evaluate \
    --split heldout \
    --embedding-model google/embeddinggemma-300m \
    --embedding-dim "$dim" \
    --bootstrap-unit audience-cell \
    >"runs/cluster_logs/s4-evaluate-forward-embeddinggemma-${dim}.log" 2>&1
done
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" -m vectorial_eval.cli \
  --data-dir "$DATA_DIR" --run-dir "$RUN_DIR" evaluate \
  --split heldout \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --embedding-dim 384 \
  --bootstrap-unit audience-cell \
  >runs/cluster_logs/s4-evaluate-forward-minilm.log 2>&1
"$PYTHON_BIN" -m vectorial_eval.cli \
  --data-dir "$DATA_DIR" --run-dir "$RUN_DIR" evaluate \
  --split heldout \
  --bootstrap-unit audience-cell \
  >runs/cluster_logs/s4-evaluate-forward-tfidf.log 2>&1

"$PYTHON_BIN" scripts/analyze_study4_spaces.py \
  "$RUN_DIR" \
  "$DATA_DIR/cells.jsonl" \
  --split heldout \
  --comparison llama_rewrite_s4_base_none:lora_s4_prior_10k \
  --comparison llm_rewrite_s4_claude_reference:llama_rewrite_s4_base_none \
  --comparison lora_s4_prior_10k:local_plan_then_transfer_s4_plan_prior_10k \
  --comparison lora_s4_target_lm:steering_s4_lora_steer_interaction \
  --comparison lora_s4_raw_10k:lora_s4_prior_10k \
  --comparison lora_s4_raw_1k:lora_s4_prior_1k \
  --comparison lora_s4_raw_10k:lora_s4_platform_then_audience \
  --comparison lora_s4_prior_10k:lora_s4_platform_then_audience \
  --comparison lora_s4_prior_10k_audience_shuffle:lora_s4_prior_10k \
  --comparison local_plan_then_transfer_s4_plan_prior_10k_audience_shuffle:local_plan_then_transfer_s4_plan_prior_10k \
  --comparison lora_s4_prior_1k:lora_s4_prior_10k \
  --comparison lora_s4_prior_10k:lora_s4_target_lm \
  --comparison lora_s4_base_lora_target_lm:lora_s4_target_lm \
  --comparison lora_s4_base_lora_target_lm_paired:lora_s4_target_lm_paired \
  --comparison lora_s4_base_lora_target_lm_aspect:lora_s4_target_lm_aspect \
  --comparison lora_s4_target_lm_audience_shuffle:lora_s4_target_lm \
  --comparison lora_s4_target_lm_paired_audience_shuffle:lora_s4_target_lm_paired \
  --comparison lora_s4_target_lm_aspect_audience_shuffle:lora_s4_target_lm_aspect \
  --comparison lora_s4_prior_10k:lora_s4_target_lm_paired \
  --comparison lora_s4_prior_10k:lora_s4_target_lm_aspect \
  --comparison lora_s4_prior_10k:soft_prompt_s4_soft_target_lm \
  --comparison soft_prompt_s4_base_soft_target_lm:soft_prompt_s4_soft_target_lm \
  --comparison soft_prompt_s4_base_soft_target_lm_paired:soft_prompt_s4_soft_target_lm_paired \
  --comparison soft_prompt_s4_base_soft_target_lm_aspect:soft_prompt_s4_soft_target_lm_aspect \
  --comparison soft_prompt_s4_soft_target_lm_audience_shuffle:soft_prompt_s4_soft_target_lm \
  --comparison soft_prompt_s4_soft_target_lm_paired_audience_shuffle:soft_prompt_s4_soft_target_lm_paired \
  --comparison soft_prompt_s4_soft_target_lm_aspect_audience_shuffle:soft_prompt_s4_soft_target_lm_aspect \
  --comparison lora_s4_prior_10k:soft_prompt_s4_soft_target_lm_paired \
  --comparison lora_s4_prior_10k:soft_prompt_s4_soft_target_lm_aspect \
  --comparison steering_s4_steer_base_target_lm:steering_s4_steer_prior_target_lm \
  --comparison steering_s4_steer_base_target_lm:lora_s4_target_lm \
  --comparison steering_s4_steer_base_target_lm_paired:steering_s4_steer_prior_target_lm_paired \
  --comparison steering_s4_steer_base_target_lm_aspect:steering_s4_steer_prior_target_lm_aspect \
  --comparison steering_s4_steer_prior_target_lm_audience_shuffle:steering_s4_steer_prior_target_lm \
  --comparison steering_s4_steer_prior_target_lm_paired_audience_shuffle:steering_s4_steer_prior_target_lm_paired \
  --comparison steering_s4_steer_prior_target_lm_aspect_audience_shuffle:steering_s4_steer_prior_target_lm_aspect

echo "Study 4 forward generation and primary evaluation finished"
