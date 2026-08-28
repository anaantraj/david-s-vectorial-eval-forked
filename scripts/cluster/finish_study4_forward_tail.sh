#!/usr/bin/env bash
# Resume the Study 4 forward pipeline from the point where it died.
#
# The Aug 23 run completed generation, the output audit, and all four
# EmbeddingGemma evaluations, then aborted in the MiniLM evaluation because
# `sentence-transformers/all-MiniLM-L6-v2` was absent from the Hugging Face
# cache while the script exports HF_HUB_OFFLINE=1. Under `set -euo pipefail`
# that single failure took the TF-IDF evaluation and the frontier analysis with
# it, and the resulting s4-generation.failed marker then blocked the reverse
# replication.
#
# This script re-runs ONLY that unfinished tail. It deliberately does not call
# `transfer`: cmd_transfer opens each outputs.*.jsonl with mode "w" and always
# re-runs the generator, so invoking continue_study4_generation.sh again would
# discard 49 completed systems and repeat roughly 33 hours of GPU work.
#
#   scripts/cluster/finish_study4_forward_tail.sh [gpu]
#
# `gpu` is the card used for the MiniLM embedding pass (default 0). Pick a card
# with free memory at launch time, per docs/09-cluster-training-config.md.
set -euo pipefail

PROJECT_ROOT="${VECTORIAL_PROJECT_ROOT:-$HOME/Projects/vectorial}"
PYTHON_BIN="${VECTORIAL_PYTHON:-$HOME/micromamba/envs/vectorial/bin/python}"
GPU="${1:-0}"
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

export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/scripts/cluster"
RUN_DIR="runs/study4_forward"
DATA_DIR="data/study3"

# Fail early and legibly if the cache gap that stopped the Aug 23 run is still
# present, rather than after the evaluation has loaded the corpus.
"$PYTHON_BIN" - <<'PY'
from sentence_transformers import SentenceTransformer

SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
print("MiniLM resolves from the local cache under HF_HUB_OFFLINE=1")
PY

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -m vectorial_eval.cli \
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
