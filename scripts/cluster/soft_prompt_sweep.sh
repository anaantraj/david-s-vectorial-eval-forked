#!/usr/bin/env bash
# Launch the soft-prompt sweep on the cluster, one job per free GPU.
#
#   scripts/cluster/soft_prompt_sweep.sh [--dry-run]
#
# Ten jobs. Eight of them are the sweep over the three hyperparameters that
# matter for prompt tuning, run on the `target_lm` variant:
#
#   virtual tokens   16, 64        capacity, against 337 training examples
#   learning rate    3e-3, 3e-2    the range prompt tuning is reported to need
#   initialisation   text, sampled_vocab
#
# The remaining two carry the middle configuration onto the other two training
# variants, so that every variant has a result without waiting for the sweep.
# Once `select_best.py` names a winner, re-launch those two at the winning
# configuration; a number is only interpretable alongside its variant.
#
# Each job is a few hundred optimiser steps over 337 examples and takes well
# under an hour on one card. Relaunching this script is safe: launch.sh refuses
# a job that is still running and a job that was killed resumes from its last
# checkpoint.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAUNCH="$ROOT/scripts/cluster/launch.sh"
SCRIPT="src/vectorial_eval/methods/soft_prompt/train.py"
DRY=""
[ "${1:-}" = "--dry-run" ] && DRY="--dry-run"

launch_one() {  # variant n_tokens lr init
  local variant="$1" n="$2" lr="$3" init="$4"
  local job="sp_${variant}_n${n}_lr${lr}_${init}"
  echo "=== $job"
  "$LAUNCH" -j "$job" $DRY -- "$SCRIPT" \
    --variant "$variant" --n-virtual-tokens "$n" --lr "$lr" --init "$init" \
    || echo "!!! launch failed for $job"
}

for n in 16 64; do
  for lr in 3e-3 3e-2; do
    for init in text sampled_vocab; do
      launch_one target_lm "$n" "$lr" "$init"
    done
  done
done

# The other two variants at the middle configuration.
launch_one target_lm_paired 32 1e-2 text
launch_one target_lm_aspect 32 1e-2 text

echo
echo "poll with: scripts/cluster/status.sh"
echo "select with: .venv/bin/python src/vectorial_eval/methods/soft_prompt/select_best.py --remote"
