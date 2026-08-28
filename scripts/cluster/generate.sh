#!/usr/bin/env bash
# Generate outputs on a cluster GPU from a trained artifact, detached.
#
#   scripts/cluster/generate.sh --fn FN --tag TAG --artifact ENV=VALUE \
#                               [--host HOST] [--gpu N] [--split SPLIT] \
#                               [--run-dir DIR] [--n-samples K]
#
# Example:
#   scripts/cluster/generate.sh --fn lora --tag lora_target_lm \
#     --artifact VECTORIAL_LORA_ADAPTER=runs/checkpoints/lora-target_lm-all-r16/best
#
# `launch.sh` runs a training script; this runs `vectorial_eval.cli transfer`,
# which needs the package importable and the artifact selected by an environment
# variable. The two are separate because a generation pass has no checkpoint and
# no resume, and conflating them made the training launcher harder to read.
#
# Three things here are load-bearing and were each discovered by getting them
# wrong:
#
#   1. The remote body is a *quoted* heredoc. With an unquoted string, `$HOME`
#      expands on the local machine, so PYTHONPATH points at a path that exists
#      only on the laptop and every job dies with ModuleNotFoundError while the
#      launcher reports success.
#   2. PYTHONPATH must include `src/`. The package is not pip-installed in the
#      cluster environment, so `python -m vectorial_eval.cli` fails without it.
#   3. The artifact environment variable name must match what the transfer
#      function reads. `VECTORIAL_STEERING_ARTIFACT`, not `..._PATH`; passing the
#      wrong name is silent, and the function falls back to a default artifact.
#      Every generated run records the artifact it loaded and its sha256 in
#      `transfer_fn.<name>.json`; check that file before trusting a comparison.
set -euo pipefail

FN=""; TAG=""; ARTIFACT=""; HOST=""; GPU=""; SPLIT="test"; RUNDIR="runs/methods"; NS=4
DATADIR="data"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --fn) FN=$2; shift 2;;
    --tag) TAG=$2; shift 2;;
    --artifact) ARTIFACT=$2; shift 2;;
    --host) HOST=$2; shift 2;;
    --gpu) GPU=$2; shift 2;;
    --split) SPLIT=$2; shift 2;;
    --run-dir) RUNDIR=$2; shift 2;;
    --data-dir) DATADIR=$2; shift 2;;
    --n-samples) NS=$2; shift 2;;
    -h|--help) sed -n '2,30p' "$0"; exit 0;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
[[ -n "$FN" && -n "$TAG" ]] || { echo "--fn and --tag are required" >&2; exit 2; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ -z "$HOST" || -z "$GPU" ]]; then
  read -r PICK_HOST PICK_GPU < <("$ROOT/.venv/bin/python" "$ROOT/scripts/cluster/pick_gpu.py" --n 1 | head -1)
  HOST=${HOST:-$PICK_HOST}
  GPU=${GPU:-$PICK_GPU}
fi
[[ -n "$HOST" && -n "$GPU" ]] || { echo "no free GPU found" >&2; exit 1; }
[[ "$HOST" == *.* ]] || HOST="${HOST}.ist.berkeley.edu"

echo "==> ${FN}/${TAG} on ${HOST} gpu ${GPU} (split=${SPLIT}, ${NS} draws)"
ssh -o BatchMode=yes "$HOST" bash -s -- \
  "$FN" "$TAG" "$ARTIFACT" "$GPU" "$SPLIT" "$RUNDIR" "$NS" "$DATADIR" <<'REMOTE'
set -euo pipefail
FN=$1; TAG=$2; ARTIFACT=$3; GPU=$4; SPLIT=$5; RUNDIR=$6; NS=$7; DATADIR=$8
cd "$HOME/Projects/vectorial"
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$HOME/Projects/vectorial/src:$HOME/Projects/vectorial/scripts/cluster"
[ -n "$ARTIFACT" ] && export "$ARTIFACT"
mkdir -p runs/cluster_logs
LOG="runs/cluster_logs/gen_${FN}_${TAG}.log"
nohup setsid "$HOME/micromamba/envs/vectorial/bin/python" -m vectorial_eval.cli \
  --data-dir "$DATADIR" --run-dir "$RUNDIR" transfer --fn "$FN" --split "$SPLIT" \
  --n-samples "$NS" --tag "$TAG" > "$LOG" 2>&1 < /dev/null &
echo "    pid $! -> $LOG"
REMOTE
