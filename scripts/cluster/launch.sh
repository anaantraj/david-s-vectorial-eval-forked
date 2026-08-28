#!/usr/bin/env bash
# Launch a training job on a cluster GPU, detached, with a log and a PID file.
#
#   scripts/cluster/launch.sh --job NAME [options] -- <script.py> [args...]
#
# Options:
#   -j, --job NAME        job name; names the log, the PID file, and the default
#                         checkpoint directory
#   -H, --host HOST       node to run on (default: chosen by pick_gpu.py)
#   -g, --gpu N           GPU index (requires --host; default: chosen too)
#   -c, --ckpt-dir DIR    checkpoint directory relative to the repo root
#                         (default runs/checkpoints/<job>)
#       --allow-download  permit Hugging Face downloads. Off by default: every
#                         model this project uses is cached and the shared disk
#                         is at 91 percent.
#       --dry-run         print the remote script instead of running it
#
# The job runs under nohup and setsid with CUDA_VISIBLE_DEVICES pinned to the
# chosen GPU, HF_HOME pointing at the existing cache, and the micromamba
# interpreter. Output is appended to runs/cluster_logs/<job>.log, the PID is
# written to <job>.pid, and the placement is recorded in <job>.meta, which
# status.sh and pick_gpu.py read.
#
# Relaunching after a kill is the intended way to continue. The same command is
# re-issued and the training script resumes from its latest checkpoint through
# checkpointing.resume_or_start. The log is appended to rather than truncated, so
# every attempt stays in one file. A job whose process is still alive is refused,
# because two writers on one checkpoint directory would corrupt the run.
#
# The job's environment includes VECTORIAL_CKPT_DIR (absolute) and VECTORIAL_JOB,
# and PYTHONPATH covers both src/ and scripts/cluster/, so a training script can
# simply `from checkpointing import resume_or_start`.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PICK="$ROOT/scripts/cluster/pick_gpu.py"
LOCAL_PYTHON="$ROOT/.venv/bin/python"
REMOTE_ROOT="Projects/vectorial"
DOMAIN="ist.berkeley.edu"

JOB=""; HOST=""; GPU=""; CKPT_DIR=""; DRY=0; ALLOW_DOWNLOAD=0
while [ $# -gt 0 ]; do
  case "$1" in
    -j|--job)         JOB="$2"; shift 2 ;;
    -H|--host)        HOST="$2"; shift 2 ;;
    -g|--gpu)         GPU="$2"; shift 2 ;;
    -c|--ckpt-dir)    CKPT_DIR="$2"; shift 2 ;;
    --allow-download) ALLOW_DOWNLOAD=1; shift ;;
    --dry-run)        DRY=1; shift ;;
    -h|--help)        sed -n '2,32p' "${BASH_SOURCE[0]}"; exit 0 ;;
    --)               shift; break ;;
    -*)               echo "error: unknown option $1" >&2; exit 2 ;;
    *)                break ;;
  esac
done

[ -n "$JOB" ] || { echo "error: --job is required" >&2; exit 2; }
[ $# -gt 0 ] || { echo "error: no command given after --" >&2; exit 2; }
case "$JOB" in
  *[!A-Za-z0-9._-]*) echo "error: job name must match [A-Za-z0-9._-]+" >&2; exit 2 ;;
esac
: "${CKPT_DIR:=runs/checkpoints/$JOB}"
case "$HOST" in "") ;; *.*) ;; *) HOST="$HOST.$DOMAIN" ;; esac

# ---- placement ------------------------------------------------------------- #
if [ -z "$GPU" ]; then
  if [ -n "$HOST" ]; then PICK_ARGS=(--n 1 --host "$HOST"); else PICK_ARGS=(--n 1 --spread); fi
  read -r PICKED_HOST PICKED_GPU < <("$LOCAL_PYTHON" "$PICK" "${PICK_ARGS[@]}") || {
    echo "error: no free GPU available" >&2; exit 1; }
  HOST="$PICKED_HOST"; GPU="$PICKED_GPU"
elif [ -z "$HOST" ]; then
  echo "error: --gpu requires --host" >&2; exit 2
fi
echo "==> job '$JOB' -> $HOST gpu $GPU"

# Liveness test, used both here and after the launch. It checks the environment
# of the process rather than only its existence, so a recycled PID cannot be
# mistaken for a job that is still running.
ALIVE_SCRIPT=''
read -r -d '' ALIVE_SCRIPT <<'EOF' || true
pid="$1"; job="$2"
[ -r "/proc/$pid/environ" ] || exit 1
tr '\0' '\n' < "/proc/$pid/environ" | grep -qx "VECTORIAL_JOB=$job"
EOF

# ---- refuse a duplicate ---------------------------------------------------- #
# The metadata file sits on the NFS-shared home, so a running instance is visible
# from any node; liveness is then tested on the node that owns the process.
# read -d is used instead of $(cat <<EOF) because bash 3.2, which is what macOS
# ships, mis-parses an unbalanced ')' inside a heredoc in a command substitution.
CHECK_SCRIPT=''
read -r -d '' CHECK_SCRIPT <<'EOF' || true
meta="$HOME/$1/runs/cluster_logs/$2.meta"
[ -f "$meta" ] || exit 0
host=''; pid=''
while IFS='=' read -r k v; do
  case "$k" in host) host=$v ;; pid) pid=$v ;; esac
done < "$meta"
echo "$host $pid"
EOF
EXISTING="$(ssh -o BatchMode=yes "$HOST" \
  "bash -s $(printf '%q ' "$REMOTE_ROOT" "$JOB")" <<< "$CHECK_SCRIPT" || true)"
if [ -n "${EXISTING// /}" ]; then
  EX_HOST="$(echo "$EXISTING" | awk '{print $1}')"
  EX_PID="$(echo "$EXISTING" | awk '{print $2}')"
  case "$EX_HOST" in *.*) EX_FQDN="$EX_HOST" ;; *) EX_FQDN="$EX_HOST.$DOMAIN" ;; esac
  if [ -n "$EX_PID" ] && ssh -o BatchMode=yes "$EX_FQDN" \
       "bash -s $(printf '%q ' "$EX_PID" "$JOB")" <<< "$ALIVE_SCRIPT"; then
    echo "error: job '$JOB' is already running on $EX_HOST as pid $EX_PID" >&2
    echo "       inspect it with scripts/cluster/status.sh --job $JOB" >&2
    exit 1
  fi
  echo "    previous attempt on $EX_HOST (pid ${EX_PID:-unknown}) is gone; it will resume"
fi

# ---- launch ---------------------------------------------------------------- #
# The remote script is single-quoted so that nothing expands locally; every value
# it needs is passed as a positional argument.
OFFLINE_ENV="HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
[ "$ALLOW_DOWNLOAD" -eq 1 ] && OFFLINE_ENV=""

LAUNCH_SCRIPT=''
read -r -d '' LAUNCH_SCRIPT <<'EOF' || true
set -eu
JOB="$1"; GPU="$2"; CKPT_REL="$3"; REL_ROOT="$4"; OFFLINE="$5"; shift 5

ROOT="$HOME/$REL_ROOT"
LOGDIR="$ROOT/runs/cluster_logs"
LOG="$LOGDIR/$JOB.log"
CKPT="$ROOT/$CKPT_REL"
PY="$HOME/micromamba/envs/vectorial/bin/python"
mkdir -p "$LOGDIR" "$CKPT"
cd "$ROOT"
[ -x "$PY" ] || { echo "error: interpreter $PY not found" >&2; exit 1; }

{
  echo
  echo "===== launch $JOB at $(date -Is) on $(hostname -s) gpu $GPU ====="
} >> "$LOG"

env CUDA_VISIBLE_DEVICES="$GPU" \
    HF_HOME="$HOME/.cache/huggingface" \
    HF_DATASETS_CACHE="$HOME/.cache/huggingface/datasets" \
    TOKENIZERS_PARALLELISM=false \
    PYTHONPATH="$ROOT/src:$ROOT/scripts/cluster" \
    VECTORIAL_JOB="$JOB" \
    VECTORIAL_CKPT_DIR="$CKPT" \
    $OFFLINE \
    nohup setsid "$PY" -u "$@" >> "$LOG" 2>&1 &
pid=$!
echo "$pid" > "$LOGDIR/$JOB.pid"
{
  echo "job=$JOB"
  echo "host=$(hostname -s)"
  echo "gpu=$GPU"
  echo "pid=$pid"
  echo "started=$(date -Is)"
  echo "ckpt_dir=$CKPT_REL"
  echo "log=$JOB.log"
  echo "cmd=$*"
} > "$LOGDIR/$JOB.meta"
echo "PID $pid"
EOF

REMOTE_ARGS="$(printf '%q ' "$JOB" "$GPU" "$CKPT_DIR" "$REMOTE_ROOT" "$OFFLINE_ENV" "$@")"

if [ "$DRY" -eq 1 ]; then
  echo "--- would run on $HOST:  bash -s $REMOTE_ARGS"
  printf '%s\n' "$LAUNCH_SCRIPT"
  exit 0
fi

OUT="$(ssh -o BatchMode=yes "$HOST" "bash -s $REMOTE_ARGS" <<< "$LAUNCH_SCRIPT")"
PID="$(echo "$OUT" | awk '/^PID /{print $2}' | tail -1)"
[ -n "$PID" ] || { echo "error: launch produced no PID:" >&2; echo "$OUT" >&2; exit 1; }

# A job that dies immediately (bad path, import error, no GPU) is worth catching
# here rather than at the next status poll.
sleep 5
if ssh -o BatchMode=yes "$HOST" \
     "bash -s $(printf '%q ' "$PID" "$JOB")" <<< "$ALIVE_SCRIPT"; then
  STATE="running"
else
  STATE="ALREADY EXITED - read the log"
fi

cat <<EOF
    host        $HOST
    gpu         $GPU
    pid         $PID  ($STATE)
    log         ~/$REMOTE_ROOT/runs/cluster_logs/$JOB.log
    checkpoints ~/$REMOTE_ROOT/$CKPT_DIR
    follow      ssh $HOST 'tail -f ~/$REMOTE_ROOT/runs/cluster_logs/$JOB.log'
    status      scripts/cluster/status.sh --job $JOB
EOF
[ "$STATE" = "running" ]
