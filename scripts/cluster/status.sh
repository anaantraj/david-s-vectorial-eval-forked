#!/usr/bin/env bash
# Compact status of every cluster job launched by launch.sh.
#
#   scripts/cluster/status.sh [--job NAME] [--tail N] [--quiet] [--hub HOST]
#
#   --job NAME   restrict to one job
#   --tail N     log lines to print per job (default 20; 0 suppresses them)
#   --quiet      table only, same as --tail 0
#   --hub HOST   node used to read the shared home directory (default cthulhu1)
#
# One table row per job: whether the process is alive, the latest completed
# checkpoint step and how long ago it was written, and whether the log ends in a
# CUDA out-of-memory error or a kill. Job metadata, logs, and checkpoints all sit
# on the NFS-shared home, so they are read from a single node; liveness is the
# only thing that has to be asked of the node that owns the process.
#
# STATE is one of:
#   RUNNING   the process is alive on its node
#   OOM       the process is gone and the log ends with a CUDA out-of-memory error
#   KILLED    the process is gone and the log shows a kill or termination signal
#   ERROR     the process is gone and the log ends in a traceback
#   STOPPED   the process is gone with no failure marker (finished, or killed
#             without a message; check the tail)
# Anything other than RUNNING is resumable: re-issue the same launch.sh command
# and the training script continues from the step in the CKPT column.

set -euo pipefail

REMOTE_ROOT="Projects/vectorial"
DOMAIN="ist.berkeley.edu"
HUB="cthulhu1"
ONLY_JOB=""
TAIL_N=20

while [ $# -gt 0 ]; do
  case "$1" in
    --job)   ONLY_JOB="$2"; shift 2 ;;
    --tail)  TAIL_N="$2"; shift 2 ;;
    --quiet) TAIL_N=0; shift ;;
    --hub)   HUB="$2"; shift 2 ;;
    -h|--help) sed -n '2,28p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "error: unknown option $1" >&2; exit 2 ;;
  esac
done
case "$HUB" in *.*) ;; *) HUB="$HUB.$DOMAIN" ;; esac

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# ---- one pass over the shared home ---------------------------------------- #
# Emits, per job, a tab-separated summary line prefixed JOB, followed by the log
# tail bracketed by TAIL/ENDTAIL markers.
GATHER="
set -u
LOGDIR=\"\$HOME/$REMOTE_ROOT/runs/cluster_logs\"
[ -d \"\$LOGDIR\" ] || exit 0
now=\$(date +%s)
for meta in \"\$LOGDIR\"/*.meta; do
  [ -e \"\$meta\" ] || continue
  job=''; host=''; gpu=''; pid=''; started=''; ckpt_dir=''
  while IFS='=' read -r k v; do
    case \"\$k\" in
      job) job=\$v ;; host) host=\$v ;; gpu) gpu=\$v ;; pid) pid=\$v ;;
      started) started=\$v ;; ckpt_dir) ckpt_dir=\$v ;;
    esac
  done < \"\$meta\"
  [ -n \"\$job\" ] || continue
  if [ -n '$ONLY_JOB' ] && [ \"\$job\" != '$ONLY_JOB' ]; then continue; fi

  step='-'; ckpt_age='-'
  if [ -n \"\$ckpt_dir\" ] && [ -d \"\$HOME/$REMOTE_ROOT/\$ckpt_dir\" ]; then
    last=\$(ls -d \"\$HOME/$REMOTE_ROOT/\$ckpt_dir\"/step_[0-9]* 2>/dev/null | sort | tail -1)
    if [ -n \"\$last\" ]; then
      step=\$(basename \"\$last\" | sed 's/^step_0*//'); [ -n \"\$step\" ] || step=0
      mt=\$(stat -c %Y \"\$last\" 2>/dev/null || echo \"\$now\")
      ckpt_age=\$(( (now - mt) / 60 ))m
    fi
  fi

  log=\"\$LOGDIR/\$job.log\"
  flags='-'; logage='-'
  if [ -f \"\$log\" ]; then
    lm=\$(stat -c %Y \"\$log\" 2>/dev/null || echo \"\$now\")
    logage=\$(( (now - lm) / 60 ))m
    t=\$(tail -n 200 \"\$log\")
    f=''
    case \"\$t\" in *'out of memory'*|*'OutOfMemoryError'*) f=OOM ;; esac
    if [ -z \"\$f\" ]; then
      case \"\$t\" in *Killed*|*SIGKILL*|*SIGTERM*|*Terminated*) f=KILLED ;; esac
    fi
    if [ -z \"\$f\" ]; then
      case \"\$t\" in *Traceback*) f=ERROR ;; esac
    fi
    [ -n \"\$f\" ] && flags=\$f
  fi
  printf 'JOB\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \\
    \"\$job\" \"\$host\" \"\$gpu\" \"\$pid\" \"\$step\" \"\$ckpt_age\" \"\$flags\" \"\$logage\" \"\$started\"
  if [ $TAIL_N -gt 0 ] && [ -f \"\$log\" ]; then
    echo \"TAIL \$job\"
    tail -n $TAIL_N \"\$log\"
    echo 'ENDTAIL'
  fi
done
"

ssh -o BatchMode=yes -o ConnectTimeout=15 "$HUB" "bash -s" <<< "$GATHER" > "$TMP/gather" || {
  echo "error: could not read job state from $HUB" >&2; exit 1; }

grep '^JOB	' "$TMP/gather" > "$TMP/jobs" || true
if [ ! -s "$TMP/jobs" ]; then
  echo "no jobs found under ~/$REMOTE_ROOT/runs/cluster_logs${ONLY_JOB:+ for job $ONLY_JOB}"
  exit 0
fi

# ---- liveness, one ssh per owning node ------------------------------------ #
: > "$TMP/alive"
cut -f3 "$TMP/jobs" | sort -u | while read -r h; do
  [ -n "$h" ] || continue
  case "$h" in *.*) fq="$h" ;; *) fq="$h.$DOMAIN" ;; esac
  # One ssh per node. Liveness is decided by the process environment rather than
  # by existence alone, so a recycled PID cannot be read as a live job.
  pairs="$(awk -F'\t' -v H="$h" '$3==H {print $5":"$2}' "$TMP/jobs" | tr '\n' ' ')"
  [ -n "${pairs// /}" ] || continue
  ssh -o BatchMode=yes -o ConnectTimeout=15 "$fq" "bash -s $pairs" <<'REMOTE' >> "$TMP/alive" 2>/dev/null || true
for pair in "$@"; do
  pid=${pair%%:*}; job=${pair#*:}
  [ -r "/proc/$pid/environ" ] || continue
  if tr '\0' '\n' < "/proc/$pid/environ" | grep -qx "VECTORIAL_JOB=$job"; then
    echo "$(hostname -s) $pid"
  fi
done
REMOTE
done

# ---- table ---------------------------------------------------------------- #
printf '%-22s %-10s %-3s %-8s %-9s %-8s %-7s %-7s\n' \
  JOB HOST GPU PID STATE CKPT CKPT_AGE LOG_AGE
printf '%.0s-' $(seq 1 84); echo

while IFS=$'\t' read -r _ job host gpu pid step ckpt_age flags logage started; do
  if grep -qx "$host $pid" "$TMP/alive" 2>/dev/null; then
    state="RUNNING"
  elif [ "$flags" != "-" ]; then
    state="$flags"
  else
    state="STOPPED"
  fi
  printf '%-22s %-10s %-3s %-8s %-9s %-8s %-7s %-7s\n' \
    "$job" "$host" "$gpu" "$pid" "$state" "$step" "$ckpt_age" "$logage"
done < "$TMP/jobs"

# ---- tails ---------------------------------------------------------------- #
if [ "$TAIL_N" -gt 0 ]; then
  awk -v n="$TAIL_N" '
    /^TAIL /   { print ""; print "=== " $2 " (last " n " log lines) ==="; next }
    /^ENDTAIL$/{ next }
    /^JOB\t/   { next }
    { print "  " $0 }
  ' "$TMP/gather"
fi
