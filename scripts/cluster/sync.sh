#!/usr/bin/env bash
# Sync this repository to the cluster.
#
#   scripts/cluster/sync.sh [node] [extra rsync args...]
#
# `node` may be a bare name (cthulhu3) or a fully qualified host; it defaults to
# cthulhu1. /home/davidchan is NFS-shared across cthulhu1..6, so one sync serves
# every node and the choice of node only affects which machine does the writing.
#
# The transfer is idempotent: running it twice in a row copies nothing the second
# time. It deletes remote files that no longer exist locally *within the synced
# subtree only*, so remote-only outputs (runs/, checkpoints, logs) are protected
# by the exclude list rather than by luck.

set -euo pipefail

NODE="${1:-cthulhu1}"
shift || true
case "$NODE" in
  *.*) HOST="$NODE" ;;
  *)   HOST="$NODE.ist.berkeley.edu" ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE_ROOT="Projects/vectorial"   # relative to the remote home directory

# Excludes. Anything produced on one side and meaningless on the other, plus
# anything large enough to be worth not copying.
#
# `runs/` is excluded in its entirety, and that exclusion is load-bearing. This
# script pushes with --delete, so any remote path under a synced directory that
# does not exist locally is removed. Training artifacts are produced on the
# cluster and never exist on the local machine, so syncing `runs/` deleted a
# completed set of steering vectors between one job finishing and the next
# launching. Adapters, soft prompts, steering vectors, checkpoints, and cluster
# logs all live under `runs/`; none of them may be exposed to --delete.
#
# The cluster's `runs/` is its own workspace. Bring results back with an explicit
# pull rather than by making this push bidirectional.
EXCLUDES=(
  --exclude '.venv/'
  --exclude '__pycache__/'
  --exclude '*.py[cod]'
  --exclude '.git/'
  --exclude '.pytest_cache/'
  --exclude '.ruff_cache/'
  --exclude 'runs/'
  --exclude '.DS_Store'
)

echo "==> syncing $ROOT/ -> $HOST:~/$REMOTE_ROOT/"
ssh -o BatchMode=yes "$HOST" "mkdir -p '$REMOTE_ROOT'"

# --delete removes remote files that no longer exist locally. It is deliberately
# NOT paired with --delete-excluded, which would delete the excluded paths on the
# remote as well and so would wipe cluster logs and checkpoints on every sync.
#
# --safe-links drops symlinks that point outside the tree. runs/ contains a few
# absolute symlinks into the local checkout; they would be broken on the cluster
# and rsync would re-send them on every run, which is also what would make this
# script non-idempotent.
rsync -az --delete --safe-links --human-readable --partial \
  "${EXCLUDES[@]}" "$@" \
  -e "ssh -o BatchMode=yes" \
  "$ROOT/" "$HOST:$REMOTE_ROOT/"

echo "==> done"
ssh -o BatchMode=yes "$HOST" "cd '$REMOTE_ROOT' && du -sh . && ls"
