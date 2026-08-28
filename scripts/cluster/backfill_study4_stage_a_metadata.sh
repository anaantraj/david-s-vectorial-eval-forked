#!/usr/bin/env bash
# Wait for the two legacy Stage A jobs, then attach reproducible input metadata.
set -euo pipefail

PROJECT_ROOT="${VECTORIAL_PROJECT_ROOT:-$HOME/Projects/vectorial}"
PYTHON_BIN="${VECTORIAL_PYTHON:-$HOME/micromamba/envs/vectorial/bin/python}"
cd "$PROJECT_ROOT"

finished() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1]) / "metrics.json"
raise SystemExit(0 if path.exists() and json.load(path.open()).get("finished") else 1)
PY
}

for run in s4-reddit-rich-raw-10k s4-reddit-rich-prior-10k; do
  until finished "runs/checkpoints/$run"; do sleep 30; done
  safety_suffix=""
  if [[ "$run" == "s4-reddit-rich-prior-10k" ]]; then
    safety_suffix=".rich_safe"
  fi
  "$PYTHON_BIN" scripts/backfill_study4_run_metadata.py \
    "runs/checkpoints/$run" \
    "data/study4_priors/reddit_10000_per_audience${safety_suffix}.train.jsonl" \
    "data/study4_priors/reddit_10000_per_audience${safety_suffix}.dev.jsonl" \
    --manifest data/study4_priors/reddit_10000_per_audience.manifest.json
  echo "backfilled $run"
done
