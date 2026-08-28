#!/usr/bin/env python
"""Read the soft-prompt runs and name the configuration to use.

    .venv/bin/python src/vectorial_eval/methods/soft_prompt/select_best.py --remote

Selection is by dev loss and nothing else. Each training run writes
`summary.json` into its checkpoint directory, holding the configuration, the
whole dev-loss curve including the value before any gradient step, and the best
epoch. This script collects them, prints one row per run, and names the best run
per training variant along with the command that generates from it.

The dev files are the only ones any of this touches. Test and heldout are not
read here or by the training script.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REMOTE_ROOT = "Projects/vectorial"
DEFAULT_HOST = "cthulhu1.ist.berkeley.edu"


def read_local(ckpt_root: Path) -> list[dict]:
    out = []
    for path in sorted(ckpt_root.glob("*/summary.json")):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        summary["job"] = path.parent.name
        out.append(summary)
    return out


def read_remote(host: str, ckpt_root: str) -> list[dict]:
    """Collect summaries from the shared home over one ssh call."""
    # The marker is printed with a leading newline, because `summary.json` is
    # written without a trailing one: concatenating the files directly glues the
    # next marker onto the last line of the previous document and the parser then
    # sees two JSON objects in one buffer.
    script = (
        f"for f in $HOME/{ckpt_root}/sp_*/summary.json; do "
        '[ -f "$f" ] || continue; '
        "printf '\\n### %s\\n' \"$(basename $(dirname $f))\"; cat \"$f\"; done"
    )
    text = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host, script],
        capture_output=True, text=True, check=True,
    ).stdout
    out, job, buf = [], None, []

    def flush(job: str | None, buf: list[str]) -> None:
        """Add one document, skipping a run whose summary is unreadable.

        A single truncated file must not stop the whole table from printing: the
        run it belongs to is reported as missing by its absence, and every other
        run is still selectable.
        """
        body = "\n".join(buf).strip()
        if not job or not body:
            return
        try:
            out.append({**json.loads(body), "job": job})
        except json.JSONDecodeError as exc:
            print(f"skipping unreadable summary for {job}: {exc}", file=sys.stderr)

    for line in text.splitlines():
        if line.startswith("### "):
            flush(job, buf)
            job, buf = line[4:].strip(), []
        else:
            buf.append(line)
    flush(job, buf)
    return out


def rows(summaries: list[dict]) -> list[dict]:
    table = []
    for s in summaries:
        cfg = s.get("config", {})
        history = s.get("history", [])
        init = next((h for h in history if h.get("epoch") == -1), None)
        table.append(
            {
                "job": s.get("job", "?"),
                "variant": cfg.get("variant", "?"),
                "n_dev": s.get("n_dev"),
                "n_train": s.get("n_train"),
                "n_tokens": cfg.get("n_virtual_tokens"),
                "lr": cfg.get("lr"),
                "init": cfg.get("init"),
                "epochs_done": max((h["epoch"] for h in history), default=-1) + 1,
                "dev_init": (init or {}).get("dev", {}).get("loss"),
                "dev_best": s.get("best_dev_loss"),
                "best_epoch": s.get("best_epoch"),
                "stopped_early": s.get("stopped_early"),
            }
        )
    return sorted(table, key=lambda r: (r["variant"], r["dev_best"] if r["dev_best"] is not None else 9e9))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--remote", action="store_true", help="read summaries from the cluster")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--ckpt-root", default="runs/checkpoints")
    args = p.parse_args(argv)

    summaries = (
        read_remote(args.host, f"{REMOTE_ROOT}/{args.ckpt_root}")
        if args.remote
        else read_local(Path(args.ckpt_root))
    )
    summaries = [s for s in summaries if s.get("job", "").startswith("sp_")]
    if not summaries:
        print("no soft-prompt runs found", file=sys.stderr)
        return 1

    table = rows(summaries)
    header = f"{'job':44} {'tokens':>6} {'lr':>7} {'init':>14} {'ep':>3} {'dev@init':>9} {'dev best':>9} {'@ep':>4}"
    print(header)
    print("-" * len(header))
    for r in table:
        init_s = "-" if r["dev_init"] is None else format(r["dev_init"], ".4f")
        best_s = "-" if r["dev_best"] is None else format(r["dev_best"], ".4f")
        print(
            f"{r['job']:44} {str(r['n_tokens']):>6} {str(r['lr']):>7} {str(r['init']):>14} "
            f"{r['epochs_done']:>3} {init_s:>9} {best_s:>9} {str(r['best_epoch']):>4}"
        )

    print()
    for variant in sorted({r["variant"] for r in table}):
        runs = [r for r in table if r["variant"] == variant and r["dev_best"] is not None]
        # A smoke run started with `--limit` scores a handful of dev examples and
        # its loss is not on the same scale as a full run's. Such a run must not
        # be able to win the comparison merely by having been evaluated on an
        # easier subset, so only runs that saw the whole dev set are eligible.
        full_dev = max((r["n_dev"] or 0) for r in runs) if runs else 0
        eligible = [r for r in runs if (r["n_dev"] or 0) == full_dev]
        for r in runs:
            if r not in eligible:
                print(f"  ignoring {r['job']}: dev set of {r['n_dev']}, not {full_dev}")
        best = min(eligible, key=lambda r: r["dev_best"], default=None)
        if best is None:
            continue
        note = ""
        if best["best_epoch"] == -1:
            note = "  (no epoch beat the initialisation; the artifact is the initial prompt)"
        print(f"{variant}: {best['job']}  dev {best['dev_best']:.4f}{note}")
        print(
            "    VECTORIAL_SOFT_PROMPT_DIR=runs/checkpoints/"
            f"{best['job']}/best python -m vectorial_eval.cli transfer "
            "--fn soft_prompt --split heldout --n-samples 4"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
