#!/usr/bin/env python
"""Audit Study 4 output cardinality, draw identity, failures, and descriptors."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def audit(
    run_dir: Path,
    tasks_path: Path,
    split: str,
    n_samples: int,
    required: set[str],
) -> dict:
    task_ids = {row["task_id"] for row in read_jsonl(tasks_path)}
    if not task_ids:
        raise RuntimeError(f"{tasks_path} contains no tasks")
    files = sorted(run_dir.glob(f"outputs.*.{split}.jsonl"))
    systems = {path.name.split(".")[1] for path in files}
    missing = sorted(required - systems)
    if missing:
        raise RuntimeError(f"missing required output systems: {missing}")
    summaries = {}
    for path in files:
        system = path.name.split(".")[1]
        rows = read_jsonl(path)
        draws = 1 if system == "identity" else n_samples
        by_task: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            if row.get("transfer_fn") != system:
                raise RuntimeError(
                    f"{path.name}: transfer_fn {row.get('transfer_fn')!r} != {system!r}"
                )
            if row.get("task_id") not in task_ids:
                raise RuntimeError(f"{path.name}: unknown task_id {row.get('task_id')!r}")
            by_task[row["task_id"]].append(row)
        absent = sorted(task_ids - set(by_task))
        if absent:
            raise RuntimeError(f"{path.name}: missing {len(absent)} task_ids")
        wrong = {
            task_id: len(task_rows)
            for task_id, task_rows in by_task.items()
            if len(task_rows) != draws
        }
        if wrong:
            raise RuntimeError(
                f"{path.name}: {len(wrong)} tasks do not have exactly {draws} draws"
            )
        if draws > 1:
            expected = set(range(draws))
            for task_id, task_rows in by_task.items():
                indices = {row.get("meta", {}).get("sample_index") for row in task_rows}
                if indices != expected:
                    raise RuntimeError(
                        f"{path.name}: task {task_id} draw indices {indices} != {expected}"
                    )
        descriptor = run_dir / f"transfer_fn.{system}.json"
        if not descriptor.exists():
            raise RuntimeError(f"{path.name}: missing {descriptor.name}")
        failures = sum(
            not bool((row.get("output_text") or "").strip())
            or row.get("meta", {}).get("ok") is False
            for row in rows
        )
        summaries[system] = {
            "n_outputs": len(rows),
            "n_tasks": len(by_task),
            "draws_per_task": draws,
            "n_failures_or_empty": failures,
            "failure_rate": failures / len(rows) if rows else 1.0,
        }
    return {
        "split": split,
        "n_tasks": len(task_ids),
        "n_systems": len(summaries),
        "systems": summaries,
        "failure_counts": dict(
            Counter(
                summary["n_failures_or_empty"] for summary in summaries.values()
            )
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("tasks", type=Path)
    parser.add_argument("--split", default="heldout")
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--require", action="append", default=[])
    args = parser.parse_args()
    result = audit(
        args.run_dir,
        args.tasks,
        args.split,
        args.n_samples,
        set(args.require),
    )
    output = args.run_dir / f"output_audit.{args.split}.json"
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
