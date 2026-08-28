"""End-to-end check of the `steering` transfer function on real tasks.

This exercises the path the harness uses: build the registered function, hand it
a small batch of validation tasks, and confirm that one output comes back per
task per draw and that the draws differ from one another. It exists because the
sweep exercises the generator directly and would not catch a fault in the
wrapper.

    scripts/cluster/launch.sh -j steer_smoke_transfer -- \
        src/vectorial_eval/methods/steering/smoke_transfer.py \
        --artifact runs/steering/steering.target_lm.pt
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from vectorial_eval.config import DEFAULT_DATA_DIR
from vectorial_eval.data.schema import TransferTask
from vectorial_eval.transfer import steering as _steering  # noqa: F401  (registers)
from vectorial_eval.transfer.base import Corpus, build_transfer_fn

log = logging.getLogger("smoke_transfer")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--n-tasks", type=int, default=3)
    p.add_argument("--n-samples", type=int, default=2)
    p.add_argument("--split", default="val")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    tasks = []
    with (args.data_dir / f"tasks.{args.split}.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                tasks.append(TransferTask.model_validate_json(line))
    tasks.sort(key=lambda t: t.task_id)
    tasks = tasks[: args.n_tasks]

    fn = build_transfer_fn(
        "steering", artifact=str(args.artifact), n_samples=args.n_samples
    )
    outs = fn.run(tasks, Corpus([]))

    expected = len(tasks) * args.n_samples
    assert len(outs) == expected, f"{len(outs)} outputs for {expected} (task, draw) pairs"
    by_task: dict[str, list[str]] = {}
    for out in outs:
        by_task.setdefault(out.task_id, []).append(out.output_text)
    identical = [tid for tid, texts in by_task.items() if len(set(texts)) == 1]

    print(json.dumps(fn.describe(), indent=2, default=str))
    for out in outs:
        print("=" * 72)
        print(f"{out.task_id} draw={out.meta['sample_index']} ok={out.meta['ok']} "
              f"scope={out.meta.get('vector_scope')} n={out.meta.get('n_target_posts')}")
        print(out.output_text[:800])
    print("=" * 72)
    print(f"outputs={len(outs)} failed={sum(1 for o in outs if not o.meta['ok'])} "
          f"tasks_with_identical_draws={len(identical)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
