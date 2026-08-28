#!/usr/bin/env python
"""Generate from a trained soft prompt on a few real tasks, on the cluster.

    cd ~/Projects/vectorial
    CUDA_VISIBLE_DEVICES=<free gpu> HF_HOME=~/.cache/huggingface PYTHONPATH=src \
      ~/micromamba/envs/vectorial/bin/python scripts/cluster/soft_prompt_gen_check.py \
      runs/checkpoints/<job>/best

This is the inference-side smoke test. It confirms that the artifact loads, that
one output is produced per task and per draw, and that two draws of the same task
differ, which is a failure this project has been bitten by once already.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from vectorial_eval.data.schema import PostRecord, TransferTask
from vectorial_eval.transfer.base import Corpus
from vectorial_eval.transfer.soft_prompt import SoftPromptTransfer


def read(path: Path, model):
    with path.open(encoding="utf-8") as fh:
        return [model(**json.loads(line)) for line in fh if line.strip()]


def main(argv: list[str]) -> int:
    adapter = argv[1] if len(argv) > 1 else "runs/checkpoints/soft_prompt_target_lm/best"
    n_tasks = int(argv[2]) if len(argv) > 2 else 3
    root = Path(".")

    tasks = read(root / "data/tasks.heldout.jsonl", TransferTask)[:n_tasks]
    posts: list[PostRecord] = []
    for split in ("train", "val", "test"):
        posts += read(root / f"data/posts.{split}.jsonl", PostRecord)

    fn = SoftPromptTransfer(adapter_dir=adapter, n_samples=2, max_new_tokens=96)
    outputs = fn.run(tasks, Corpus(posts))

    print(f"outputs {len(outputs)} (expected {len(tasks) * 2})")
    for out in outputs:
        print(f"--- {out.task_id[:48]} ok={out.meta['ok']} draw={out.meta['sample_index']}")
        print(repr(out.output_text[:200]))
    by_task: dict[str, set[str]] = {}
    for out in outputs:
        by_task.setdefault(out.task_id, set()).add(out.output_text)
    identical = [t for t, texts in by_task.items() if len(texts) == 1]
    print(f"tasks whose two draws are identical: {len(identical)} of {len(by_task)}")
    print(json.dumps(fn.describe(), indent=2, sort_keys=True)[:800])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
