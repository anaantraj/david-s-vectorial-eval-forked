"""Non-LLM baselines that bracket the achievable score range.

A metric suite requires anchors before its figures admit interpretation. The
following three provide them.

`identity`
    Copy the source post unchanged. This is the *lower* bound on platform
    transfer and simultaneously the *upper* bound on semantic preservation.
    This tension constitutes the most diagnostic reading in the harness: a
    transfer function that improves on identity with respect to the platform
    classifier while approaching its semantic similarity is performing
    substantive work, whereas one that improves on the classifier solely by
    discarding content will exhibit a corresponding decline here.

`target_sample`
    Emit a real target-platform post drawn from the cell's *scored* split. This
    constitutes the practical ceiling, being authentic target-distribution
    text, and it calibrates every distributional metric. A metric unable to
    separate `target_sample` from `identity` is not measuring platform transfer
    and should not be relied upon for substantive systems.

`shuffle_control`
    Emit a real target post from a *different* cell. Same marginal platform
    style, wrong topic. Any metric that scores this as highly as
    `target_sample` is style-indexed and blind to semantics — which is exactly
    the failure mode the July 16 discussion flagged as the project's main risk.
"""

from __future__ import annotations

import random

from ..data.schema import TransferOutput, TransferTask
from .base import Corpus, TransferFunction, register


@register("identity")
class IdentityTransfer(TransferFunction):
    name = "identity"

    def __init__(self, name: str | None = None, **_):
        if name:
            self.name = name

    def run(self, tasks: list[TransferTask], corpus: Corpus) -> list[TransferOutput]:
        return [
            TransferOutput(
                task_id=t.task_id,
                cell_id=t.cell_id,
                transfer_fn=self.name,
                output_text=t.source_text,
                meta={"strategy": "copy_source"},
            )
            for t in tasks
        ]


@register("target_sample")
class TargetSampleTransfer(TransferFunction):
    """Oracle calibration: a real target post from the scored cell and split."""

    name = "target_sample"

    def __init__(self, seed: int = 20260722, n_samples: int = 1,
                 name: str | None = None, **_):
        self.seed = seed
        self.n_samples = max(1, int(n_samples))
        if name:
            self.name = name

    def run(self, tasks: list[TransferTask], corpus: Corpus) -> list[TransferOutput]:
        rng = random.Random(self.seed)
        out = []
        for t in tasks:
            pool = [pid for pid in t.target_reference_ids if pid in corpus]
            if not pool:
                for k in range(self.n_samples):
                    out.append(
                        TransferOutput(
                            task_id=t.task_id,
                            cell_id=t.cell_id,
                            transfer_fn=self.name,
                            output_text="",
                            meta={
                                "error": "no_scored_references",
                                "heldout_cell": t.heldout_cell,
                                "sample_index": k,
                            },
                        )
                    )
                continue
            # Draw without replacement until the authentic pool is exhausted.
            # This gives the stochastic oracle the same candidate count as the
            # systems while never pretending repeated draws are distinct.
            picks = rng.sample(pool, min(self.n_samples, len(pool)))
            if len(picks) < self.n_samples:
                picks.extend(rng.choice(pool) for _ in range(self.n_samples - len(picks)))
            for k, pid in enumerate(picks):
                out.append(TransferOutput(
                    task_id=t.task_id,
                    cell_id=t.cell_id,
                    transfer_fn=self.name,
                    output_text=corpus.text(pid),
                    meta={"sampled_post_id": pid, "sample_index": k},
                ))
        return out


@register("shuffle_control")
class ShuffleControlTransfer(TransferFunction):
    """Negative control: correct platform, incorrect topic.

    Separates acquisition of platform register from preservation of topic.
    """

    name = "shuffle_control"

    def __init__(self, seed: int = 20260722, n_samples: int = 1,
                 name: str | None = None, **_):
        self.seed = seed
        self.n_samples = max(1, int(n_samples))
        if name:
            self.name = name

    def run(self, tasks: list[TransferTask], corpus: Corpus) -> list[TransferOutput]:
        rng = random.Random(self.seed)
        by_cell: dict[str, list[str]] = {}
        for t in tasks:
            by_cell.setdefault(t.cell_id, []).extend(
                pid for pid in t.target_reference_ids if pid in corpus
            )
        cells = [c for c, pool in by_cell.items() if pool]

        out = []
        for t in tasks:
            others = [c for c in cells if c != t.cell_id]
            if not others:
                for k in range(self.n_samples):
                    out.append(
                        TransferOutput(
                            task_id=t.task_id,
                            cell_id=t.cell_id,
                            transfer_fn=self.name,
                            output_text="",
                            meta={"error": "no_other_cells", "sample_index": k},
                        )
                    )
                continue
            for k in range(self.n_samples):
                donor = rng.choice(others)
                pid = rng.choice(by_cell[donor])
                out.append(TransferOutput(
                    task_id=t.task_id,
                    cell_id=t.cell_id,
                    transfer_fn=self.name,
                    output_text=corpus.text(pid),
                    meta={"donor_cell": donor, "sampled_post_id": pid, "sample_index": k},
                ))
        return out
