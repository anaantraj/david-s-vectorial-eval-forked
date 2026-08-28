"""Aspect-prevalence matching from one frozen, uniformly applied rater.

The rater is run before evaluation and saved as ``aspect_ratings.jsonl`` in the
run directory. Generated, authentic target, authentic source, and baseline text
are all read from that same artifact. This avoids mixing the collaborators'
historic real-post scores with a newly implemented generated-post rater.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .base import EvalContext, Metric, MetricResult, jensen_shannon, register

MIN_ASPECT_POOL = 5


def rating_key(cell_id: str, text: str) -> str:
    payload = f"{cell_id}\0{text}".encode()
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def bernoulli_jsd(p: float, q: float) -> float:
    return jensen_shannon(np.array([p, 1.0 - p]), np.array([q, 1.0 - q]))


def aspect_prevalence_scores(
    generated: np.ndarray,
    target: np.ndarray,
    source: np.ndarray,
) -> dict[str, float]:
    """Compare per-aspect prevalence while retaining both presence and absence."""
    if min(len(generated), len(target), len(source)) < MIN_ASPECT_POOL:
        return {}
    if generated.ndim != 2 or target.ndim != 2 or source.ndim != 2:
        raise ValueError("aspect arrays must be two-dimensional")
    if not (generated.shape[1] == target.shape[1] == source.shape[1]):
        raise ValueError("aspect arrays must share one frozen vocabulary")

    p_gen, p_target, p_source = generated.mean(0), target.mean(0), source.mean(0)
    target_gap = np.abs(p_gen - p_target)
    source_target_gap = np.abs(p_source - p_target)
    denom = float(source_target_gap.mean())
    return {
        "prevalence_jsd": float(
            np.mean(
                [
                    bernoulli_jsd(float(a), float(b))
                    for a, b in zip(p_gen, p_target, strict=True)
                ]
            )
        ),
        "prevalence_mae": float(target_gap.mean()),
        "shift_progress": float(1.0 - target_gap.mean() / denom) if denom > 0 else float("nan"),
        "aspect_count_gap": float(abs(generated.sum(1).mean() - target.sum(1).mean())),
        "source_target_prevalence_mae": denom,
    }


@register("aspect")
class AspectPrevalenceMetric(Metric):
    directions = {
        "prevalence_jsd": -1,
        "prevalence_mae": -1,
        "shift_progress": 1,
        "aspect_count_gap": -1,
        "source_target_prevalence_mae": 0,
        "rating_coverage": 0,
        "target_resample_jsd": 0,
    }

    def score(self, transfer_fn, outputs, ctx: EvalContext) -> MetricResult:
        path = Path(ctx.extras["_run_dir"]) / "aspect_ratings.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is required; run scripts/rate_aspects.py before the aspect metric"
            )
        ratings = {}
        vocab_by_cell = {}
        rater = None
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                ratings[row["key"]] = row
                vocab_by_cell[row["cell_id"]] = tuple(row["vocabulary"])
                rater = row.get("rater", rater)

        by_cell = defaultdict(list)
        for out in outputs:
            by_cell[out.cell_id].append(out)

        result = MetricResult(metric=self.name, transfer_fn=transfer_fn)
        for cell_id, cell_outputs in by_cell.items():
            vocab = vocab_by_cell.get(cell_id)
            if not vocab:
                continue

            def vectors(
                texts: list[str], *, current_cell: str = cell_id, current_vocab=vocab
            ) -> tuple[np.ndarray, float]:
                rows = []
                for text in texts:
                    row = ratings.get(rating_key(current_cell, text))
                    if row and row.get("ok"):
                        rows.append(
                            [float(row["scores"].get(name, 0.0) > 0) for name in current_vocab]
                        )
                coverage = len(rows) / len(texts) if texts else 0.0
                return np.asarray(rows, dtype=float), coverage

            gen_texts = [o.output_text for o in cell_outputs if o.output_text]
            gen, gen_cov = vectors(gen_texts)
            target, target_cov = vectors(ctx.target_pool(cell_id))
            source, source_cov = vectors(ctx.source_pool(cell_id))
            scores = aspect_prevalence_scores(gen, target, source)
            if not scores:
                continue
            seed = int.from_bytes(hashlib.blake2b(cell_id.encode(), digest_size=8).digest(), "big")
            rng = np.random.default_rng(seed)
            resampled = target[rng.choice(len(target), size=len(target), replace=True)]
            scores["target_resample_jsd"] = float(
                np.mean(
                    [
                        bernoulli_jsd(float(a), float(b))
                        for a, b in zip(resampled.mean(0), target.mean(0), strict=True)
                    ]
                )
            )
            scores["rating_coverage"] = min(gen_cov, target_cov, source_cov)
            result.cell_scores[cell_id] = scores

        result.notes = {
            "minimum_pool": MIN_ASPECT_POOL,
            "rater": rater,
            "artifact": str(path),
            "definition": "mean per-aspect JSD between Bernoulli prevalence values",
            "caveat": (
                "The existing platform-visible vocabulary is used as supplied. The rater is "
                "a frozen Study 3 implementation and is applied uniformly to every text."
            ),
        }
        return result
