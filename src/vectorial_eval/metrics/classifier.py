"""Platform-classifier confusion — the headline transfer metric.

Method
------
A LinkedIn-versus-Reddit classifier is trained on authentic posts drawn from the
train split alone, then applied to each transfer function's generated posts to
determine how frequently they are assigned to the target platform.

Interpretation
--------------
The naive interpretation, that a higher target-platform rate is preferable, is
incorrect in isolation, and this metric is constructed to expose that. A
function emitting the single most Reddit-like string available would approach a
rate of 1.0 while destroying the content entirely. Three quantities are
therefore reported jointly:

`target_rate`
    Share of generated posts classified as target platform.
`calibration_gap`
    |target_rate(generated) - target_rate(authentic target posts)|. The
    authentic target pool does not itself classify at 1.0; the objective is to
    *match* the authentic distribution rather than to saturate the classifier.
    This is the score to be read.
`mean_target_prob`
    Mean predicted probability, which is more sensitive than the hard rate when
    a function moves outputs across the boundary only marginally.

A confusion matrix is additionally emitted, because the informative failure is
asymmetric: outputs that remain LinkedIn-like indicate under-transfer, which
calls for a different remedy than over-transfer.

Two classifiers are fit deliberately:

`lexical`
    TF-IDF with logistic regression. Strong, but keys substantially on topic
    vocabulary and platform artefacts such as hashtags and URLs.
`stylometric`
    The interpretable surface features alone. Weaker, but unable to exploit
    topic vocabulary, so agreement between the two constitutes evidence that the
    transfer is stylistic rather than an artefact of word choice.
"""

from __future__ import annotations

import logging

import numpy as np

from ..features import stylometry
from .base import EvalContext, Metric, MetricResult, group_by_cell, register

log = logging.getLogger(__name__)


def _fit_classifiers(ctx: EvalContext, seed: int):
    """Fit both platform classifiers on authentic train-split posts."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    texts, labels = [], []
    for cell_id in ctx.cells:
        for platform in (ctx.source_platform, ctx.target_platform):
            for t in ctx.pools[cell_id][platform]["train"]:
                texts.append(t)
                labels.append(1 if platform == ctx.target_platform else 0)

    y = np.array(labels)
    if len(set(labels)) < 2 or len(texts) < 20:
        raise RuntimeError(
            f"Not enough train-split data to fit a platform classifier "
            f"({len(texts)} posts). Lower min_posts_per_platform or widen splits."
        )

    lexical = make_pipeline(
        TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2), max_features=50000),
        LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=seed),
    ).fit(texts, y)

    stylo = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=seed),
    ).fit(stylometry.matrix(texts), y)

    # Held-out accuracy is reported so that a weak classifier is not relied
    # upon implicitly.
    val_texts, val_y = [], []
    for cell_id in ctx.cells:
        for platform in (ctx.source_platform, ctx.target_platform):
            for t in ctx.pools[cell_id][platform]["val"]:
                val_texts.append(t)
                val_y.append(1 if platform == ctx.target_platform else 0)
    acc = {}
    if val_texts and len(set(val_y)) > 1:
        acc["lexical_val_acc"] = float((lexical.predict(val_texts) == np.array(val_y)).mean())
        acc["stylometric_val_acc"] = float(
            (stylo.predict(stylometry.matrix(val_texts)) == np.array(val_y)).mean()
        )
    return lexical, stylo, acc, len(texts)


@register("classifier")
class ClassifierConfusion(Metric):
    directions = {
        # Diagnostic rather than a quality score: the objective is to match the
        # authentic rate, not to maximise it.
        "target_rate": 0,
        "calibration_gap": -1,
        "mean_target_prob": 0,
        "prob_gap_vs_real": -1,
    }

    def score(self, transfer_fn, outputs, ctx: EvalContext) -> MetricResult:
        cache_key = "_classifier_models"
        if cache_key not in ctx.extras:
            ctx.extras[cache_key] = _fit_classifiers(ctx, ctx.seed)
        lexical, stylo, acc, n_train = ctx.extras[cache_key]

        pools = group_by_cell(outputs)
        result = MetricResult(metric=self.name, transfer_fn=transfer_fn)
        result.notes = {"classifier_val_accuracy": acc, "n_train_posts": n_train}

        all_gen, all_real = [], []
        for cell_id, gen in pools.items():
            real = ctx.target_pool(cell_id)
            if not gen or not real:
                continue
            all_gen.extend(gen)
            all_real.extend(real)

            gen_p = lexical.predict_proba(gen)[:, 1]
            real_p = lexical.predict_proba(real)[:, 1]
            gen_s = stylo.predict_proba(stylometry.matrix(gen))[:, 1]
            real_s = stylo.predict_proba(stylometry.matrix(real))[:, 1]

            result.cell_scores[cell_id] = {
                "target_rate": float((gen_p > 0.5).mean()),
                "real_target_rate": float((real_p > 0.5).mean()),
                "calibration_gap": float(abs((gen_p > 0.5).mean() - (real_p > 0.5).mean())),
                "mean_target_prob": float(gen_p.mean()),
                "prob_gap_vs_real": float(abs(gen_p.mean() - real_p.mean())),
                "stylometric_target_rate": float((gen_s > 0.5).mean()),
                "stylometric_calibration_gap": float(
                    abs((gen_s > 0.5).mean() - (real_s > 0.5).mean())
                ),
                "n_generated": len(gen),
                "n_real": len(real),
            }

        if all_gen:
            p = lexical.predict_proba(all_gen)[:, 1]
            r = lexical.predict_proba(all_real)[:, 1]
            pred = (p > 0.5).astype(int)
            # Confusion of the generated pool against the intended label, under
            # which every output should be assigned to the target platform.
            result.overall = {
                "target_rate": float(pred.mean()),
                "real_target_rate": float((r > 0.5).mean()),
                "calibration_gap": float(abs(pred.mean() - (r > 0.5).mean())),
                "mean_target_prob": float(p.mean()),
                "n_classified_target": int(pred.sum()),
                "n_classified_source": int((1 - pred).sum()),
                "n_total": int(len(pred)),
            }
        return result
