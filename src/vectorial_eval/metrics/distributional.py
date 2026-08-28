"""Embedding-space distribution matching (the KLD/JSD-over-post-pools axis).

Compares the generated pool against the real target pool in embedding space,
per cell, with three complementary statistics:

`mmd2`
    Squared maximum mean discrepancy, RBF kernel. Sensitive to distribution
    shape, so mode collapse is penalised even when the mean is right.
`centroid_distance`
    Cosine distance between pool centroids. Coarse but very legible.
`topic_jsd`
    JSD between the pools' distributions over a shared k-means vocabulary of
    the embedding space — the closest analogue of the "KLD/JSD over post pools"
    the meeting described, computed on discretised semantic clusters.

Every statistic is reported twice: in the raw embedding space, and in the
**style-residual** space with platform-discriminating directions projected out.

This pairing is the central diagnostic. In the raw space, a function that merely
adopts Reddit register will appear to have matched the distribution; in the
residual space only topical agreement survives. A function whose raw score is
favourable but whose residual score is no better than the shuffle control has
acquired register alone, which is the risk identified on July 16.
"""

from __future__ import annotations

import logging

import numpy as np

from .base import (
    EvalContext,
    Metric,
    MetricResult,
    ensure_shared_reference,
    group_by_cell,
    jensen_shannon,
    rbf_mmd2,
    rbf_mmd2_biased,
    register,
)

log = logging.getLogger(__name__)
N_TOPIC_CLUSTERS = 24


def _cosine_centroid_distance(a: np.ndarray, b: np.ndarray) -> float:
    ca, cb = a.mean(0), b.mean(0)
    na, nb = np.linalg.norm(ca), np.linalg.norm(cb)
    if na < 1e-9 or nb < 1e-9:
        return float("nan")
    return float(1.0 - (ca @ cb) / (na * nb))


def _project_out(x: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Remove the span of `basis` from `x` and renormalise."""
    residual = x - (x @ basis) @ basis.T
    norms = np.linalg.norm(residual, axis=1, keepdims=True)
    return residual / np.clip(norms, 1e-9, None)


def _cluster_jsd(a: np.ndarray, b: np.ndarray, centers: np.ndarray) -> float:
    def hist(x):
        d = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        assign = d.argmin(1)
        return np.bincount(assign, minlength=len(centers)).astype(float) + 1e-9

    return jensen_shannon(hist(a), hist(b))


@register("distributional")
class DistributionMatch(Metric):
    directions = {
        "mmd2": -1,
        "mmd2_biased": -1,
        "centroid_distance": -1,
        "topic_jsd": -1,
        "mmd2_style_residual": -1,
        "mmd2_biased_style_residual": -1,
        "centroid_distance_style_residual": -1,
    }

    def score(self, transfer_fn, outputs, ctx: EvalContext) -> MetricResult:
        space = ctx.embedding_space
        if space is None:
            raise RuntimeError("distributional metric requires ctx.embedding_space")

        pools = group_by_cell(outputs)
        result = MetricResult(metric=self.name, transfer_fn=transfer_fn)

        # Fit the style subspace and the topic vocabulary once per evaluation,
        # on real posts only, and share across transfer functions so scores are
        # directly comparable.
        shared = ensure_shared_reference(ctx)
        basis = shared["style_basis"]

        for cell_id, gen in pools.items():
            real = ctx.target_pool(cell_id)
            if len(gen) < 2 or len(real) < 2:
                continue
            g = space.encode(gen)
            r = space.encode(real)

            scores = {
                "mmd2": rbf_mmd2(g, r),
                "mmd2_biased": rbf_mmd2_biased(g, r),
                "centroid_distance": _cosine_centroid_distance(g, r),
                "topic_jsd": _cluster_jsd(g, r, shared["centers"]),
                "n_generated": float(len(gen)),
                "n_real": float(len(real)),
            }

            # Style-residual view: remove the platform-discriminating
            # directions from both pools, leaving similarity that is closer to
            # topical. A function whose raw score is good but whose residual
            # score is no better than the shuffle control learned style only.
            gr, rr = _project_out(g, basis), _project_out(r, basis)
            scores["mmd2_style_residual"] = rbf_mmd2(gr, rr)
            scores["mmd2_biased_style_residual"] = rbf_mmd2_biased(gr, rr)
            scores["centroid_distance_style_residual"] = _cosine_centroid_distance(gr, rr)

            result.cell_scores[cell_id] = scores

        result.notes = {"embedding_backend": getattr(space, "backend", "unknown")}
        return result
