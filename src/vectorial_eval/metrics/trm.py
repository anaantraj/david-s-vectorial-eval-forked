"""Distribution-aware metrics: Triangle-Rank Metrics and Kernel-Based Metrics.

Implements the metric families from:

    Chan, Ni, Ross, Vijayanarasimhan, Myers, Canny.
    "Distribution Aware Metrics for Conditional Natural Language Generation."
    LREC-COLING 2024, pp. 5064-5095.

Motivation
----------
The setting addressed by the paper corresponds closely to the present one. It
concerns conditional generation in which (a) reference diversity carries
information rather than noise, and (b) only tens of references are available per
condition. Both conditions obtain here: a bilateral cell contains between 5 and
150 authentic target posts, and the *dispersion* of an audience's expression on
Reddit constitutes the signal to be reproduced rather than noise to be averaged
away.

The paper further identifies the failure mode this harness must detect. Metrics
based on distance to the nearest reference reward generation at the mode of the
reference distribution, yielding bland and central output. Prompt-based
rewriting is strongly disposed toward this behaviour, emitting a generically
Reddit-like post for every task. Such a system attains a favourable centroid
distance while conveying little information about the audience. TRM penalises
it, because it measures location and dispersion jointly.

MAUVE addresses the same concern but requires thousands of samples for its
k-means density estimate, and the paper demonstrates that it degrades in the
low-reference regime. This excludes it for cells of the present size, which is
the circumstance TRM was introduced to address.

The statistic
-------------
For candidates C and references R under a pairwise distance S, enumerate every
triangle with one vertex in C and two in R. Call the R-R edge the *in* edge and
the two C-R edges the *cross* edges. Record which rank the in-edge takes:

    I0 : in-edge is shortest       (candidates lie outside/between references)
    I1 : in-edge is middle
    I2 : in-edge is longest        (candidates are more dispersed than references)

Under the null hypothesis that C and R are drawn from the same distribution
each rank is equiprobable, so each indicator has expectation 1/3. The directed
statistic is the squared deviation from that uniform profile:

    Q(C,R) = Σ_k ( mean(I_k) - 1/3 )^2

and the reported undirected statistic symmetrises it:

    TRM(C,R) = Q(C,R) + Q(R,C)

TRM is 0 when the two sets are distributionally indistinguishable and grows as
they diverge in either locus or spread. **Lower is better.**

Base distance
-------------
TRM is a meta-metric rather than a distance in its own right: it is defined over
an underlying pairwise distance between texts. The choice of that distance
determines what the statistic is able to detect, so it is recorded in the result
notes of every run.

This implementation uses **cosine distance in the harness embedding space**,
which is TF-IDF with truncated SVD by default and a Sentence-Transformers
checkpoint when one is configured. The style-residual variant applies the same
cosine distance after the platform-discriminating directions have been projected
out.

The statistic requires only that d(x,x) = 0, imposing neither symmetry nor the
triangle inequality, so any text distance may be substituted, including
asymmetric learned distances such as BERTScore.

Significance
------------
A raw TRM value has no natural scale, so each cell additionally receives a
permutation-test p-value: C and R are pooled, reshuffled into groups of the
original sizes, and the proportion of permuted statistics at least as large as
the observed statistic is recorded. Per-cell p-values are combined using the
harmonic mean p-value (Wilson, 2019), following the paper. The harmonic mean
remains valid under arbitrary dependence between tests, which is the relevant
situation here, since cells share audience rooms, an embedding space, and a
generating model.

A large p-value is the favourable outcome for a transfer function: it indicates
that the generated pool is statistically indistinguishable from authentic target
posts.
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
    register,
)

log = logging.getLogger(__name__)

#: TRM requires sufficient points to form informative triangles. Below this
#: threshold the permutation null distribution is too coarse to support
#: inference.
MIN_POOL = 4
UNIFORM = 1.0 / 3.0


def _pairwise_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine distance matrix. Rows index `a`, columns index `b`."""
    an = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-9, None)
    bn = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-9, None)
    return 1.0 - an @ bn.T


def _directed_q(d_cross: np.ndarray, d_in: np.ndarray) -> tuple[float, np.ndarray]:
    """Q(C,R) given the C-R distance matrix and the R-R distances.

    `d_cross` is (n_c, n_r); `d_in` is the (n_r, n_r) within-reference matrix.
    Returns the statistic and the three indicator rates.
    """
    n_r = d_in.shape[0]
    iu = np.triu_indices(n_r, k=1)
    if len(iu[0]) == 0:
        return float("nan"), np.array([np.nan] * 3)

    in_edges = d_in[iu]                 # (P,)
    e0 = d_cross[:, iu[0]]              # (n_c, P)
    e1 = d_cross[:, iu[1]]              # (n_c, P)
    lo = np.minimum(e0, e1)
    hi = np.maximum(e0, e1)

    # The indicators are rendered mutually exclusive so that they sum to unity
    # even under tied distances, which occur frequently among short posts.
    i0 = in_edges[None, :] <= lo
    i2 = (hi <= in_edges[None, :]) & ~i0
    i1 = ~(i0 | i2)

    rates = np.array([i0.mean(), i1.mean(), i2.mean()])
    return float(np.sum((rates - UNIFORM) ** 2)), rates


def triangle_rank(d_cc: np.ndarray, d_rr: np.ndarray, d_cr: np.ndarray) -> dict:
    """Undirected TRM from the three blocks of the pooled distance matrix."""
    q_cr, rates_cr = _directed_q(d_cr, d_rr)
    q_rc, rates_rc = _directed_q(d_cr.T, d_cc)
    return {
        "trm": q_cr + q_rc,
        "q_candidate_ref": q_cr,
        "q_ref_candidate": q_rc,
        # Rank profile of the reference in-edge. An elevated i2 indicates that
        # the candidate pool is under-dispersed relative to the references, the
        # signature of mode collapse; an elevated i0 indicates that candidates
        # lie outside the reference cloud.
        "rank_i0": rates_cr[0],
        "rank_i1": rates_cr[1],
        "rank_i2": rates_cr[2],
    }


def trm_permutation_test(
    embeddings_c: np.ndarray,
    embeddings_r: np.ndarray,
    n_permutations: int = 200,
    seed: int = 0,
) -> tuple[float, float, dict]:
    """Observed TRM, permutation p-value, and the rank profile.

    Pools both sets, computes the full distance matrix once, then permutes
    group membership. Permuting indices rather than recomputing distances keeps
    the procedure inexpensive enough to run on every cell.
    """
    n_c, n_r = len(embeddings_c), len(embeddings_r)
    pooled = np.vstack([embeddings_c, embeddings_r])
    d = _pairwise_cosine(pooled, pooled)
    np.fill_diagonal(d, 0.0)

    def stat(idx_c: np.ndarray, idx_r: np.ndarray) -> float:
        return triangle_rank(
            d[np.ix_(idx_c, idx_c)], d[np.ix_(idx_r, idx_r)], d[np.ix_(idx_c, idx_r)]
        )["trm"]

    idx_c0 = np.arange(n_c)
    idx_r0 = np.arange(n_c, n_c + n_r)
    detail = triangle_rank(
        d[np.ix_(idx_c0, idx_c0)], d[np.ix_(idx_r0, idx_r0)], d[np.ix_(idx_c0, idx_r0)]
    )
    observed = detail["trm"]
    if not np.isfinite(observed):
        return float("nan"), float("nan"), detail

    rng = np.random.default_rng(seed)
    total = n_c + n_r
    exceed = 0
    for _ in range(n_permutations):
        perm = rng.permutation(total)
        s = stat(perm[:n_c], perm[n_c:])
        if np.isfinite(s) and s >= observed:
            exceed += 1
    # Add-one smoothing ensures p is never exactly zero, which is required
    # because the harmonic mean p-value divides by p.
    p = (exceed + 1) / (n_permutations + 1)
    return observed, p, detail


def harmonic_mean_p(p_values: list[float]) -> float:
    """Harmonic mean p-value (Wilson, 2019).

    Employed by the paper to pool per-item tests. Unlike Fisher's method it
    remains valid under arbitrary dependence between tests, which is the
    situation here: cells share audience rooms, an embedding space, and a
    generating model.
    """
    ps = [p for p in p_values if np.isfinite(p) and p > 0]
    if not ps:
        return float("nan")
    return float(len(ps) / np.sum([1.0 / p for p in ps]))


def frechet_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    """Frechet (2-Wasserstein between fitted Gaussians) distance — the FID-BERT
    style kernel-based metric from the paper, computed in the harness's
    embedding space rather than BERT's.

    Less expensive and more numerically stable than TRM at small sample sizes,
    but sensitive only to the first two moments; it is therefore reported as a
    companion statistic rather than a replacement.
    """
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    mu_a, mu_b = a.mean(0), b.mean(0)
    cov_a = np.cov(a, rowvar=False) + eps * np.eye(a.shape[1])
    cov_b = np.cov(b, rowvar=False) + eps * np.eye(b.shape[1])

    # tr((AB)^{1/2}) is computed from the eigenvalues of the product rather than
    # via scipy.linalg.sqrtm. The matrix square root of a non-symmetric product
    # is expensive and numerically delicate, and the `disp` argument that
    # previously suppressed its warnings was removed in recent SciPy releases.
    # Only the trace is required, and for the product of two positive
    # semi-definite matrices that equals the sum of the square roots of the
    # eigenvalues, which are real and non-negative up to rounding.
    eigenvalues = np.linalg.eigvals(cov_a @ cov_b)
    trace_sqrt = float(np.sum(np.sqrt(np.clip(eigenvalues.real, 0.0, None))))

    diff = mu_a - mu_b
    return float(diff @ diff + np.trace(cov_a) + np.trace(cov_b) - 2 * trace_sqrt)


@register("trm")
class TriangleRankMetric(Metric):
    directions = {
        "trm": -1,
        "trm_pvalue": +1,          # high p = indistinguishable from real = good
        "trm_style_residual": -1,
        "trm_pvalue_style_residual": +1,
        "q_candidate_ref": -1,
        "q_ref_candidate": -1,
        "frechet": -1,
        "rank_i0": 0,              # diagnostic: candidates outside reference cloud
        "rank_i2": 0,              # diagnostic: candidate under-dispersion
    }

    def __init__(self, n_permutations: int = 200, style_residual: bool = True):
        self.n_permutations = n_permutations
        self.style_residual = style_residual

    def score(self, transfer_fn, outputs, ctx: EvalContext) -> MetricResult:
        space = ctx.embedding_space
        if space is None:
            raise RuntimeError("trm metric requires ctx.embedding_space")

        pools = group_by_cell(outputs)
        result = MetricResult(metric=self.name, transfer_fn=transfer_fn)
        p_values: list[float] = []

        basis = ensure_shared_reference(ctx)["style_basis"] if self.style_residual else None

        for cell_id, gen in pools.items():
            real = ctx.target_pool(cell_id)
            if len(gen) < MIN_POOL or len(real) < MIN_POOL:
                continue

            g = space.encode(gen)
            r = space.encode(real)
            trm, p, detail = trm_permutation_test(
                g, r, n_permutations=self.n_permutations, seed=ctx.seed
            )
            p_values.append(p)

            scores = {
                "trm": trm,
                "trm_pvalue": p,
                "q_candidate_ref": detail["q_candidate_ref"],
                "q_ref_candidate": detail["q_ref_candidate"],
                "rank_i0": detail["rank_i0"],
                "rank_i1": detail["rank_i1"],
                "rank_i2": detail["rank_i2"],
                "frechet": frechet_distance(g, r),
                "n_generated": float(len(gen)),
                "n_real": float(len(real)),
            }

            # The same raw-versus-residual pairing as the distributional
            # metric: a TRM that is favourable only before style is projected
            # out indicates that the function matched register, not content.
            if basis is not None:
                proj = lambda x: x - (x @ basis) @ basis.T  # noqa: E731
                gr, rr = proj(g), proj(r)
                trm_res, p_res, _ = trm_permutation_test(
                    gr, rr, n_permutations=self.n_permutations, seed=ctx.seed
                )
                scores["trm_style_residual"] = trm_res
                scores["trm_pvalue_style_residual"] = p_res

            result.cell_scores[cell_id] = scores

        result.overall = {
            "harmonic_mean_pvalue": harmonic_mean_p(p_values),
            "n_cells_tested": float(len(p_values)),
            "n_cells_indistinguishable_p05": float(
                sum(1 for p in p_values if np.isfinite(p) and p > 0.05)
            ),
        }
        backend = getattr(space, "backend", "unknown")
        model = getattr(space, "model_name", None)
        result.notes = {
            "reference": "Chan et al., LREC-COLING 2024, 'Distribution Aware "
                         "Metrics for Conditional Natural Language Generation'",
            # TRM is a meta-metric over a pairwise distance. The choice of base
            # distance determines what the statistic can detect, so it is
            # recorded alongside every result rather than left implicit.
            "pairwise_distance": "cosine",
            "distance_space": f"{backend}"
                              + (f" ({model})" if model and model != backend else "")
                              + f", dim={getattr(space, 'dim', '?')}",
            "n_permutations": self.n_permutations,
            "interpretation": (
                "trm: 0 = indistinguishable, higher = more divergent. "
                "trm_pvalue: high = candidate pool statistically indistinguishable "
                "from real target posts. rank_i2 high = candidates under-dispersed "
                "relative to references (mode collapse)."
            ),
        }
        return result
