"""Metric interface and the evaluation context shared by all metrics.

Every metric here is *distributional*: it compares a pool of generated posts
against a pool of real target posts within one bilateral cell. This follows
directly from the constraint in the meeting notes — there is no one-to-one post
matching, so pairwise scores (BLEU against a reference, etc.) are not defined.

Metrics return cell-level scores. Aggregation to a headline number happens once,
in the reporting layer, so that per-cell variance stays visible rather than
being averaged away — with only ~30 viable cells, a single dense cell can
otherwise dominate a mean.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ..data.schema import CellRecord, TransferOutput


@dataclass
class EvalContext:
    """Everything a metric may need, assembled once and shared.

    Building the embedding space once across all transfer functions is load
    bearing: fitting a separate space per function would make their scores
    incomparable.
    """

    cells: dict[str, CellRecord]
    #: cell_id -> platform -> split -> list[text]
    pools: dict[str, dict[str, dict[str, list[str]]]]
    #: post_id -> text, for the whole built corpus
    texts_by_id: dict[str, str]
    split: str
    source_platform: str
    target_platform: str
    embedding_space: object | None = None
    seed: int = 0
    extras: dict = field(default_factory=dict)

    def target_pool(self, cell_id: str, split: str | None = None) -> list[str]:
        """Real target-platform posts for a cell in the given split."""
        return self.pools[cell_id][self.target_platform][split or self.split]

    def source_pool(self, cell_id: str, split: str | None = None) -> list[str]:
        return self.pools[cell_id][self.source_platform][split or self.split]


@dataclass
class MetricResult:
    """Scores from one metric for one transfer function.

    `cell_scores` maps cell_id -> {metric_name: value}; `overall` holds
    corpus-level values that are not decomposable per cell (e.g. a single
    classifier's confusion matrix over all outputs).
    """

    metric: str
    transfer_fn: str
    cell_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    overall: dict[str, float] = field(default_factory=dict)
    notes: dict = field(default_factory=dict)


class Metric(ABC):
    name: str = "unnamed"
    #: Direction for each score this metric emits: +1 if higher is better,
    #: -1 if lower is better, 0 if it is diagnostic rather than a quality score.
    directions: dict[str, int] = {}

    @abstractmethod
    def score(
        self,
        transfer_fn: str,
        outputs: list[TransferOutput],
        ctx: EvalContext,
    ) -> MetricResult:
        ...


_REGISTRY: dict[str, type[Metric]] = {}


def register(name: str):
    def deco(cls):
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def build_metric(name: str, **kwargs) -> Metric:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown metric {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def available() -> list[str]:
    return sorted(_REGISTRY)


# --- shared numeric helpers -------------------------------------------------


def ensure_shared_reference(ctx: EvalContext) -> dict:
    """Fit the once-per-evaluation artifacts derived from real train posts.

    Holds the platform-style basis and the topic-cluster centers. Computed here
    rather than inside whichever metric happens to run first, so that results do
    not depend on metric ordering and every metric shares one basis.
    """
    if "_dist_shared" in ctx.extras:
        return ctx.extras["_dist_shared"]

    from sklearn.cluster import KMeans

    from ..features.embeddings import style_directions

    texts, labels = [], []
    for cell_id in ctx.cells:
        for platform in (ctx.source_platform, ctx.target_platform):
            for t in ctx.pools[cell_id][platform]["train"]:
                texts.append(t)
                labels.append(platform)

    emb = ctx.embedding_space.encode(texts)
    k = int(min(24, max(2, len(texts) // 10)))
    km = KMeans(n_clusters=k, n_init=10, random_state=ctx.seed).fit(emb)
    ctx.extras["_dist_shared"] = {
        "centers": km.cluster_centers_,
        "style_basis": style_directions(emb, labels, k=2),
    }
    return ctx.extras["_dist_shared"]


def group_by_cell(outputs: list[TransferOutput]) -> dict[str, list[str]]:
    """Collect non-empty generated texts per cell.

    Empty strings are dropped from the *pool* (they are not posts) but counted
    separately by the degeneracy metric, so a function that fails half its
    generations cannot hide behind a clean-looking pool.
    """
    pools: dict[str, list[str]] = {}
    for o in outputs:
        if o.output_text and o.output_text.strip():
            pools.setdefault(o.cell_id, []).append(o.output_text)
    return pools


def jensen_shannon(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """JSD between two discrete distributions, in bits (0 = identical, 1 = disjoint)."""
    p = np.clip(np.asarray(p, dtype=float), eps, None)
    q = np.clip(np.asarray(q, dtype=float), eps, None)
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a * np.log2(a / b)))  # noqa: E731
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Standardised mean difference, pooled SD. Sign: positive if a > b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if pooled < 1e-12:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


#: Minimum pool size on each side for MMD to be reported.
#: The unbiased estimator is extremely noisy on pools of 2-4 — it returns
#: negative values and occasional huge outliers that dominate any average over
#: cells. Given how sparse the bilateral cells are, silently averaging those
#: produced a table where the oracle looked worse than the identity baseline.
#: Undersized cells now yield NaN and are excluded rather than adding noise.
MIN_MMD_POOL = 5


def rbf_mmd2(x: np.ndarray, y: np.ndarray, gamma: float | None = None) -> float:
    """Unbiased squared MMD with an RBF kernel.

    Preferred over comparing pool means because it is sensitive to differences
    in the *shape* of a distribution, not just its centre — a transfer function
    that hits the right average register while collapsing all variety should not
    score as a match.

    Returns NaN when either pool is smaller than `MIN_MMD_POOL`.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < MIN_MMD_POOL or len(y) < MIN_MMD_POOL:
        return float("nan")
    if gamma is None:
        # Median heuristic on the pooled sample.
        z = np.vstack([x, y])
        d2 = np.sum((z[:, None, :] - z[None, :, :]) ** 2, axis=-1)
        med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
        gamma = 1.0 / max(med, 1e-9)

    def k(a, b):
        d2 = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
        return np.exp(-gamma * d2)

    kxx, kyy, kxy = k(x, x), k(y, y), k(x, y)
    n, m = len(x), len(y)
    np.fill_diagonal(kxx, 0.0)
    np.fill_diagonal(kyy, 0.0)
    return float(
        kxx.sum() / (n * (n - 1)) + kyy.sum() / (m * (m - 1)) - 2 * kxy.mean()
    )


def rbf_mmd2_biased(x: np.ndarray, y: np.ndarray, gamma: float | None = None) -> float:
    """Non-negative biased (V-statistic) squared MMD with the same pool guard."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < MIN_MMD_POOL or len(y) < MIN_MMD_POOL:
        return float("nan")
    if gamma is None:
        z = np.vstack([x, y])
        d2 = np.sum((z[:, None, :] - z[None, :, :]) ** 2, axis=-1)
        med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
        gamma = 1.0 / max(med, 1e-9)

    def k(a, b):
        d2 = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
        return np.exp(-gamma * d2)

    value = k(x, x).mean() + k(y, y).mean() - 2 * k(x, y).mean()
    return float(max(0.0, value))


def bootstrap_ci(
    values: list[float], n_boot: int = 500, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Mean and percentile CI over cell-level scores.

    Resampling is over *cells*, not posts: cells are the independent unit here,
    and with ~30 of them the uncertainty on any headline number is large enough
    that reporting it without an interval would be misleading.
    """
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(vals) == 1:
        return float(vals[0]), float(vals[0]), float(vals[0])
    rng = np.random.default_rng(seed)
    means = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_boot)]
    return (
        float(vals.mean()),
        float(np.percentile(means, 100 * alpha / 2)),
        float(np.percentile(means, 100 * (1 - alpha / 2))),
    )


def audience_cell_bootstrap_ci(
    values_by_audience: dict[str, list[float]],
    n_boot: int = 500,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Audience-balanced mean and two-stage audience/cell percentile interval."""
    clean = {
        audience: np.asarray([v for v in values if np.isfinite(v)], dtype=float)
        for audience, values in values_by_audience.items()
    }
    clean = {audience: values for audience, values in clean.items() if len(values)}
    if not clean:
        return float("nan"), float("nan"), float("nan")
    audiences = sorted(clean)
    point = float(np.mean([clean[audience].mean() for audience in audiences]))
    if len(audiences) == 1 and len(clean[audiences[0]]) == 1:
        return point, point, point
    rng = np.random.default_rng(seed)
    replicates = []
    for _ in range(n_boot):
        sampled_audiences = rng.choice(audiences, size=len(audiences), replace=True)
        audience_means = []
        for audience in sampled_audiences:
            cells = clean[str(audience)]
            audience_means.append(
                rng.choice(cells, size=len(cells), replace=True).mean()
            )
        replicates.append(float(np.mean(audience_means)))
    return (
        point,
        float(np.percentile(replicates, 100 * alpha / 2)),
        float(np.percentile(replicates, 100 * (1 - alpha / 2))),
    )
