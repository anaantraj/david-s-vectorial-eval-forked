"""Feature-level distributional comparison over interpretable style features.

Where the classifier metric yields a single figure, this metric provides the
decomposition that addresses the client-facing question raised in the meetings:
which properties specifically distinguish these audiences, and which of those
differences did the transfer function reproduce?

For each stylometric feature, three things are computed per cell:

  gen_mean / real_mean   Raw means, for direct inspection.
  jsd                    Jensen-Shannon divergence between the binned
                         distributions, hence sensitive to shape rather than
                         location alone (0 = identical, 1 = disjoint).
  effect_gap             |d(gen, source) - d(real, source)|, where d denotes
                         Cohen's d. This assesses whether the function displaced
                         the feature by the correct magnitude relative to the
                         authentic platform difference, a stricter criterion
                         than displacement in the correct direction.

`feature_coverage` summarises the result as the proportion of features whose
authentic source-to-target shift the function reproduced to within half its
magnitude.
"""

from __future__ import annotations

import numpy as np

from ..features import stylometry
from .base import (
    EvalContext,
    Metric,
    MetricResult,
    cohens_d,
    group_by_cell,
    jensen_shannon,
    register,
)

N_BINS = 10


def _binned(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges)
    return counts.astype(float) + 1e-9


@register("structural")
class StructuralFeatureMatch(Metric):
    directions = {
        "mean_feature_jsd": -1,
        "mean_effect_gap": -1,
        "feature_coverage": +1,
    }

    def score(self, transfer_fn, outputs, ctx: EvalContext) -> MetricResult:
        pools = group_by_cell(outputs)
        result = MetricResult(metric=self.name, transfer_fn=transfer_fn)
        per_feature_gap: dict[str, list[float]] = {f: [] for f in stylometry.FEATURE_NAMES}

        for cell_id, gen in pools.items():
            real = ctx.target_pool(cell_id)
            source = ctx.source_pool(cell_id)
            if len(gen) < 2 or len(real) < 2 or len(source) < 2:
                continue

            g, r, s = (stylometry.matrix(x) for x in (gen, real, source))

            jsds, effect_gaps, covered = [], [], []
            detail = {}
            for i, fname in enumerate(stylometry.FEATURE_NAMES):
                gi, ri, si = g[:, i], r[:, i], s[:, i]

                # Bin edges are shared across all three pools; otherwise the
                # JSD would compare distributions on incompatible supports.
                lo = float(min(gi.min(), ri.min(), si.min()))
                hi = float(max(gi.max(), ri.max(), si.max()))
                edges = np.linspace(lo, hi + 1e-9, N_BINS + 1) if hi > lo else None
                jsd = jensen_shannon(_binned(gi, edges), _binned(ri, edges)) if edges is not None else 0.0
                jsds.append(jsd)

                d_real = cohens_d(ri, si)      # the authentic platform shift
                d_gen = cohens_d(gi, si)       # the shift produced by the function
                gap = abs(d_gen - d_real) if np.isfinite(d_real) and np.isfinite(d_gen) else np.nan
                if np.isfinite(gap):
                    effect_gaps.append(gap)
                    per_feature_gap[fname].append(gap)
                    # Reproduction is defined as agreement to within half the
                    # magnitude of the authentic shift.
                    covered.append(gap <= max(0.5 * abs(d_real), 0.1))

                detail[f"{fname}__gen_mean"] = float(gi.mean())
                detail[f"{fname}__real_mean"] = float(ri.mean())
                detail[f"{fname}__source_mean"] = float(si.mean())
                detail[f"{fname}__jsd"] = float(jsd)
                detail[f"{fname}__effect_gap"] = float(gap)

            result.cell_scores[cell_id] = {
                "mean_feature_jsd": float(np.mean(jsds)) if jsds else np.nan,
                "mean_effect_gap": float(np.mean(effect_gaps)) if effect_gaps else np.nan,
                "feature_coverage": float(np.mean(covered)) if covered else np.nan,
                **detail,
            }

        # Features this transfer function systematically fails to reproduce.
        # This constitutes the table appropriate for a client asking which
        # properties the model failed to capture.
        ranked = sorted(
            ((f, float(np.mean(v))) for f, v in per_feature_gap.items() if v),
            key=lambda kv: -kv[1],
        )
        result.notes = {
            "worst_features": ranked[:8],
            "best_features": ranked[-8:][::-1],
        }
        return result
