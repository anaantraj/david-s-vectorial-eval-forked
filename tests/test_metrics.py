"""Tests for the statistical core.

These target the properties on which the harness's conclusions rest: that TRM
recovers its theoretical null, that it detects the specific failure mode it was
adopted to detect, and that the shared numerical helpers behave correctly at the
boundaries. Defects here would not raise an exception; they would produce a
plausible but wrong result table, which is the failure mode most difficult to
notice.
"""

from __future__ import annotations

import numpy as np
import pytest

from vectorial_eval.metrics.aspect import (
    MIN_ASPECT_POOL,
    aspect_prevalence_scores,
    bernoulli_jsd,
)
from vectorial_eval.metrics.base import (
    MIN_MMD_POOL,
    audience_cell_bootstrap_ci,
    bootstrap_ci,
    cohens_d,
    jensen_shannon,
    rbf_mmd2,
    rbf_mmd2_biased,
)
from vectorial_eval.metrics.semantic import _entity_surfaces, _numbers, _set_recall
from vectorial_eval.metrics.trm import (
    UNIFORM,
    frechet_distance,
    harmonic_mean_p,
    triangle_rank,
    trm_permutation_test,
)


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-9, None)


def test_detail_surface_extractors_are_normalized_and_exact():
    source = "OpenAI shipped GPT-5 for 1,250 users at 42.5% coverage."
    generated = "OpenAI says GPT-5 reached 1250 users, but only 40%."
    assert _entity_surfaces(source) == {"openai", "gpt-5"}
    assert _numbers(source) == {"1250", "42.5%"}
    assert _set_recall(_numbers(source), _numbers(generated)) == pytest.approx(1 / 2)


def _blocks(a: np.ndarray, b: np.ndarray):
    """Cosine-distance blocks for the triangle_rank signature."""
    an, bn = _unit(a), _unit(b)
    return 1.0 - an @ an.T, 1.0 - bn @ bn.T, 1.0 - an @ bn.T


class TestTriangleRank:
    def test_identical_distributions_approach_zero(self):
        """Samples from one distribution should be near-indistinguishable."""
        rng = np.random.default_rng(0)
        a = rng.normal(size=(60, 16))
        b = rng.normal(size=(60, 16))
        d_cc, d_rr, d_cr = _blocks(a, b)
        assert triangle_rank(d_cc, d_rr, d_cr)["trm"] < 0.02

    def test_separated_distributions_are_penalised(self):
        """A location shift must increase the statistic substantially."""
        rng = np.random.default_rng(1)
        a = rng.normal(size=(60, 16))
        b = rng.normal(size=(60, 16)) + 6.0
        same = triangle_rank(*_blocks(a, rng.normal(size=(60, 16))))["trm"]
        shifted = triangle_rank(*_blocks(a, b))["trm"]
        assert shifted > same * 5

    def test_under_dispersion_raises_rank_i2(self):
        """The mode-collapse signature: candidates tighter than references.

        This is the property for which TRM was adopted. A collapsed candidate
        pool sits at the reference centroid, so reference-to-reference edges are
        long relative to the cross edges, driving I2 above its 1/3 expectation.
        """
        rng = np.random.default_rng(2)
        references = rng.normal(size=(60, 16))
        collapsed = rng.normal(size=(60, 16)) * 0.02  # near-identical candidates
        detail = triangle_rank(*_blocks(collapsed, references))
        assert detail["rank_i2"] > UNIFORM
        assert detail["rank_i2"] > detail["rank_i0"]

    def test_indicators_form_a_distribution(self):
        rng = np.random.default_rng(3)
        detail = triangle_rank(*_blocks(rng.normal(size=(20, 8)), rng.normal(size=(20, 8))))
        total = detail["rank_i0"] + detail["rank_i1"] + detail["rank_i2"]
        assert total == pytest.approx(1.0)

    def test_symmetry_of_the_undirected_statistic(self):
        rng = np.random.default_rng(4)
        a, b = rng.normal(size=(30, 8)), rng.normal(size=(30, 8)) + 1.0
        d_aa, d_bb, d_ab = _blocks(a, b)
        forward = triangle_rank(d_aa, d_bb, d_ab)["trm"]
        reverse = triangle_rank(d_bb, d_aa, d_ab.T)["trm"]
        assert forward == pytest.approx(reverse)


class TestPermutationTest:
    def test_null_is_not_rejected_for_matched_samples(self):
        rng = np.random.default_rng(5)
        a, b = rng.normal(size=(40, 12)), rng.normal(size=(40, 12))
        _, p, _ = trm_permutation_test(a, b, n_permutations=200, seed=0)
        assert p > 0.05

    def test_null_is_rejected_for_separated_samples(self):
        rng = np.random.default_rng(6)
        a, b = rng.normal(size=(40, 12)), rng.normal(size=(40, 12)) + 5.0
        _, p, _ = trm_permutation_test(a, b, n_permutations=200, seed=0)
        assert p < 0.05

    def test_p_value_is_never_zero(self):
        """Add-one smoothing is required: the harmonic mean divides by p."""
        rng = np.random.default_rng(7)
        a, b = rng.normal(size=(30, 8)), rng.normal(size=(30, 8)) + 50.0
        _, p, _ = trm_permutation_test(a, b, n_permutations=50, seed=0)
        assert p > 0.0

    def test_deterministic_given_a_seed(self):
        rng = np.random.default_rng(8)
        a, b = rng.normal(size=(25, 8)), rng.normal(size=(25, 8)) + 0.5
        first = trm_permutation_test(a, b, n_permutations=100, seed=42)
        second = trm_permutation_test(a, b, n_permutations=100, seed=42)
        assert first[0] == second[0] and first[1] == second[1]


class TestHarmonicMeanP:
    def test_dominated_by_the_smallest_value(self):
        assert harmonic_mean_p([0.001, 0.9, 0.9]) < 0.01

    def test_equal_inputs_return_that_value(self):
        assert harmonic_mean_p([0.3, 0.3, 0.3]) == pytest.approx(0.3)

    def test_ignores_non_finite_and_zero(self):
        assert harmonic_mean_p([0.5, float("nan"), 0.0]) == pytest.approx(0.5)

    def test_empty_input_is_nan(self):
        assert np.isnan(harmonic_mean_p([]))


class TestFrechet:
    def test_zero_for_a_distribution_against_itself(self):
        rng = np.random.default_rng(9)
        x = rng.normal(size=(80, 10))
        assert frechet_distance(x, x) == pytest.approx(0.0, abs=1e-3)

    def test_increases_with_separation(self):
        rng = np.random.default_rng(10)
        x = rng.normal(size=(80, 10))
        near = frechet_distance(x, rng.normal(size=(80, 10)) + 0.5)
        far = frechet_distance(x, rng.normal(size=(80, 10)) + 5.0)
        assert far > near

    def test_is_finite_and_real(self):
        """Regression: the eigenvalue formulation must not leak complex values."""
        rng = np.random.default_rng(11)
        value = frechet_distance(rng.normal(size=(20, 30)), rng.normal(size=(20, 30)))
        assert np.isfinite(value) and not isinstance(value, complex)


class TestSharedHelpers:
    def test_jsd_bounds(self):
        assert jensen_shannon(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == pytest.approx(0.0)
        assert jensen_shannon(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(1.0, abs=1e-6)

    def test_cohens_d_sign_and_magnitude(self):
        rng = np.random.default_rng(12)
        a, b = rng.normal(2.0, 1.0, 400), rng.normal(0.0, 1.0, 400)
        assert cohens_d(a, b) == pytest.approx(2.0, abs=0.3)
        assert cohens_d(b, a) < 0

    def test_cohens_d_handles_zero_variance(self):
        constant = np.ones(10)
        assert cohens_d(constant, constant) == 0.0

    def test_mmd_requires_a_minimum_pool(self):
        """Undersized pools must yield NaN rather than a noisy value.

        Averaging the unbiased estimator over 2-4 point pools previously ranked
        the oracle below the identity baseline.
        """
        rng = np.random.default_rng(13)
        small = rng.normal(size=(MIN_MMD_POOL - 1, 8))
        large = rng.normal(size=(30, 8))
        assert np.isnan(rbf_mmd2(small, large))
        assert np.isfinite(rbf_mmd2(large, rng.normal(size=(30, 8))))

    def test_mmd_is_near_zero_for_matched_samples(self):
        rng = np.random.default_rng(14)
        value = rbf_mmd2(rng.normal(size=(60, 8)), rng.normal(size=(60, 8)))
        assert abs(value) < 0.05

    def test_biased_mmd_is_nonnegative_and_guarded(self):
        rng = np.random.default_rng(140)
        small = rng.normal(size=(MIN_MMD_POOL - 1, 8))
        large = rng.normal(size=(30, 8))
        assert np.isnan(rbf_mmd2_biased(small, large))
        assert rbf_mmd2_biased(large, rng.normal(size=(30, 8))) >= 0.0

    def test_bootstrap_ci_brackets_the_mean(self):
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        mean, low, high = bootstrap_ci(values, n_boot=400, seed=0)
        assert mean == pytest.approx(0.3)
        assert low <= mean <= high

    def test_bootstrap_ci_tolerates_degenerate_input(self):
        assert all(np.isnan(v) for v in bootstrap_ci([], n_boot=10))
        assert bootstrap_ci([0.5], n_boot=10) == (0.5, 0.5, 0.5)
        # Non-finite entries must be discarded rather than propagated.
        mean, _, _ = bootstrap_ci([0.2, float("nan"), 0.4], n_boot=100, seed=0)
        assert mean == pytest.approx(0.3)

    def test_audience_cell_bootstrap_balances_audiences_before_cells(self):
        values = {"large": [0.0] * 9, "small": [1.0]}
        mean, low, high = audience_cell_bootstrap_ci(values, n_boot=400, seed=0)
        assert mean == pytest.approx(0.5)
        assert low <= mean <= high
        assert mean != pytest.approx(np.mean(values["large"] + values["small"]))

    def test_audience_cell_bootstrap_handles_empty_and_singleton(self):
        assert all(np.isnan(v) for v in audience_cell_bootstrap_ci({}, n_boot=10))
        assert audience_cell_bootstrap_ci({"a": [0.25]}, n_boot=10) == (
            0.25,
            0.25,
            0.25,
        )


class TestAspectPrevalence:
    def test_bernoulli_jsd_retains_presence_and_absence(self):
        assert bernoulli_jsd(0.2, 0.2) == pytest.approx(0.0)
        assert bernoulli_jsd(0.0, 1.0) == pytest.approx(1.0)

    def test_target_sample_is_best_and_inverted_prevalence_is_worse(self):
        target = np.array([[1, 0], [1, 0], [1, 0], [1, 0], [1, 0]], dtype=float)
        source = 1.0 - target
        matched = aspect_prevalence_scores(target, target, source)
        inverted = aspect_prevalence_scores(source, target, source)
        assert matched["prevalence_jsd"] == pytest.approx(0.0)
        assert matched["shift_progress"] == pytest.approx(1.0)
        assert inverted["prevalence_jsd"] > matched["prevalence_jsd"]
        assert inverted["shift_progress"] == pytest.approx(0.0)

    def test_minimum_pool_and_vocabulary_order_invariance(self):
        small = np.zeros((MIN_ASPECT_POOL - 1, 2))
        large = np.array([[1, 0]] * MIN_ASPECT_POOL, dtype=float)
        assert aspect_prevalence_scores(small, large, 1.0 - large) == {}
        first = aspect_prevalence_scores(large, large, 1.0 - large)
        second = aspect_prevalence_scores(large[:, ::-1], large[:, ::-1], (1.0 - large)[:, ::-1])
        assert first == second

    def test_all_zero_target_does_not_crash(self):
        zeros = np.zeros((MIN_ASPECT_POOL, 3))
        ones = np.ones((MIN_ASPECT_POOL, 3))
        scores = aspect_prevalence_scores(zeros, zeros, ones)
        assert scores["prevalence_jsd"] == pytest.approx(0.0)
