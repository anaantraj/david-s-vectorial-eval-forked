"""Tests for the selectable embedding space.

Every embedding-derived score is defined relative to the space it was computed
in, and TRM is not comparable at all across spaces. Two things therefore have to
hold, and neither would raise an exception if it broke: the per-text encoding
cache must return exactly what the backend would have returned, and a report
computed in a second space must not overwrite one computed in the first.

Only the offline TF-IDF backend is exercised here, so the suite still runs
without the optional neural dependency.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from vectorial_eval.config import HarnessConfig
from vectorial_eval.evaluate import (
    check_space_matches,
    default_report_tag,
    recorded_space_slug,
    report_path,
)
from vectorial_eval.features.embeddings import (
    EmbeddingSpace,
    fit_embedding_space,
    space_slug,
)

CORPUS = [
    "shipping a new backend service this quarter, thrilled to share the news",
    "does anyone actually enjoy on-call rotations or is it just me",
    "excited to announce our latest product milestone with the team",
    "my code editor keeps crashing on large files, any suggestions",
    "grateful for the mentorship that got me here, hiring now",
    "spent the weekend rewriting the deployment pipeline in rust",
]


def _space():
    return fit_embedding_space(CORPUS, backend="tfidf-svd", dim=4, seed=0)


class TestEncodeCache:
    def test_cached_vectors_match_the_backend(self):
        """The cache is a speed measure and must not alter a single value."""
        space = _space()
        uncached = space.transform(CORPUS)
        assert np.allclose(space.encode(CORPUS), uncached)
        # Second call is served entirely from the cache.
        assert np.allclose(space.encode(CORPUS), uncached)

    def test_repeated_and_reordered_texts_line_up(self):
        space = _space()
        texts = [CORPUS[2], CORPUS[0], CORPUS[2]]
        rows = space.encode(texts)
        assert rows.shape == (3, space.dim)
        assert np.allclose(rows[0], rows[2])
        assert np.allclose(rows[1], space.encode([CORPUS[0]])[0])

    def test_backend_sees_each_text_once(self):
        calls: list[list[str]] = []

        def transform(texts):
            calls.append(list(texts))
            return np.ones((len(texts), 3))

        space = EmbeddingSpace(backend="stub", transform=transform, dim=3)
        space.encode(["a", "b", "a"])
        space.encode(["b", "c"])
        assert calls == [["a", "b"], ["c"]]

    def test_empty_input_returns_an_empty_matrix(self):
        space = _space()
        assert space.encode([]).shape == (0, space.dim)


class TestSlug:
    def test_checkpoint_name_is_reduced_to_its_stem(self):
        assert space_slug("google/embeddinggemma-300m", 256) == "embeddinggemma-300m-d256"

    def test_default_backend(self):
        assert space_slug("tfidf-svd", 256) == "tfidf-svd-d256"

    def test_space_reports_its_own_slug(self):
        space = _space()
        assert space.slug == "tfidf-svd-d4"
        assert space.describe()["slug"] == space.slug


class TestReportNaming:
    def test_default_space_keeps_the_plain_filename(self):
        cfg = HarnessConfig()
        assert default_report_tag(cfg) is None
        assert report_path("runs/x", "heldout", None).name == "report.heldout.json"

    def test_a_second_space_lands_in_a_second_file(self):
        cfg = HarnessConfig()
        cfg.eval.embedding_backend = "sentence-transformers"
        cfg.eval.embedding_model = "google/embeddinggemma-300m"
        assert default_report_tag(cfg) == "embeddinggemma-300m-d256"
        assert (
            report_path("runs/x", "heldout", default_report_tag(cfg)).name
            == "report.heldout.embeddinggemma-300m-d256.json"
        )

    def test_a_non_default_dimension_also_qualifies_the_name(self):
        cfg = HarnessConfig()
        cfg.eval.embedding_dim = 128
        assert default_report_tag(cfg) == "tfidf-svd-d128"


class TestSpaceGuard:
    def test_slug_is_read_from_an_explicit_record(self):
        assert recorded_space_slug({"embedding_space": {"slug": "abc-d8"}}) == "abc-d8"

    def test_slug_is_recovered_from_an_older_report(self):
        """Reports predating the explicit record still state their config."""
        older = {"config": {"eval": {"embedding_backend": "tfidf-svd", "embedding_dim": 256}}}
        assert recorded_space_slug(older) == "tfidf-svd-d256"

    def test_unknown_when_nothing_is_recorded(self):
        assert recorded_space_slug({}) is None

    def test_mismatched_space_is_refused(self, tmp_path):
        path = tmp_path / "report.heldout.json"
        path.write_text(json.dumps({"embedding_space": {"slug": "tfidf-svd-d256"}}))
        with pytest.raises(RuntimeError, match="not comparable"):
            check_space_matches(path, "embeddinggemma-300m-d256")

    def test_matching_space_is_allowed(self, tmp_path):
        path = tmp_path / "report.heldout.json"
        path.write_text(json.dumps({"embedding_space": {"slug": "tfidf-svd-d256"}}))
        check_space_matches(path, "tfidf-svd-d256")

    def test_absent_or_unreadable_report_is_not_an_error(self, tmp_path):
        check_space_matches(tmp_path / "missing.json", "tfidf-svd-d256")
        broken = tmp_path / "broken.json"
        broken.write_text("{not json")
        check_space_matches(broken, "tfidf-svd-d256")


class TestBackendSelection:
    def test_unknown_backend_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown embedding backend"):
            fit_embedding_space(CORPUS, backend="word2vec")

    def test_tfidf_rank_is_capped_by_the_corpus(self):
        """A rank above the corpus size would fail inside the SVD."""
        space = fit_embedding_space(CORPUS, backend="tfidf-svd", dim=4096, seed=0)
        assert space.dim <= len(CORPUS)
        assert space.encode(CORPUS).shape == (len(CORPUS), space.dim)
