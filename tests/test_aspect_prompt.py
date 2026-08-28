"""Tests for the aspect-aware prompting transfer function.

These are offline: they exercise the join, the skew computation, the prompt
construction and the fallback path, none of which call an LLM.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vectorial_eval.data.schema import TransferTask
from vectorial_eval.transfer.aspect_prompt import (
    DEFAULT_ASPECT_DIR,
    AspectIndex,
    AspectSkew,
    build_user_prompt,
    select_aspects,
)

pytestmark = pytest.mark.skipif(
    not (Path(DEFAULT_ASPECT_DIR) / "aspect_join_index.json").exists(),
    reason="aspect package not present; run scripts/build_aspect_index.py",
)


def _task(**kw) -> TransferTask:
    base = dict(
        task_id="t1",
        cell_id="backend_engineer::kubernetes",
        room="backend_engineer",
        topic="kubernetes",
        domain="Software Engineering",
        split="test",
        source_platform="linkedin",
        target_platform="reddit",
        source_post_id="p1",
        source_text="We moved our services to Kubernetes and cut incident time.",
        target_reference_ids=[],
        exemplar_ids=[],
        heldout_cell=False,
    )
    base.update(kw)
    return TransferTask(**base)


def _skew(name: str, src_w: float, tgt_w: float) -> AspectSkew:
    return AspectSkew(name, "", 0.0, 0.0, source_weight=src_w, target_weight=tgt_w)


def test_shares_are_relative_to_each_platform_pool():
    # kubernetes ships 29 LinkedIn posts against 15 Reddit posts, so a raw
    # count is not comparable across platforms.
    index = AspectIndex()
    look = index.lookup(_task())
    assert look.ok
    scaling = {a.name: (a.source_share, a.target_share) for a in look.aspects}
    # deployment complexity: 10/29 LinkedIn against 6/15 Reddit.
    src, tgt = scaling["deployment complexity"]
    assert src == pytest.approx(10 / 29)
    assert tgt == pytest.approx(6 / 15)


def test_weights_remove_the_platform_aspect_mass_difference():
    # LinkedIn raises more aspects per post than Reddit does, so the weights
    # rather than the shares are what the selection has to read.
    index = AspectIndex()
    look = index.lookup(_task())
    assert look.ok
    assert sum(a.source_weight for a in look.aspects) == pytest.approx(1.0)
    assert sum(a.target_weight for a in look.aspects) == pytest.approx(1.0)
    assert sum(a.skew for a in look.aspects) == pytest.approx(0.0, abs=1e-9)
    assert any(a.skew > 0 for a in look.aspects)


def test_select_aspects_respects_threshold_and_caps():
    aspects = tuple(_skew(f"a{i}", 0.0, i / 10) for i in range(10))
    fore, back = select_aspects(aspects, min_skew=0.1, max_foreground=3, max_background=2)
    assert len(fore) == 3
    assert [a.name for a in fore] == ["a9", "a8", "a7"]
    assert back == []


def test_selection_is_deterministic_under_ties():
    aspects = (_skew("b", 0.0, 0.5), _skew("a", 0.0, 0.5))
    fore, _ = select_aspects(aspects, min_skew=0.1)
    assert [a.name for a in fore] == ["a", "b"]


def test_covered_cell_is_aspect_conditioned():
    index = AspectIndex()
    task = _task()
    look = index.lookup(task)
    assert look.ok and look.cluster == "kubernetes"
    prompt, info = build_user_prompt(task, look)
    assert info["aspect_path"] == "aspect_conditioned"
    assert info["n_aspects_foregrounded"] > 0
    assert "Evaluative emphasis for this topic" in prompt
    assert task.source_text in prompt


def test_uncovered_cell_degrades_to_zero_shot():
    index = AspectIndex()
    # `career development` has no shipped vocabulary.
    task = _task(topic="career development", domain="Career")
    look = index.lookup(task)
    assert not look.ok
    prompt, info = build_user_prompt(task, look)
    assert info["aspect_path"] == "zero_shot_fallback"
    assert info["aspect_reason"] == "no_vocabulary_for_domain_topic"
    assert "Evaluative emphasis" not in prompt
    assert task.source_text in prompt


def test_join_requires_domain_as_well_as_topic():
    index = AspectIndex()
    # The topic is real but this domain never appears with it. Joining on topic
    # alone would silently hand back another cluster's vocabulary.
    assert not index.lookup(_task(topic="kubernetes", domain="EdTech")).ok


def test_domain_merged_clusters_are_flagged():
    index = AspectIndex()
    look = index.lookup(_task(topic="rust programming language", domain="Open Source"))
    assert look.ok and look.domain_merged


def test_missing_index_falls_back_rather_than_raising(tmp_path):
    index = AspectIndex(tmp_path)
    look = index.lookup(_task())
    assert not look.ok and look.reason == "index_missing"
    _, info = build_user_prompt(_task(), look)
    assert info["aspect_path"] == "zero_shot_fallback"


def test_every_index_entry_points_at_a_readable_cluster():
    index = AspectIndex()
    for key, entry in index.join.items():
        path = Path(index.aspect_dir) / "clusters" / entry["file"]
        assert path.exists(), key
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["cluster"] == entry["cluster"]
        assert data["n_linkedin"] > 0 and data["n_reddit"] > 0
