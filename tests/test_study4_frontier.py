"""Statistical checks for the Study 4 frontier analyzer."""

from __future__ import annotations

import pytest

from scripts.analyze_study4_frontier import (
    analyze,
    audience_balanced_mean,
    frontier,
    paired_audience_cell_ci,
    point_hypervolume,
    reference_bounds,
)


def test_audience_balanced_point_does_not_weight_a_large_audience_more():
    rooms = {**{f"a{i}": "large" for i in range(9)}, "b": "small"}
    values = {**{f"a{i}": 0.0 for i in range(9)}, "b": 1.0}
    assert audience_balanced_mean(values, rooms) == pytest.approx(0.5)


def test_paired_interval_is_over_within_cell_differences():
    rooms = {"a1": "a", "a2": "a", "b1": "b", "b2": "b"}
    baseline = {cell: 0.2 for cell in rooms}
    candidate = {cell: 0.3 for cell in rooms}
    result = paired_audience_cell_ci(
        candidate, baseline, rooms, n_boot=100, seed=0
    )
    assert result["mean_delta"] == pytest.approx(0.1)
    assert result["ci_low"] == pytest.approx(0.1)
    assert result["ci_high"] == pytest.approx(0.1)


def test_frontier_uses_higher_source_and_lower_trm():
    points = {
        "dominated": {
            "source_similarity": 0.7,
            "content_word_retention": 0.7,
            "trm": 0.4,
        },
        "content": {
            "source_similarity": 0.9,
            "content_word_retention": 0.8,
            "trm": 0.4,
        },
        "style": {
            "source_similarity": 0.7,
            "content_word_retention": 0.7,
            "trm": 0.2,
        },
    }
    assert frontier(points) == ["content", "style"]


def test_reference_hypervolume_uses_all_three_frozen_objectives():
    references = {
        "identity": {
            "source_similarity": 1.0,
            "content_word_retention": 1.0,
            "trm": 0.8,
        },
        "shuffle_control": {
            "source_similarity": 0.2,
            "content_word_retention": 0.1,
            "trm": 0.2,
        },
        "target_sample": {
            "source_similarity": 0.6,
            "content_word_retention": 0.5,
            "trm": 0.1,
        },
    }
    bounds = reference_bounds(references)
    strong = {"source_similarity": 0.8, "content_word_retention": 0.8, "trm": 0.3}
    weak_content = dict(strong, content_word_retention=0.4)
    assert point_hypervolume(strong, bounds) > point_hypervolume(
        weak_content, bounds
    )


def _report() -> dict:
    systems = {
        "identity": (1.0, 0.8),
        "shuffle_control": (0.2, 0.2),
        "target_sample": (0.6, 0.1),
        "base": (0.70, 0.40),
        "candidate": (0.70, 0.30),
    }
    per_cell = {}
    for name, (source, trm) in systems.items():
        per_cell[f"{name}.semantic"] = {
            cell: {
                "source_similarity": source,
                "content_word_retention": source,
            }
            for cell in ("a1", "a2", "b1", "b2")
        }
        per_cell[f"{name}.trm"] = {
            cell: {"trm": trm} for cell in ("a1", "a2", "b1", "b2")
        }
    return {"table": {name: {} for name in systems}, "per_cell": per_cell}


def test_supported_gain_requires_separation_and_noninferiority():
    rooms = {"a1": "a", "a2": "a", "b1": "b", "b2": "b"}
    result = analyze(
        _report(),
        rooms,
        [("base", "candidate")],
        source_margin=0.01,
        trm_margin=0.01,
        n_boot=100,
        seed=0,
    )
    comparison = result["comparisons"]["base:candidate"]
    assert comparison["frontier_improvement_supported"] is True
    assert result["reference_bounds"]["estimated_from"] == [
        "identity",
        "shuffle_control",
        "target_sample",
    ]


def test_trm_gain_is_rejected_when_content_retention_drops():
    report = _report()
    for values in report["per_cell"]["candidate.semantic"].values():
        values["content_word_retention"] = 0.5
    rooms = {"a1": "a", "a2": "a", "b1": "b", "b2": "b"}
    result = analyze(
        report,
        rooms,
        [("base", "candidate")],
        source_margin=0.01,
        trm_margin=0.01,
        n_boot=50,
        seed=0,
    )
    comparison = result["comparisons"]["base:candidate"]
    assert comparison["frontier_improvement_supported"] is False
    assert comparison["content_word_retention_delta"]["ci_high"] < 0


def test_extract_then_transfer_uses_its_stricter_retention_rule():
    report = _report()
    plan = "local_plan_then_transfer_s4_plan_prior_10k"
    report["table"][plan] = {}
    report["per_cell"][f"{plan}.semantic"] = {}
    report["per_cell"][f"{plan}.trm"] = {}
    for cell in ("a1", "a2", "b1", "b2"):
        report["per_cell"]["base.semantic"][cell].update(
            {"entity_surface_recall": 0.5, "number_recall": 0.5}
        )
        report["per_cell"][f"{plan}.semantic"][cell] = {
            "source_similarity": 0.7,
            "content_word_retention": 0.8,
            "entity_surface_recall": 0.7,
            "number_recall": 0.7,
        }
        report["per_cell"][f"{plan}.trm"][cell] = {"trm": 0.3}
    rooms = {"a1": "a", "a2": "a", "b1": "b", "b2": "b"}
    result = analyze(
        report,
        rooms,
        [("base", plan)],
        source_margin=0.01,
        trm_margin=0.01,
        n_boot=30,
        seed=0,
    )
    special = result["comparisons"][f"base:{plan}"]["extract_then_transfer"]
    assert special["n_common_cells"] == 4
    assert special["retention_improvement_supported"] is True


def test_interval_axes_use_the_same_cross_metric_cells():
    report = _report()
    rooms = {"a1": "a", "a2": "a", "b1": "b", "b2": "b", "extra": "c"}
    report["per_cell"]["base.semantic"]["extra"] = {"source_similarity": 0.0}
    report["per_cell"]["candidate.semantic"]["extra"] = {
        "source_similarity": 1.0
    }
    result = analyze(
        report,
        rooms,
        [("base", "candidate")],
        source_margin=0.01,
        trm_margin=0.01,
        n_boot=20,
        seed=0,
    )
    comparison = result["comparisons"]["base:candidate"]
    assert comparison["n_common_cross_metric_cells"] == 4
    assert "extra" not in comparison["common_cross_metric_cells"]
    assert comparison["source_similarity_delta"]["mean_delta"] == pytest.approx(0.0)


def test_frontier_obeys_reported_common_system_cell_base():
    report = _report()
    rooms = {"a1": "a", "a2": "a", "b1": "b", "b2": "b", "extra": "c"}
    for system in ("base", "candidate"):
        report["per_cell"][f"{system}.semantic"]["extra"] = {
            "source_similarity": 1.0 if system == "candidate" else 0.0,
            "content_word_retention": 1.0 if system == "candidate" else 0.0,
        }
        report["per_cell"][f"{system}.trm"]["extra"] = {
            "trm": 0.0 if system == "candidate" else 1.0
        }
    report["common_cell_ids"] = {
        "semantic": ["a1", "a2", "b1", "b2"],
        "trm": ["a1", "a2", "b1", "b2"],
    }
    result = analyze(
        report,
        rooms,
        [("base", "candidate")],
        source_margin=0.01,
        trm_margin=0.01,
        n_boot=20,
        seed=0,
    )
    comparison = result["comparisons"]["base:candidate"]
    assert comparison["n_common_cross_metric_cells"] == 4
    assert result["points"]["candidate"]["n_cells"] == 4


def test_missing_trained_system_is_recorded_without_a_false_verdict():
    rooms = {"a1": "a", "a2": "a", "b1": "b", "b2": "b"}
    result = analyze(
        _report(),
        rooms,
        [("base", "not_selected")],
        source_margin=0.01,
        trm_margin=0.01,
        n_boot=20,
        seed=0,
    )
    comparison = result["comparisons"]["base:not_selected"]
    assert comparison["status"] == "missing_system"
    assert comparison["missing"] == ["not_selected"]
    assert comparison["frontier_improvement_supported"] is False
