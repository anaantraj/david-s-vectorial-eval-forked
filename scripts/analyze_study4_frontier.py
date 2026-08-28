#!/usr/bin/env python
"""Compute the preregistered Study 4 content-versus-TRM frontier evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SOURCE_KEY = "semantic.source_similarity"
CONTENT_KEY = "semantic.content_word_retention"
TRM_KEY = "trm.trm"
REFERENCE_COLUMNS = ("identity", "shuffle_control", "target_sample")


def metric_cells(report: dict, system: str, family: str, score: str) -> dict[str, float]:
    rows = report.get("per_cell", {}).get(f"{system}.{family}", {})
    allowed = None
    if system not in REFERENCE_COLUMNS:
        recorded = report.get("common_cell_ids", {}).get(family)
        if recorded is not None:
            allowed = set(recorded)
    return {
        cell_id: float(values[score])
        for cell_id, values in rows.items()
        if (allowed is None or cell_id in allowed)
        and score in values
        and np.isfinite(values[score])
    }


def audience_balanced_mean(values: dict[str, float], room_by_cell: dict[str, str]) -> float:
    grouped: dict[str, list[float]] = {}
    for cell_id, value in values.items():
        grouped.setdefault(room_by_cell[cell_id], []).append(value)
    if not grouped:
        return float("nan")
    return float(np.mean([np.mean(grouped[room]) for room in sorted(grouped)]))


def paired_audience_cell_ci(
    candidate: dict[str, float],
    baseline: dict[str, float],
    room_by_cell: dict[str, str],
    *,
    n_boot: int,
    seed: int,
) -> dict:
    shared = sorted(set(candidate) & set(baseline))
    grouped: dict[str, list[float]] = {}
    for cell_id in shared:
        grouped.setdefault(room_by_cell[cell_id], []).append(
            candidate[cell_id] - baseline[cell_id]
        )
    if not grouped:
        return {
            "mean_delta": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_cells": 0,
            "n_audiences": 0,
        }
    audiences = sorted(grouped)
    point = float(np.mean([np.mean(grouped[room]) for room in audiences]))
    if len(shared) == 1:
        low = high = point
    else:
        rng = np.random.default_rng(seed)
        draws = []
        for _ in range(n_boot):
            sampled_rooms = rng.choice(audiences, size=len(audiences), replace=True)
            room_means = []
            for room in sampled_rooms:
                values = np.asarray(grouped[str(room)], dtype=float)
                room_means.append(
                    rng.choice(values, size=len(values), replace=True).mean()
                )
            draws.append(float(np.mean(room_means)))
        low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "mean_delta": point,
        "ci_low": float(low),
        "ci_high": float(high),
        "n_cells": len(shared),
        "n_audiences": len(grouped),
    }


def frontier(points: dict[str, dict[str, float]]) -> list[str]:
    """Return systems not dominated on preservation and target-match estimates."""
    selected = []
    for name, point in points.items():
        dominated = any(
            other != name
            and candidate["source_similarity"] >= point["source_similarity"]
            and candidate["content_word_retention"] >= point["content_word_retention"]
            and candidate["trm"] <= point["trm"]
            and (
                candidate["source_similarity"] > point["source_similarity"]
                or candidate["content_word_retention"]
                > point["content_word_retention"]
                or candidate["trm"] < point["trm"]
            )
            for other, candidate in points.items()
        )
        if not dominated:
            selected.append(name)
    return sorted(selected)


def reference_bounds(points: dict[str, dict[str, float]]) -> dict[str, float]:
    missing = [name for name in REFERENCE_COLUMNS if name not in points]
    if missing:
        raise ValueError(f"report lacks required reference columns: {missing}")
    refs = [points[name] for name in REFERENCE_COLUMNS]
    return {
        "source_worst": min(point["source_similarity"] for point in refs),
        "source_best": max(point["source_similarity"] for point in refs),
        "content_worst": min(point["content_word_retention"] for point in refs),
        "content_best": max(point["content_word_retention"] for point in refs),
        "trm_best": min(point["trm"] for point in refs),
        "trm_worst": max(point["trm"] for point in refs),
        "estimated_from": list(REFERENCE_COLUMNS),
    }


def point_hypervolume(point: dict[str, float], bounds: dict[str, float]) -> float:
    source_span = bounds["source_best"] - bounds["source_worst"]
    content_span = bounds["content_best"] - bounds["content_worst"]
    trm_span = bounds["trm_worst"] - bounds["trm_best"]
    if source_span <= 0 or content_span <= 0 or trm_span <= 0:
        return float("nan")
    source_gain = np.clip(
        (point["source_similarity"] - bounds["source_worst"]) / source_span, 0, 1
    )
    trm_gain = np.clip((bounds["trm_worst"] - point["trm"]) / trm_span, 0, 1)
    content_gain = np.clip(
        (point["content_word_retention"] - bounds["content_worst"])
        / content_span,
        0,
        1,
    )
    return float(source_gain * content_gain * trm_gain)


def analyze(
    report: dict,
    room_by_cell: dict[str, str],
    comparisons: list[tuple[str, str]],
    *,
    source_margin: float,
    trm_margin: float,
    n_boot: int,
    seed: int,
    content_margin: float = 0.01,
) -> dict:
    points = {}
    for system in sorted(report.get("table", {})):
        source = metric_cells(report, system, "semantic", "source_similarity")
        content = metric_cells(
            report, system, "semantic", "content_word_retention"
        )
        trm = metric_cells(report, system, "trm", "trm")
        shared = set(source) & set(content) & set(trm)
        if not shared:
            continue
        points[system] = {
            "source_similarity": audience_balanced_mean(
                {cell: source[cell] for cell in shared}, room_by_cell
            ),
            "trm": audience_balanced_mean(
                {cell: trm[cell] for cell in shared}, room_by_cell
            ),
            "content_word_retention": audience_balanced_mean(
                {cell: content[cell] for cell in shared}, room_by_cell
            ),
            "n_cells": len(shared),
            "n_audiences": len({room_by_cell[cell] for cell in shared}),
        }
    bounds = reference_bounds(points)
    for point in points.values():
        point["reference_normalized_hypervolume"] = point_hypervolume(point, bounds)

    evidence = {}
    for baseline, candidate in comparisons:
        if baseline not in points or candidate not in points:
            missing = [name for name in (baseline, candidate) if name not in points]
            evidence[f"{baseline}:{candidate}"] = {
                "status": "missing_system",
                "missing": missing,
                "frontier_improvement_supported": False,
                "rule": (
                    "no verdict is computed when a required system has no scored "
                    "outputs; a null training result remains in training metadata"
                ),
            }
            continue
        candidate_source = metric_cells(
            report, candidate, "semantic", "source_similarity"
        )
        baseline_source = metric_cells(
            report, baseline, "semantic", "source_similarity"
        )
        candidate_content = metric_cells(
            report, candidate, "semantic", "content_word_retention"
        )
        baseline_content = metric_cells(
            report, baseline, "semantic", "content_word_retention"
        )
        candidate_trm = metric_cells(report, candidate, "trm", "trm")
        baseline_trm = metric_cells(report, baseline, "trm", "trm")
        comparison_cells = set(candidate_source) & set(baseline_source)
        comparison_cells &= set(candidate_content) & set(baseline_content)
        comparison_cells &= set(candidate_trm) & set(baseline_trm)
        source_delta = paired_audience_cell_ci(
            {cell: candidate_source[cell] for cell in comparison_cells},
            {cell: baseline_source[cell] for cell in comparison_cells},
            room_by_cell,
            n_boot=n_boot,
            seed=seed,
        )
        trm_delta = paired_audience_cell_ci(
            {cell: candidate_trm[cell] for cell in comparison_cells},
            {cell: baseline_trm[cell] for cell in comparison_cells},
            room_by_cell,
            n_boot=n_boot,
            seed=seed + 1,
        )
        content_delta = paired_audience_cell_ci(
            {cell: candidate_content[cell] for cell in comparison_cells},
            {cell: baseline_content[cell] for cell in comparison_cells},
            room_by_cell,
            n_boot=n_boot,
            seed=seed + 2,
        )
        lower_trm = trm_delta["ci_high"] < 0
        higher_source = source_delta["ci_low"] > 0
        higher_content = content_delta["ci_low"] > 0
        source_noninferior = source_delta["ci_low"] >= -source_margin
        content_noninferior = content_delta["ci_low"] >= -content_margin
        trm_noninferior = trm_delta["ci_high"] <= trm_margin
        improved = (
            lower_trm and source_noninferior and content_noninferior
        ) or (
            higher_source and trm_noninferior and content_noninferior
        ) or (
            higher_content and source_noninferior and trm_noninferior
        )
        record = {
            "status": "scored",
            "source_similarity_delta": source_delta,
            "trm_delta": trm_delta,
            "content_word_retention_delta": content_delta,
            "source_noninferiority_margin": source_margin,
            "content_noninferiority_margin": content_margin,
            "trm_noninferiority_margin": trm_margin,
            "common_cross_metric_cells": sorted(comparison_cells),
            "n_common_cross_metric_cells": len(comparison_cells),
            "frontier_improvement_supported": bool(improved),
            "rule": (
                "one of source cosine, content-word retention, or TRM has a "
                "favourable separated interval and both remaining objectives "
                "are non-inferior under frozen margins"
            ),
        }
        if candidate.startswith("local_plan_then_transfer_"):
            candidate_entity = metric_cells(
                report, candidate, "semantic", "entity_surface_recall"
            )
            baseline_entity = metric_cells(
                report, baseline, "semantic", "entity_surface_recall"
            )
            candidate_number = metric_cells(
                report, candidate, "semantic", "number_recall"
            )
            baseline_number = metric_cells(
                report, baseline, "semantic", "number_recall"
            )
            plan_cells = comparison_cells & set(candidate_entity) & set(baseline_entity)
            plan_cells &= set(candidate_number) & set(baseline_number)
            entity_delta = paired_audience_cell_ci(
                {cell: candidate_entity[cell] for cell in plan_cells},
                {cell: baseline_entity[cell] for cell in plan_cells},
                room_by_cell,
                n_boot=n_boot,
                seed=seed + 3,
            )
            number_delta = paired_audience_cell_ci(
                {cell: candidate_number[cell] for cell in plan_cells},
                {cell: baseline_number[cell] for cell in plan_cells},
                room_by_cell,
                n_boot=n_boot,
                seed=seed + 4,
            )
            plan_content_delta = paired_audience_cell_ci(
                {cell: candidate_content[cell] for cell in plan_cells},
                {cell: baseline_content[cell] for cell in plan_cells},
                room_by_cell,
                n_boot=n_boot,
                seed=seed + 5,
            )
            plan_trm_delta = paired_audience_cell_ci(
                {cell: candidate_trm[cell] for cell in plan_cells},
                {cell: baseline_trm[cell] for cell in plan_cells},
                room_by_cell,
                n_boot=n_boot,
                seed=seed + 6,
            )
            record["extract_then_transfer"] = {
                "common_cells": sorted(plan_cells),
                "n_common_cells": len(plan_cells),
                "entity_surface_recall_delta": entity_delta,
                "number_recall_delta": number_delta,
                "content_word_retention_delta": plan_content_delta,
                "trm_delta": plan_trm_delta,
                "retention_improvement_supported": bool(
                    entity_delta["ci_low"] > 0
                    and number_delta["ci_low"] > 0
                    and plan_content_delta["ci_low"] > 0
                    and plan_trm_delta["ci_high"] <= trm_margin
                ),
                "rule": (
                    "entity, number, and content-word intervals must each "
                    "separate favourably; TRM must be non-inferior"
                ),
            }
        evidence[f"{baseline}:{candidate}"] = record
    return {
        "embedding_space": report.get("embedding_space"),
        "bootstrap": "audiences, then eligible cells within sampled audience",
        "points": points,
        "point_estimate_frontier": frontier(points),
        "reference_bounds": bounds,
        "comparisons": evidence,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("cells", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--comparison",
        action="append",
        default=[],
        help="BASELINE:CANDIDATE; may be supplied more than once",
    )
    parser.add_argument("--source-margin", type=float, default=0.01)
    parser.add_argument("--trm-margin", type=float, default=0.01)
    parser.add_argument("--content-margin", type=float, default=0.01)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    with args.cells.open(encoding="utf-8") as fh:
        cells = [json.loads(line) for line in fh if line.strip()]
    room_by_cell = {cell["cell_id"]: cell["room"] for cell in cells}
    comparisons = []
    for raw in args.comparison:
        baseline, sep, candidate = raw.partition(":")
        if not sep or not baseline or not candidate:
            raise ValueError(f"invalid comparison {raw!r}; expected BASELINE:CANDIDATE")
        comparisons.append((baseline, candidate))
    result = analyze(
        report,
        room_by_cell,
        comparisons,
        source_margin=args.source_margin,
        trm_margin=args.trm_margin,
        n_boot=args.n_bootstrap,
        seed=args.seed,
        content_margin=args.content_margin,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
