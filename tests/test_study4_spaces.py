"""Checks for cross-space Study 4 verdict summarization."""

from __future__ import annotations

import json

from scripts.analyze_study4_spaces import analyze_spaces


def _report(slug: str, candidate_trm: float) -> dict:
    systems = {
        "identity": (1.0, 0.8),
        "shuffle_control": (0.2, 0.2),
        "target_sample": (0.6, 0.1),
        "base": (0.7, 0.4),
        "candidate": (0.7, candidate_trm),
    }
    per_cell = {}
    for name, (source, trm) in systems.items():
        per_cell[f"{name}.semantic"] = {
            cell: {
                "source_similarity": source,
                "content_word_retention": source,
            }
            for cell in ("a1", "b1")
        }
        per_cell[f"{name}.trm"] = {
            cell: {"trm": trm} for cell in ("a1", "b1")
        }
    return {
        "embedding_space": {"slug": slug},
        "table": {name: {} for name in systems},
        "per_cell": per_cell,
    }


def test_space_summary_keeps_values_separate_and_counts_verdicts(tmp_path):
    cells = tmp_path / "cells.jsonl"
    cells.write_text(
        json.dumps({"cell_id": "a1", "room": "a"})
        + "\n"
        + json.dumps({"cell_id": "b1", "room": "b"})
        + "\n"
    )
    for slug, trm in (("space-a", 0.3), ("space-b", 0.3)):
        (tmp_path / f"report.heldout.{slug}.json").write_text(
            json.dumps(_report(slug, trm))
        )
    summary = analyze_spaces(
        tmp_path,
        cells,
        "heldout",
        [("base", "candidate")],
        source_margin=0.01,
        trm_margin=0.01,
        n_boot=20,
        seed=0,
    )
    assert summary["n_spaces"] == 2
    assert summary["values_pooled_across_spaces"] is False
    assert summary["comparisons"]["base:candidate"]["n_supported"] == 2
    assert (tmp_path / "frontier.heldout.space-a.json").exists()
