from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).parents[1] / "experimental-notes/build_study4.py"
    spec = importlib.util.spec_from_file_location("build_study4", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_row_computes_per_audience_shortfalls():
    report = _module()
    row = report.manifest_row(
        "target",
        {
            "n_total": 19,
            "n_train": 17,
            "n_dev": 2,
            "per_audience": 10,
            "available_per_audience": {"a": 10, "b": 9},
        },
    )
    assert "b: 9" in row
    assert "a: 10" not in row


def test_scale_figure_and_training_table_read_artifacts(tmp_path, monkeypatch):
    report = _module()
    monkeypatch.setattr(report, "ROOT", tmp_path)
    names = [
        "s4-reddit-rich-raw-1k",
        "s4-reddit-rich-prior-1k",
        "s4-reddit-rich-raw-10k",
        "s4-reddit-rich-prior-10k",
        "s4-linkedin-rich-platform_lm-1000",
        "s4-linkedin-rich-platform_prior_lm-1000",
        "s4-linkedin-rich-platform_lm-10000",
        "s4-linkedin-rich-platform_prior_lm-10000",
        "s4-reddit-coarse-platform-half",
        "s4-reddit-platform-then-audience",
        "s4-linkedin-coarse-platform-half",
        "s4-linkedin-platform-then-audience",
    ]
    for index, name in enumerate(names):
        root = tmp_path / "runs/checkpoints" / name
        root.mkdir(parents=True)
        n_train = 1_000 if "1k" in name or "1000" in name else 10_000
        (root / "run.json").write_text(
            json.dumps(
                {
                    "variant": "platform_lm",
                    "n_train": n_train,
                    "steps_per_epoch": 10,
                    "train_completion_tokens": 12_345,
                    "quantisation": "nf4-4bit-double",
                }
            ),
            encoding="utf-8",
        )
        (root / "metrics.json").write_text(
            json.dumps(
                {
                    "best_nll": 3.5 - index * 0.05,
                    "train_completion_tokens": 12_345,
                }
            ),
            encoding="utf-8",
        )
    figure = report.scale_svg()
    table = report.training_table()
    assert "Unsupervised development loss" in figure
    assert "training records (log scale)" in figure
    report.audit_svg_layout(figure)
    assert table.count("nf4-4bit-double") == 12
    assert "12,345" in table


def test_svg_layout_audit_rejects_overflow():
    report = _module()
    with pytest.raises(ValueError, match="overflows"):
        report.audit_svg_layout(
            '<svg viewBox="0 0 100 50"><text x="90" y="10">far too long</text></svg>'
        )


def test_findings_never_promote_ungated_aspect_result():
    report = _module()
    frontier = {
        "comparisons": {
            "lora_s4_base_lora_target_lm_aspect:lora_s4_target_lm_aspect": {
                "frontier_improvement_supported": True
            }
        }
    }
    assert "No preregistered comparison" in report.findings(frontier, {})
