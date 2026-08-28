"""Checks for output cardinality and failure auditing."""

from __future__ import annotations

import json

import pytest

from scripts.audit_study4_outputs import audit


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_output_audit_pins_every_task_and_draw(tmp_path):
    tasks = tmp_path / "tasks.heldout.jsonl"
    write_jsonl(tasks, [{"task_id": "a"}, {"task_id": "b"}])
    rows = [
        {
            "task_id": task,
            "transfer_fn": "system",
            "output_text": "text",
            "meta": {"sample_index": draw, "ok": True},
        }
        for task in ("a", "b")
        for draw in range(2)
    ]
    write_jsonl(tmp_path / "outputs.system.heldout.jsonl", rows)
    (tmp_path / "transfer_fn.system.json").write_text("{}")
    result = audit(tmp_path, tasks, "heldout", 2, {"system"})
    assert result["systems"]["system"]["n_outputs"] == 4
    assert result["systems"]["system"]["failure_rate"] == 0


def test_output_audit_rejects_a_repeated_draw(tmp_path):
    tasks = tmp_path / "tasks.heldout.jsonl"
    write_jsonl(tasks, [{"task_id": "a"}])
    rows = [
        {
            "task_id": "a",
            "transfer_fn": "system",
            "output_text": "text",
            "meta": {"sample_index": 0},
        }
        for _ in range(2)
    ]
    write_jsonl(tmp_path / "outputs.system.heldout.jsonl", rows)
    (tmp_path / "transfer_fn.system.json").write_text("{}")
    with pytest.raises(RuntimeError, match="draw indices"):
        audit(tmp_path, tasks, "heldout", 2, {"system"})
