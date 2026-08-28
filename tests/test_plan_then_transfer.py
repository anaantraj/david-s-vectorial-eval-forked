"""Offline checks for the explicit content-plan transfer baseline."""

from __future__ import annotations

import json

import pytest

from vectorial_eval.data.schema import TransferTask
from vectorial_eval.transfer.plan_then_transfer import parse_plan, plan_prompt, render_prompt


def _task() -> TransferTask:
    return TransferTask(
        task_id="t1",
        cell_id="backend::databases",
        room="backend_engineer",
        topic="databases",
        domain="Software Engineering",
        split="test",
        source_platform="linkedin",
        target_platform="reddit",
        source_post_id="p1",
        source_text="We cut latency by 42% after moving from MySQL to PostgreSQL.",
        target_reference_ids=[],
        exemplar_ids=[],
        heldout_cell=False,
    )


def _plan() -> dict[str, list[str]]:
    return {
        "claims": ["Latency fell after the database migration"],
        "entities": [],
        "numbers": ["42%"],
        "named_tools": ["MySQL", "PostgreSQL"],
        "stance": ["Positive about the migration"],
        "requested_actions": [],
        "aspects": ["database performance"],
    }


def test_plan_parser_accepts_json_fences_and_normalizes_strings():
    payload = _plan()
    payload["numbers"] = [" 42% ", ""]
    parsed = parse_plan(f"```json\n{json.dumps(payload)}\n```")
    assert parsed["numbers"] == ["42%"]


def test_plan_parser_rejects_missing_or_nonlist_fields():
    payload = _plan()
    del payload["aspects"]
    with pytest.raises(ValueError, match="exactly"):
        parse_plan(json.dumps(payload))
    payload = _plan()
    payload["claims"] = "not a list"
    with pytest.raises(ValueError, match="list of strings"):
        parse_plan(json.dumps(payload))


def test_render_uses_the_plan_not_the_source_post():
    task = _task()
    extraction = plan_prompt(task)
    rendering = render_prompt(task, _plan())
    assert task.source_text in extraction
    assert task.source_text not in rendering
    assert "42%" in rendering
    assert "backend engineer" in rendering
