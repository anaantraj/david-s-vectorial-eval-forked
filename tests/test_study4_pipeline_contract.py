from __future__ import annotations

import re
import shlex
from pathlib import Path

from scripts.audit_study4_completion import (
    FORWARD_COMPARISONS,
    FORWARD_SYSTEMS,
    REVERSE_COMPARISONS,
    REVERSE_SYSTEMS,
)

ROOT = Path(__file__).parents[1]


def comparisons(path: Path) -> set[str]:
    return set(re.findall(r"--comparison\s+(\S+)", path.read_text(encoding="utf-8")))


def generated_systems(path: Path) -> set[str]:
    systems = {
        "identity",
        "target_sample",
        "shuffle_control",
        "llama_rewrite_s4_base_none",
        "llm_rewrite_s4_claude_reference",
    }
    prefixes = {
        "generate_lora": "lora_",
        "generate_soft_prompt": "soft_prompt_",
        "generate_steering": "steering_",
        "generate_local_plan": "local_plan_then_transfer_",
    }
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        for function, prefix in prefixes.items():
            if not line.startswith(function + " "):
                continue
            parts = shlex.split(line)
            name = prefix + parts[2]
            if parts[-1] == "yes":
                name += "_audience_shuffle"
            systems.add(name)
    return systems


def test_forward_pipeline_contains_every_completion_gate_comparison():
    got = comparisons(ROOT / "scripts/cluster/continue_study4_generation.sh")
    assert FORWARD_COMPARISONS <= got


def test_reverse_pipeline_contains_every_completion_gate_comparison():
    got = comparisons(ROOT / "scripts/cluster/continue_study4_reverse.sh")
    assert REVERSE_COMPARISONS <= got


def test_scale_handoff_records_and_checks_process_liveness():
    stage_a = (ROOT / "scripts/cluster/continue_study4_stage_a.sh").read_text()
    stage_b = (ROOT / "scripts/cluster/continue_study4_stage_b.sh").read_text()
    assert 'echo "$job_pid" >"$pid_file"' in stage_a
    assert "s4-stage-a.failed" in stage_a
    assert "s4-stage-a.failed" in stage_b
    assert "parent run exited without a finished metrics artifact" in stage_a
    assert 'kill -0 "$run_pid"' in stage_b
    assert "died without a finished metrics artifact" in stage_b


def test_dependent_pipelines_propagate_failure_markers():
    stage_b = (ROOT / "scripts/cluster/continue_study4_stage_b.sh").read_text()
    generation = (
        ROOT / "scripts/cluster/continue_study4_generation.sh"
    ).read_text()
    reverse = (ROOT / "scripts/cluster/continue_study4_reverse.sh").read_text()
    assert "s4-stage-b.failed" in stage_b
    assert "s4-stage-b.failed" in generation
    assert "s4-generation.failed" in generation
    assert "s4-generation.failed" in reverse
    assert "s4-reverse.failed" in reverse


def test_output_audits_use_the_authoritative_complete_system_sets():
    forward = (ROOT / "scripts/cluster/continue_study4_generation.sh").read_text()
    reverse = (ROOT / "scripts/cluster/continue_study4_reverse.sh").read_text()
    assert "from scripts.audit_study4_completion import FORWARD_SYSTEMS" in forward
    assert '"${REQUIRE_ARGS[@]}"' in forward
    assert "from scripts.audit_study4_completion import REVERSE_SYSTEMS" in reverse
    assert '"${REQUIRE_ARGS[@]}"' in reverse


def test_generation_calls_exactly_cover_the_required_system_contracts():
    forward = ROOT / "scripts/cluster/continue_study4_generation.sh"
    reverse = ROOT / "scripts/cluster/continue_study4_reverse.sh"
    assert generated_systems(forward) == FORWARD_SYSTEMS
    assert generated_systems(reverse) == REVERSE_SYSTEMS
