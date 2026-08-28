"""Two-call baseline that separates content extraction from platform rendering."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from ..config import LLMConfig
from ..data.schema import TransferOutput, TransferTask
from ..llm import LLMClient, LLMResult
from .base import Corpus, TransferFunction, register
from .llm_rewrite import PLATFORM_HINT, PLATFORM_LABEL
from .lora import LoraTransfer, _draw_seed

log = logging.getLogger(__name__)

PLAN_SYSTEM = """Extract an explicit factual content plan from a social post.
Return one JSON object and nothing else. This is not a request for reasoning or
analysis. Record only information stated in the source. Use exactly these keys,
each containing a JSON array of short strings: claims, entities, numbers,
named_tools, stance, requested_actions, aspects. Do not infer private motives,
new facts, or target-platform wording."""

RENDER_SYSTEM = """Write a social post from a supplied factual content plan.
Express the plan naturally for the named platform and occupational audience.
Preserve every claim, entity, number, named tool, stance, and requested action.
Do not add facts or experiences. Output only the post body, with no explanation,
header, quotation marks, or JSON."""

PLAN_FIELDS = (
    "claims",
    "entities",
    "numbers",
    "named_tools",
    "stance",
    "requested_actions",
    "aspects",
)


def parse_plan(text: str) -> dict[str, list[str]]:
    """Parse the public factual plan and reject schema drift."""
    candidate = (text or "").strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    payload = json.loads(candidate)
    if not isinstance(payload, dict) or set(payload) != set(PLAN_FIELDS):
        raise ValueError(f"plan must contain exactly {PLAN_FIELDS}")
    normalized: dict[str, list[str]] = {}
    for field in PLAN_FIELDS:
        value = payload[field]
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"plan field {field!r} must be a list of strings")
        normalized[field] = [item.strip() for item in value if item.strip()]
    return normalized


def plan_prompt(task: TransferTask) -> str:
    return (
        f"Source platform: {PLATFORM_LABEL.get(task.source_platform, task.source_platform)}\n"
        f"Topic: {task.topic}\n\n--- source post ---\n{task.source_text}"
    )


def render_prompt(task: TransferTask, plan: dict[str, list[str]]) -> str:
    target = PLATFORM_LABEL.get(task.target_platform, task.target_platform)
    return "\n".join(
        [
            f"Audience: {task.room.replace('_', ' ')}",
            f"Domain: {task.domain}",
            f"Topic: {task.topic}",
            f"Target platform: {target}",
            f"How this audience writes there: {PLATFORM_HINT.get(task.target_platform, '')}",
            "",
            "--- factual content plan ---",
            json.dumps(plan, ensure_ascii=False, sort_keys=True),
            "",
            f"Write the {target} post now.",
        ]
    )


def _usage(result: LLMResult) -> dict[str, Any] | None:
    return result.usage if isinstance(result.usage, dict) else None


@register("plan_then_transfer")
class PlanThenTransfer(TransferFunction):
    name = "plan_then_transfer"

    def __init__(
        self,
        llm: LLMConfig | None = None,
        n_samples: int = 1,
        name: str | None = None,
        **_,
    ):
        self.cfg = llm or LLMConfig()
        self.n_samples = max(1, int(n_samples))
        if name:
            self.name = name
        self.client = LLMClient(self.cfg)

    def run(self, tasks: list[TransferTask], corpus: Corpus) -> list[TransferOutput]:
        del corpus
        jobs = [(index, draw) for index in range(len(tasks)) for draw in range(self.n_samples)]
        plan_results = self.client.map(
            jobs,
            lambda job: (PLAN_SYSTEM, plan_prompt(tasks[job[0]])),
            variants=[2 * draw for _, draw in jobs],
        )
        parsed: dict[tuple[int, int], dict[str, list[str]]] = {}
        plan_errors: dict[tuple[int, int], str] = {}
        for job, result in zip(jobs, plan_results, strict=True):
            if not result.ok:
                plan_errors[job] = result.error or "plan_generation_failed"
                continue
            try:
                parsed[job] = parse_plan(result.text)
            except (ValueError, json.JSONDecodeError) as exc:
                plan_errors[job] = f"plan_parse_error: {exc}"

        render_jobs = [job for job in jobs if job in parsed]
        render_results = self.client.map(
            render_jobs,
            lambda job: (
                RENDER_SYSTEM,
                render_prompt(tasks[job[0]], parsed[job]),
            ),
            variants=[2 * draw + 1 for _, draw in render_jobs],
        )
        rendered = dict(zip(render_jobs, render_results, strict=True))

        outputs = []
        for job, plan_result in zip(jobs, plan_results, strict=True):
            index, draw = job
            task = tasks[index]
            result = rendered.get(job)
            ok = result is not None and result.ok and bool(result.text.strip())
            error = plan_errors.get(job)
            if result is not None and not ok:
                error = result.error or "empty_render"
            outputs.append(
                TransferOutput(
                    task_id=task.task_id,
                    cell_id=task.cell_id,
                    transfer_fn=self.name,
                    output_text=result.text.strip() if ok and result else "",
                    meta={
                        "model": self.cfg.model,
                        "sample_index": draw,
                        "ok": ok,
                        "error": error,
                        "content_plan": parsed.get(job),
                        "plan_usage": _usage(plan_result),
                        "render_usage": _usage(result) if result else None,
                        "calls_per_success": 2,
                    },
                )
            )
        return outputs

    def describe(self) -> dict:
        return {
            "name": self.name,
            "class": type(self).__name__,
            "method": "two-call factual-plan extraction then platform rendering",
            "n_samples": self.n_samples,
            "llm": self.cfg.to_dict(),
            "plan_fields": list(PLAN_FIELDS),
            "calls_per_success": 2,
            "private_reasoning_collected": False,
        }


@register("local_plan_then_transfer")
class LocalPlanThenTransfer(LoraTransfer):
    """Two-pass factual planning with the same local Llama/adapter as transfer.

    The public JSON plan is an auditable intermediate representation, not hidden
    reasoning.  It is generated once per candidate and retained on the output.
    A malformed plan is a recorded failed candidate rather than being silently
    replaced by a direct rewrite.
    """

    name = "local_plan_then_transfer"

    def _plan_model_prompt(self, task: TransferTask) -> str:
        return f"{PLAN_SYSTEM}\n\n{plan_prompt(task)}\n\nJSON plan:"

    def _render_model_prompt(
        self, task: TransferTask, plan: dict[str, list[str]]
    ) -> str:
        return f"{RENDER_SYSTEM}\n\n{render_prompt(task, plan)}"

    def run(self, tasks: list[TransferTask], corpus: Corpus) -> list[TransferOutput]:
        del corpus
        jobs = [(i, k) for i in range(len(tasks)) for k in range(self.n_samples)]
        results: dict[tuple[int, int], tuple[str, bool, str | None]] = {}
        plans: dict[tuple[int, int], dict[str, list[str]]] = {}
        seeds: dict[tuple[int, int], dict[str, int]] = {}
        token_counts: dict[tuple[int, int], dict[str, int]] = {}
        started = time.time()

        # Planning is deliberately single-item.  Independent seeds are then
        # exact rather than merely recorded as the first draw's batch seed.
        for position, (i, k) in enumerate(jobs):
            task = tasks[i]
            plan_seed = _draw_seed(self.seed, f"{task.task_id}:plan", k)
            render_seed = _draw_seed(self.seed, f"{task.task_id}:render", k)
            seeds[(i, k)] = {"plan": plan_seed, "render": render_seed}
            try:
                plan_model_prompt = self._plan_model_prompt(task)
                raw_plan = self._generate([plan_model_prompt], plan_seed)[0]
                plan = parse_plan(raw_plan)
                plans[(i, k)] = plan
                render_model_prompt = self._render_model_prompt(task, plan)
                rendered = self._generate(
                    [render_model_prompt], render_seed
                )[0].strip()
                if not rendered:
                    raise ValueError("empty_render")
                _, tok = self._load()
                token_counts[(i, k)] = {
                    "plan_input_tokens": len(
                        tok.encode(plan_model_prompt, add_special_tokens=False)
                    ),
                    "plan_output_tokens": len(tok.encode(raw_plan, add_special_tokens=False)),
                    "render_input_tokens": len(
                        tok.encode(render_model_prompt, add_special_tokens=False)
                    ),
                    "render_output_tokens": len(tok.encode(rendered, add_special_tokens=False)),
                }
                token_counts[(i, k)]["total_tokens"] = sum(
                    token_counts[(i, k)].values()
                )
                results[(i, k)] = (rendered, True, None)
            except Exception as exc:  # noqa: BLE001 - failures remain in the sample base
                log.warning(
                    "local plan transfer: task %s draw %d failed: %s",
                    task.task_id,
                    k,
                    exc,
                )
                results[(i, k)] = (
                    "",
                    False,
                    f"{type(exc).__name__}: {exc}",
                )
            if position % 10 == 0:
                log.info(
                    "local plan transfer: %d/%d draws (%.0fs)",
                    position + 1,
                    len(jobs),
                    time.time() - started,
                )

        outputs = []
        for i, k in jobs:
            task = tasks[i]
            text, ok, error = results[(i, k)]
            outputs.append(
                TransferOutput(
                    task_id=task.task_id,
                    cell_id=task.cell_id,
                    transfer_fn=self.name,
                    output_text=text,
                    meta={
                        "base_model": self.base_model,
                        "adapter_dir": str(self.adapter_dir) if self.adapter_dir else None,
                        "training_variant": self.variant,
                        "sample_index": k,
                        "seeds": seeds[(i, k)],
                        "ok": ok,
                        "error": error,
                        "content_plan": plans.get((i, k)),
                        "calls_per_success": 2,
                        **token_counts.get((i, k), {}),
                    },
                )
            )
        return outputs

    def describe(self) -> dict:
        description = super().describe()
        description.update(
            {
                "name": self.name,
                "class": type(self).__name__,
                "method": "local two-pass factual-plan extraction then rendering",
                "plan_fields": list(PLAN_FIELDS),
                "calls_per_success": 2,
                "private_reasoning_collected": False,
            }
        )
        return description
