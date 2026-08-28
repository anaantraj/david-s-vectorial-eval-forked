"""LLM-as-a-judge, in the one form that is valid without paired references.

A rubric formulation — presenting a generated post and requesting a quality
rating — would measure the judge's prior regarding the platform rather than
whether the transfer matched this audience on this topic, and is moreover
susceptible to length and fluency bias.

The judge is therefore framed as a **discrimination task**, the natural
LLM analogue of a distributional metric:

  Present N authentic target-platform posts from the cell alongside one
  candidate, and require the judge to identify the machine-written item.

If the transfer function is effective, the judge cannot exceed chance,
1/(N+1). `detection_rate` is consequently the score, for which lower is better,
and `chance_rate` is reported alongside it so that the reader is never comparing
against an implied zero.

This design possesses three properties the rubric formulation lacks:

  * It is grounded in the cell's authentic posts, and therefore assesses
    audience and topic agreement rather than generic platform register.
  * Position bias is controlled by randomising the candidate's slot per item.
  * It has meaningful floor and ceiling values: chance and unity.

The judge additionally names the cue by which it identified the candidate. These
aggregate into a frequency table constituting the qualitative counterpart to the
structural metric's worst-feature list, and directly addressing the
interpretability requirement raised on the client side.
"""

from __future__ import annotations

import json
import logging
import random
import re

import numpy as np

from ..config import LLMConfig
from ..llm import LLMClient
from .base import EvalContext, Metric, MetricResult, register

log = logging.getLogger(__name__)

SYSTEM = """You are an expert analyst of online writing communities. You can \
tell machine-written text from genuine community posts by register, structure, \
and what the author chooses to foreground.

You will see several real posts from one online community, plus one candidate \
that may be machine-written. Identify the machine-written one.

Respond with ONLY a JSON object, no markdown fence:
{"impostor": <1-based index>, "confidence": <0.0-1.0>, "tell": "<the single \
strongest cue, under 12 words>"}"""


def _build_prompt(item: dict) -> tuple[str, str]:
    lines = [
        f"Community: {item['room'].replace('_', ' ')} on {item['platform']}",
        f"Topic under discussion: {item['topic']}",
        "",
        "Posts:",
    ]
    for i, post in enumerate(item["candidates"], 1):
        snippet = post if len(post) <= 1500 else post[:1500] + " […]"
        lines.append(f"\n[{i}]\n{snippet}")
    lines += [
        "",
        f"Exactly one of these {len(item['candidates'])} posts is machine-written. "
        "Which one? Respond with the JSON object only.",
    ]
    return SYSTEM, "\n".join(lines)


def _parse(text: str, n: int) -> dict | None:
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    idx = data.get("impostor")
    if not isinstance(idx, int) or not 1 <= idx <= n:
        return None
    conf = data.get("confidence")
    return {
        "impostor": idx,
        "confidence": float(conf) if isinstance(conf, (int, float)) else None,
        "tell": str(data.get("tell", ""))[:120],
    }


@register("judge")
class LLMJudge(Metric):
    directions = {
        "detection_rate": -1,      # lower = more indistinguishable = better
        "excess_over_chance": -1,
        "mean_confidence": 0,
    }

    def __init__(
        self,
        llm: LLMConfig | None = None,
        n_distractors: int = 3,
        samples_per_cell: int = 8,
        panel: list[str] | None = None,
    ):
        self.cfg = llm or LLMConfig()
        self.n_distractors = n_distractors
        self.samples_per_cell = samples_per_cell
        self.panel = panel or []

    def _build_items(self, outputs, ctx: EvalContext, rng) -> list[dict]:
        by_cell: dict[str, list[str]] = {}
        for o in outputs:
            if o.output_text and o.output_text.strip():
                by_cell.setdefault(o.cell_id, []).append(o.output_text)

        items = []
        for cell_id, gens in by_cell.items():
            real = ctx.target_pool(cell_id)
            if len(real) < self.n_distractors + 1:
                continue  # not enough real posts to build a discrimination set
            cell = ctx.cells[cell_id]
            chosen = rng.sample(gens, min(self.samples_per_cell, len(gens)))
            for gen in chosen:
                distractors = rng.sample(real, self.n_distractors)
                slot = rng.randrange(self.n_distractors + 1)  # controls position bias
                candidates = list(distractors)
                candidates.insert(slot, gen)
                items.append(
                    {
                        "cell_id": cell_id,
                        "room": cell.room,
                        "topic": cell.topic,
                        "platform": ctx.target_platform,
                        "candidates": candidates,
                        "answer": slot + 1,
                    }
                )
        return items

    def _run_judge(self, items, model: str) -> list[dict | None]:
        cfg = LLMConfig(**{**self.cfg.to_dict(), "model": model, "cache_dir": self.cfg.cache_dir})
        client = LLMClient(cfg)
        results = client.map(items, _build_prompt)
        return [_parse(r.text, len(it["candidates"])) for r, it in zip(results, items, strict=True)]

    def score(self, transfer_fn, outputs, ctx: EvalContext) -> MetricResult:
        rng = random.Random(ctx.seed)
        items = self._build_items(outputs, ctx, rng)
        result = MetricResult(metric=self.name, transfer_fn=transfer_fn)
        chance = 1.0 / (self.n_distractors + 1)

        if not items:
            result.notes = {"skipped": "no cell had enough real posts for discrimination"}
            return result

        models = self.panel or [self.cfg.model]
        log.info("judge: %d items x %d model(s) for %s", len(items), len(models), transfer_fn)

        per_model: dict[str, list[dict | None]] = {m: self._run_judge(items, m) for m in models}

        # Majority vote across the panel; single-model panels pass through.
        votes = []
        for i in range(len(items)):
            picks = [per_model[m][i]["impostor"] for m in models if per_model[m][i]]
            votes.append(max(set(picks), key=picks.count) if picks else None)

        by_cell: dict[str, list[tuple[bool, float | None]]] = {}
        tells: dict[str, int] = {}
        # The positional index is carried explicitly rather than recovered via
        # list.index(), which performs an equality search and would resolve to
        # the wrong element whenever two items compare equal.
        for i, (item, vote) in enumerate(zip(items, votes, strict=True)):
            if vote is None:
                continue
            correct = vote == item["answer"]
            responses = [per_model[m][i] for m in models if per_model[m][i]]
            confs = [r["confidence"] for r in responses if r["confidence"] is not None]
            by_cell.setdefault(item["cell_id"], []).append(
                (correct, float(np.mean(confs)) if confs else None)
            )
            if correct:
                for r in responses:
                    if r["tell"]:
                        tells[r["tell"].lower()] = tells.get(r["tell"].lower(), 0) + 1

        for cell_id, records in by_cell.items():
            detected = [c for c, _ in records]
            confs = [f for _, f in records if f is not None]
            rate = float(np.mean(detected))
            result.cell_scores[cell_id] = {
                "detection_rate": rate,
                "excess_over_chance": float(max(0.0, rate - chance)),
                "mean_confidence": float(np.mean(confs)) if confs else float("nan"),
                "n_trials": float(len(records)),
            }

        all_detected = [c for recs in by_cell.values() for c, _ in recs]
        parsed = sum(1 for v in votes if v is not None)
        result.overall = {
            "detection_rate": float(np.mean(all_detected)) if all_detected else float("nan"),
            "chance_rate": chance,
            "n_trials": float(len(all_detected)),
            "parse_success_rate": parsed / max(len(items), 1),
        }
        result.notes = {
            "models": models,
            "n_distractors": self.n_distractors,
            "top_tells": sorted(tells.items(), key=lambda kv: -kv[1])[:12],
        }
        return result
