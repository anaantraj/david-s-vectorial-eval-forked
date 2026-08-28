"""Semantic preservation and degeneracy guards.

These two metrics render the platform-transfer scores resistant to exploitation.

Semantic preservation
---------------------
`source_similarity` is the cosine similarity between each generated post and the
source post it was derived from. The identity baseline scores 1.0 by
construction; the shuffle control scores near the corpus floor. A useful
transfer function must sit high here *while* moving on the classifier metric.

Reported alongside it is `content_word_retention` — the share of the source's
distinctive content words (rare, non-stopword) that survive into the output.
Embedding similarity may remain high while specific details are discarded; this
measure detects that case, and is necessary because transfer is required to
preserve topic and stance rather than register alone.

`entity_surface_recall` and `number_recall` are exact, transparent guards for
specific details. They are deliberately named as surface-form measures: this
harness does not pretend that a capitalization heuristic is a full named-entity
recognizer, and a paraphrased entity can therefore count as missed.

Degeneracy
----------
Low-effort failure modes that would otherwise inflate distributional scores:
`empty_rate` and `failure_rate` detect a function that silently discards
difficult cases; `copy_rate` detects near-verbatim reproduction of the source,
that is, the absence of transfer; and `distinct_2` together with
`pool_self_similarity` detect mode collapse, in which a single conservative
target-register post is emitted for every task. The latter can match a pool
centroid while conveying no information.
"""

from __future__ import annotations

import re

import numpy as np

from ..data.schema import TransferOutput
from .base import EvalContext, Metric, MetricResult, group_by_cell, register

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]+")
NUMBER_RE = re.compile(
    r"(?<![\w-])[+-]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)%?(?![\w-])"
)
ENTITY_RE = re.compile(
    r"\b(?:[A-Z]{2,}(?:[-./][A-Z0-9]+)*|[A-Z][a-z]+[A-Z][A-Za-z]*|"
    r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"
)
STOPWORDS = set(
    """a an the and or but if then than that this these those is are was were be been being
    of to in on for with at by from as it its it's i you he she they we me my your our their
    not no so do does did have has had will would can could should may might must about into
    over under more most some any all just like get got make made use used using what which
    who when where how why there here out up down off very much many one two also new
    """.split()
)


def _content_words(text: str) -> set[str]:
    return {w.lower() for w in WORD_RE.findall(text or "") if w.lower() not in STOPWORDS and len(w) > 3}


def _numbers(text: str) -> set[str]:
    return {match.replace(",", "") for match in NUMBER_RE.findall(text or "")}


def _entity_surfaces(text: str) -> set[str]:
    return {" ".join(match.lower().split()) for match in ENTITY_RE.findall(text or "")}


def _set_recall(source: set[str], generated: set[str]) -> float:
    return len(source & generated) / len(source) if source else float("nan")


def _distinct_n(texts: list[str], n: int = 2) -> float:
    grams, total = set(), 0
    for t in texts:
        toks = WORD_RE.findall((t or "").lower())
        for i in range(len(toks) - n + 1):
            grams.add(tuple(toks[i : i + n]))
            total += 1
    return len(grams) / total if total else float("nan")


@register("semantic")
class SemanticPreservation(Metric):
    directions = {
        "source_similarity": +1,
        "content_word_retention": +1,
        "entity_surface_recall": +1,
        "number_recall": +1,
        "plan_content_word_recall": +1,
        "plan_entity_surface_recall": +1,
        "plan_number_recall": +1,
        "target_pool_similarity": +1,
    }

    def score(self, transfer_fn, outputs, ctx: EvalContext) -> MetricResult:
        space = ctx.embedding_space
        if space is None:
            raise RuntimeError("semantic metric requires ctx.embedding_space")

        result = MetricResult(metric=self.name, transfer_fn=transfer_fn)
        by_cell: dict[str, list[TransferOutput]] = {}
        for o in outputs:
            if o.output_text and o.output_text.strip():
                by_cell.setdefault(o.cell_id, []).append(o)

        source_text = ctx.extras.get("_source_text_by_task", {})

        for cell_id, outs in by_cell.items():
            paired = [
                (output, output.output_text, source_text.get(output.task_id, ""))
                for output in outs
                if source_text.get(output.task_id, "")
            ]
            if not paired:
                continue
            g_txt = [generated for _, generated, _ in paired]
            s_txt = [source for _, _, source in paired]

            g_emb = space.encode(list(g_txt))
            s_emb = space.encode(list(s_txt))
            gn = g_emb / np.clip(np.linalg.norm(g_emb, axis=1, keepdims=True), 1e-9, None)
            sn = s_emb / np.clip(np.linalg.norm(s_emb, axis=1, keepdims=True), 1e-9, None)
            sims = np.sum(gn * sn, axis=1)

            retention = []
            entity_recall = []
            number_recall = []
            plan_content_recall = []
            plan_entity_recall = []
            plan_number_recall = []
            for output, g, s in paired:
                cw = _content_words(s)
                if cw:
                    retention.append(len(cw & _content_words(g)) / len(cw))
                entities = _entity_surfaces(s)
                if entities:
                    entity_recall.append(_set_recall(entities, _entity_surfaces(g)))
                numbers = _numbers(s)
                if numbers:
                    number_recall.append(_set_recall(numbers, _numbers(g)))
                plan = output.meta.get("content_plan")
                if isinstance(plan, dict):
                    plan_text = " ".join(
                        item
                        for value in plan.values()
                        if isinstance(value, list)
                        for item in value
                        if isinstance(item, str)
                    )
                    if cw:
                        plan_content_recall.append(
                            _set_recall(cw, _content_words(plan_text))
                        )
                    if entities:
                        plan_entity_recall.append(
                            _set_recall(entities, _entity_surfaces(plan_text))
                        )
                    if numbers:
                        plan_number_recall.append(
                            _set_recall(numbers, _numbers(plan_text))
                        )

            real = ctx.target_pool(cell_id)
            tgt_sim = float("nan")
            if real:
                r_emb = space.encode(real)
                rn = r_emb / np.clip(np.linalg.norm(r_emb, axis=1, keepdims=True), 1e-9, None)
                tgt_sim = float((gn @ rn.T).mean())

            result.cell_scores[cell_id] = {
                "source_similarity": float(sims.mean()),
                "content_word_retention": float(np.mean(retention)) if retention else float("nan"),
                "entity_surface_recall": (
                    float(np.mean(entity_recall)) if entity_recall else float("nan")
                ),
                "number_recall": (
                    float(np.mean(number_recall)) if number_recall else float("nan")
                ),
                "plan_content_word_recall": (
                    float(np.mean(plan_content_recall))
                    if plan_content_recall
                    else float("nan")
                ),
                "plan_entity_surface_recall": (
                    float(np.mean(plan_entity_recall))
                    if plan_entity_recall
                    else float("nan")
                ),
                "plan_number_recall": (
                    float(np.mean(plan_number_recall))
                    if plan_number_recall
                    else float("nan")
                ),
                "target_pool_similarity": tgt_sim,
                "n_scored": float(len(paired)),
            }
        return result


@register("degeneracy")
class Degeneracy(Metric):
    directions = {
        "empty_rate": -1,
        "failure_rate": -1,
        "copy_rate": -1,
        "distinct_2": +1,
        "pool_self_similarity": -1,
        "length_ratio_vs_real": 0,
    }

    def score(self, transfer_fn, outputs, ctx: EvalContext) -> MetricResult:
        space = ctx.embedding_space
        result = MetricResult(metric=self.name, transfer_fn=transfer_fn)
        source_text = ctx.extras.get("_source_text_by_task", {})

        by_cell_all: dict[str, list[TransferOutput]] = {}
        for o in outputs:
            by_cell_all.setdefault(o.cell_id, []).append(o)
        pools = group_by_cell(outputs)

        n_empty = n_fail = 0
        for cell_id, all_outs in by_cell_all.items():
            gen = pools.get(cell_id, [])
            empties = sum(1 for o in all_outs if not (o.output_text or "").strip())
            fails = sum(1 for o in all_outs if o.meta.get("ok") is False)
            n_empty += empties
            n_fail += fails
            if not gen:
                result.cell_scores[cell_id] = {
                    "empty_rate": 1.0,
                    "failure_rate": fails / max(len(all_outs), 1),
                    "copy_rate": float("nan"),
                    "distinct_2": float("nan"),
                }
                continue

            # Near-verbatim reproduction: high bidirectional content-word
            # overlap with the source indicates that no transfer occurred.
            copies = 0
            for o in all_outs:
                s = source_text.get(o.task_id, "")
                g = o.output_text or ""
                if not s or not g:
                    continue
                cs, cg = _content_words(s), _content_words(g)
                if cs and cg:
                    jaccard = len(cs & cg) / len(cs | cg)
                    if jaccard > 0.85:
                        copies += 1

            self_sim = float("nan")
            if space is not None and len(gen) > 1:
                e = space.encode(gen)
                en = e / np.clip(np.linalg.norm(e, axis=1, keepdims=True), 1e-9, None)
                sim = en @ en.T
                np.fill_diagonal(sim, np.nan)
                self_sim = float(np.nanmean(sim))

            real = ctx.target_pool(cell_id)
            real_len = np.mean([len(t) for t in real]) if real else np.nan
            gen_len = np.mean([len(t) for t in gen])

            result.cell_scores[cell_id] = {
                "empty_rate": empties / max(len(all_outs), 1),
                "failure_rate": fails / max(len(all_outs), 1),
                "copy_rate": copies / max(len(all_outs), 1),
                "distinct_2": _distinct_n(gen, 2),
                "pool_self_similarity": self_sim,
                "length_ratio_vs_real": float(gen_len / real_len) if real_len else float("nan"),
            }

        result.overall = {
            "n_outputs": float(len(outputs)),
            "n_empty": float(n_empty),
            "n_failed": float(n_fail),
        }
        return result
