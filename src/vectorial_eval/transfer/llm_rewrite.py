"""LLM rewrite transfer functions — the prompt-engineering arm of the study.

Two variants, which differ only in what context the prompt carries:

`llm_rewrite`
    Zero-shot. The model is told the audience, topic, and target platform, and
    asked to re-express the post. No target-platform text is shown. This
    measures how much of the transfer is recoverable from the model's prior
    knowledge of platform norms alone.

`llm_fewshot`
    Shows `k` real target-platform posts from the *same cell's train split* as
    exemplars. This is the trait-free version of what the meeting called
    prompt-engineered transfer, and the gap between it and `llm_rewrite`
    isolates how much comes from observing the actual target distribution
    versus from generic priors.

Held-out cells have no exemplars by construction, so `llm_fewshot` degrades to
zero-shot there. That is intentional: it makes the zero-shot generalization
penalty directly measurable rather than hidden.

A note on what these are *not*: both are surface rewriters. The meeting was
explicit that reducing transfer to surface rewriting would undershoot the
research framing, and that Vectorial's own direction is trait-mediated. These
exist to be the baseline that a trait-mediated function must beat — which is
why the prompt below deliberately does not attempt trait extraction.
"""

from __future__ import annotations

import logging

from ..config import LLMConfig
from ..data.schema import TransferOutput, TransferTask
from ..llm import LLMClient
from .base import Corpus, TransferFunction, register

log = logging.getLogger(__name__)

PLATFORM_LABEL = {"linkedin": "LinkedIn", "reddit": "Reddit"}

SYSTEM_PROMPT = """You rewrite social posts across platforms for audience research.

You are given a post written by a member of a specific occupational audience on \
one platform. Rewrite it as the same kind of person would have written it on the \
target platform, discussing the same underlying subject.

Rules:
- Preserve the substantive content: the topic, the specific claims, the technical \
details, and the author's actual stance. Do not invent facts, products, numbers, \
or experiences that are not in the source.
- Change how it is expressed: tone, register, structure, length, formatting, and \
what the author foregrounds. Different platforms reward different things and the \
same person performs differently on each.
- Do not translate mechanically sentence by sentence. A post can legitimately \
become shorter, blunter, or reorganised around a question.
- Output only the rewritten post body. No preamble, no quotation marks, no \
explanation, no markdown headers."""

PLATFORM_HINT = {
    "reddit": (
        "Reddit posts in these communities are execution-focused and "
        "detail-oriented. They are frequently framed as questions or as "
        "concrete problem reports, are willing to be critical or blunt, assume "
        "the reader is a practitioner, and carry no self-promotional framing."
    ),
    "linkedin": (
        "LinkedIn posts in these communities are strategic and business-case "
        "oriented. They tend to be positive, framed around lessons or "
        "outcomes, often self-promotional, addressed to a mixed professional "
        "audience, and formatted for skimming."
    ),
}


def _build_user_prompt(task: TransferTask, exemplars: list[str]) -> str:
    src = PLATFORM_LABEL.get(task.source_platform, task.source_platform)
    tgt = PLATFORM_LABEL.get(task.target_platform, task.target_platform)
    room = task.room.replace("_", " ")

    parts = [
        f"Audience: {room}",
        f"Domain: {task.domain}",
        f"Topic: {task.topic}",
        f"Source platform: {src}",
        f"Target platform: {tgt}",
        "",
        f"How this audience writes on {tgt}: {PLATFORM_HINT.get(task.target_platform, '')}",
    ]

    if exemplars:
        parts += [
            "",
            f"Real examples of this audience posting on {tgt} about this topic. "
            f"Match their register and shape, not their specific content:",
        ]
        for i, ex in enumerate(exemplars, 1):
            # Truncate exemplars rather than the source: a clipped source would
            # change the content being transferred, but a clipped exemplar only
            # weakens the style signal.
            snippet = ex if len(ex) <= 1200 else ex[:1200] + " […]"
            parts.append(f"\n--- example {i} ---\n{snippet}")

    parts += [
        "",
        f"--- source post ({src}) ---",
        task.source_text,
        "",
        f"Rewrite this as a {tgt} post. Output only the post body.",
    ]
    return "\n".join(parts)


class _BaseLLMRewrite(TransferFunction):
    def __init__(
        self,
        llm: LLMConfig | None = None,
        n_shot: int = 0,
        name: str | None = None,
        n_samples: int = 1,
    ):
        self.cfg = llm or LLMConfig()
        self.n_shot = n_shot
        #: Candidates drawn per task. The distribution-aware metrics compare a
        #: pool of candidates against a pool of references, so drawing several
        #: candidates per source post enlarges the candidate pool without
        #: requiring additional corpus data. Sampling at temperature 1.0 is what
        #: the metric's originating work prescribes for this purpose.
        self.n_samples = max(1, n_samples)
        if name:
            self.name = name
        self.client = LLMClient(self.cfg)

    def describe(self) -> dict:
        return {
            "name": self.name,
            "class": type(self).__name__,
            "n_shot": self.n_shot,
            "n_samples": self.n_samples,
            "llm": self.cfg.to_dict(),
        }

    def _exemplars(self, task: TransferTask, corpus: Corpus) -> list[str]:
        if self.n_shot <= 0:
            return []
        # Longest-first: very short posts carry little style signal, and the
        # exemplar budget is small.
        texts = corpus.texts(task.exemplar_ids)
        return sorted(texts, key=len, reverse=True)[: self.n_shot]

    def run(self, tasks: list[TransferTask], corpus: Corpus) -> list[TransferOutput]:
        exemplars = [self._exemplars(t, corpus) for t in tasks]
        # Each task is expanded into `n_samples` independent draws.
        jobs = [(i, k) for i in range(len(tasks)) for k in range(self.n_samples)]
        variants = [k for _, k in jobs]

        def build(job):
            i, _k = job
            return SYSTEM_PROMPT, _build_user_prompt(tasks[i], exemplars[i])

        log.info(
            "%s: generating %d candidates (%d tasks x %d samples) with %s",
            self.name, len(jobs), len(tasks), self.n_samples, self.cfg.model,
        )
        results = self.client.map(jobs, build, variants=variants)

        out = []
        for (i, k), res in zip(jobs, results, strict=True):
            task, ex = tasks[i], exemplars[i]
            out.append(
                TransferOutput(
                    task_id=task.task_id,
                    cell_id=task.cell_id,
                    transfer_fn=self.name,
                    output_text=res.text,
                    meta={
                        "model": self.cfg.model,
                        "n_exemplars": len(ex),
                        "sample_index": k,
                        "ok": res.ok,
                        "error": res.error,
                        "cached": res.cached,
                        "usage": res.usage,
                    },
                )
            )
        n_failed = sum(1 for o in out if not o.meta.get("ok"))
        if n_failed:
            log.warning("%s: %d/%d generations failed", self.name, n_failed, len(out))
        return out


@register("llm_rewrite")
class ZeroShotRewrite(_BaseLLMRewrite):
    name = "llm_rewrite"

    def __init__(self, llm: LLMConfig | None = None, **kw):
        super().__init__(
            llm=llm, n_shot=0, name=kw.pop("name", None),
            n_samples=kw.pop("n_samples", 1),
        )


@register("llm_fewshot")
class FewShotRewrite(_BaseLLMRewrite):
    name = "llm_fewshot"

    def __init__(self, llm: LLMConfig | None = None, n_shot: int = 4, **kw):
        super().__init__(
            llm=llm, n_shot=n_shot, name=kw.pop("name", None),
            n_samples=kw.pop("n_samples", 1),
        )
