"""Aspect-aware prompting — conditioning the rewrite on the platform aspect skew.

`llm_rewrite` and `llm_fewshot` treat transfer as a change of manner: the same
content, expressed the way the target platform expresses things. The aspect
package Vectorial delivered says that is only part of it. Within one topic
cluster the two platforms foreground *different evaluative dimensions*, and the
gap is positive in 27 of the 36 clusters they measured. Transfer therefore also
involves a shift in what the post is about evaluatively, which none of the
existing rewriters model.

`aspect_prompt` adds exactly one thing to the zero-shot prompt: a statement of
which evaluative aspects the target platform foregrounds in this topic relative
to the source platform, and which it plays down. Nothing else differs from
`llm_rewrite` — no exemplars, no change of model, no change of sampling — so
the difference between the two functions isolates the value of conditioning on
the aspect skew.

How the skew is computed
------------------------
Each shipped cluster file lists aspects with `linkedin_count` and
`reddit_count`, the number of posts on each platform that invoked the aspect,
alongside the cluster's `n_linkedin` and `n_reddit`. Two corrections are applied
before the counts can be read as emphasis. First, the raw counts are not
comparable because the two pools differ in size, often by a factor of two or
more, so each count is divided by its own platform's pool to give a share.
Second, LinkedIn posts invoke more aspects each than Reddit posts do — 2.75
against 1.96 per post on average, and more in 33 of the 36 clusters — so a
difference of shares is negative almost everywhere and reports that LinkedIn
says more, not that the platforms emphasise different things. Each share is
therefore divided by its platform's total aspect mass in the cluster, which
turns it into a share of emphasis that sums to one per platform. The skew is the
difference of those weights in the direction of the target platform, and it sums
to zero over a cluster's aspects. Aspects above a threshold are named as ones to
bring forward, aspects below its negative as ones to play down, and the prompt
quotes the raw shares because they are the interpretable quantity.

Joining to the harness
----------------------
Aspect clusters are keyed by `(domain, final_topic)`, not by `final_topic`
alone: some topics occur under two domains, and a join on topic alone silently
merges two clusters. `phase_d_similarity.csv` carries no domain column and the
per-cluster files carry only the topic name, so the join is mediated by
`data/aspects/aspect_join_index.json`, built by `scripts/build_aspect_index.py`
from `final_dataset_v1.csv`. A `(domain, topic)` pair that is not in that index
is not covered, and a pair served by a vocabulary that also serves another
domain is covered but flagged `domain_merged`. Both facts are recorded per
output so coverage is measurable from the run rather than assumed.

Cells with no vocabulary fall back to the zero-shot prompt rather than failing,
and record `aspect_path="zero_shot_fallback"` with a reason.

Validity caveat
---------------
The aspect vocabularies were extracted by `gpt-4.1-mini` **with the platform
labels visible to the extracting model**. The vocabulary may therefore have been
constructed to separate the platforms, which is the same thing this function
conditions on. A blind rerun of the extraction is pending. Every result produced
by this function is provisional until that rerun lands, and this caveat belongs
in any write-up that quotes its numbers.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from ..config import DEFAULT_DATA_DIR, LLMConfig
from ..data.schema import TransferOutput, TransferTask
from ..llm import LLMClient
from .base import Corpus, TransferFunction, register
from .llm_rewrite import PLATFORM_HINT, PLATFORM_LABEL

log = logging.getLogger(__name__)

DEFAULT_ASPECT_DIR = Path(DEFAULT_DATA_DIR) / "aspects"

#: Minimum difference in emphasis weight for an aspect to be named in the
#: prompt. A cluster carries eight to ten aspects, so uniform emphasis is 0.10
#: to 0.125 of the weight; five points is a substantial re-weighting and less
#: than that is within the resolution of counts drawn from a few dozen posts.
MIN_WEIGHT_SKEW = 0.05
#: Cap on how many aspects are named in each direction. The prompt has to leave
#: the source content room to dominate; naming eight dimensions would turn the
#: rewrite into a checklist.
MAX_FOREGROUND = 4
MAX_BACKGROUND = 3

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
- Change what is emphasised. The two platforms judge the same subject along \
different evaluative dimensions, and you will be told which dimensions the target \
platform foregrounds. Re-weight the post toward those dimensions using material \
that is already in the source: draw out what the source implies or mentions in \
passing about them, and give less room to the dimensions the target platform \
plays down. If the source says nothing that bears on a dimension, leave it out \
rather than inventing content for it.
- Do not translate mechanically sentence by sentence. A post can legitimately \
become shorter, blunter, or reorganised around a question.
- Output only the rewritten post body. No preamble, no quotation marks, no \
explanation, no markdown headers."""


@dataclass(frozen=True)
class AspectSkew:
    """One aspect, with each platform's share of posts invoking it.

    `source_share` and `target_share` are the raw shares, which is what the
    prompt quotes because they are interpretable. `source_weight` and
    `target_weight` are those shares expressed as a fraction of the platform's
    total aspect mass in this cluster, which is what selection uses.
    """

    name: str
    description: str
    source_share: float
    target_share: float
    source_weight: float
    target_weight: float

    @property
    def skew(self) -> float:
        """Positive when the target platform foregrounds this aspect.

        The difference of composition weights rather than of raw shares.
        LinkedIn posts invoke more aspects each than Reddit posts do (2.75
        against 1.96 per post, averaged over the shipped clusters, and higher in
        33 of 36), so a difference of raw shares is negative for nearly every
        aspect and would tell the model to play everything down. Dividing by
        each platform's total aspect mass removes that and leaves the
        re-weighting, which is the thing being modelled: the weights sum to one
        on each side, so the skews sum to zero.
        """
        return self.target_weight - self.source_weight


@dataclass(frozen=True)
class AspectLookup:
    """Result of resolving a task against the aspect index."""

    ok: bool
    reason: str
    cluster: str | None = None
    aspects: tuple[AspectSkew, ...] = ()
    domain_merged: bool = False
    domain_contributed: bool = True


class AspectIndex:
    """The shipped aspect vocabularies, joined on `(domain, final_topic)`."""

    def __init__(self, aspect_dir: Path | str = DEFAULT_ASPECT_DIR):
        self.aspect_dir = Path(aspect_dir)
        self.index_path = self.aspect_dir / "aspect_join_index.json"
        self.available = self.index_path.exists()
        self.join: dict[str, dict] = {}
        self.note = ""
        if not self.available:
            log.warning(
                "aspect index not found at %s; every task will fall back to "
                "zero-shot. Run scripts/build_aspect_index.py.",
                self.index_path,
            )
            return
        payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        self.join = payload.get("join", {})
        self.note = payload.get("note", "")

    @staticmethod
    def _key(domain: str, topic: str) -> str:
        return f"{domain}||{topic}"

    def _load_cluster(self, filename: str) -> dict:
        return _read_cluster(self.aspect_dir / "clusters" / filename)

    def lookup(self, task: TransferTask) -> AspectLookup:
        if not self.available:
            return AspectLookup(ok=False, reason="index_missing")
        entry = self.join.get(self._key(task.domain, task.topic))
        if entry is None:
            # Either the topic has no shipped vocabulary, or it has one under a
            # different domain. Both are treated as no coverage: reusing another
            # domain's vocabulary is exactly the silent merge the index exists
            # to prevent.
            return AspectLookup(ok=False, reason="no_vocabulary_for_domain_topic")
        try:
            data = self._load_cluster(entry["file"])
        except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover
            log.warning("aspect file %s unreadable: %s", entry["file"], exc)
            return AspectLookup(ok=False, reason="cluster_file_unreadable")

        n = {"linkedin": data.get("n_linkedin", 0), "reddit": data.get("n_reddit", 0)}
        n_src, n_tgt = n.get(task.source_platform, 0), n.get(task.target_platform, 0)
        if n_src <= 0 or n_tgt <= 0:
            # A share is undefined against an empty pool, and a zero count would
            # read as "this platform never discusses it" rather than "unknown".
            return AspectLookup(ok=False, reason="empty_platform_pool")

        aspects = data.get("aspects", [])
        if not aspects:
            return AspectLookup(ok=False, reason="empty_vocabulary")

        def _share(a: dict, platform: str, n: int) -> float:
            return a.get(f"{platform}_count", 0) / n

        src_mass = sum(_share(a, task.source_platform, n_src) for a in aspects)
        tgt_mass = sum(_share(a, task.target_platform, n_tgt) for a in aspects)
        if src_mass <= 0 or tgt_mass <= 0:
            # One platform invoked no aspect at all, so there is no emphasis to
            # compare and the weights would be undefined.
            return AspectLookup(ok=False, reason="empty_aspect_mass")

        skews = tuple(
            AspectSkew(
                name=a["name"],
                description=a.get("description", ""),
                source_share=_share(a, task.source_platform, n_src),
                target_share=_share(a, task.target_platform, n_tgt),
                source_weight=_share(a, task.source_platform, n_src) / src_mass,
                target_weight=_share(a, task.target_platform, n_tgt) / tgt_mass,
            )
            for a in aspects
        )
        return AspectLookup(
            ok=True,
            reason="ok",
            cluster=entry["cluster"],
            aspects=skews,
            domain_merged=bool(entry.get("domain_merged")),
            domain_contributed=bool(entry.get("domain_contributed", True)),
        )


@cache
def _read_cluster(path: Path) -> dict:
    """Cluster files are read once per process; there are only 36 of them."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def select_aspects(
    aspects: tuple[AspectSkew, ...],
    min_skew: float = MIN_WEIGHT_SKEW,
    max_foreground: int = MAX_FOREGROUND,
    max_background: int = MAX_BACKGROUND,
) -> tuple[list[AspectSkew], list[AspectSkew]]:
    """Split a vocabulary into aspects to bring forward and to play down.

    Sorted by skew, then by name so the prompt, and therefore the cache key, is
    stable across runs.
    """
    ordered = sorted(aspects, key=lambda a: (-a.skew, a.name))
    fore = [a for a in ordered if a.skew >= min_skew][:max_foreground]
    back = [a for a in reversed(ordered) if a.skew <= -min_skew][:max_background]
    return fore, back


def _pct(x: float) -> str:
    return f"{round(100 * x):d}%"


def _aspect_block(
    look: AspectLookup, src: str, tgt: str, min_skew: float = MIN_WEIGHT_SKEW
) -> tuple[str, int, int]:
    fore, back = select_aspects(look.aspects, min_skew=min_skew)
    if not fore and not back:
        return "", 0, 0

    lines = [
        f"Evaluative emphasis for this topic. Across real posts on this topic, "
        f"{src} and {tgt} judge the subject along different dimensions. The "
        f"percentages are the share of posts on each platform that raised the "
        f"dimension at all."
    ]
    if fore:
        lines.append(f"\nBring forward, because {tgt} foregrounds these:")
        for a in fore:
            lines.append(
                f"- {a.name}: {a.description} "
                f"({tgt} {_pct(a.target_share)} of posts, {src} {_pct(a.source_share)})"
            )
    if back:
        lines.append(f"\nGive less room, because {tgt} raises these far less than {src}:")
        for a in back:
            lines.append(
                f"- {a.name}: {a.description} "
                f"({tgt} {_pct(a.target_share)} of posts, {src} {_pct(a.source_share)})"
            )
    lines.append(
        "\nRe-weight the emphasis using what the source already contains. Do not "
        "add claims or experiences to cover a dimension the source is silent on."
    )
    return "\n".join(lines), len(fore), len(back)


def build_user_prompt(
    task: TransferTask, look: AspectLookup, min_skew: float = MIN_WEIGHT_SKEW
) -> tuple[str, dict]:
    """Return the user prompt and the provenance recorded for the task."""
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

    info: dict = {
        "aspect_cluster": look.cluster,
        "aspect_reason": look.reason,
        "n_aspects_in_vocabulary": len(look.aspects),
        "aspect_domain_merged": look.domain_merged,
        "aspect_domain_contributed": look.domain_contributed,
    }

    block, n_fore, n_back = ("", 0, 0)
    if look.ok:
        block, n_fore, n_back = _aspect_block(look, src, tgt, min_skew)
    if block:
        parts += ["", block]
        info["aspect_path"] = "aspect_conditioned"
    else:
        # Either no vocabulary at all, or one whose aspects are all balanced
        # between the platforms. Both leave the prompt identical to zero-shot,
        # and the two cases are distinguished by `aspect_reason`.
        info["aspect_path"] = "zero_shot_fallback"
        if look.ok:
            info["aspect_reason"] = "no_aspect_above_threshold"
    info["n_aspects_foregrounded"] = n_fore
    info["n_aspects_backgrounded"] = n_back

    parts += [
        "",
        f"--- source post ({src}) ---",
        task.source_text,
        "",
        f"Rewrite this as a {tgt} post. Output only the post body.",
    ]
    return "\n".join(parts), info


@register("aspect_prompt")
class AspectAwarePrompt(TransferFunction):
    """Zero-shot rewrite conditioned on the target platform's aspect emphasis."""

    name = "aspect_prompt"

    def __init__(
        self,
        llm: LLMConfig | None = None,
        aspect_dir: Path | str = DEFAULT_ASPECT_DIR,
        name: str | None = None,
        n_samples: int = 1,
        min_skew: float = MIN_WEIGHT_SKEW,
    ):
        self.cfg = llm or LLMConfig()
        self.n_samples = max(1, n_samples)
        self.min_skew = min_skew
        if name:
            self.name = name
        self.index = AspectIndex(aspect_dir)
        self.client = LLMClient(self.cfg)

    def describe(self) -> dict:
        return {
            "name": self.name,
            "class": type(self).__name__,
            "n_shot": 0,
            "n_samples": self.n_samples,
            "llm": self.cfg.to_dict(),
            "aspect_index": str(self.index.index_path),
            "aspect_index_available": self.index.available,
            "n_join_keys": len(self.index.join),
            "min_weight_skew": self.min_skew,
            "max_foreground": MAX_FOREGROUND,
            "max_background": MAX_BACKGROUND,
            "caveat": self.index.note,
        }

    def run(self, tasks: list[TransferTask], corpus: Corpus) -> list[TransferOutput]:
        prompts, infos = [], []
        for t in tasks:
            prompt, info = build_user_prompt(t, self.index.lookup(t), self.min_skew)
            prompts.append(prompt)
            infos.append(info)

        jobs = [(i, k) for i in range(len(tasks)) for k in range(self.n_samples)]
        variants = [k for _, k in jobs]

        def build(job):
            i, _k = job
            return SYSTEM_PROMPT, prompts[i]

        n_cond = sum(1 for x in infos if x["aspect_path"] == "aspect_conditioned")
        log.info(
            "%s: generating %d candidates (%d tasks x %d samples) with %s; "
            "%d/%d tasks aspect-conditioned",
            self.name, len(jobs), len(tasks), self.n_samples, self.cfg.model,
            n_cond, len(tasks),
        )
        results = self.client.map(jobs, build, variants=variants)

        out = []
        for (i, k), res in zip(jobs, results, strict=True):
            task = tasks[i]
            out.append(
                TransferOutput(
                    task_id=task.task_id,
                    cell_id=task.cell_id,
                    transfer_fn=self.name,
                    output_text=res.text,
                    meta={
                        "model": self.cfg.model,
                        "n_exemplars": 0,
                        "sample_index": k,
                        "ok": res.ok,
                        "error": res.error,
                        "cached": res.cached,
                        "usage": res.usage,
                        **infos[i],
                    },
                )
            )
        n_failed = sum(1 for o in out if not o.meta.get("ok"))
        if n_failed:
            log.warning("%s: %d/%d generations failed", self.name, n_failed, len(out))
        return out
