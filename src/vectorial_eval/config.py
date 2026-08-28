"""Central configuration for the transfer-function experimentation harness.

Everything that a researcher is likely to want to sweep lives here, so that an
experiment is reproducible from (config, git sha) alone. Nothing in this module
imports pandas or any model library, so that it remains inexpensive to import.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = PROJECT_ROOT / "final_dataset_v1.csv"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_RUN_DIR = PROJECT_ROOT / "runs"

# The dataset ships one `audience_room_name` per (room, platform) scrape, with
# inconsistent naming: "CTOs" is the LinkedIn side of "CTO Reddit". Bilateral
# cells can only be reconstructed after collapsing these to a canonical id.
#
# Only well-supported merges are listed. Rooms whose canonical identifier is
# unique to a single platform simply never produce a bilateral cell, which is
# the correct outcome rather than a forced pairing.
ROOM_CANONICAL_MAP: dict[str, str] = {
    "Backend Engineer (linkedin+reddit)": "backend_engineer",
    "CTOs": "cto",
    "CTO Reddit": "cto",
    "Engineering Manager": "engineering_manager",
    "Engineering Manager reddit": "engineering_manager",
    "Fullstack Engineers": "fullstack_engineer",
    "Full Stack Developer Reddit": "fullstack_engineer",
    "Instructional Designer": "instructional_designer",
    "[LinkedIn] OpenAI Community": "openai_community",
    "[Reddit] OpenAI Community": "openai_community",
    "Product Leadership Audience Room": "product_leadership",
    "Product Manager Reddit": "product_leadership",
    "Engineering Org in Edtech Co": "edtech_engineering",
    "Edtech Developers": "edtech_engineering",
    "Director/ VP engineering/ CTO in Edtech Co": "edtech_leadership",
    "Staff Engineer": "staff_engineer",
    "VP Engineering": "vp_engineering",
    "Engineering Leadership Audience Room": "engineering_leadership",
    "Devops/Infra Engineer": "devops_infra",
    "Frontend Engineer Reddit": "frontend_engineer",
    "Knowledge Graph Devs": "knowledge_graph_dev",
    "QA Engineer Reddit": "qa_engineer",
    "First Responders": "first_responders",
    "Programming reddit": "programming_general",
}

# Fields the July 16 meeting flagged as artifacts of earlier extraction passes.
# Carried through into the JSONL as `legacy_labels` so nothing is lost, but kept
# out of the primary record surface so they are not accidentally trained on.
LEGACY_LABEL_FIELDS = ("sentiment", "stance", "original_topic", "topic_narrow")


@dataclass
class DatasetConfig:
    """Controls how the raw CSV becomes train/val/test JSONL."""

    csv_path: Path = DEFAULT_CSV
    out_dir: Path = DEFAULT_DATA_DIR

    #: `strict` groups on the raw `audience_room_name`; `canonical` applies
    #: ROOM_CANONICAL_MAP first. Canonical roughly doubles bilateral coverage
    #: but asserts that e.g. "CTOs" and "CTO Reddit" sample the same audience.
    room_mode: str = "canonical"

    #: Cluster field used as the topic axis of a bilateral cell. The meeting
    #: notes designate `final_topic` as the current clustering output.
    topic_field: str = "final_topic"

    #: A cell needs at least this many posts on *each* platform to be modelled.
    min_posts_per_platform: int = 5

    #: Cells that clear `min_posts_per_platform` but not this are retained and
    #: marked `tier="sparse"` for qualitative inspection only.
    dense_min_posts_per_platform: int = 10

    #: Post-level split proportions, applied within each (cell, platform)
    #: stratum so every cell keeps references in every split.
    split_ratios: tuple[float, float, float] = (0.70, 0.15, 0.15)

    #: Fraction of qualifying cells reserved entirely for zero-shot
    #: generalization (`heldout_cell=True`); these appear only in test.
    heldout_cell_frac: float = 0.15

    #: Seed for the deterministic hash salt. Splits are a pure function of
    #: (post_id, seed), so adding data never reshuffles existing assignments.
    seed: int = 20260722

    min_chars: int = 30
    max_chars: int = 8000

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["csv_path"] = str(self.csv_path)
        d["out_dir"] = str(self.out_dir)
        return d


# Model slugs verified live against the OpenRouter catalogue (2026-07-22).
# Keeping them here rather than inline at call sites means a model sweep is a
# config edit. Refresh with `python -m vectorial_eval.cli models`.
MODELS = {
    # Frontier rewriters. gpt-5.6 ships in three size tiers (sol > terra >
    # luna); `-pro` variants trade latency for quality at the same price.
    "claude-opus": "anthropic/claude-opus-4.8",
    "claude-sonnet": "anthropic/claude-sonnet-5",
    "claude-fable": "anthropic/claude-fable-5",
    "gpt-5.6": "openai/gpt-5.6-sol",
    "gpt-5.6-mid": "openai/gpt-5.6-terra",
    "gpt-5.6-small": "openai/gpt-5.6-luna",
    "grok": "x-ai/grok-4.5",
    # Cheap, fast models — appropriate for judging and for large sweeps where
    # the per-call cost dominates.
    "gemini-flash": "google/gemini-3.6-flash",
    "gemini-flash-lite": "google/gemini-3.5-flash-lite",
    "claude-haiku": "anthropic/claude-haiku-4.5",
}


@dataclass
class LLMConfig:
    """Shared settings for any component that calls a hosted model.

    Defaults to OpenRouter (`OPENROUTER_API_KEY` in `.env`), which reaches both
    Claude and GPT behind a single OpenAI-compatible endpoint. This is material
    for the present project: a GPT rewrite was proposed as a baseline, and
    routing every model through one client keeps the rewrite baselines and the
    LLM judge mutually comparable rather than confounded by differing SDK
    defaults for retry, sampling, and message formatting.

    `provider="anthropic"` / `"openai"` use those vendors' own endpoints
    directly if a native key is ever preferred.
    """

    provider: str = "openrouter"
    model: str = MODELS["claude-opus"]
    max_tokens: int = 2048
    temperature: float = 1.0
    max_concurrency: int = 10
    max_retries: int = 4
    #: Cache the (prompt -> completion) mapping on disk so re-running an
    #: evaluation after a metric bugfix does not re-pay for generation.
    cache_dir: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cache_dir"] = str(self.cache_dir) if self.cache_dir else None
        return d


#: The offline default embedding space. A report computed in this space keeps
#: the plain `report.<split>.json` name; any other space is written to a
#: slug-qualified file so that two spaces cannot overwrite one another.
DEFAULT_EMBEDDING_BACKEND = "tfidf-svd"
DEFAULT_EMBEDDING_DIM = 256


@dataclass
class EvalConfig:
    """Controls which metrics run and how they are parameterised."""

    metrics: list[str] = field(
        default_factory=lambda: [
            "structural",
            "classifier",
            "distributional",
            "trm",
            "semantic",
            "degeneracy",
        ]
    )
    #: Embedding backend for the semantic, distributional, and TRM metrics.
    #: "tfidf-svd" requires no network access or additional dependency and is
    #: therefore the default; "sentence-transformers" is substantially stronger
    #: and is recommended for reported results. Every embedding-derived score,
    #: TRM above all, is only comparable within one space, so a report is named
    #: after the space it was computed in whenever that space is not the default
    #: (see `evaluate.report_path`).
    embedding_backend: str = "tfidf-svd"
    #: Checkpoint identifier, used only by the sentence-transformers backend.
    #: EmbeddingGemma is gated: its licence must be accepted on the Hugging Face
    #: model page and the client authenticated before first use.
    embedding_model: str = "google/embeddinggemma-300m"
    #: Output dimensionality. For TF-IDF this is the SVD rank; for EmbeddingGemma
    #: it selects a Matryoshka width and must be one of 768/512/256/128.
    embedding_dim: int = 256
    #: Task prompt for checkpoints that declare one. Pool-versus-pool comparison
    #: is symmetric, so a clustering prompt is appropriate rather than the
    #: asymmetric retrieval prompts.
    embedding_prompt: str = "Clustering"
    #: Encoding batch size, used only by the sentence-transformers backend.
    embedding_batch_size: int = 32
    #: Bootstrap resamples for confidence intervals on cell-level metrics.
    n_bootstrap: int = 500
    #: Independent-unit policy. Study 3 uses cells. Study 4 uses a two-stage
    #: audience-then-cell bootstrap because cells from one audience share
    #: adaptation data and are not independent audience replications.
    bootstrap_unit: str = "cell"
    #: Optional multi-model judge panel. When non-empty these override the
    #: single `judge` model and results are reported per-judge plus as a
    #: majority vote, which provides the least expensive available defence
    #: against the idiosyncratic bias of any single judge. Cross-family by
    #: construction.
    judge_panel: list[str] = field(default_factory=list)
    #: Cap on judge calls per (transfer_fn, cell) to bound cost.
    judge_samples_per_cell: int = 8
    seed: int = 20260722


@dataclass
class HarnessConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    #: Model used by LLM-backed transfer functions (the rewrite baselines).
    llm: LLMConfig = field(default_factory=LLMConfig)
    #: Model used by the LLM-as-a-judge metric. Deliberately from a *different
    #: family* than the default rewriter: LLM judges show measurable
    #: self-preference bias, so evaluating a Claude rewrite with a Claude judge
    #: would inflate its score. Gemini Flash is additionally inexpensive enough
    #: to support a three-judge panel over every cell (see
    #: EvalConfig.judge_panel).
    judge: LLMConfig = field(
        default_factory=lambda: LLMConfig(
            model=MODELS["gemini-flash"], max_tokens=1024, temperature=0.0
        )
    )
    eval: EvalConfig = field(default_factory=EvalConfig)
    run_dir: Path = DEFAULT_RUN_DIR

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset.to_dict(),
            "llm": self.llm.to_dict(),
            "judge": self.judge.to_dict(),
            "eval": asdict(self.eval),
            "run_dir": str(self.run_dir),
        }
