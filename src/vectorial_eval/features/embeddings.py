"""Text embedding backends for the semantic and distributional metrics.

The embedding space is a first-order confound in this project rather than an
implementation detail. The July 16 discussion identified the central risk that
general-purpose embeddings may organise LinkedIn and Reddit posts by *register*
rather than by *topic*; if that holds, every embedding-derived metric measures
style agreement while appearing to measure semantic agreement. The backend is
therefore configurable, and results should be replicated across at least two
backends before being reported.

Available backends
------------------
``tfidf-svd`` (default)
    TF-IDF over word unigrams and bigrams, reduced by truncated SVD. Requires no
    network access, no accelerator, and no additional dependency, so the harness
    remains fully executable offline. It is fit on the evaluation corpus itself,
    which is admissible because every metric is a *relative* comparison between
    transfer functions scored within a single fitted space.

``sentence-transformers``
    Any Sentence-Transformers-compatible checkpoint, selected via
    ``EvalConfig.embedding_model``. The default is ``google/embeddinggemma-300m``
    (768 dimensions, Matryoshka-truncatable to 512/256/128). This backend
    provides substantially stronger semantics than TF-IDF and is the recommended
    configuration for reported results.

Selecting a backend
-------------------
The space is chosen on the ``evaluate`` command and applies to every
embedding-based metric at once::

    vectorial-eval evaluate --split heldout                      # tfidf-svd, dim 256
    vectorial-eval evaluate --split heldout \
        --embedding-model google/embeddinggemma-300m --embedding-dim 256

Because a fitted space is shared by every transfer function and every metric,
the same text is encoded many times over a run. ``EmbeddingSpace`` therefore
memoises encodings per text. This is a pure speed measure: both backends encode
each document independently of the others in its batch, so a cached vector is
the vector the backend would have returned.

Style residualisation
---------------------
``style_residual_space`` projects out the linear directions that best separate
the two platforms, yielding a representation in which similarity is closer to
topical. This operationalises, in its minimal form, the proposal to construct a
style representation and factor it out of total similarity. It is a lower bound
on style removal rather than a solution: a small number of linear directions
cannot capture register in full. Its purpose is to permit every embedding-based
metric to be reported both raw and residualised, where a large discrepancy
between the two constitutes evidence that a result is stylistic rather than
semantic.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

#: Selectable backends, in the order the documentation introduces them.
BACKENDS = ("tfidf-svd", "sentence-transformers")

DEFAULT_ST_MODEL = "google/embeddinggemma-300m"

#: Prompt name passed to Sentence-Transformers checkpoints that expose task
#: prompts. Pool-versus-pool comparison is a symmetric similarity problem, so
#: the clustering prompt is appropriate; the asymmetric retrieval prompts
#: ("query" / "document") would be incorrect here.
DEFAULT_ST_PROMPT = "Clustering"

#: Dimensions to which EmbeddingGemma may be Matryoshka-truncated. Truncation to
#: an unsupported width degrades the representation without raising an error.
MATRYOSHKA_DIMS = (768, 512, 256, 128)


@dataclass
class EmbeddingSpace:
    """A fitted embedding model together with its encode function."""

    backend: str
    transform: Callable[[list[str]], np.ndarray]
    dim: int
    model_name: str | None = None
    prompt_name: str | None = None
    #: Memoised encodings, keyed on the text. One space is shared by every
    #: metric and every transfer function, so the authentic pools are re-encoded
    #: on the order of thirty times per run; under a neural backend that is the
    #: dominant cost. Both backends encode a document independently of the rest
    #: of its batch, so this changes runtime and nothing else.
    _cache: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim))
        # `dict.fromkeys` preserves first-seen order, so the batch handed to the
        # backend is a deterministic function of the request.
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            vectors = self.transform(missing)
            for text, vector in zip(missing, np.asarray(vectors), strict=True):
                self._cache[text] = vector
        return np.vstack([self._cache[t] for t in texts])

    @property
    def slug(self) -> str:
        """Filesystem-safe identifier for the space, used to name reports.

        TRM and every other embedding-derived value is only comparable within
        one space, so a report computed in a second space must not overwrite the
        first. The slug is what keeps the two files apart.
        """
        return space_slug(self.model_name or self.backend, self.dim)

    def describe(self) -> dict:
        return {
            "backend": self.backend,
            "model": self.model_name,
            "dim": self.dim,
            "prompt": self.prompt_name,
            "slug": self.slug,
        }


def space_slug(model_name: str, dim: int) -> str:
    """Slug for a (model, dimension) pair, e.g. ``embeddinggemma-300m-d256``."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", str(model_name).split("/")[-1])
    return f"{stem.strip('-').lower()}-d{int(dim)}"


def _fit_tfidf_svd(corpus_texts: list[str], dim: int, seed: int) -> EmbeddingSpace:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import Normalizer

    n_docs = max(len(corpus_texts), 1)
    dim = int(min(dim, max(2, n_docs - 1)))
    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        min_df=2 if n_docs > 50 else 1,
        max_df=0.9,
        ngram_range=(1, 2),
        # Stop words are retained deliberately: function-word frequency is one
        # of the strongest register signals distinguishing the two platforms,
        # and removing it would suppress precisely the variation of interest.
        stop_words=None,
        max_features=60000,
    )
    pipeline = make_pipeline(
        vectorizer, TruncatedSVD(n_components=dim, random_state=seed), Normalizer()
    )
    pipeline.fit(corpus_texts or ["placeholder"])
    return EmbeddingSpace(
        backend="tfidf-svd", transform=pipeline.transform, dim=dim, model_name="tfidf-svd"
    )


def _fit_sentence_transformer(
    model_name: str, dim: int | None, prompt_name: str | None, batch_size: int
) -> EmbeddingSpace:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "The sentence-transformers backend requires the package to be "
            "installed:\n    pip install -U sentence-transformers\n"
            f"Alternatively, set embedding_backend='tfidf-svd'. ({exc})"
        ) from exc

    try:
        model = SentenceTransformer(model_name)
    except Exception as exc:  # noqa: BLE001 - re-raised with actionable guidance
        raise RuntimeError(
            f"Failed to load '{model_name}'. Note that EmbeddingGemma is a gated "
            "checkpoint: its licence must be accepted on the Hugging Face model "
            "page and the client must be authenticated (`huggingface-cli login`, "
            "or the HF_TOKEN environment variable).\n"
            f"Underlying error: {type(exc).__name__}: {exc}"
        ) from exc

    # Renamed in sentence-transformers 5.6; the old name still works but warns.
    get_dim = getattr(model, "get_embedding_dimension", None) or (
        model.get_sentence_embedding_dimension
    )
    native_dim = get_dim()
    target_dim = native_dim
    if dim and dim < native_dim:
        if "embeddinggemma" in model_name.lower() and dim not in MATRYOSHKA_DIMS:
            nearest = min(MATRYOSHKA_DIMS, key=lambda d: abs(d - dim))
            log.warning(
                "embedding_dim=%d is not a supported Matryoshka width for %s; "
                "using %d instead. Supported widths: %s",
                dim, model_name, nearest, MATRYOSHKA_DIMS,
            )
            target_dim = nearest
        else:
            target_dim = dim

    # The prompt argument is only accepted by checkpoints that declare prompts;
    # it is probed once rather than assumed, so that arbitrary checkpoints
    # remain usable.
    supported_prompt = None
    if prompt_name:
        available = getattr(model, "prompts", None) or {}
        if prompt_name in available:
            supported_prompt = prompt_name
        else:
            log.info(
                "Prompt %r is not declared by %s (available: %s); encoding without a prompt.",
                prompt_name, model_name, sorted(available) or "none",
            )

    def transform(texts: list[str]) -> np.ndarray:
        kwargs = {
            "batch_size": batch_size,
            "show_progress_bar": False,
            "convert_to_numpy": True,
            # Normalisation is applied after truncation below, so it is
            # disabled here to keep Matryoshka truncation valid.
            "normalize_embeddings": False,
        }
        if supported_prompt:
            kwargs["prompt_name"] = supported_prompt
        vectors = np.asarray(model.encode(texts, **kwargs), dtype=np.float32)
        if target_dim < vectors.shape[1]:
            vectors = vectors[:, :target_dim]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.clip(norms, 1e-9, None)

    log.info(
        "Loaded %s (native dim %d, using %d%s)",
        model_name, native_dim, target_dim,
        f", prompt={supported_prompt}" if supported_prompt else "",
    )
    return EmbeddingSpace(
        backend="sentence-transformers",
        transform=transform,
        dim=target_dim,
        model_name=model_name,
        prompt_name=supported_prompt,
    )


def fit_embedding_space(
    corpus_texts: list[str],
    backend: str = "tfidf-svd",
    dim: int = 256,
    seed: int = 0,
    model_name: str | None = None,
    prompt_name: str | None = DEFAULT_ST_PROMPT,
    batch_size: int = 32,
) -> EmbeddingSpace:
    """Construct the embedding space used by all embedding-based metrics.

    A single space is fit once per evaluation and shared across every transfer
    function; fitting separately per function would render their scores
    mutually incomparable.
    """
    if backend == "tfidf-svd":
        return _fit_tfidf_svd(corpus_texts, dim, seed)
    if backend == "sentence-transformers":
        return _fit_sentence_transformer(
            model_name or DEFAULT_ST_MODEL, dim, prompt_name, batch_size
        )
    raise ValueError(
        f"Unknown embedding backend {backend!r}; expected one of "
        f"{', '.join(repr(b) for b in BACKENDS)}."
    )


def style_directions(
    embeddings: np.ndarray, platform_labels: list[str], k: int = 2
) -> np.ndarray:
    """Return an orthonormal basis, shape ``(dim, r)``, for the style subspace.

    Two related but non-identical characterisations of platform separation are
    combined: the decision direction of a linear platform classifier, and the
    between-platform difference of means. The result is orthonormalised.
    """
    from sklearn.linear_model import LogisticRegression

    y = np.array([1 if p == "reddit" else 0 for p in platform_labels])
    classifier = LogisticRegression(max_iter=2000, C=1.0)
    classifier.fit(embeddings, y)

    mean_difference = embeddings[y == 1].mean(0) - embeddings[y == 0].mean(0)
    candidates = np.vstack([classifier.coef_, mean_difference[None, :]])
    basis, _ = np.linalg.qr(candidates.T)
    return basis[:, : min(k, basis.shape[1])]


def project_out(x: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Remove the span of ``basis`` from ``x`` and renormalise the result."""
    residual = x - (x @ basis) @ basis.T
    norms = np.linalg.norm(residual, axis=1, keepdims=True)
    return residual / np.clip(norms, 1e-9, None)


def style_residual_space(
    embeddings: np.ndarray, platform_labels: list[str], n_directions: int = 2
) -> np.ndarray:
    """Project out the linear directions that separate the two platforms."""
    if embeddings.shape[0] < 4 or len(set(platform_labels)) < 2:
        return embeddings
    return project_out(embeddings, style_directions(embeddings, platform_labels, n_directions))
