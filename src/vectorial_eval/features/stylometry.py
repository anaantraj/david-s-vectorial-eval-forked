"""Interpretable surface features separating LinkedIn from Reddit register.

These are the "lexical / stylistic feature comparison" axis from the meeting,
and they carry a second job beyond scoring: the client-facing interpretability
requirement ("what distinguishes a LinkedIn audience from a Reddit one?") is
much easier to answer with named, human-readable features than with an
embedding distance.

Feature choice is driven by the platform differences the team actually observed:
LinkedIn skews promotional, first-person-singular, emoji- and hashtag-heavy,
with short punchy line breaks; Reddit skews question-heavy, second-person, more
hedged, more code/URL-bearing, and more profane.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable

import numpy as np

# Vocabulary probes. Deliberately small and legible: these are meant to be read
# and argued about by the team, not to maximise a classifier.
PROMO_TERMS = (
    "excited to", "thrilled", "proud to", "delighted", "honored", "we're hiring",
    "check it out", "learn more", "sign up", "register", "join us", "announcing",
    "launch", "webinar", "free trial", "dm me", "link in", "grateful",
)
CRITIQUE_TERMS = (
    "terrible", "garbage", "broken", "sucks", "awful", "useless", "hate",
    "waste of", "overrated", "buggy", "nightmare", "footgun", "bloated",
)
HEDGE_TERMS = (
    "maybe", "probably", "i think", "seems like", "kind of", "sort of", "afaik",
    "iirc", "not sure", "might be", "in my experience", "ymmv",
)
FIRST_PERSON_SG = (" i ", "i'm", "i've", "my ", "me ", "mine")
FIRST_PERSON_PL = ("we ", "we're", "we've", "our ", "us ")
SECOND_PERSON = ("you ", "you're", "your ", "yall", "y'all")

EMOJI_RE = re.compile(
    "[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f1e6-\U0001f1ff]"
)
URL_RE = re.compile(r"https?://\S+")
HASHTAG_RE = re.compile(r"(?<!\w)#\w+")
MENTION_RE = re.compile(r"(?<!\w)@\w+")
CODE_RE = re.compile(r"`[^`]+`|```|\b\w+\(\)|\b[A-Za-z_]+\.[a-z]{2,4}\b")
SENT_SPLIT_RE = re.compile(r"[.!?]+(?:\s|$)")
WORD_RE = re.compile(r"[A-Za-z']+")

FEATURE_NAMES = (
    "n_chars",
    "n_words",
    "mean_word_len",
    "mean_sentence_len",
    "n_paragraphs",
    "question_rate",
    "exclaim_rate",
    "uppercase_ratio",
    "emoji_per_100w",
    "hashtag_per_100w",
    "mention_per_100w",
    "url_per_100w",
    "code_per_100w",
    "list_marker_rate",
    "promo_per_100w",
    "critique_per_100w",
    "hedge_per_100w",
    "first_person_sg_per_100w",
    "first_person_pl_per_100w",
    "second_person_per_100w",
    "type_token_ratio",
)


def _count_terms(haystack: str, terms: Iterable[str]) -> int:
    return sum(haystack.count(t) for t in terms)


def extract(text: str) -> dict[str, float]:
    """Compute the feature dict for one post."""
    raw = text or ""
    lowered = f" {raw.lower()} "
    words = WORD_RE.findall(raw)
    n_words = max(len(words), 1)
    per100 = 100.0 / n_words

    sentences = [s for s in SENT_SPLIT_RE.split(raw) if s.strip()]
    n_sentences = max(len(sentences), 1)
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    letters = [c for c in raw if c.isalpha()]

    return {
        "n_chars": float(len(raw)),
        "n_words": float(len(words)),
        "mean_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
        "mean_sentence_len": len(words) / n_sentences,
        "n_paragraphs": float(len(lines)),
        # Question rate is per-sentence, not per-post: the meeting's finding was
        # that Reddit is question-heavy, which shows up as a higher share of
        # interrogative sentences rather than merely "contains a ?".
        "question_rate": raw.count("?") / n_sentences,
        "exclaim_rate": raw.count("!") / n_sentences,
        "uppercase_ratio": (
            sum(1 for c in letters if c.isupper()) / len(letters) if letters else 0.0
        ),
        "emoji_per_100w": len(EMOJI_RE.findall(raw)) * per100,
        "hashtag_per_100w": len(HASHTAG_RE.findall(raw)) * per100,
        "mention_per_100w": len(MENTION_RE.findall(raw)) * per100,
        "url_per_100w": len(URL_RE.findall(raw)) * per100,
        "code_per_100w": len(CODE_RE.findall(raw)) * per100,
        "list_marker_rate": (
            sum(1 for ln in lines if re.match(r"^\s*([-*•\d]+[.)]?\s)", ln)) / max(len(lines), 1)
        ),
        "promo_per_100w": _count_terms(lowered, PROMO_TERMS) * per100,
        "critique_per_100w": _count_terms(lowered, CRITIQUE_TERMS) * per100,
        "hedge_per_100w": _count_terms(lowered, HEDGE_TERMS) * per100,
        "first_person_sg_per_100w": _count_terms(lowered, FIRST_PERSON_SG) * per100,
        "first_person_pl_per_100w": _count_terms(lowered, FIRST_PERSON_PL) * per100,
        "second_person_per_100w": _count_terms(lowered, SECOND_PERSON) * per100,
        "type_token_ratio": len({w.lower() for w in words}) / n_words,
    }


def matrix(texts: Iterable[str]) -> np.ndarray:
    """Stack `extract` over many texts into an (n, n_features) array."""
    rows = [extract(t) for t in texts]
    if not rows:
        return np.zeros((0, len(FEATURE_NAMES)))
    return np.array([[r[name] for name in FEATURE_NAMES] for r in rows], dtype=float)


def summarize(texts: Iterable[str]) -> dict[str, float]:
    """Mean of every feature over a pool of texts."""
    m = matrix(texts)
    if m.shape[0] == 0:
        return {n: math.nan for n in FEATURE_NAMES}
    return dict(zip(FEATURE_NAMES, m.mean(axis=0), strict=True))
