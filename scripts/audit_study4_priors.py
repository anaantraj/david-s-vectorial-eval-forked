#!/usr/bin/env python
"""Fail closed on Study 4 prior-corpus leakage and schema mistakes."""

from __future__ import annotations

import argparse
import html
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

REQUIRED = {
    "record_id",
    "author_id",
    "platform",
    "audience",
    "record_type",
    "text",
    "split",
    "audience_prior",
    "profile_weak_labels",
}
RICH_FIELDS = (
    "subreddit",
    "community",
    "title",
    "thread_context",
    "context_summary",
    "engagement",
    "dimensions",
    "profile_weak_labels",
    "audience_prior",
)


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def normalized_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", html.unescape(str(value or ""))).lower()
    return re.sub(r"\s+", " ", text).strip()


def contains_scored(text: str, scored_texts: set[str]) -> bool:
    return any(
        scored in text or (len(text) >= 80 and text in scored)
        for scored in scored_texts
    )


def iter_strings(value: object):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, nested in value.items():
            yield str(key)
            yield from iter_strings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from iter_strings(nested)


def model_visible_strings(row: dict):
    """Yield completion text and metadata exposed by rich-priors-v3."""
    yield str(row.get("text") or "")
    for value in (
        row.get("subreddit") or row.get("community"),
        {
            "title": row.get("title"),
            "context": row.get("thread_context") or row.get("context_summary"),
        },
        row.get("engagement"),
        row.get("dimensions"),
    ):
        yield from iter_strings(value)
    profile = dict(row.get("profile_weak_labels") or {})
    profile.pop("summary", None)
    profile.pop("highlights", None)
    yield from iter_strings(profile)
    yield from iter_strings(row.get("audience_prior") or {})


def audit_scored_overlap(
    prior_paths: list[Path], scored_paths: list[Path]
) -> dict:
    """Fail if a scored post is present in an unsupervised train or dev corpus."""
    scored = [row for path in scored_paths for row in load(path)]
    prior = [row for path in prior_paths for row in load(path)]
    scored_by_platform = {
        platform: {
            text
            for row in scored
            if row.get("platform") == platform
            and len(text := normalized_text(row.get("text"))) >= 40
        }
        for platform in ("reddit", "linkedin")
    }
    prior_text = set()
    for row in prior:
        for value in model_visible_strings(row):
            text = normalized_text(value)
            if len(text) >= 40:
                prior_text.add((row.get("platform"), text))
    text_overlap = {
        (platform, text)
        for platform, text in prior_text
        if contains_scored(text, scored_by_platform.get(platform, set()))
    }

    scored_reddit_ids = {
        str(row.get("post_id") or "").removeprefix("t3_")
        for row in scored
        if row.get("platform") == "reddit"
    }
    prior_reddit_ids = {
        str(row.get("record_id") or "").removeprefix("reddit:post:")
        for row in prior
        if row.get("platform") == "reddit" and row.get("record_type") == "post"
    }
    id_overlap = (scored_reddit_ids - {""}) & (prior_reddit_ids - {""})
    if text_overlap or id_overlap:
        raise RuntimeError(
            "scored-post leakage into Study 4 priors: "
            f"{len(text_overlap)} normalized texts, {len(id_overlap)} Reddit IDs"
        )
    return {
        "n_scored_posts_checked": len(scored),
        "n_prior_records_checked": len(prior),
        "normalized_text_overlaps": 0,
        "normalized_containment_overlaps": 0,
        "reddit_post_id_overlaps": 0,
        "prompt_schema": "study4-rich-priors-v3",
    }


def audit_pair(train_path: Path, dev_path: Path) -> dict:
    train, dev = load(train_path), load(dev_path)
    if not train or not dev:
        raise RuntimeError(f"empty split: {train_path} / {dev_path}")
    for split, rows in (("train", train), ("dev", dev)):
        wrong = [row["record_id"] for row in rows if row.get("split") != split]
        if wrong:
            raise RuntimeError(f"{len(wrong)} rows have the wrong {split} split label")
        missing = REQUIRED - set(rows[0])
        if missing:
            raise RuntimeError(f"{split} missing fields {sorted(missing)}")
        ids = [row["record_id"] for row in rows]
        if len(ids) != len(set(ids)):
            duplicates = [key for key, n in Counter(ids).items() if n > 1]
            raise RuntimeError(
                f"{split} contains {len(duplicates)} duplicate record IDs; "
                "cross-audience clones must be resolved before training"
            )
    train_authors = {row["author_id"] for row in train}
    dev_authors = {row["author_id"] for row in dev}
    if train_authors & dev_authors:
        raise RuntimeError("author leakage between train and dev")
    if {row["record_id"] for row in train} & {row["record_id"] for row in dev}:
        raise RuntimeError("record leakage between train and dev")
    contaminated = [row["record_id"] for row in dev if row["profile_weak_labels"]]
    if contaminated:
        raise RuntimeError(
            f"{len(contaminated)} dev rows retain their author's profile weak labels"
        )
    platforms = {row["platform"] for row in train + dev}
    audiences = {row["audience"] for row in train + dev}
    prior_sources = {
        row["audience_prior"].get("estimated_from") for row in train + dev
    }
    if prior_sources != {"train authors only"}:
        raise RuntimeError(f"unfrozen or unknown prior provenance: {prior_sources}")
    rich_coverage = {
        field: sum(row.get(field) not in (None, "", [], {}) for row in train)
        for field in RICH_FIELDS
    }
    if rich_coverage["audience_prior"] != len(train):
        raise RuntimeError("not every training row has a frozen audience prior")
    if not rich_coverage["profile_weak_labels"]:
        raise RuntimeError("training corpus contains no train-author weak labels")
    return {
        "train": len(train),
        "dev": len(dev),
        "train_authors": len(train_authors),
        "dev_authors": len(dev_authors),
        "platforms": sorted(platforms),
        "audiences": sorted(audiences),
        "record_types": dict(Counter(row["record_type"] for row in train + dev)),
        "rich_train_field_coverage": rich_coverage,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("train", type=Path)
    parser.add_argument("dev", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_pair(args.train, args.dev), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
