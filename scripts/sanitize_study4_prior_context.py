#!/usr/bin/env python
"""Remove scored-post text from metadata exposed to Study 4 prior training."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

try:
    from .audit_study4_priors import contains_scored, load, normalized_text
    from .normalize_study4_corpus import write_jsonl_atomic
except ImportError:
    from audit_study4_priors import contains_scored, load, normalized_text
    from normalize_study4_corpus import write_jsonl_atomic


VISIBLE_METADATA = (
    "subreddit",
    "community",
    "thread_context",
    "context_summary",
    "engagement",
    "dimensions",
    "profile_weak_labels",
    "audience_prior",
    "title",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def completion_projection_sha256(path: Path) -> str:
    """Hash fields that sanitization is forbidden to change."""
    digest = hashlib.sha256()
    for row in load(path):
        projection = {
            key: row.get(key)
            for key in (
                "record_id",
                "author_id",
                "platform",
                "audience",
                "record_type",
                "text",
                "split",
            )
        }
        digest.update(
            (json.dumps(projection, sort_keys=True, ensure_ascii=False) + "\n").encode()
        )
    return digest.hexdigest()


def scrub(value: object, forbidden: set[str]) -> tuple[object, int]:
    if isinstance(value, str):
        text = normalized_text(value)
        return (
            ("", 1)
            if len(text) >= 40 and contains_scored(text, forbidden)
            else (value, 0)
        )
    if isinstance(value, list):
        values, count = [], 0
        for item in value:
            clean, removed = scrub(item, forbidden)
            values.append(clean)
            count += removed
        return values, count
    if isinstance(value, dict):
        values, count = {}, 0
        for key, item in value.items():
            clean, removed = scrub(item, forbidden)
            values[key] = clean
            count += removed
        return values, count
    return value, 0


def sanitize(source: Path, output: Path, scored_paths: list[Path]) -> dict:
    scored = [row for path in scored_paths for row in load(path)]
    forbidden = {
        platform: {
            text
            for row in scored
            if row.get("platform") == platform
            and len(text := normalized_text(row.get("text"))) >= 40
        }
        for platform in ("reddit", "linkedin")
    }
    rows = load(source)
    removals = Counter()
    for row in rows:
        platform_forbidden = forbidden.get(row.get("platform"), set())
        completion = normalized_text(row.get("text"))
        if len(completion) >= 40 and contains_scored(completion, platform_forbidden):
            raise RuntimeError(
                f"{row.get('record_id')}: completion itself overlaps a scored post"
            )
        for field in VISIBLE_METADATA:
            clean, count = scrub(row.get(field), platform_forbidden)
            row[field] = clean
            removals[field] += count
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(output, rows)
    manifest = {
        "source": str(source),
        "source_sha256": sha256(source),
        "output": str(output),
        "output_sha256": sha256(output),
        "source_completion_projection_sha256": completion_projection_sha256(source),
        "output_completion_projection_sha256": completion_projection_sha256(output),
        "n_records": len(rows),
        "scored_splits": [str(path) for path in scored_paths],
        "scored_split_sha256": {str(path): sha256(path) for path in scored_paths},
        "removed_exact_metadata_strings": {
            key: value for key, value in sorted(removals.items()) if value
        },
        "completion_records_removed": 0,
        "policy": (
            "normalized scored-post strings and substantial contained excerpts "
            "are blanked only in metadata; "
            "authentic completion text and record cardinality are unchanged"
        ),
    }
    manifest_path = output.with_suffix(output.suffix + ".sanitization.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scored", action="append", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(sanitize(args.source, args.output, args.scored), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
