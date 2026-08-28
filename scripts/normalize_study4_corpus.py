#!/usr/bin/env python
"""Resolve cross-room clones and recompute train-only priors in a built corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_study4_unsupervised import label_values  # noqa: E402


def choose_record(rows: list[dict]) -> dict:
    authors = {row["author_id"] for row in rows}
    splits = {row["split"] for row in rows}
    texts = {row["text"] for row in rows}
    if len(splits) != 1 or len(texts) != 1:
        raise RuntimeError(
            f"clone {rows[0]['record_id']} disagrees on author or split: "
            f"{len(authors)} authors, {splits} splits, {len(texts)} texts"
        )
    audiences = sorted({row["audience"] for row in rows})
    winner = min(
        rows,
        key=lambda row: hashlib.blake2b(
            f"{row['record_id']}:{row['audience']}".encode(), digest_size=8
        ).digest(),
    ).copy()
    winner["audience_memberships"] = audiences
    if len(authors) > 1:
        winner["author_aliases"] = sorted(authors)
    return winner


def deduplicate(rows: list[dict]) -> tuple[list[dict], int]:
    grouped = {}
    for row in rows:
        grouped.setdefault(row["record_id"], []).append(row)
    duplicates = sum(len(group) - 1 for group in grouped.values())
    resolved = [choose_record(group) for _, group in sorted(grouped.items())]
    return resolved, duplicates


def recompute_priors(rows: list[dict], platform: str) -> dict:
    train = [row for row in rows if row["split"] == "train"]
    priors = {}
    for audience in sorted({row["audience"] for row in rows}):
        audience_train = [row for row in train if row["audience"] == audience]
        author_rows = {row["author_id"]: row for row in audience_train}.values()
        if platform == "reddit":
            keyword_counts = Counter(
                label
                for row in author_rows
                for label in label_values(
                    row["profile_weak_labels"].get("keywords")
                )
            )
            communities = Counter(
                row.get("subreddit") for row in audience_train if row.get("subreddit")
            )
        else:
            keyword_counts = Counter(
                label
                for row in author_rows
                for field in ("role", "domain", "industry", "keywords", "highlights")
                for label in label_values(row["profile_weak_labels"].get(field))
            )
            communities = Counter()
        dimensions = Counter(
            label for row in audience_train for label in label_values(row["dimensions"])
        )
        priors[audience] = {
            "keywords": [label for label, _ in keyword_counts.most_common(32)],
            "dimensions": [label for label, _ in dimensions.most_common(32)],
            ("subreddits" if platform == "reddit" else "communities"): [
                label for label, _ in communities.most_common(16)
            ],
            "estimated_from": "train authors only",
        }
    return priors


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    fd, raw_temp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    temp = Path(raw_temp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for row in sorted(rows, key=lambda item: item["record_id"]):
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("train", type=Path)
    parser.add_argument("dev", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = []
    for path in (args.train, args.dev):
        # Iterate on physical JSONL newlines. `str.splitlines()` also splits at
        # valid Unicode separators such as U+2028 inside a JSON string and can
        # turn one valid record into two invalid fragments.
        with path.open(encoding="utf-8") as fh:
            rows.extend(json.loads(line) for line in fh if line.strip())
    rows, duplicates = deduplicate(rows)
    priors = recompute_priors(rows, manifest["platform"])
    for row in rows:
        row["audience_prior"] = priors[row["audience"]]
        if row["split"] == "dev":
            row["profile_weak_labels"] = {}
    train = [row for row in rows if row["split"] == "train"]
    dev = [row for row in rows if row["split"] == "dev"]
    write_jsonl_atomic(args.train, train)
    write_jsonl_atomic(args.dev, dev)
    manifest.update(
        {
            "n_total": len(rows),
            "n_train": len(train),
            "n_dev": len(dev),
            "cross_room_duplicate_rows_removed": duplicates,
            "cross_room_policy": (
                "one deterministic primary audience plus all audience_memberships"
            ),
            "audience_priors": priors,
            "record_types": dict(Counter(row["record_type"] for row in rows)),
        }
    )
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
