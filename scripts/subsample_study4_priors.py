#!/usr/bin/env python
"""Create a deterministic nested rung from an audited Study 4 prior corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from normalize_study4_corpus import recompute_priors, write_jsonl_atomic


def rank(record_id: str) -> bytes:
    """Stable sampling rank shared by every rung."""
    return hashlib.blake2b(record_id.encode(), digest_size=16).digest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def nested_sample(rows: list[dict], per_audience: int) -> list[dict]:
    """Take the first stable-ranked records within each primary audience."""
    if per_audience <= 0:
        raise ValueError("per_audience must be positive")
    selected = []
    for audience in sorted({row["audience"] for row in rows}):
        pool = [row for row in rows if row["audience"] == audience]
        selected.extend(
            sorted(pool, key=lambda row: (rank(row["record_id"]), row["record_id"]))[
                :per_audience
            ]
        )
    return selected


def file_hash(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def build_rung(
    source_manifest: Path,
    source_train: Path,
    source_dev: Path,
    output_stem: Path,
    per_audience: int,
) -> dict:
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    rows = read_jsonl(source_train) + read_jsonl(source_dev)
    if len({row["record_id"] for row in rows}) != len(rows):
        raise RuntimeError("source corpus contains duplicate record_ids")
    selected = nested_sample(rows, per_audience)
    priors = recompute_priors(selected, manifest["platform"])
    for row in selected:
        row["audience_prior"] = priors[row["audience"]]
        if row["split"] == "dev":
            row["profile_weak_labels"] = {}

    train = [row for row in selected if row["split"] == "train"]
    dev = [row for row in selected if row["split"] == "dev"]
    train_path = output_stem.with_suffix(".train.jsonl")
    dev_path = output_stem.with_suffix(".dev.jsonl")
    manifest_path = output_stem.with_suffix(".manifest.json")
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(train_path, train)
    write_jsonl_atomic(dev_path, dev)

    source_output_hashes = {
        item["path"]: item["sha256"] for item in manifest.get("output_files", [])
    }
    result = {
        **manifest,
        "requested_per_audience": per_audience,
        "n_total": len(selected),
        "n_train": len(train),
        "n_dev": len(dev),
        "available_counts": dict(Counter(row["audience"] for row in rows)),
        "selected_counts": dict(Counter(row["audience"] for row in selected)),
        "record_types": dict(Counter(row["record_type"] for row in selected)),
        "audience_priors": priors,
        "nested_sampling": {
            "algorithm": "lowest blake2b-128(record_id) ranks within primary audience",
            "source_manifest": source_manifest.name,
            "source_manifest_sha256": hashlib.sha256(
                source_manifest.read_bytes()
            ).hexdigest(),
            "source_output_sha256": source_output_hashes,
        },
    }
    # These describe the newly written rung, not its parent corpus.
    result["output_files"] = [file_hash(path) for path in (dev_path, train_path)]
    manifest_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("source_train", type=Path)
    parser.add_argument("source_dev", type=Path)
    parser.add_argument("output_stem", type=Path)
    parser.add_argument("--per-audience", type=int, required=True)
    args = parser.parse_args()
    build_rung(
        args.source_manifest,
        args.source_train,
        args.source_dev,
        args.output_stem,
        args.per_audience,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
