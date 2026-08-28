#!/usr/bin/env python
"""Partition a Study 4 training corpus for matched coarse-to-fine adaptation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

try:
    from .normalize_study4_corpus import write_jsonl_atomic
except ImportError:  # direct ``python scripts/...`` execution
    from normalize_study4_corpus import write_jsonl_atomic


def rank(record_id: str) -> bytes:
    return hashlib.blake2b(record_id.encode(), digest_size=16).digest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def partition(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return disjoint stable-ranked halves within every primary audience."""
    platform, audience = [], []
    for label in sorted({row["audience"] for row in rows}):
        group = sorted(
            (row for row in rows if row["audience"] == label),
            key=lambda row: (rank(row["record_id"]), row["record_id"]),
        )
        cut = (len(group) + 1) // 2
        platform.extend(group[:cut])
        audience.extend(group[cut:])
    return platform, audience


def build(train_path: Path, dev_path: Path, output_dir: Path, platform: str) -> dict:
    train, dev = read_jsonl(train_path), read_jsonl(dev_path)
    first, second = partition(train)
    first_ids = {row["record_id"] for row in first}
    second_ids = {row["record_id"] for row in second}
    if first_ids & second_ids or first_ids | second_ids != {
        row["record_id"] for row in train
    }:
        raise RuntimeError("coarse-to-fine partition is not a disjoint union")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "platform_train": output_dir / f"{platform}.platform_half.train.jsonl",
        "audience_train": output_dir / f"{platform}.audience_half.train.jsonl",
        "shared_dev": output_dir / f"{platform}.shared.dev.jsonl",
    }
    write_jsonl_atomic(paths["platform_train"], first)
    write_jsonl_atomic(paths["audience_train"], second)
    write_jsonl_atomic(paths["shared_dev"], dev)
    manifest = {
        "platform": platform,
        "algorithm": (
            "stable blake2b-128 rank within audience; first ceil(n/2) records "
            "to platform stage, remainder to audience stage"
        ),
        "training_sets_are_disjoint": True,
        "training_union_equals_parent": True,
        "shared_development_set": True,
        "n_parent_train": len(train),
        "n_platform_train": len(first),
        "n_audience_train": len(second),
        "n_dev": len(dev),
        "per_audience_parent": dict(Counter(row["audience"] for row in train)),
        "per_audience_platform": dict(Counter(row["audience"] for row in first)),
        "per_audience_audience": dict(Counter(row["audience"] for row in second)),
        "source": {
            "train": str(train_path),
            "train_sha256": sha256(train_path),
            "dev": str(dev_path),
            "dev_sha256": sha256(dev_path),
        },
        "outputs": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in paths.items()
        },
    }
    manifest_path = output_dir / f"{platform}.coarse_to_fine.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train", type=Path)
    parser.add_argument("dev", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--platform", required=True, choices=["reddit", "linkedin"])
    args = parser.parse_args()
    result = build(args.train, args.dev, args.output_dir, args.platform)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
