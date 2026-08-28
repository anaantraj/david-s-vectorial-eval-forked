#!/usr/bin/env python
"""Build the de-identified LinkedIn half of the Study 4 prior corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BUCKET = "audience-rooms-beta"
ROOMS = {
    "backend_engineer": "b2fffd34-421b-4dad-a45b-4ec9c76e140b",
    "fullstack_engineer": "cc0b0f6e-00f6-45b9-850a-1712e5b884f7",
    "instructional_designer": "40c7509c-e988-42b4-892a-3b75cca66440",
    "edtech_engineering": "0ff36a9c-6534-4fe2-af7b-8557d371b312",
}


def aws_json(*args: str) -> dict:
    proc = subprocess.run(
        ["aws", "s3api", *args, "--region", "us-west-2", "--output", "json"],
        check=True,
        capture_output=True,
        text=True,
        env=os.environ,
    )
    return json.loads(proc.stdout)


def object_keys(prefix: str) -> list[str]:
    token = None
    keys = []
    while True:
        args = [
            "list-objects-v2", "--bucket", BUCKET, "--prefix", prefix,
            "--max-keys", "1000",
        ]
        if token:
            args += ["--continuation-token", token]
        page = aws_json(*args)
        keys.extend(x["Key"] for x in page.get("Contents", []))
        token = page.get("NextContinuationToken")
        if not token:
            return keys


def fetch_json(key: str, temp_dir: Path):
    target = temp_dir / hashlib.sha256(key.encode()).hexdigest()
    subprocess.run(
        [
            "aws", "s3api", "get-object", "--bucket", BUCKET, "--key", key,
            "--region", "us-west-2", str(target), "--output", "json",
        ],
        check=True,
        capture_output=True,
        env=os.environ,
    )
    return key, json.loads(target.read_text(encoding="utf-8", errors="replace"))


def digest(value: str) -> str:
    return hashlib.blake2b(
        value.encode(), digest_size=16, person=b"vectorial-s4"
    ).hexdigest()


def author_split(author_id: str) -> str:
    value = int(
        hashlib.blake2b(
            author_id.encode(), digest_size=8, person=b"s4-split"
        ).hexdigest(),
        16,
    )
    return "dev" if value % 10 == 0 else "train"


def normal_url(value: str) -> str:
    return str(value or "").split("?", 1)[0].rstrip("/")


def evaluation_keys(data_dir: Path) -> tuple[set[str], set[str]]:
    ids, urls = set(), set()
    for name in ("posts.val.jsonl", "posts.test.jsonl"):
        path = data_dir / name
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                if row.get("platform") != "linkedin":
                    continue
                ids.add(str(row.get("post_id") or "").strip())
                urls.add(normal_url(row.get("url")))
    return ids - {""}, urls - {""}


def label_values(value) -> list[str]:
    if isinstance(value, str):
        value = value.strip()
        return [value] if value else []
    if isinstance(value, list):
        return [item for child in value for item in label_values(child)]
    if isinstance(value, dict):
        return [
            item
            for key, child in sorted(value.items())
            for item in [str(key), *label_values(child)]
            if item
        ]
    return []


def post_timestamp(post: dict) -> int:
    value = int(post.get("postedAtTimestamp") or 0)
    return value // 1000 if value > 10_000_000_000 else value


def linkedin_author_id(profile: dict, profile_id: str) -> str:
    """Stable across room-specific export UUIDs whenever the source URL exists."""
    raw_author = normal_url(profile.get("linkedin_profile_url")) or profile_id
    return digest("linkedin:" + raw_author)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data/study4_priors"))
    parser.add_argument("--per-audience", type=int, default=10000)
    parser.add_argument("--stem", default="")
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--allow-shortfall", action="store_true")
    parser.add_argument(
        "--evaluation-data-dir", type=Path, default=Path("data/study3")
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    heldout_ids, heldout_urls = evaluation_keys(args.evaluation_data_dir)
    selected = []
    available_counts = {}
    excluded_profiles_total = 0
    room_weak_labels = {}

    with tempfile.TemporaryDirectory(prefix="vectorial-s4-linkedin-") as raw_temp:
        temp_dir = Path(raw_temp)
        for audience, room_id in sorted(ROOMS.items()):
            try:
                _, description = fetch_json(
                    f"linkedin-audience/{room_id}/description.json", temp_dir
                )
            except subprocess.CalledProcessError:
                description = {}
            room_weak_labels[audience] = {
                "summary": description.get("summary") or "",
                "traits": description.get("traits") or [],
                "policy": "audit only; includes authors outside reconstructed train",
            }

            prefix = f"linkedin-audience/{room_id}/profiles/"
            keys = sorted(
                key
                for key in object_keys(prefix)
                if key.endswith(("/posts.json", "/comment.json", "/profile.json"))
            )
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                documents = list(pool.map(lambda key: fetch_json(key, temp_dir), keys))
            profiles = {
                key.split("/")[3]: document
                for key, document in documents
                if key.endswith("/profile.json")
            }
            excluded_profiles = {
                key.split("/")[3]
                for key, document in documents
                if key.endswith("/posts.json")
                for post in document.get("posts", [])
                if str(post.get("urn") or "").strip() in heldout_ids
                or normal_url(post.get("url")) in heldout_urls
            }
            excluded_profiles_total += len(excluded_profiles)
            records = {}
            for key, document in documents:
                profile_id = key.split("/")[3]
                if profile_id in excluded_profiles or key.endswith("/profile.json"):
                    continue
                profile = profiles.get(profile_id, {})
                weak_profile = {
                    field: profile.get(field)
                    for field in (
                        "role", "domain", "industry", "education", "experience",
                        "keywords", "highlights", "summary", "additional_info",
                    )
                }
                if key.endswith("/posts.json"):
                    items = document.get("posts", [])
                    record_type = "post"
                else:
                    items = document.get("comments", [])
                    record_type = "comment"
                for item in items:
                    text = (
                        item.get("text")
                        if record_type == "post"
                        else item.get("comment_body")
                    )
                    text = (text or "").strip()
                    raw_id = (
                        str(item.get("urn") or item.get("comment_url") or "").strip()
                    )
                    if not raw_id or len(text) < 30:
                        continue
                    created = post_timestamp(item) if record_type == "post" else 0
                    if created > 1787270399:
                        continue
                    record_id = f"linkedin:{record_type}:{digest(raw_id)}"
                    records[record_id] = {
                        "record_id": record_id,
                        "author_id": linkedin_author_id(profile, profile_id),
                        "platform": "linkedin",
                        "audience": audience,
                        "source_room_id": room_id,
                        "record_type": record_type,
                        "created_utc": created,
                        "text": text,
                        "title": "",
                        "community": (item.get("authorHeadline") or "").strip(),
                        "thread_context": {
                            "post_body": item.get("post_body"),
                            "parent_comment_body": item.get("parent_comment_body"),
                        },
                        "context_summary": (item.get("context_summary") or "").strip(),
                        "engagement": {
                            "likes": item.get("numLikes"),
                            "comments": item.get("numComments"),
                            "shares": item.get("numShares"),
                        },
                        "dimensions": item.get("labels") or {},
                        "profile_weak_labels": weak_profile,
                    }
            ordered = sorted(records.values(), key=lambda row: digest(row["record_id"]))
            available_counts[audience] = len(ordered)
            if len(ordered) < args.per_audience and not args.allow_shortfall:
                raise RuntimeError(
                    f"{audience} has {len(ordered)} usable unique records, "
                    f"needs {args.per_audience}"
                )
            selected.extend(ordered[: args.per_audience])

    for record in selected:
        record["split"] = author_split(record["author_id"])
    train = sorted(
        (record for record in selected if record["split"] == "train"),
        key=lambda row: row["record_id"],
    )
    dev = sorted(
        (record for record in selected if record["split"] == "dev"),
        key=lambda row: row["record_id"],
    )
    if {row["author_id"] for row in train} & {row["author_id"] for row in dev}:
        raise RuntimeError("author leakage between train and dev")

    priors = {}
    for audience in sorted(ROOMS):
        audience_train = [row for row in train if row["audience"] == audience]
        author_rows = {row["author_id"]: row for row in audience_train}.values()
        profile_counts = Counter(
            label
            for row in author_rows
            for field in ("role", "domain", "industry", "keywords", "highlights")
            for label in label_values(row["profile_weak_labels"].get(field))
        )
        dimension_counts = Counter(
            label
            for row in audience_train
            for label in label_values(row["dimensions"])
        )
        priors[audience] = {
            "keywords": [label for label, _ in profile_counts.most_common(32)],
            "dimensions": [label for label, _ in dimension_counts.most_common(32)],
            "communities": [],
            "estimated_from": "train authors only",
        }
    for record in selected:
        record["audience_prior"] = priors[record["audience"]]
    for record in dev:
        record["profile_weak_labels"] = {}

    stem = args.stem or f"linkedin_{args.per_audience}_per_audience"
    for split, records in (("train", train), ("dev", dev)):
        path = args.out_dir / f"{stem}.{split}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    manifest = {
        "bucket": BUCKET,
        "region": "us-west-2",
        "platform": "linkedin",
        "audiences": sorted(ROOMS),
        "room_crosswalk": ROOMS,
        "per_audience": args.per_audience,
        "available_per_audience": available_counts,
        "n_total": len(selected),
        "n_train": len(train),
        "n_dev": len(dev),
        "record_types": {
            kind: sum(row["record_type"] == kind for row in selected)
            for kind in ("post", "comment")
        },
        "split_unit": "author_id",
        "selection": "nested deterministic blake2b over record_id",
        "heldout_profile_exclusions": excluded_profiles_total,
        "derived_fields_retained": True,
        "derived_fields_used_as_lm_targets": False,
        "weak_label_policy": (
            "retained only on train records; dev receives only frozen "
            "train-author aggregates"
        ),
        "audience_priors": priors,
        "room_weak_labels": room_weak_labels,
    }
    (args.out_dir / f"{stem}.manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
