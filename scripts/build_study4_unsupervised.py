#!/usr/bin/env python
"""Build a de-identified, author-grouped Study 4 Reddit prior corpus from S3.

Posts are not the whole audience export.  This builder also retains comments,
thread context, subreddit and engagement fields, post dimensions, and
train-profile weak labels.  The latter remain structured metadata rather than
being concatenated into target text, so generated prose is not trained to emit
profile summaries.
"""

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
    "backend_engineer": "4dc6850a-027f-45d3-9b96-5140dbcbae36",
    "fullstack_engineer": "9d9464a8-abc7-44ea-bf85-c43d699eceed",
    "instructional_designer": "69104124-9677-4f13-8452-9e863edbccf3",
    "edtech_engineering": "3bd77eca-d93a-434d-9a7a-751a03c52db4",
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
        args = ["list-objects-v2", "--bucket", BUCKET, "--prefix", prefix, "--max-keys", "1000"]
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
    return hashlib.blake2b(value.encode(), digest_size=16, person=b"vectorial-s4").hexdigest()


def author_split(author_id: str) -> str:
    value = int(hashlib.blake2b(author_id.encode(), digest_size=8, person=b"s4-split").hexdigest(), 16)
    return "dev" if value % 10 == 0 else "train"


def normal_id(value: str) -> str:
    value = str(value or "").strip()
    return value.split("_", 1)[-1] if value.startswith(("t1_", "t3_")) else value


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
                if row.get("platform") != "reddit":
                    continue
                ids.add(normal_id(row.get("post_id")))
                urls.add(normal_url(row.get("url")))
    return ids - {""}, urls - {""}


def label_values(value) -> list[str]:
    """Flatten weak-label structures without copying profile prose into prompts."""
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data/study4_unsupervised"))
    parser.add_argument("--per-audience", type=int, default=250)
    parser.add_argument("--stem", default="", help="output stem; defaults to the actual rung")
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument(
        "--evaluation-data-dir",
        type=Path,
        default=Path("data/study3"),
        help="held-out benchmark whose matching profiles are excluded",
    )
    parser.add_argument(
        "--allow-shortfall",
        action="store_true",
        help="use every available record when an audience is below the requested rung",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    heldout_ids, heldout_urls = evaluation_keys(args.evaluation_data_dir)
    selected = []
    excluded_profiles_total = 0
    available_counts = {}
    room_weak_labels = {}
    with tempfile.TemporaryDirectory(prefix="vectorial-s4-") as raw_temp:
        temp_dir = Path(raw_temp)
        for audience, room_id in sorted(ROOMS.items()):
            room_documents = {}
            for name in ("description", "content_themes", "signals_breakdown"):
                key = f"reddit-audience/{room_id}/{name}.json"
                try:
                    _, room_documents[name] = fetch_json(key, temp_dir)
                except subprocess.CalledProcessError:
                    room_documents[name] = {}
            description = room_documents["description"]
            themes = room_documents["content_themes"]
            room_weak_labels[audience] = {
                "summary": description.get("summary") or "",
                "traits": description.get("traits") or [],
                "theme_categories": sorted((themes.get("categories") or {}).keys()),
                "signals_summary": room_documents["signals_breakdown"].get("summary") or {},
                "policy": "audit only; includes authors outside the reconstructed train split",
            }
            prefix = f"reddit-audience/{room_id}/profiles/"
            keys = sorted(
                k
                for k in object_keys(prefix)
                if k.endswith(("/posts.json", "/comments.json", "/profile.json"))
            )
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                documents = list(pool.map(lambda k: fetch_json(k, temp_dir), keys))
            records = {}
            profiles = {
                key.split("/")[3]: document
                for key, document in documents
                if key.endswith("/profile.json")
            }
            excluded_profiles = {
                key.split("/")[3]
                for key, document in documents
                if key.endswith("/posts.json")
                for item in (
                    document if isinstance(document, list) else document.get("posts", [])
                )
                if normal_id(item.get("id")) in heldout_ids
                or normal_url(item.get("url")) in heldout_urls
            }
            excluded_profiles_total += len(excluded_profiles)
            for key, document in documents:
                profile_id = key.split("/")[3]
                if profile_id in excluded_profiles:
                    continue
                if key.endswith("/profile.json"):
                    continue
                record_type = "post" if key.endswith("/posts.json") else "comment"
                items = document if isinstance(document, list) else document.get(
                    "posts" if record_type == "post" else "comments", []
                )
                for item in items:
                    text = (
                        item.get("text")
                        if record_type == "post"
                        else item.get("body")
                    )
                    text = (text or "").strip()
                    raw_id = str(item.get("id") or "").strip()
                    raw_author = str(
                        item.get("userId")
                        or item.get("username")
                        or item.get("author")
                        or profile_id
                    )
                    if raw_author.strip().lower() in {"", "[deleted]", "[removed]", "none"}:
                        raw_author = profile_id
                    created = int(item.get("created_utc") or 0)
                    if not raw_id or len(text) < 30 or created > 1787270399:
                        continue
                    author_id = digest("reddit:" + raw_author)
                    profile = profiles.get(profile_id, {})
                    record_id = f"reddit:{record_type}:{raw_id}"
                    records[record_id] = {
                        "record_id": record_id,
                        "author_id": author_id,
                        "platform": "reddit",
                        "audience": audience,
                        "source_room_id": room_id,
                        "record_type": record_type,
                        "created_utc": created,
                        "text": text,
                        "title": (item.get("title") or item.get("postTitle") or "").strip(),
                        "subreddit": (item.get("subreddit") or "").strip(),
                        "thread_context": item.get("context") or {},
                        "context_summary": (item.get("context_summary") or "").strip(),
                        "engagement": {
                            "score": item.get("score"),
                            "upvote_ratio": item.get("upvoteRatio"),
                            "comment_count": item.get("commentCount"),
                        },
                        "dimensions": item.get("dimensions") or {},
                        "profile_weak_labels": {
                            "summary": (profile.get("summary") or "").strip(),
                            "highlights": profile.get("highlights") or [],
                            "keywords": profile.get("keywords") or [],
                        },
                    }
            ordered = sorted(records.values(), key=lambda x: digest(x["record_id"]))
            available_counts[audience] = len(ordered)
            if len(ordered) < args.per_audience:
                if not args.allow_shortfall:
                    raise RuntimeError(
                        f"{audience} has {len(ordered)} usable unique records, "
                        f"needs {args.per_audience}"
                    )
            selected.extend(ordered[: args.per_audience])

    for record in selected:
        record["split"] = author_split(record["author_id"])
    train = sorted((r for r in selected if r["split"] == "train"), key=lambda x: x["record_id"])
    dev = sorted((r for r in selected if r["split"] == "dev"), key=lambda x: x["record_id"])
    if not dev:
        raise RuntimeError("deterministic author split produced no dev records")
    if {r["author_id"] for r in train} & {r["author_id"] for r in dev}:
        raise RuntimeError("author leakage between train and dev")

    # Priors are estimated from train authors only, then frozen and attached to
    # both splits.  This makes the dev loss representative of inference, where
    # an audience prior is available but the scored author's own profile is not.
    priors = {}
    for audience in sorted(ROOMS):
        audience_train = [r for r in train if r["audience"] == audience]
        author_rows = {r["author_id"]: r for r in audience_train}.values()
        keyword_counts = Counter(
            label
            for r in author_rows
            for label in label_values(r["profile_weak_labels"].get("keywords"))
        )
        dimension_counts = Counter(
            label for r in audience_train for label in label_values(r["dimensions"])
        )
        subreddit_counts = Counter(
            r["subreddit"] for r in audience_train if r["subreddit"]
        )
        priors[audience] = {
            "keywords": [x for x, _ in keyword_counts.most_common(32)],
            "dimensions": [x for x, _ in dimension_counts.most_common(32)],
            "subreddits": [x for x, _ in subreddit_counts.most_common(16)],
            "estimated_from": "train authors only",
        }
    for record in selected:
        record["audience_prior"] = priors[record["audience"]]
    for record in dev:
        record["profile_weak_labels"] = {}

    stem = args.stem or f"reddit_{args.per_audience}_per_audience"
    for split, records in (("train", train), ("dev", dev)):
        path = args.out_dir / f"{stem}.{split}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "bucket": BUCKET,
        "region": "us-west-2",
        "platform": "reddit",
        "audiences": sorted(ROOMS),
        "room_crosswalk": ROOMS,
        "per_audience": args.per_audience,
        "available_per_audience": available_counts,
        "n_total": len(selected),
        "n_train": len(train),
        "n_dev": len(dev),
        "split_unit": "author_id",
        "heldout_profile_exclusions": excluded_profiles_total,
        "heldout_exclusion_rule": (
            "drop the complete S3 profile when any post ID or normalized URL "
            "matches Study 3 val/test"
        ),
        "selection": "nested deterministic blake2b over record_id",
        "cutoff_utc": 1787270399,
        "derived_fields_retained": True,
        "derived_fields_used_as_lm_targets": False,
        "record_types": {
            kind: sum(r["record_type"] == kind for r in selected)
            for kind in ("post", "comment")
        },
        "structured_prior_fields": [
            "subreddit",
            "thread_context",
            "context_summary",
            "engagement",
            "dimensions",
            "profile_weak_labels",
        ],
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
