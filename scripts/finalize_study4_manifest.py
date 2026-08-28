#!/usr/bin/env python
"""Attach source-inventory and output hashes to a Study 4 corpus manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def aws_page(prefix: str, token: str | None = None) -> dict:
    args = [
        "aws", "s3api", "list-objects-v2", "--bucket", "audience-rooms-beta",
        "--prefix", prefix, "--max-keys", "1000", "--region", "us-west-2",
        "--output", "json",
    ]
    if token:
        args += ["--continuation-token", token]
    proc = subprocess.run(
        args, check=True, capture_output=True, text=True, env=os.environ
    )
    return json.loads(proc.stdout)


def inventory(prefix: str) -> dict:
    objects, token = [], None
    while True:
        page = aws_page(prefix, token)
        objects.extend(page.get("Contents", []))
        token = page.get("NextContinuationToken")
        if not token:
            break
    canonical = [
        {
            "relative_key_hash": hashlib.sha256(
                item["Key"].removeprefix(prefix).encode()
            ).hexdigest(),
            "etag": str(item.get("ETag") or "").strip('"'),
            "size": int(item.get("Size") or 0),
        }
        for item in sorted(objects, key=lambda row: row["Key"])
    ]
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return {
        "n_objects": len(canonical),
        "n_bytes": sum(item["size"] for item in canonical),
        "inventory_sha256": hashlib.sha256(payload).hexdigest(),
        "entries": canonical,
    }


def file_hash(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("files", type=Path, nargs="+")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    platform = manifest["platform"]
    crosswalk = manifest.get("room_crosswalk")
    if not crosswalk:
        if platform == "reddit":
            from build_study4_unsupervised import ROOMS as crosswalk
        else:
            from build_study4_linkedin_priors import ROOMS as crosswalk
    manifest["source_inventory"] = {
        audience: inventory(f"{platform}-audience/{room_id}/")
        for audience, room_id in sorted(crosswalk.items())
    }
    manifest["output_files"] = [file_hash(path) for path in sorted(args.files)]
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
