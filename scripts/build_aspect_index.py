#!/usr/bin/env python
"""Build the (domain, final_topic) join index for the shipped aspect vocabularies.

The Vectorial aspect package keys its per-cluster files by `final_topic` alone,
while the harness keys clusters by `(domain, final_topic)`. Several topics occur
under two domains, so a join on topic alone silently merges two source clusters
into one vocabulary. This script makes that merge explicit rather than implicit:
it writes one entry per `(domain, final_topic)` pair observed in
`final_dataset_v1.csv`, records which cluster file serves that pair, whether the
same file also serves another domain (`domain_merged`), and whether posts from
this particular domain were in the bilateral subset the vocabulary was extracted
from (`domain_contributed`). A `(domain, topic)` pair absent from the index is
not covered, which is what stops an unverified topic-name match from being
treated as a hit.

The output is committed alongside the vocabularies so that the transfer function
performs no CSV parsing at inference time.

    python scripts/build_aspect_index.py
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASPECT_DIR = ROOT / "data" / "aspects"
CLUSTER_DIR = ASPECT_DIR / "clusters"
CSV_PATH = ROOT / "final_dataset_v1.csv"
OUT_PATH = ASPECT_DIR / "aspect_join_index.json"

NOTE = (
    "Aspect vocabularies were extracted by gpt-4.1-mini with platform labels "
    "visible to the extracting model. A blind rerun is pending; any result "
    "conditioned on these vocabularies is provisional until it lands."
)


def _is_bilateral(row: dict) -> bool:
    return row["bilateral_flag"].strip().lower() in ("true", "1", "yes")


def main() -> int:
    clusters = {}
    for path in sorted(CLUSTER_DIR.glob("aspects_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        clusters[data["cluster"]] = path.name

    total: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    bilateral: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    with CSV_PATH.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            topic = row["final_topic"]
            if topic not in clusters:
                continue
            total[topic][row["domain"]] += 1
            if _is_bilateral(row):
                bilateral[topic][row["domain"]] += 1

    join: dict[str, dict] = {}
    for topic, domain_counts in total.items():
        contributing = sorted(d for d in domain_counts if bilateral[topic].get(d))
        for domain in sorted(domain_counts):
            join[f"{domain}||{topic}"] = {
                "cluster": topic,
                "file": clusters[topic],
                # True when this one vocabulary serves more than one source
                # cluster, i.e. the shipped file already merged them.
                "domain_merged": len(contributing) > 1,
                "contributing_domains": contributing,
                # False when no post from this domain was in the bilateral
                # subset the vocabulary was extracted from.
                "domain_contributed": domain in contributing,
                "n_posts_this_domain": domain_counts[domain],
                "n_bilateral_posts_this_domain": bilateral[topic].get(domain, 0),
            }

    unmatched = sorted(set(clusters) - set(total))
    payload = {
        "note": NOTE,
        "source_package": "aspect_stylistics (Vectorial, received 2026-07-29)",
        "n_clusters": len(clusters),
        "n_join_keys": len(join),
        "clusters_with_no_matching_topic": unmatched,
        "join": dict(sorted(join.items())),
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_PATH} — {len(clusters)} clusters, {len(join)} (domain, topic) keys")
    merged = sorted({v["cluster"] for v in join.values() if v["domain_merged"]})
    print(f"domain-merged clusters ({len(merged)}): {', '.join(merged)}")
    nc = sorted(k for k, v in join.items() if not v["domain_contributed"])
    print(f"keys whose domain did not contribute to the vocabulary ({len(nc)}): {nc}")
    if unmatched:
        print(f"clusters with no matching final_topic in the CSV: {unmatched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
