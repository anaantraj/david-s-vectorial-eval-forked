#!/usr/bin/env python
"""Analyze every Study 4 embedding-space report and summarize verdict stability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .analyze_study4_frontier import analyze
except ImportError:  # executed directly as ``python scripts/...``
    from analyze_study4_frontier import analyze


def parse_comparisons(raw_values: list[str]) -> list[tuple[str, str]]:
    comparisons = []
    for raw in raw_values:
        baseline, sep, candidate = raw.partition(":")
        if not sep or not baseline or not candidate:
            raise ValueError(f"invalid comparison {raw!r}; expected BASELINE:CANDIDATE")
        comparisons.append((baseline, candidate))
    return comparisons


def analyze_spaces(
    run_dir: Path,
    cells_path: Path,
    split: str,
    comparisons: list[tuple[str, str]],
    *,
    source_margin: float,
    trm_margin: float,
    n_boot: int,
    seed: int,
    content_margin: float = 0.01,
) -> dict:
    with cells_path.open(encoding="utf-8") as fh:
        room_by_cell = {
            row["cell_id"]: row["room"]
            for line in fh
            if line.strip()
            for row in [json.loads(line)]
        }
    reports = sorted(run_dir.glob(f"report.{split}*.json"))
    if not reports:
        raise FileNotFoundError(f"no report.{split}*.json files in {run_dir}")
    results = {}
    for report_path in reports:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        space = report.get("embedding_space") or {}
        slug = space.get("slug")
        if not slug:
            raise ValueError(f"{report_path} does not record its embedding-space slug")
        if slug in results:
            raise ValueError(f"two reports claim embedding space {slug!r}")
        result = analyze(
            report,
            room_by_cell,
            comparisons,
            source_margin=source_margin,
            trm_margin=trm_margin,
            n_boot=n_boot,
            seed=seed,
            content_margin=content_margin,
        )
        output = run_dir / f"frontier.{split}.{slug}.json"
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        results[slug] = result

    sensitivity = {}
    for baseline, candidate in comparisons:
        key = f"{baseline}:{candidate}"
        verdicts = {
            slug: result["comparisons"][key]["frontier_improvement_supported"]
            for slug, result in sorted(results.items())
        }
        sensitivity[key] = {
            "by_embedding_space": verdicts,
            "n_supported": sum(verdicts.values()),
            "n_spaces": len(verdicts),
            "unanimous": len(set(verdicts.values())) == 1,
        }
        plan_verdicts = {
            slug: result["comparisons"][key]
            .get("extract_then_transfer", {})
            .get("retention_improvement_supported")
            for slug, result in sorted(results.items())
        }
        if any(value is not None for value in plan_verdicts.values()):
            sensitivity[key]["extract_then_transfer"] = {
                "by_embedding_space": plan_verdicts,
                "n_supported": sum(value is True for value in plan_verdicts.values()),
                "n_spaces": len(plan_verdicts),
                "unanimous": len(set(plan_verdicts.values())) == 1,
            }
    summary = {
        "split": split,
        "spaces": sorted(results),
        "n_spaces": len(results),
        "values_pooled_across_spaces": False,
        "noninferiority_margins": {
            "source_similarity": source_margin,
            "content_word_retention": content_margin,
            "trm": trm_margin,
        },
        "comparisons": sensitivity,
    }
    (run_dir / f"frontier_sensitivity.{split}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("cells", type=Path)
    parser.add_argument("--split", default="heldout")
    parser.add_argument("--comparison", action="append", default=[])
    parser.add_argument("--source-margin", type=float, default=0.01)
    parser.add_argument("--trm-margin", type=float, default=0.01)
    parser.add_argument("--content-margin", type=float, default=0.01)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()
    summary = analyze_spaces(
        args.run_dir,
        args.cells,
        args.split,
        parse_comparisons(args.comparison),
        source_margin=args.source_margin,
        trm_margin=args.trm_margin,
        n_boot=args.n_bootstrap,
        seed=args.seed,
        content_margin=args.content_margin,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
