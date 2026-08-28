#!/usr/bin/env python3
"""Rate every real and generated Study text against its cell aspect vocabulary.

The supplied aspect pipeline contains vocabularies and historic ratings but not
the executable rater. This freezes a replacement for Study 3 and, critically,
applies it uniformly to generated text, authentic source/target text, and all
baselines. Results are cached by the full prompt and written as a versioned
artifact consumed by the offline ``aspect`` metric.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from vectorial_eval.config import LLMConfig
from vectorial_eval.llm import LLMClient
from vectorial_eval.metrics.aspect import rating_key
from vectorial_eval.transfer.aspect_prompt import AspectIndex

RATER_ID = "study3-aspect-presence-gpt-4.1-mini-v1"
MODEL = "openai/gpt-4.1-mini"
SYSTEM = """You label which evaluative aspects a social post actually invokes.

An aspect is an evaluative lens used to judge or discuss the topic, not a word-match and not a
latent author trait. Mark an aspect present only when the post explicitly raises it or makes a
claim that directly bears on it. Do not infer an aspect merely because it is generally relevant
to the topic. Return only a JSON object mapping every supplied aspect name to one numeric score:
0 absent; 0.3 indirectly invoked; 0.5 clearly present; 0.7 substantive; 1.0 central. Use 0 when
uncertain. Do not omit keys and do not add keys."""


def _parse(text: str, vocabulary: list[str]) -> dict[str, float]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError("rater response is not an object")
    expected = set(vocabulary)
    if set(obj) != expected:
        raise ValueError("rater response keys do not match the frozen vocabulary")
    scores = {name: float(obj[name]) for name in vocabulary}
    if any(value not in {0.0, 0.3, 0.5, 0.7, 1.0} for value in scores.values()):
        raise ValueError("rater response contains an unsupported score")
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--aspect-dir", type=Path, default=Path("data/aspects"))
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--max-concurrency", type=int, default=20)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cells = [json.loads(line) for line in (args.data_dir / "cells.jsonl").read_text().splitlines()]
    posts = []
    for split in ("train", "val", "test"):
        posts.extend(
            json.loads(line)
            for line in (args.data_dir / f"posts.{split}.jsonl").read_text().splitlines()
        )
    texts_by_cell: dict[str, set[str]] = {cell["cell_id"]: set() for cell in cells}
    for post in posts:
        texts_by_cell[post["cell_id"]].add(post["text"])
    for path in sorted(args.run_dir.glob("outputs.*.heldout.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("output_text"):
                texts_by_cell[row["cell_id"]].add(row["output_text"])

    index = AspectIndex(args.aspect_dir)
    jobs = []
    for cell in cells:
        entry = None
        for domain in cell.get("domains", []):
            entry = index.join.get(index._key(domain, cell["topic"]))  # noqa: SLF001
            if entry:
                break
        if not entry:
            continue
        cluster = index._load_cluster(entry["file"])  # noqa: SLF001
        aspects = cluster.get("aspects", [])
        vocabulary = [a["name"] for a in aspects]
        definitions = "\n".join(
            f'- {a["name"]}: {a.get("description", "")}' for a in aspects
        )
        for text in sorted(texts_by_cell[cell["cell_id"]]):
            user = (
                f'Topic: {cell["topic"]}\n\nAspects:\n{definitions}\n\n'
                f"Post:\n{text}\n\nReturn the JSON object."
            )
            jobs.append((cell["cell_id"], text, vocabulary, user))

    cfg = LLMConfig(
        model=args.model,
        temperature=0.0,
        max_tokens=1024,
        max_concurrency=args.max_concurrency,
        cache_dir=args.run_dir / "aspect_rater_cache",
    )
    client = LLMClient(cfg)
    results = client.map(jobs, lambda job: (SYSTEM, job[3]))
    records = []
    failures = 0
    for job, result in zip(jobs, results, strict=True):
        cell_id, text, vocabulary, _user = job
        ok, error, scores = result.ok, result.error, {}
        if ok:
            try:
                scores = _parse(result.text, vocabulary)
            except (ValueError, json.JSONDecodeError) as exc:
                retry = client.complete(SYSTEM, _user, variant=1)
                ok, error = retry.ok, retry.error
                if ok:
                    try:
                        scores = _parse(retry.text, vocabulary)
                    except (ValueError, json.JSONDecodeError) as retry_exc:
                        ok, error = False, f"parse after retry: {retry_exc}"
                elif error is None:
                    error = f"parse: {exc}"
        if not ok:
            failures += 1
        records.append(
            {
                "key": rating_key(cell_id, text),
                "cell_id": cell_id,
                "vocabulary": vocabulary,
                "scores": scores,
                "ok": ok,
                "error": error,
                "rater": {
                    "id": RATER_ID,
                    "model": args.model,
                    "temperature": 0.0,
                    "system_prompt": SYSTEM,
                },
            }
        )

    out = args.run_dir / "aspect_ratings.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for row in records:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    logging.info("wrote %s ratings to %s (%s failures)", len(records), out, failures)
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
