#!/usr/bin/env python
"""Backfill token and input provenance for Study 4 jobs launched before logging it."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict) -> None:
    fd, raw_temp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    temp = Path(raw_temp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, default=str)
            fh.write("\n")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def token_metadata(tokenizer, records: list[dict], max_length: int, use_priors: bool) -> dict:
    from vectorial_eval.methods.lora.train import encode_unsupervised

    encoded = [
        encode_unsupervised(tokenizer, record, max_length, use_priors=use_priors)
        for record in records
    ]
    return {
        "prompt_tokens": sum(row["n_prompt"] for row in encoded),
        "completion_tokens": sum(row["n_completion"] for row in encoded),
        "examples_truncated": sum(bool(row["truncated"]) for row in encoded),
    }


def patch_artifacts(checkpoint_dir: Path, additions: dict) -> None:
    run_path = checkpoint_dir / "run.json"
    metrics_path = checkpoint_dir / "metrics.json"
    if not run_path.exists() or not metrics_path.exists():
        raise FileNotFoundError("run.json and metrics.json must exist before metadata backfill")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not metrics.get("finished"):
        raise RuntimeError("refusing to alter metadata while training is unfinished")
    run.update(additions)
    metrics["run"] = run
    write_json_atomic(run_path, run)
    write_json_atomic(metrics_path, metrics)
    training_path = checkpoint_dir / "best" / "training_config.json"
    if training_path.exists():
        training = json.loads(training_path.read_text(encoding="utf-8"))
        training.update(additions)
        write_json_atomic(training_path, training)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("train_file", type=Path)
    parser.add_argument("dev_file", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from vectorial_eval.methods.lora.train import BASE_MODEL, load_unsupervised_records

    run = json.loads((args.checkpoint_dir / "run.json").read_text(encoding="utf-8"))
    if run.get("objective") != "unsupervised":
        raise ValueError("this backfill applies only to unsupervised Study 4 runs")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    train = load_unsupervised_records(args.train_file, "train")
    dev = load_unsupervised_records(args.dev_file, "dev")
    use_priors = run.get("variant") == "platform_prior_lm"
    train_tokens = token_metadata(tokenizer, train, int(run["max_length"]), use_priors)
    dev_tokens = token_metadata(tokenizer, dev, int(run["max_length"]), use_priors)
    additions = {
        "input_files": {
            "train": {
                "path": str(args.train_file),
                "sha256": sha256_file(args.train_file),
            },
            "dev": {
                "path": str(args.dev_file),
                "sha256": sha256_file(args.dev_file),
            },
        },
        "train_prompt_tokens": train_tokens["prompt_tokens"],
        "train_completion_tokens": train_tokens["completion_tokens"],
        "train_examples_truncated": train_tokens["examples_truncated"],
        "dev_prompt_tokens": dev_tokens["prompt_tokens"],
        "dev_completion_tokens": dev_tokens["completion_tokens"],
        "dev_examples_truncated": dev_tokens["examples_truncated"],
        "metadata_reconstruction": (
            "deterministically recomputed with the frozen tokenizer and input files "
            "after training; model weights and metric history were not changed"
        ),
    }
    if args.manifest:
        additions["input_manifest"] = {
            "path": str(args.manifest),
            "sha256": sha256_file(args.manifest),
        }
    patch_artifacts(args.checkpoint_dir, additions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
