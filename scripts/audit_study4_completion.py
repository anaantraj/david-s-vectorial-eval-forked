#!/usr/bin/env python
"""Fail unless the complete two-direction Study 4 artifact set is present."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

try:
    from .audit_study4_priors import audit_pair, audit_scored_overlap
    from .sanitize_study4_prior_context import completion_projection_sha256
except ImportError:  # direct ``python scripts/...`` execution
    from audit_study4_priors import audit_pair, audit_scored_overlap
    from sanitize_study4_prior_context import completion_projection_sha256

FORWARD_SYSTEMS = {
    "identity",
    "target_sample",
    "shuffle_control",
    "llama_rewrite_s4_base_none",
    "llm_rewrite_s4_claude_reference",
    "lora_s4_raw_1k",
    "lora_s4_prior_1k",
    "lora_s4_raw_10k",
    "lora_s4_prior_10k",
    "lora_s4_target_lm",
    "lora_s4_target_lm_paired",
    "lora_s4_target_lm_aspect",
    "soft_prompt_s4_soft_target_lm",
    "soft_prompt_s4_soft_target_lm_paired",
    "soft_prompt_s4_soft_target_lm_aspect",
    "local_plan_then_transfer_s4_plan_prior_10k",
    "steering_s4_steer_base_target_lm",
    "steering_s4_steer_prior_target_lm",
    "steering_s4_steer_base_target_lm_paired",
    "steering_s4_steer_prior_target_lm_paired",
    "steering_s4_steer_base_target_lm_aspect",
    "steering_s4_steer_prior_target_lm_aspect",
    "steering_s4_lora_steer_interaction",
    "lora_s4_platform_then_audience",
    "lora_s4_platform_then_audience_audience_shuffle",
    "lora_s4_raw_10k_audience_shuffle",
    "lora_s4_prior_10k_audience_shuffle",
    "local_plan_then_transfer_s4_plan_prior_10k_audience_shuffle",
    *(f"lora_s4_{variant}_audience_shuffle" for variant in (
        "target_lm", "target_lm_paired", "target_lm_aspect"
    )),
    *(f"soft_prompt_s4_soft_{variant}_audience_shuffle" for variant in (
        "target_lm", "target_lm_paired", "target_lm_aspect"
    )),
    *(f"steering_s4_steer_prior_{variant}_audience_shuffle" for variant in (
        "target_lm", "target_lm_paired", "target_lm_aspect"
    )),
    *(f"lora_s4_base_lora_{variant}" for variant in (
        "target_lm", "target_lm_paired", "target_lm_aspect"
    )),
    *(f"lora_s4_base_lora_{variant}_audience_shuffle" for variant in (
        "target_lm", "target_lm_paired", "target_lm_aspect"
    )),
    *(f"soft_prompt_s4_base_soft_{variant}" for variant in (
        "target_lm", "target_lm_paired", "target_lm_aspect"
    )),
    *(f"soft_prompt_s4_base_soft_{variant}_audience_shuffle" for variant in (
        "target_lm", "target_lm_paired", "target_lm_aspect"
    )),
}

REVERSE_SYSTEMS = {
    name.replace("raw_10k", "raw_full")
    .replace("prior_10k", "prior_full")
    .replace("plan_prior_10k", "plan_prior_full")
    for name in FORWARD_SYSTEMS
    if name != "steering_s4_lora_steer_interaction"
}

FORWARD_COMPARISONS = {
    "llm_rewrite_s4_claude_reference:llama_rewrite_s4_base_none",
    "llama_rewrite_s4_base_none:lora_s4_prior_10k",
    "lora_s4_raw_1k:lora_s4_prior_1k",
    "lora_s4_raw_10k:lora_s4_prior_10k",
    "lora_s4_prior_1k:lora_s4_prior_10k",
    "lora_s4_raw_10k:lora_s4_platform_then_audience",
    "lora_s4_prior_10k:lora_s4_platform_then_audience",
    "lora_s4_prior_10k:local_plan_then_transfer_s4_plan_prior_10k",
    "steering_s4_steer_base_target_lm:lora_s4_target_lm",
    "lora_s4_target_lm:steering_s4_lora_steer_interaction",
    *(f"lora_s4_base_lora_{variant}:lora_s4_{variant}" for variant in (
        "target_lm", "target_lm_paired", "target_lm_aspect"
    )),
    *(f"soft_prompt_s4_base_soft_{variant}:soft_prompt_s4_soft_{variant}" for variant in (
        "target_lm", "target_lm_paired", "target_lm_aspect"
    )),
    *(f"steering_s4_steer_base_{variant}:steering_s4_steer_prior_{variant}" for variant in (
        "target_lm", "target_lm_paired", "target_lm_aspect"
    )),
    *(f"lora_s4_{variant}_audience_shuffle:lora_s4_{variant}" for variant in (
        "target_lm", "target_lm_paired", "target_lm_aspect"
    )),
}

REVERSE_COMPARISONS = {
    key.replace("raw_10k", "raw_full")
    .replace("prior_10k", "prior_full")
    .replace("plan_prior_10k", "plan_prior_full")
    for key in FORWARD_COMPARISONS
    if "lora_steer_interaction" not in key
}

TRAINING_RUNS = {
    "s4-reddit-rich-raw-10k",
    "s4-reddit-rich-prior-10k",
    "s4-reddit-rich-raw-1k",
    "s4-reddit-rich-prior-1k",
    *(f"s4-prior10k-lora-{v}" for v in ("target_lm", "target_lm_paired", "target_lm_aspect")),
    "s4-linkedin-rich-platform_lm-10000",
    "s4-linkedin-rich-platform_prior_lm-10000",
    "s4-linkedin-rich-platform_lm-1000",
    "s4-linkedin-rich-platform_prior_lm-1000",
    *(f"s4-reverse-prior-lora-{v}" for v in ("target_lm", "target_lm_paired", "target_lm_aspect")),
    "s4-reddit-coarse-platform-half",
    "s4-reddit-platform-then-audience",
    "s4-linkedin-coarse-platform-half",
    "s4-linkedin-platform-then-audience",
}

SOFT_RUNS = {
    *(f"s4-prior10k-soft-prompt-{v}" for v in (
        "target_lm", "target_lm_paired", "target_lm_aspect"
    )),
    *(f"s4-reverse-prior-soft-prompt-{v}" for v in (
        "target_lm", "target_lm_paired", "target_lm_aspect"
    )),
}


def read(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(f"missing required artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_direction(
    run_dir: Path, required: set[str], required_comparisons: set[str]
) -> dict:
    output_audit = read(run_dir / "output_audit.heldout.json")
    systems = set(output_audit.get("systems", {}))
    missing = sorted(required - systems)
    if missing:
        raise RuntimeError(f"{run_dir}: missing output systems: {missing}")

    reports = sorted(run_dir.glob("report.heldout*.json"))
    if len(reports) != 6:
        raise RuntimeError(f"{run_dir}: expected 6 reports, found {len(reports)}")
    spaces = set()
    for path in reports:
        report = read(path)
        missing_table = sorted(required - set(report.get("table", {})))
        if missing_table:
            raise RuntimeError(
                f"{path}: required systems missing from scored table: {missing_table}"
            )
        space = report.get("embedding_space") or {}
        slug = space.get("slug")
        if not slug:
            raise RuntimeError(f"{path}: missing embedding-space slug")
        spaces.add(slug)
        if report.get("config", {}).get("eval", {}).get("bootstrap_unit") != "audience-cell":
            raise RuntimeError(f"{path}: bootstrap unit is not audience-cell")
        common_ids = report.get("common_cell_ids") or {}
        for family in ("semantic", "trm"):
            if not common_ids.get(family):
                raise RuntimeError(f"{path}: missing exact common cells for {family}")
        for system, metrics in report.get("table", {}).items():
            if system in {"identity", "shuffle_control", "target_sample"}:
                continue
            for family, score in (
                ("semantic", "semantic.source_similarity"),
                ("semantic", "semantic.content_word_retention"),
                ("trm", "trm.trm"),
            ):
                entry = metrics.get(score)
                if entry and entry.get("n_cells") != len(common_ids[family]):
                    raise RuntimeError(
                        f"{path}: {system} {family} is not on the common cell base"
                    )
        notes = report.get("notes", {})
        trm_notes = [
            value
            for key, value in notes.items()
            if key.endswith(".trm") and isinstance(value, dict)
        ]
        if not trm_notes or any(
            not note.get("pairwise_distance") or not note.get("distance_space")
            for note in trm_notes
        ):
            raise RuntimeError(f"{path}: incomplete TRM distance provenance")
    if len(spaces) != 6:
        raise RuntimeError(f"{run_dir}: reports contain only {len(spaces)} distinct spaces")

    sensitivity = read(run_dir / "frontier_sensitivity.heldout.json")
    if sensitivity.get("n_spaces") != 6 or sensitivity.get("values_pooled_across_spaces") is not False:
        raise RuntimeError(f"{run_dir}: invalid embedding-space sensitivity summary")
    missing_comparisons = sorted(
        required_comparisons - set(sensitivity.get("comparisons", {}))
    )
    if missing_comparisons:
        raise RuntimeError(
            f"{run_dir}: missing preregistered comparisons: {missing_comparisons}"
        )
    plan_keys = [
        key for key in required_comparisons if "local_plan_then_transfer" in key
    ]
    for key in plan_keys:
        special = sensitivity["comparisons"][key].get("extract_then_transfer")
        if not special or special.get("n_spaces") != 6:
            raise RuntimeError(f"{run_dir}: incomplete plan-retention sensitivity for {key}")
    return {
        "n_systems": len(systems),
        "required_systems": len(required),
        "n_embedding_spaces": len(spaces),
        "spaces": sorted(spaces),
        "n_tasks": output_audit.get("n_tasks"),
    }


def audit_training(checkpoint_dir: Path) -> dict:
    summaries = {}
    for name in sorted(TRAINING_RUNS):
        root = checkpoint_dir / name
        metrics = read(root / "metrics.json")
        if not metrics.get("finished"):
            raise RuntimeError(f"{root}: training is not marked finished")
        run = read(root / "run.json")
        for input_file in (run.get("input_files") or {}).values():
            path = Path(input_file["path"])
            if not path.exists() or sha256(path) != input_file.get("sha256"):
                raise RuntimeError(f"{root}: training input hash no longer matches {path}")
        if run.get("max_length") != 768 or run.get("per_device_batch_size") != 1:
            raise RuntimeError(f"{root}: violates fixed batch/length configuration")
        if run.get("quantisation") != "nf4-4bit-double":
            raise RuntimeError(f"{root}: does not record 4-bit NF4")
        if (
            run.get("variant") == "platform_prior_lm"
            and run.get("prior_prompt_schema") != "study4-rich-priors-v3"
        ):
            raise RuntimeError(f"{root}: does not use the rich-prior prompt schema")
        summaries[name] = {
            "variant": run.get("variant"),
            "n_train": run.get("n_train"),
            "n_dev": run.get("n_dev"),
            "best_nll": metrics.get("best_nll"),
            "train_completion_tokens": run.get("train_completion_tokens"),
            "dev_completion_tokens": run.get("dev_completion_tokens"),
            "train_prompt_tokens": run.get("train_prompt_tokens"),
            "dev_prompt_tokens": run.get("dev_prompt_tokens"),
        }
    matched_pairs = (
        ("s4-reddit-rich-raw-1k", "s4-reddit-rich-prior-1k"),
        ("s4-reddit-rich-raw-10k", "s4-reddit-rich-prior-10k"),
        (
            "s4-linkedin-rich-platform_lm-1000",
            "s4-linkedin-rich-platform_prior_lm-1000",
        ),
        (
            "s4-linkedin-rich-platform_lm-10000",
            "s4-linkedin-rich-platform_prior_lm-10000",
        ),
    )
    for raw_name, prior_name in matched_pairs:
        raw, prior = summaries[raw_name], summaries[prior_name]
        for field in (
            "n_train",
            "n_dev",
            "train_completion_tokens",
            "dev_completion_tokens",
        ):
            if raw[field] != prior[field]:
                raise RuntimeError(
                    f"{raw_name} and {prior_name} differ on matched exposure: {field}"
                )
        if raw["train_prompt_tokens"] != 0 or raw["dev_prompt_tokens"] != 0:
            raise RuntimeError(f"{raw_name}: raw arm unexpectedly contains prompt tokens")
        if prior["train_prompt_tokens"] <= 0 or prior["dev_prompt_tokens"] <= 0:
            raise RuntimeError(f"{prior_name}: structured arm has no prior prompt tokens")
    for platform, full, first, second in (
        (
            "reddit",
            "s4-reddit-rich-prior-10k",
            "s4-reddit-coarse-platform-half",
            "s4-reddit-platform-then-audience",
        ),
        (
            "linkedin",
            "s4-linkedin-rich-platform_prior_lm-10000",
            "s4-linkedin-coarse-platform-half",
            "s4-linkedin-platform-then-audience",
        ),
    ):
        full_tokens = summaries[full]["train_completion_tokens"]
        partition_tokens = (
            summaries[first]["train_completion_tokens"]
            + summaries[second]["train_completion_tokens"]
        )
        if partition_tokens != full_tokens:
            raise RuntimeError(
                f"{platform}: coarse-to-fine completion tokens {partition_tokens} "
                f"!= primary {full_tokens}"
            )
        if summaries[second]["variant"] != "platform_prior_lm":
            raise RuntimeError(f"{platform}: audience stage has the wrong variant")
    soft_summaries = {}
    for name in sorted(SOFT_RUNS):
        root = checkpoint_dir / name
        summary = read(root / "summary.json")
        # The soft-prompt trainer writes summary.json only after the training
        # loop has completed, and unlike LoRA's metrics.json it emits no
        # redundant `finished` field and no `best_dev_nll`. Requiring either
        # rejects every correctly finished run. Check the artifacts the trainer
        # does write, which is the same criterion `soft_finished` applies in
        # scripts/cluster/continue_study4_generation.sh: a selected epoch, a
        # finite held-out loss, a non-empty history, and a selected adapter.
        complete = (
            isinstance(summary.get("best_epoch"), int)
            and isinstance(summary.get("best_dev_loss"), (int, float))
            and math.isfinite(summary.get("best_dev_loss", math.nan))
            and bool(summary.get("history"))
            and (root / "best" / "adapter_config.json").exists()
        )
        if not complete:
            raise RuntimeError(f"{root}: soft-prompt training is not finished")
        soft_summaries[name] = {
            "variant": (summary.get("config") or {}).get("variant"),
            "best_epoch": summary.get("best_epoch"),
            "best_dev_loss": summary.get("best_dev_loss"),
            "stopped_early": summary.get("stopped_early"),
        }
    return {
        "n_lora_runs": len(summaries),
        "lora_runs": summaries,
        "n_soft_prompt_runs": len(soft_summaries),
        "soft_prompt_runs": soft_summaries,
    }


def audit_steering(root: Path) -> dict:
    groups = {
        "forward": root / "study4_steering",
        "reverse": root / "study4_reverse_steering",
    }
    result = {}
    for direction, directory in groups.items():
        variants = ["target_lm", "target_lm_paired", "target_lm_aspect"]
        if direction == "forward":
            variants.append("lora_interaction")
        entries = {}
        for variant in variants:
            artifact = directory / f"steering.{variant}.pt"
            chosen = directory / f"chosen.steering.{variant}.json"
            if not artifact.exists() or not chosen.exists():
                raise RuntimeError(f"missing steering artifact or selection for {variant}")
            selection = read(chosen)
            entries[variant] = {
                "layer": selection.get("layer"),
                "alpha": selection.get("alpha"),
                "scope": selection.get("scope"),
                "audience_alpha": selection.get("audience_alpha"),
            }
        result[direction] = entries
    return result


def audit_partitions(data_dir: Path = Path("data/study4_priors/coarse_to_fine")) -> dict:
    result = {}
    for platform in ("reddit", "linkedin"):
        manifest = read(data_dir / f"{platform}.coarse_to_fine.manifest.json")
        if manifest.get("training_sets_are_disjoint") is not True:
            raise RuntimeError(f"{platform}: coarse-to-fine stages are not disjoint")
        if manifest.get("training_union_equals_parent") is not True:
            raise RuntimeError(f"{platform}: coarse-to-fine stages do not cover the parent")
        if (
            manifest.get("n_platform_train", 0)
            + manifest.get("n_audience_train", 0)
            != manifest.get("n_parent_train")
        ):
            raise RuntimeError(f"{platform}: partition counts do not add to parent")
        for output in manifest.get("outputs", {}).values():
            path = Path(output["path"])
            if not path.exists() or sha256(path) != output["sha256"]:
                raise RuntimeError(f"{platform}: partition output hash mismatch for {path}")
        result[platform] = {
            key: manifest[key]
            for key in (
                "n_parent_train",
                "n_platform_train",
                "n_audience_train",
                "n_dev",
            )
        }
    return result


def audit_priors(data_dir: Path = Path("data/study4_priors")) -> dict:
    result = {
        platform: audit_pair(
            data_dir / f"{platform}_10000_per_audience.rich_safe.train.jsonl",
            data_dir / f"{platform}_10000_per_audience.rich_safe.dev.jsonl",
        )
        for platform in ("reddit", "linkedin")
    }
    result["scored_overlap"] = audit_scored_overlap(
        [
            data_dir / f"{platform}_10000_per_audience.rich_safe.{split}.jsonl"
            for platform in ("reddit", "linkedin")
            for split in ("train", "dev")
        ],
        [Path("data/study3/posts.val.jsonl"), Path("data/study3/posts.test.jsonl")],
    )
    sanitized_paths = sorted(data_dir.rglob("*.rich_safe.*.jsonl"))
    if len(sanitized_paths) != 12:
        raise RuntimeError(
            "expected 12 sanitized prior corpora "
            f"(full, 1k, and coarse-to-fine); found {len(sanitized_paths)}"
        )
    sanitized = {}
    for path in sanitized_paths:
        manifest_path = path.with_suffix(path.suffix + ".sanitization.json")
        manifest = read(manifest_path)
        if manifest.get("output_sha256") != sha256(path):
            raise RuntimeError(f"sanitized prior hash mismatch: {path}")
        source = Path(manifest["source"])
        if manifest.get("source_sha256") != sha256(source):
            raise RuntimeError(f"sanitized prior source hash mismatch: {source}")
        source_projection = completion_projection_sha256(source)
        output_projection = completion_projection_sha256(path)
        if source_projection != output_projection:
            raise RuntimeError(f"sanitization changed completion exposure: {path}")
        if (
            manifest.get("source_completion_projection_sha256")
            != source_projection
            or manifest.get("output_completion_projection_sha256")
            != output_projection
        ):
            raise RuntimeError(f"sanitization projection manifest mismatch: {path}")
        sanitized[str(path)] = manifest.get("removed_exact_metadata_strings", {})
    result["sanitized_inputs"] = sanitized
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward", type=Path, default=Path("runs/study4_forward"))
    parser.add_argument("--reverse", type=Path, default=Path("runs/study4_reverse"))
    parser.add_argument("--checkpoints", type=Path, default=Path("runs/checkpoints"))
    parser.add_argument("--out", type=Path, default=Path("runs/study4_completion_audit.json"))
    args = parser.parse_args()
    result = {
        "status": "complete",
        "forward": audit_direction(
            args.forward, FORWARD_SYSTEMS, FORWARD_COMPARISONS
        ),
        "reverse": audit_direction(
            args.reverse, REVERSE_SYSTEMS, REVERSE_COMPARISONS
        ),
        "training": audit_training(args.checkpoints),
        "steering": audit_steering(args.checkpoints.parent),
        "coarse_to_fine_partitions": audit_partitions(),
        "prior_corpora": audit_priors(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
