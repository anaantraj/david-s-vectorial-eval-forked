"""Evaluation driver: load a run's outputs, score them, aggregate, report."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .config import (
    DEFAULT_EMBEDDING_BACKEND,
    DEFAULT_EMBEDDING_DIM,
    HarnessConfig,
)
from .data.schema import CellRecord, PostRecord, TransferOutput, TransferTask
from .features.embeddings import fit_embedding_space, space_slug
from .metrics import base as metric_base
from .metrics.base import (
    EvalContext,
    MetricResult,
    audience_cell_bootstrap_ci,
    bootstrap_ci,
)

log = logging.getLogger(__name__)


def read_jsonl(path: Path, model_cls):
    with Path(path).open(encoding="utf-8") as fh:
        return [model_cls.model_validate_json(line) for line in fh if line.strip()]


# --- report naming ----------------------------------------------------------
#
# Every embedding-derived score is defined relative to the space it was computed
# in, and TRM is not comparable at all across spaces. A run scored under a second
# embedding model must therefore land in a second file rather than replacing the
# first. The default space keeps the unqualified name so that existing tooling
# and the report site are unaffected.


def default_report_tag(cfg: HarnessConfig) -> str | None:
    """The report suffix implied by the configured embedding space.

    ``None`` for the default space, otherwise its slug.
    """
    is_default = (
        cfg.eval.embedding_backend == DEFAULT_EMBEDDING_BACKEND
        and cfg.eval.embedding_dim == DEFAULT_EMBEDDING_DIM
    )
    if is_default:
        return None
    name = (
        cfg.eval.embedding_model
        if cfg.eval.embedding_backend != DEFAULT_EMBEDDING_BACKEND
        else cfg.eval.embedding_backend
    )
    return space_slug(name, cfg.eval.embedding_dim)


def report_path(run_dir: Path, split: str, tag: str | None = None) -> Path:
    suffix = f".{tag}" if tag else ""
    return Path(run_dir) / f"report.{split}{suffix}.json"


def recorded_space_slug(report: dict) -> str | None:
    """The embedding space a saved report was computed in, if it states one."""
    space = report.get("embedding_space")
    if isinstance(space, dict) and space.get("slug"):
        return str(space["slug"])
    # Reports written before the space was recorded explicitly still carry the
    # configuration that produced them.
    eval_cfg = report.get("config", {}).get("eval")
    if not isinstance(eval_cfg, dict) or "embedding_backend" not in eval_cfg:
        return None
    backend = eval_cfg["embedding_backend"]
    name = eval_cfg.get("embedding_model") if backend != DEFAULT_EMBEDDING_BACKEND else backend
    return space_slug(name or backend, eval_cfg.get("embedding_dim", DEFAULT_EMBEDDING_DIM))


def check_space_matches(path: Path, slug: str) -> None:
    """Refuse to overwrite a report computed in a different embedding space.

    The slug-qualified filename makes this near-unreachable in normal use. It is
    kept as a backstop for reports written by an earlier version of the harness,
    because the failure it guards against — a TRM column silently replaced by
    values from an incomparable space — produces no error of its own.
    """
    if not path.exists():
        return
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    existing = recorded_space_slug(previous)
    if existing and existing != slug:
        raise RuntimeError(
            f"{path.name} was computed in embedding space {existing!r}, not "
            f"{slug!r}. Values from two spaces are not comparable, so writing "
            "here would silently replace one with the other. Pass "
            "--report-tag to name this run's report separately."
        )


def build_context(
    data_dir: Path,
    split: str,
    cfg: HarnessConfig,
    tasks: list[TransferTask],
) -> EvalContext:
    """Assemble the shared evaluation context.

    The embedding space is fit once here over every real post plus every
    generated post seen so far, so all transfer functions are scored in one
    space. Fitting per-function would make their numbers incomparable.
    """
    cells = {c.cell_id: c for c in read_jsonl(data_dir / "cells.jsonl", CellRecord)}
    posts: list[PostRecord] = []
    for s in ("train", "val", "test"):
        p = data_dir / f"posts.{s}.jsonl"
        if p.exists():
            posts.extend(read_jsonl(p, PostRecord))

    pools: dict[str, dict[str, dict[str, list[str]]]] = {
        cid: {
            p: {"train": [], "val": [], "test": [], "heldout": []}
            for p in ("linkedin", "reddit")
        }
        for cid in cells
    }
    texts_by_id = {}
    for post in posts:
        texts_by_id[post.post_id] = post.text
        if post.cell_id in pools:
            pools[post.cell_id][post.platform][post.split].append(post.text)
    # The merged held-out pool is the union of val and test.
    for cid in pools:
        for platform in pools[cid]:
            pools[cid][platform]["heldout"] = (
                pools[cid][platform]["val"] + pools[cid][platform]["test"]
            )

    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    direction = manifest.get("direction", {"source": "linkedin", "target": "reddit"})

    log.info(
        "Fitting embedding space: backend=%s model=%s dim=%d",
        cfg.eval.embedding_backend,
        cfg.eval.embedding_model if cfg.eval.embedding_backend != "tfidf-svd" else "-",
        cfg.eval.embedding_dim,
    )
    space = fit_embedding_space(
        [p.text for p in posts],
        backend=cfg.eval.embedding_backend,
        dim=cfg.eval.embedding_dim,
        seed=cfg.eval.seed,
        model_name=cfg.eval.embedding_model,
        prompt_name=cfg.eval.embedding_prompt,
        batch_size=cfg.eval.embedding_batch_size,
    )

    ctx = EvalContext(
        cells=cells,
        pools=pools,
        texts_by_id=texts_by_id,
        split=split,
        source_platform=direction["source"],
        target_platform=direction["target"],
        embedding_space=space,
        seed=cfg.eval.seed,
    )
    # Metrics that compare a generation to *its* source need this lookup.
    ctx.extras["_source_text_by_task"] = {t.task_id: t.source_text for t in tasks}
    ctx.extras["_task_by_id"] = {t.task_id: t for t in tasks}
    ctx.extras["_data_dir"] = str(data_dir)
    return ctx


def _make_metrics(cfg: HarnessConfig):
    metrics = []
    for name in cfg.eval.metrics:
        if name == "judge":
            metrics.append(
                metric_base.build_metric(
                    "judge",
                    llm=cfg.judge,
                    samples_per_cell=cfg.eval.judge_samples_per_cell,
                    panel=cfg.eval.judge_panel,
                )
            )
        else:
            metrics.append(metric_base.build_metric(name))
    return metrics


REFERENCE_COLUMNS = {"identity", "shuffle_control", "target_sample"}


def _is_reference(name: str) -> bool:
    return name in REFERENCE_COLUMNS


def _common_cells(results: list[MetricResult]) -> dict[str, set[str]]:
    """Per metric, the cells scored for *every* transfer function.

    Without this the table is not a like-for-like comparison: `target_sample`
    produces nothing for held-out cells (they have no train exemplars by
    construction), so it would otherwise be averaged over an easier subset than
    `identity`. Restricting to the intersection costs a few cells and buys a
    column-comparable table.
    """
    by_metric: dict[str, list[set[str]]] = {}
    for res in results:
        if not _is_reference(res.transfer_fn):
            by_metric.setdefault(res.metric, []).append(set(res.cell_scores))
    return {
        metric: set.intersection(*sets) if sets else set()
        for metric, sets in by_metric.items()
    }


def aggregate(results: list[MetricResult], cfg: HarnessConfig, ctx: EvalContext) -> dict:
    """Collapse cell-level scores into a comparison table with CIs.

    Aggregation is an unweighted mean over cells with a bootstrap CI, restricted
    to cells every transfer function was scored on. Dense and held-out subsets
    are reported separately.

    Weighting by cell size was rejected: the largest cell holds ~15x the posts
    of the median, so a size-weighted mean would mostly report performance on
    one topic.
    """
    shared = _common_cells(results)
    table: dict[str, dict[str, dict]] = {}
    for res in results:
        fn = res.transfer_fn
        table.setdefault(fn, {})
        keep = (
            set(res.cell_scores)
            if _is_reference(res.transfer_fn)
            else shared.get(res.metric, set())
        )
        score_names = {k for cs in res.cell_scores.values() for k in cs}
        for score in sorted(score_names):
            if score.startswith("n_") or "__" in score:
                continue  # counts and per-feature detail stay out of the headline
            all_vals, dense_vals, heldout_vals = [], [], []
            values_by_audience: dict[str, list[float]] = {}
            for cell_id, scores in res.cell_scores.items():
                if cell_id not in keep:
                    continue
                v = scores.get(score)
                if v is None or not np.isfinite(v):
                    continue
                all_vals.append(v)
                cell = ctx.cells.get(cell_id)
                if cell:
                    values_by_audience.setdefault(cell.room, []).append(v)
                if cell and cell.tier == "dense":
                    dense_vals.append(v)
                if cell and cell.heldout_cell:
                    heldout_vals.append(v)

            cell_mean, cell_lo, cell_hi = bootstrap_ci(
                all_vals, cfg.eval.n_bootstrap, cfg.eval.seed
            )
            if cfg.eval.bootstrap_unit == "audience-cell":
                mean, lo, hi = audience_cell_bootstrap_ci(
                    values_by_audience, cfg.eval.n_bootstrap, cfg.eval.seed
                )
            elif cfg.eval.bootstrap_unit == "cell":
                mean, lo, hi = cell_mean, cell_lo, cell_hi
            else:
                raise ValueError(
                    f"unknown bootstrap unit {cfg.eval.bootstrap_unit!r}"
                )
            entry = {
                "mean": mean,
                "ci_low": lo,
                "ci_high": hi,
                "n_cells": len(all_vals),
                "n_dense_cells": len(dense_vals),
                "n_heldout_cells": len(heldout_vals),
                "n_audiences": len(values_by_audience),
                "bootstrap_unit": cfg.eval.bootstrap_unit,
                "cell_bootstrap_mean": cell_mean,
                "cell_bootstrap_ci_low": cell_lo,
                "cell_bootstrap_ci_high": cell_hi,
                "direction": next(
                    (m.directions.get(score, 0) for m in [res] if hasattr(m, "directions")), 0
                ),
            }
            if dense_vals:
                entry["dense_mean"] = float(np.mean(dense_vals))
            if heldout_vals:
                entry["heldout_mean"] = float(np.mean(heldout_vals))
            table[fn][f"{res.metric}.{score}"] = entry

        if res.overall:
            for k, v in res.overall.items():
                table[fn][f"{res.metric}.overall.{k}"] = {"mean": float(v)}
    return table, {m: sorted(c) for m, c in shared.items()}


def run(
    data_dir: Path,
    run_dir: Path,
    split: str,
    cfg: HarnessConfig,
    transfer_fns: list[str] | None = None,
    report_tag: str | None = None,
) -> dict:
    """Score every transfer function that has outputs in `run_dir`.

    `report_tag` names the report file. When omitted it is derived from the
    embedding space, so a run scored in a second space is written alongside the
    first rather than over it.
    """
    data_dir, run_dir = Path(data_dir), Path(run_dir)
    tasks = read_jsonl(data_dir / f"tasks.{split}.jsonl", TransferTask)
    ctx = build_context(data_dir, split, cfg, tasks)
    ctx.extras["_run_dir"] = str(run_dir)

    out_files = sorted(run_dir.glob(f"outputs.*.{split}.jsonl"))
    if transfer_fns:
        out_files = [f for f in out_files if f.name.split(".")[1] in transfer_fns]
    if not out_files:
        raise FileNotFoundError(
            f"No outputs.*.{split}.jsonl in {run_dir}. Run `transfer` first."
        )

    metrics = _make_metrics(cfg)
    results: list[MetricResult] = []
    directions: dict[str, int] = {}
    for m in metrics:
        directions.update({f"{m.name}.{k}": v for k, v in m.directions.items()})

    for path in out_files:
        fn_name = path.name.split(".")[1]
        outputs = read_jsonl(path, TransferOutput)
        log.info("Scoring %s (%d outputs)", fn_name, len(outputs))
        for metric in metrics:
            try:
                results.append(metric.score(fn_name, outputs, ctx))
            except Exception as exc:  # noqa: BLE001
                log.error("metric %s failed on %s: %s", metric.name, fn_name, exc)

    table, common_cells = aggregate(results, cfg, ctx)
    for scores in table.values():
        for key, entry in scores.items():
            base = key.rsplit(".overall.", 1)[0] if ".overall." in key else key
            entry["direction"] = directions.get(base, 0)

    report = {
        "split": split,
        "config": cfg.to_dict(),
        #: The space every embedding-derived score in this report was computed
        #: in. TRM in particular is a meta-metric over cosine distance here, so
        #: no value below is comparable with a value from another space.
        "embedding_space": ctx.embedding_space.describe(),
        "direction": {"source": ctx.source_platform, "target": ctx.target_platform},
        "n_cells_scored": len(ctx.cells),
        #: Cells common to every transfer function, per metric. The headline
        #: table is restricted to these so columns are comparable.
        "common_cells": {m: len(c) for m, c in common_cells.items()},
        "common_cell_ids": common_cells,
        "table": table,
        "notes": {
            f"{r.transfer_fn}.{r.metric}": r.notes for r in results if r.notes
        },
        "per_cell": {
            f"{r.transfer_fn}.{r.metric}": r.cell_scores for r in results if r.cell_scores
        },
    }
    out_path = report_path(run_dir, split, report_tag or default_report_tag(cfg))
    check_space_matches(out_path, ctx.embedding_space.slug)
    out_path.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    log.info("Wrote %s", out_path)
    return report


def format_table(report: dict, keys: list[str] | None = None) -> str:
    """Human-readable comparison table for the terminal."""
    table = report["table"]
    fns = sorted(table)
    if keys is None:
        preferred = [
            "classifier.calibration_gap",
            "classifier.target_rate",
            "semantic.source_similarity",
            "semantic.content_word_retention",
            "structural.mean_feature_jsd",
            "structural.feature_coverage",
            "trm.trm",
            "trm.trm_style_residual",
            "trm.trm_pvalue",
            "trm.rank_i2",
            "trm.frechet",
            "distributional.mmd2",
            "distributional.topic_jsd",
            "judge.detection_rate",
            "degeneracy.copy_rate",
            "degeneracy.distinct_2",
            "degeneracy.pool_self_similarity",
        ]
        present = {k for fn in fns for k in table[fn]}
        keys = [k for k in preferred if k in present]

    arrow = {1: "↑", -1: "↓", 0: "·"}
    width = max((len(k) for k in keys), default=10) + 4
    header = "metric".ljust(width) + "".join(f"{fn:>22}" for fn in fns)
    lines = [header, "-" * len(header)]
    for key in keys:
        d = next((table[fn][key].get("direction", 0) for fn in fns if key in table[fn]), 0)
        row = f"{key} {arrow[d]}".ljust(width)
        for fn in fns:
            entry = table[fn].get(key)
            row += f"{entry['mean']:>22.3f}" if entry and np.isfinite(entry["mean"]) else f"{'—':>22}"
        lines.append(row)
    lines.append("")
    lines.append("↑ higher is better   ↓ lower is better   · diagnostic, not a score")
    # TRM and the other embedding-derived rows are only comparable within one
    # space, so the space is printed with them rather than left to be looked up.
    space = report.get("embedding_space")
    if isinstance(space, dict):
        model = space.get("model") or space.get("backend")
        lines.append(
            f"embedding space: {model}, dim {space.get('dim')} "
            "— embedding-derived rows are not comparable across spaces"
        )
    return "\n".join(lines)
