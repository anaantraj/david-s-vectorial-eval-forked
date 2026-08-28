"""Sweep the steering layer and strength on the dev split. Runs on the cluster.

What is being selected
----------------------
Steering has two free parameters, the layer at which the offset is added and its
strength alpha, and nothing in the fit chooses them. They are chosen here on the
validation split, which is the split's purpose. The test and held-out splits are
never read by this file.

The objective
-------------
The reason to expect steering to help in this project is specific. The harness's
TRM diagnostics on the existing rewriters report `rank_i2` between 0.068 and
0.129 against 0.261 for `target_sample`, with `rank_i0` between 0.68 and 0.73.
That combination says the candidates sit outside the reference cloud rather than
collapsed inside it, which is an error of location. A translation of the residual
stream acts on location directly, so the sweep is scored on location:

    centroid_cos    cosine between the centre of the generated pool and the
                    centre of the authentic dev Reddit pool. Higher is better,
                    and its ceiling is estimated by splitting the reference pool
                    in half and measuring the two halves against each other.
    energy_dist     energy distance between the two pools, which is sensitive to
                    the spread as well as the centre. Lower is better.
    source_cos      mean cosine between a generation and the source post it was
                    given, so that a gain in location bought by discarding the
                    content is visible rather than hidden.
    degeneracy      empty generations and within-output repetition, which is how
                    an alpha that has broken the model announces itself.

These are computed in an embedding space chosen here and are used only to pick
hyperparameters. They are not the study's metrics and must not be quoted beside
them; the selected configuration is then scored by the harness in the usual way.

Resuming
--------
One checkpoint per grid point, holding every generation produced so far. A kill
costs at most one grid point.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts" / "cluster"))

import numpy as np  # noqa: E402

from vectorial_eval.config import DEFAULT_DATA_DIR  # noqa: E402
from vectorial_eval.data.build_training import render_prompt  # noqa: E402
from vectorial_eval.data.schema import PostRecord, TransferTask  # noqa: E402
from vectorial_eval.methods.steering.generate import GenConfig, SteeredGenerator  # noqa: E402
from vectorial_eval.methods.steering.runtime import (  # noqa: E402
    SteeringSpec,
    load_artifact,
    parse_layers,
)

log = logging.getLogger("sweep_steering")

TARGET_PLATFORM = "reddit"
EMBED_MODEL = "google/embeddinggemma-300m"


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #


def read_jsonl(path: Path, model):
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(model.model_validate_json(line))
    return out


def selection_source(data_dir: Path) -> str:
    manifest = data_dir / "training" / "manifest.json"
    if not manifest.exists():
        return "val"
    return json.loads(manifest.read_text()).get("selection", {}).get("source", "val")


def dev_tasks(data_dir: Path, limit: int) -> list[TransferTask]:
    """A deterministic, cell-balanced subsample of the validation tasks."""
    split = selection_source(data_dir)
    tasks = read_jsonl(data_dir / f"tasks.{split}.jsonl", TransferTask)
    off = [t.task_id for t in tasks if t.split != split]
    if off:
        raise RuntimeError(f"tasks.{split}.jsonl holds {len(off)} tasks from another split")
    tasks.sort(key=lambda t: t.task_id)
    if not limit or limit >= len(tasks):
        return tasks
    by_cell: dict[str, list[TransferTask]] = {}
    for task in tasks:
        by_cell.setdefault(task.cell_id, []).append(task)
    picked: list[TransferTask] = []
    round_ = 0
    while len(picked) < limit:
        added = False
        for cell in sorted(by_cell):
            if round_ < len(by_cell[cell]) and len(picked) < limit:
                picked.append(by_cell[cell][round_])
                added = True
        if not added:
            break
        round_ += 1
    picked.sort(key=lambda t: t.task_id)
    return picked


def dev_reference_texts(data_dir: Path) -> list[str]:
    if selection_source(data_dir) == "train":
        path = data_dir / "training" / "target_lm.dev.jsonl"
        return [json.loads(line)["completion"] for line in path.read_text().splitlines() if line]
    posts = read_jsonl(data_dir / "posts.val.jsonl", PostRecord)
    return [p.text for p in posts if p.platform == TARGET_PLATFORM and p.split == "val"]


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #


class Embedder:
    """Sentence embeddings for the sweep objective, with a TF-IDF fallback."""

    def __init__(self, model_id: str = EMBED_MODEL, device: str = "cpu"):
        self.backend = "tfidf"
        self.model = None
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_id, device=device)
            self.backend = f"sentence-transformers:{model_id}"
        except Exception as exc:  # noqa: BLE001
            log.warning("falling back to TF-IDF embeddings: %s", exc)

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.model is not None:
            vecs = self.model.encode(texts, batch_size=8, show_progress_bar=False)
            vecs = np.asarray(vecs, dtype=np.float64)
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer

            vecs = TfidfVectorizer(min_df=1, sublinear_tf=True).fit_transform(texts).toarray()
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.clip(norms, 1e-9, None)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / max(na * nb, 1e-9))


def energy_distance(x: np.ndarray, y: np.ndarray) -> float:
    def mean_pairwise(a, b, same):
        d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
        if same:
            n = len(a)
            if n < 2:
                return 0.0
            return float((d.sum() - np.trace(d)) / (n * (n - 1)))
        return float(d.mean())

    return float(
        2 * mean_pairwise(x, y, False)
        - mean_pairwise(x, x, True)
        - mean_pairwise(y, y, True)
    )


def repetition(text: str) -> float:
    """Share of repeated word trigrams. 0 for clean text, near 1 for a loop."""
    words = text.split()
    if len(words) < 6:
        return 0.0
    grams = [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]
    return 1.0 - len(set(grams)) / len(grams)


def score_pool(
    emb: Embedder,
    generated: list[str],
    sources: list[str],
    references: list[str],
    ref_matrix: np.ndarray,
) -> dict:
    kept = [g for g in generated if g.strip()]
    empty_rate = 1.0 - len(kept) / max(1, len(generated))
    rep = [repetition(g) for g in generated]
    out = {
        "n": len(generated),
        "n_nonempty": len(kept),
        "empty_rate": round(empty_rate, 4),
        "repetition_mean": round(float(np.mean(rep)) if rep else 0.0, 4),
        "repetition_p90": round(float(np.percentile(rep, 90)) if rep else 0.0, 4),
        "chars_mean": round(float(np.mean([len(g) for g in generated])), 1),
        "degeneracy": round(
            empty_rate + float(np.mean([1.0 for r in rep if r > 0.5] or [0.0])) * 0, 4
        ),
    }
    # degeneracy: an output is degenerate if it is empty or mostly repeated text.
    bad = sum(1 for g, r in zip(generated, rep, strict=True) if not g.strip() or r > 0.5)
    out["degeneracy"] = round(bad / max(1, len(generated)), 4)
    if not kept:
        out.update({"centroid_cos": float("nan"), "energy_dist": float("nan"),
                    "source_cos": float("nan")})
        return out

    gen_m = emb.encode(kept)
    out["centroid_cos"] = round(_cos(gen_m.mean(0), ref_matrix.mean(0)), 4)
    out["energy_dist"] = round(energy_distance(gen_m, ref_matrix), 4)

    pairs = [(g, s) for g, s in zip(generated, sources, strict=True) if g.strip()]
    src_m = emb.encode([s for _, s in pairs])
    gen_for_src = emb.encode([g for g, _ in pairs])
    out["source_cos"] = round(
        float(np.mean([_cos(a, b) for a, b in zip(gen_for_src, src_m, strict=True)])), 4
    )
    del references
    return out


def reference_ceiling(emb: Embedder, ref_matrix: np.ndarray) -> dict:
    """Split the reference pool in half and measure the halves against each other.

    This is the value `centroid_cos` would take for a pool of authentic posts,
    and it is what makes an absolute cosine interpretable.
    """
    n = len(ref_matrix)
    if n < 6:
        return {"centroid_cos": float("nan"), "energy_dist": float("nan"), "n": n}
    idx = np.arange(n)
    a, b = ref_matrix[idx % 2 == 0], ref_matrix[idx % 2 == 1]
    del emb
    return {
        "centroid_cos": round(_cos(a.mean(0), b.mean(0)), 4),
        "energy_dist": round(energy_distance(a, b), 4),
        "n": n,
    }


def select_pareto_point(
    results: list[dict], max_degeneracy: float
) -> tuple[dict | None, dict | None, list[dict]]:
    """Choose only points non-inferior to the unsteered content/style axes."""
    baseline = next((r for r in results if r["alpha"] == 0.0), None)
    eligible: list[dict] = []
    if baseline:
        eligible = [
            r
            for r in results
            if r["alpha"] != 0.0
            and r["degeneracy"] <= max_degeneracy
            and r["centroid_cos"] == r["centroid_cos"]
            and r["source_cos"] == r["source_cos"]
            and r["centroid_cos"] >= baseline["centroid_cos"]
            and r["source_cos"] >= baseline["source_cos"]
        ]
    chosen = (
        max(
            eligible,
            key=lambda r: (
                r["centroid_cos"] - baseline["centroid_cos"],
                r["source_cos"] - baseline["source_cos"],
            ),
        )
        if eligible
        else baseline
    )
    return chosen, baseline, eligible


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--layers", default="8,12,16,20,24")
    p.add_argument("--alphas", default="0.5,1.0,2.0")
    p.add_argument(
        "--scope",
        default="global",
        choices=["global", "cell", "decomposed", "both", "all"],
    )
    p.add_argument(
        "--audience-alphas",
        default="0.5,1.0,2.0",
        help="relative audience coefficients for decomposed steering",
    )
    p.add_argument("--limit", type=int, default=16, help="dev tasks per grid point")
    p.add_argument("--max-new-tokens", type=int, default=320)
    p.add_argument("--batch-size", type=int, default=None,
                   help="default: 4 locally, 8 on NDIF, which charges wall-clock "
                        "per trace rather than per token")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--source-chars", type=int, default=0,
                   help="source character cap; 0 passes the complete post")
    p.add_argument("--load-in-4bit", dest="load_in_4bit", action="store_true", default=True)
    p.add_argument("--bf16", dest="load_in_4bit", action="store_false")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--ckpt-dir", default=os.environ.get("VECTORIAL_CKPT_DIR", ""))
    p.add_argument("--max-degeneracy", type=float, default=0.15,
                   help="a grid point above this is not eligible for selection")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    from checkpointing import resume_or_start, save_checkpoint

    root = Path(__file__).resolve().parents[4]
    artifact = load_artifact(args.artifact)
    variant = artifact["variant"]
    out = args.out or (root / "runs" / "steering" / f"sweep.{args.artifact.stem}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.ckpt_dir or str(root / "runs" / "checkpoints" / f"sweep_{args.artifact.stem}")

    tasks = dev_tasks(args.data_dir, args.limit)
    refs = dev_reference_texts(args.data_dir)
    dev_split = selection_source(args.data_dir)
    log.info("dev tasks=%d cells=%d reference posts=%d",
             len(tasks), len({t.cell_id for t in tasks}), len(refs))

    items = [
        {
            "task_id": t.task_id,
            "cell_id": t.cell_id,
            "audience_id": t.room,
            "source": t.source_text,
            "prompt": render_prompt(
                room=t.room,
                topic=t.topic,
                domain=t.domain,
                target_platform=t.target_platform,
                source_text=(t.source_text[: args.source_chars]
                             if args.source_chars else t.source_text),
                source_platform=t.source_platform,
            ),
        }
        for t in tasks
    ]

    if args.scope == "both":
        scopes = ["global", "cell"]
    elif args.scope == "all":
        scopes = ["global", "cell", "decomposed"]
    else:
        scopes = [args.scope]
    grid: list[dict] = [
        {"layer": 0, "alpha": 0.0, "scope": "global", "audience_alpha": 0.0}
    ]  # unsteered baseline
    for scope in scopes:
        for layer in parse_layers(args.layers):
            for alpha in [float(a) for a in args.alphas.split(",") if a.strip()]:
                audience_alphas = (
                    [float(a) for a in args.audience_alphas.split(",") if a.strip()]
                    if scope == "decomposed"
                    else [0.0]
                )
                for audience_alpha in audience_alphas:
                    grid.append(
                        {
                            "layer": layer,
                            "alpha": alpha,
                            "scope": scope,
                            "audience_alpha": audience_alpha,
                        }
                    )

    results: list[dict] = []

    def restore(ckpt):
        results.extend(ckpt.payload["results"])

    start_step, _ = resume_or_start(ckpt_dir, restore=restore)
    log.info("start_step=%d of %d grid points", start_step, len(grid))

    # A remote artifact must be swept on the residual stream it was read from.
    # The vectors carry the width of their own model (8192 at 70B, 16384 at
    # 405B), so loading `base_model` locally is not merely too large to fit, it
    # is the wrong stream. Dispatch on the backend the fit recorded rather than
    # on a flag, so the sweep cannot be pointed at the wrong one by hand.
    # `load_in_4bit` is not forwarded: NDIF serves its own precision and
    # `NDIFConfig.to_dict` files it as None so a remote grid point is never
    # recorded as sharing the study's quantisation constant.
    backend = (artifact.get("fit_config") or {}).get("backend", "local")
    if backend == "ndif":
        from vectorial_eval.methods.steering.ndif_backend import (
            NDIFConfig,
            NDIFSteeredGenerator,
        )

        gen = NDIFSteeredGenerator(
            artifact,
            NDIFConfig(
                model_id=artifact["base_model"],
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                **({} if args.batch_size is None else {"batch_size": args.batch_size}),
            ),
        )
    else:
        gen = SteeredGenerator(
            artifact,
            GenConfig(
                model_id=artifact["base_model"],
                load_in_4bit=args.load_in_4bit,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                **({} if args.batch_size is None else {"batch_size": args.batch_size}),
            ),
        )
    emb = Embedder()
    ref_matrix = emb.encode(refs)
    ceiling = reference_ceiling(emb, ref_matrix)
    log.info("reference ceiling: %s (embeddings: %s)", ceiling, emb.backend)

    t0 = time.time()
    for step in range(start_step, len(grid)):
        point = grid[step]
        spec = SteeringSpec(
            layers=(point["layer"],) if point["alpha"] else (),
            alpha=point["alpha"],
            scope=point["scope"],
            audience_alpha=point["audience_alpha"],
        )
        outs = gen.generate(items, spec, draw=0)
        texts = [o["text"] for o in outs]
        scored = score_pool(emb, texts, [i["source"] for i in items], refs, ref_matrix)
        record = {
            **point,
            **scored,
            "n_failed": sum(1 for o in outs if not o["ok"]),
            "example": next((t for t in texts if t.strip()), ""),
        }
        results.append(record)
        log.info(
            "grid %d/%d layer=%s alpha=%s scope=%s centroid_cos=%s energy=%s "
            "source_cos=%s degeneracy=%s (%.0fs)",
            step + 1, len(grid), point["layer"], point["alpha"], point["scope"],
            scored.get("centroid_cos"), scored.get("energy_dist"),
            scored.get("source_cos"), scored.get("degeneracy"), time.time() - t0,
        )
        save_checkpoint(
            ckpt_dir, step + 1, payload={"results": results},
            meta={"grid_point": point}, keep_last=2,
        )

    chosen, baseline, eligible = select_pareto_point(results, args.max_degeneracy)

    payload = {
        "artifact": {
            "path": str(args.artifact),
            "sha256": artifact["sha256"],
            "variant": variant,
            "base_model": artifact["base_model"],
            "counts": artifact["counts"],
        },
        "sweep": {
            "split": dev_split,
            "n_dev_tasks": len(tasks),
            "n_dev_cells": len({t.cell_id for t in tasks}),
            "n_reference_posts": len(refs),
            "embedding_backend": emb.backend,
            "objective": (
                "Pareto improvement over unsteered centroid_cos and source_cos, "
                "subject to degeneracy <= max_degeneracy"
            ),
            "max_degeneracy": args.max_degeneracy,
            "generation": gen.cfg.to_dict(),
            "note": (
                "These numbers select hyperparameters on the validation split. "
                "They are not the harness metrics and are not comparable with them."
            ),
        },
        "reference_ceiling": ceiling,
        "baseline_unsteered": baseline,
        "chosen": chosen,
        "results": sorted(
            results,
            key=lambda r: (
                r["scope"], r["layer"], r["alpha"], r.get("audience_alpha", 0.0)
            ),
        ),
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # `SteeringTransfer` reads its layer, alpha and scope from
    # `chosen.<artifact stem>.json` beside the artifact, and warns and falls back
    # to unselected defaults when that file is absent. Writing the sweep result
    # to `--out` alone therefore left the selection where nothing would read it,
    # and a full run generated at alpha=1.0, which the grid shows destroys the
    # model. Emit the file the transfer function actually looks for.
    if chosen:
        chosen_path = Path(args.artifact).parent / f"chosen.{Path(args.artifact).stem}.json"
        chosen_path.write_text(
            json.dumps(
                {
                    "layer": chosen["layer"],
                    "alpha": chosen["alpha"],
                    "scope": chosen["scope"],
                    "audience_alpha": chosen.get("audience_alpha", 0.0),
                    "selected_on": dev_split,
                    "objective": (
                        "Pareto non-inferior centroid_cos and source_cos versus "
                        "unsteered, subject to degeneracy <= "
                        f"{args.max_degeneracy}"
                    ),
                    "centroid_cos": chosen.get("centroid_cos"),
                    "baseline_unsteered_centroid_cos": (baseline or {}).get("centroid_cos"),
                    "reference_ceiling": ceiling,
                    "sweep_file": str(out),
                    "artifact_sha256": artifact["sha256"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        log.info("wrote %s (layer=%s alpha=%s)", chosen_path, chosen["layer"], chosen["alpha"])
    else:
        log.warning(
            "no grid point satisfied degeneracy <= %s; no chosen file written, so "
            "the transfer function will keep warning and using defaults",
            args.max_degeneracy,
        )
    log.info("wrote %s", out)
    if chosen:
        chosen_path = out.parent / f"chosen.{args.artifact.stem}.json"
        chosen_path.write_text(
            json.dumps(
                {
                    "artifact": str(args.artifact),
                    "variant": variant,
                    "layer": chosen["layer"],
                    "alpha": chosen["alpha"],
                    "scope": chosen["scope"],
                    "audience_alpha": chosen.get("audience_alpha", 0.0),
                    "dev": {k: chosen[k] for k in
                            ("centroid_cos", "energy_dist", "source_cos", "degeneracy")},
                    "baseline_unsteered": (
                        {k: baseline[k] for k in
                         ("centroid_cos", "energy_dist", "source_cos", "degeneracy")}
                        if baseline else None
                    ),
                    "reference_ceiling": ceiling,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        log.info("chosen: layer=%s alpha=%s scope=%s -> %s",
                 chosen["layer"], chosen["alpha"], chosen["scope"], chosen_path)
    else:
        log.error("no grid point met the degeneracy bound; nothing selected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
