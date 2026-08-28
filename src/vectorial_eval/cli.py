"""Command-line entry point.

    python -m vectorial_eval.cli build
    python -m vectorial_eval.cli build-training
    python -m vectorial_eval.cli transfer --fn identity target_sample shuffle_control
    python -m vectorial_eval.cli transfer --fn llm_rewrite llm_fewshot --limit 100
    python -m vectorial_eval.cli evaluate --split test
    python -m vectorial_eval.cli evaluate --split test \
        --embedding-model google/embeddinggemma-300m --embedding-dim 256
    python -m vectorial_eval.cli report --split test
    python -m vectorial_eval.cli models
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import sys
from pathlib import Path

from .config import DEFAULT_DATA_DIR, DEFAULT_RUN_DIR, MODELS, HarnessConfig
from .data import build_dataset, build_training
from .data.schema import PostRecord, TransferTask
from .evaluate import format_table, read_jsonl, report_path
from .evaluate import run as run_eval
from .features.embeddings import BACKENDS
from .transfer import Corpus, build_transfer_fn
from .transfer import available as available_fns

# Transfer functions that call an LLM and therefore take `llm=` and
# `n_samples=`. Most are named `llm_*`; this set carries the ones that are not,
# so that `--model`, `--n-samples` and the shared cache reach them too. A
# function omitted here would silently run on the default model, which would
# make its results incomparable with the rest of the sweep.
_LLM_BACKED_FNS = {"aspect_prompt", "plan_then_transfer"}


def _is_llm_backed(name: str) -> bool:
    return name.startswith("llm_") or name in _LLM_BACKED_FNS


def _accepts(name: str, param: str) -> bool:
    """Whether the registered function's constructor takes `param`.

    `--n-samples` is not an LLM concern. The trained methods draw several
    candidates per source post too, and the group-level metrics need at least
    four candidates in a cell to score it at all. Gating the argument on
    `_is_llm_backed` silently gave those methods one draw per task while the
    language-model columns got four, which is both an unfair comparison and
    enough to drop cells out of the triangle score. Ask the constructor instead
    of maintaining a second list that can fall out of step with the registry.
    """
    from .transfer.base import _REGISTRY

    cls = _REGISTRY.get(name)
    if cls is None:
        return False
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return False
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return param in sig.parameters
    return param in sig.parameters


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_posts(data_dir: Path) -> list[PostRecord]:
    posts = []
    for split in ("train", "val", "test"):
        path = data_dir / f"posts.{split}.jsonl"
        if path.exists():
            posts.extend(read_jsonl(path, PostRecord))
    return posts


def _derange_audiences(
    tasks: list[TransferTask],
) -> tuple[list[TransferTask], dict[str, str]]:
    """Replace each task's audience with a deterministic different audience.

    Task and cell identifiers remain unchanged deliberately: evaluation must
    compare the resulting text with the original task's target pool. Only the
    audience conditioning visible to the transfer method changes. Rotating the
    sorted audience names is a frozen derangement and does not inspect method
    outputs or held-out scores.
    """
    rooms = sorted({task.room for task in tasks})
    if len(rooms) < 2:
        raise ValueError("audience shuffle requires at least two audience rooms")
    mapping = {room: rooms[(index + 1) % len(rooms)] for index, room in enumerate(rooms)}
    if any(source == target for source, target in mapping.items()):
        raise RuntimeError("audience derangement contains a fixed point")
    return [
        task.model_copy(update={"room": mapping[task.room]}) for task in tasks
    ], mapping


def cmd_build(args, cfg: HarnessConfig) -> int:
    cfg.dataset.out_dir = Path(args.data_dir)
    if args.csv:
        cfg.dataset.csv_path = Path(args.csv)
    if args.room_mode:
        cfg.dataset.room_mode = args.room_mode
    if args.min_posts:
        cfg.dataset.min_posts_per_platform = args.min_posts

    manifest = build_dataset.build(
        cfg.dataset, source_platform=args.source, target_platform=args.target
    )
    print(f"\nCells: {manifest['n_cells']} "
          f"({manifest['n_dense_cells']} dense, {manifest['n_heldout_cells']} held out)")
    print(f"Posts: {manifest['n_posts']}   Tasks: {manifest['n_tasks']}")
    print(f"Per split — posts: {manifest['counts']['posts']}")
    print(f"Per split — tasks: {manifest['counts']['tasks']}")
    print("\nLargest cells:")
    for c in manifest["cells"][:10]:
        flag = " [held out]" if c["heldout_cell"] else ""
        print(f"  {c['n_linkedin']:>4} li / {c['n_reddit']:>4} rd  {c['tier']:<7} "
              f"{c['room']} :: {c['topic']}{flag}")
    return 0


def cmd_build_training(args, cfg: HarnessConfig) -> int:
    manifest = build_training.build(
        data_dir=Path(args.data_dir),
        out_dir=Path(args.out_dir) if args.out_dir else None,
        target_platform=args.target,
        cfg=cfg.dataset,
        selection_from_train=args.selection_from_train,
        selection_fraction=args.selection_fraction,
    )
    counts = manifest["counts"]
    print(f"\nConditional-LM data -> {manifest['out_dir']}")
    print(f"Target platform: {manifest['target_platform']}   seed: {manifest['seed']}")
    print("Examples per file: " + "  ".join(
        f"{k}={v}" for k, v in counts["examples"].items()
    ))
    print("Cells per file:    " + "  ".join(
        f"{k}={v}" for k, v in counts["n_cells"].items()
    ))
    print(f"Dropped: {manifest['dropped']}")
    lk = manifest["leakage_check"]
    print(f"Leakage check: train file is train-split only = {lk['train_file_is_train_split_only']}, "
          f"no post_id in two splits = {not lk['post_ids_in_more_than_one_split']}")
    return 0


def cmd_transfer(args, cfg: HarnessConfig) -> int:
    data_dir, run_dir = Path(args.data_dir), Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    tasks = read_jsonl(data_dir / f"tasks.{args.split}.jsonl", TransferTask)
    if args.limit:
        # Deterministic subsample for an inexpensive smoke run. Sorting by
        # task_id ensures the same subset is used across transfer functions, so
        # that the resulting comparison remains like-for-like.
        tasks = sorted(tasks, key=lambda t: t.task_id)[: args.limit]
    audience_mapping = None
    original_room_by_task = {task.task_id: task.room for task in tasks}
    if args.audience_shuffle:
        tasks, audience_mapping = _derange_audiences(tasks)
    corpus = Corpus(_load_posts(data_dir))
    print(f"{len(tasks)} tasks in split={args.split}")

    if args.model:
        cfg.llm.model = MODELS.get(args.model, args.model)
    if args.cache:
        cfg.llm.cache_dir = run_dir / "llm_cache"

    for name in args.fn:
        kwargs = {}
        if _is_llm_backed(name):
            kwargs["llm"] = cfg.llm
        # The language-model rewriters absorb `n_samples` through **kw, so they
        # are matched by name; the trained methods declare it explicitly. The
        # baselines take neither and are single-draw by construction.
        if _is_llm_backed(name) or _accepts(name, "n_samples"):
            kwargs["n_samples"] = args.n_samples
        if name == "llm_fewshot":
            kwargs["n_shot"] = args.n_shot
        # `--tag` gives the run a distinct label so the same function can be run
        # with several models side by side without clobbering earlier outputs.
        label = f"{name}_{args.tag}" if args.tag else name
        if audience_mapping:
            label = f"{label}_audience_shuffle"
        kwargs["name"] = label
        fn = build_transfer_fn(name, **kwargs)

        outputs = fn.run(tasks, corpus)
        if audience_mapping:
            conditioned_room_by_task = {task.task_id: task.room for task in tasks}
            for output in outputs:
                output.meta.update(
                    {
                        "audience_shuffle": True,
                        "original_room": original_room_by_task[output.task_id],
                        "conditioned_room": conditioned_room_by_task[output.task_id],
                        "audience_derangement": audience_mapping,
                    }
                )
        name = label
        path = run_dir / f"outputs.{name}.{args.split}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for o in outputs:
                fh.write(o.model_dump_json() + "\n")
        description = fn.describe()
        if audience_mapping:
            description["task_intervention"] = {
                "kind": "audience_shuffle",
                "mapping": audience_mapping,
                "policy": "sorted cyclic derangement frozen before generation",
            }
        (run_dir / f"transfer_fn.{name}.json").write_text(
            json.dumps(description, indent=2), encoding="utf-8"
        )
        n_ok = sum(1 for o in outputs if (o.output_text or "").strip())
        print(f"  {name}: {n_ok}/{len(outputs)} non-empty -> {path.name}")
    return 0


def cmd_evaluate(args, cfg: HarnessConfig) -> int:
    if args.metrics:
        cfg.eval.metrics = args.metrics
    if args.judge_model:
        cfg.judge.model = MODELS.get(args.judge_model, args.judge_model)
    if args.judge_panel:
        cfg.eval.judge_panel = [MODELS.get(m, m) for m in args.judge_panel]
    if args.embedding_backend:
        cfg.eval.embedding_backend = args.embedding_backend
    if args.embedding_model:
        cfg.eval.embedding_model = args.embedding_model
        cfg.eval.embedding_backend = "sentence-transformers"
    if args.embedding_dim:
        cfg.eval.embedding_dim = args.embedding_dim
    if args.embedding_batch_size:
        cfg.eval.embedding_batch_size = args.embedding_batch_size
    if args.bootstrap_unit:
        cfg.eval.bootstrap_unit = args.bootstrap_unit
    if args.cache:
        cfg.judge.cache_dir = Path(args.run_dir) / "llm_cache"

    report = run_eval(
        Path(args.data_dir),
        Path(args.run_dir),
        args.split,
        cfg,
        transfer_fns=args.fn,
        report_tag=args.report_tag,
    )
    print()
    print(format_table(report))
    return 0


def cmd_report(args, cfg: HarnessConfig) -> int:
    path = report_path(Path(args.run_dir), args.split, args.report_tag)
    if not path.exists():
        print(f"No report at {path}. Run `evaluate` first.", file=sys.stderr)
        return 1
    report = json.loads(path.read_text(encoding="utf-8"))
    print(format_table(report))

    for key, note in report.get("notes", {}).items():
        if "structural" in key and isinstance(note, dict) and note.get("worst_features"):
            print(f"\nLeast-reproduced style features — {key.split('.')[0]}:")
            for feat, gap in note["worst_features"][:6]:
                print(f"  {gap:6.3f}  {feat}")
        if "judge" in key and isinstance(note, dict) and note.get("top_tells"):
            print(f"\nMost common judge tells — {key.split('.')[0]}:")
            for tell, n in note["top_tells"][:6]:
                print(f"  {n:>4}x  {tell}")
    return 0


def cmd_models(args, cfg: HarnessConfig) -> int:
    """Show configured aliases, and optionally refresh from OpenRouter."""
    print("Configured aliases:")
    for alias, slug in MODELS.items():
        print(f"  {alias:<20} {slug}")
    if args.refresh:
        import datetime
        import urllib.request

        data = json.load(
            urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=30)
        )["data"]
        rows = sorted(data, key=lambda m: m.get("created", 0), reverse=True)
        print(f"\n{args.top} newest on OpenRouter:")
        for m in rows[: args.top]:
            d = datetime.datetime.utcfromtimestamp(m.get("created", 0)).strftime("%Y-%m-%d")
            p = m.get("pricing", {})
            print(f"  {d}  {m['id']:<45} in={p.get('prompt')} out={p.get('completion')}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="vectorial_eval", description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR / "default"))
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build", help="CSV -> train/val/test JSONL")
    p.add_argument("--csv")
    p.add_argument("--room-mode", choices=["strict", "canonical"])
    p.add_argument("--min-posts", type=int)
    p.add_argument("--source", default="linkedin")
    p.add_argument("--target", default="reddit")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser(
        "build-training",
        help="project the existing splits into conditional target-LM training data",
    )
    p.add_argument("--out-dir", help="default: <data-dir>/training")
    p.add_argument("--target", default="reddit",
                   help="target platform whose distribution is being modelled")
    p.add_argument("--selection-from-train", action="store_true",
                   help="carve dev deterministically from train so heldout remains untouched")
    p.add_argument("--selection-fraction", type=float, default=0.15)
    p.set_defaults(func=cmd_build_training)

    p = sub.add_parser("transfer", help="run transfer functions over tasks")
    p.add_argument("--fn", nargs="+", required=True,
                   help=f"one or more of: {', '.join(available_fns())}")
    p.add_argument("--split", default="test", choices=["train", "val", "test", "heldout"],
                   help="'heldout' merges val and test, which widens the reference pools")
    p.add_argument("--limit", type=int, help="cap tasks (deterministic) for a smoke run")
    p.add_argument("--n-samples", type=int, default=1,
                   help="candidates drawn per task; >1 enlarges the candidate pool")
    p.add_argument("--model", help=f"alias or slug; aliases: {', '.join(MODELS)}")
    p.add_argument("--n-shot", type=int, default=4)
    p.add_argument("--tag", help="suffix for the output label, e.g. the model name")
    p.add_argument(
        "--audience-shuffle",
        action="store_true",
        help=(
            "negative control: condition each task on a frozen different audience "
            "while scoring it against the original task's target pool"
        ),
    )
    p.add_argument("--cache", action="store_true", default=True,
                   help="cache LLM calls under <run-dir>/llm_cache (default on)")
    p.add_argument("--no-cache", dest="cache", action="store_false")
    p.set_defaults(func=cmd_transfer)

    p = sub.add_parser("evaluate", help="score outputs and write a report")
    p.add_argument("--split", default="test", choices=["train", "val", "test", "heldout"])
    p.add_argument("--fn", nargs="+", help="restrict to these transfer functions")
    p.add_argument("--metrics", nargs="+",
                   help="default: structural classifier distributional semantic degeneracy; "
                        "add 'judge' to run the LLM judge (costs money)")
    p.add_argument("--judge-model")
    p.add_argument("--judge-panel", nargs="+", help="multi-model judge panel")
    p.add_argument("--embedding-backend", choices=list(BACKENDS))
    p.add_argument("--embedding-model",
                   help="Sentence-Transformers checkpoint, e.g. google/embeddinggemma-300m; "
                        "implies --embedding-backend sentence-transformers")
    p.add_argument("--embedding-dim", type=int,
                   help="SVD rank (tfidf-svd) or Matryoshka width (EmbeddingGemma: 768/512/256/128)")
    p.add_argument("--embedding-batch-size", type=int,
                   help="encoding batch size for the sentence-transformers backend")
    p.add_argument(
        "--bootstrap-unit",
        choices=("cell", "audience-cell"),
        help=(
            "independent-unit policy; Study 4 uses audience-cell and records "
            "the cell-only bootstrap alongside it as a sensitivity analysis"
        ),
    )
    p.add_argument("--report-tag",
                   help="suffix for the report filename, giving report.<split>.<tag>.json. "
                        "Defaults to the embedding-space slug for any space other than the "
                        "default, so two spaces never overwrite one another")
    p.add_argument("--cache", action="store_true", default=True)
    p.add_argument("--no-cache", dest="cache", action="store_false")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("report", help="re-print a saved report")
    p.add_argument("--split", default="test", choices=["train", "val", "test", "heldout"])
    p.add_argument("--report-tag", help="read report.<split>.<tag>.json instead")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("models", help="list model aliases; --refresh queries OpenRouter")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--top", type=int, default=25)
    p.set_defaults(func=cmd_models)

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args, HarnessConfig())


if __name__ == "__main__":
    sys.exit(main())
