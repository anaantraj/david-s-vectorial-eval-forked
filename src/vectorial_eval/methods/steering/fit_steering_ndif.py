"""Fit steering vectors on an NDIF-hosted model.

This is `fit_steering` with the forward pass moved to NDIF. It exists because a
steering vector is tied to the residual stream it was read from: the local
artifacts have 4096 columns and 70B's stream has 8192, 405B's has 16384. There is
no way to *apply* the existing artifacts at scale, so the scale arm has to fit
its own.

Everything that defines the artifact is imported from `fit_steering` rather than
reimplemented — `build_examples`, `State`, `finalise` and the artifact format. The
only differences are where the activations come from and that posts are read
concurrently, because a remote forward pass costs seconds of network rather than
milliseconds of GPU and 674 posts read serially is most of an hour.

    NDIF_API_KEY=... .venv/bin/python -m vectorial_eval.methods.steering.fit_steering_ndif \\
        --model meta-llama/Llama-3.1-70B-Instruct --variant target_lm --concurrency 8

The artifact records `base_model` as the NDIF model id, so `SteeringTransfer`'s
existing guard refuses to apply it to any other model, and `fit_config` carries
the NDIF deployment and the nnsight version, without which a remote number is not
reproducible.

Two properties of the local fit deliberately do *not* hold here, and both are
written into the artifact rather than left implicit:

* `load_in_4bit` is None, not False. NDIF serves its own precision, so the
  study's "every method under 4-bit NF4" constant does not cover this arm.
* The checkpoint is NDIF's. A rerun after they redeploy is not guaranteed to
  reproduce, which is why the deployment state is recorded.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from vectorial_eval.config import DEFAULT_DATA_DIR
from vectorial_eval.methods.steering import fit_steering as fs
from vectorial_eval.methods.steering.ndif_backend import (
    NDIF_STATUS_URL,
    _nnsight_version,
    check_model,
    probe_block_layout,
    remote_blocks,
    remote_embeddings,
    remote_pooled_activations,
    resolve_api_key,
)

log = logging.getLogger("fit_steering_ndif")

DEFAULT_MODEL = "meta-llama/Llama-3.1-70B-Instruct"


class _Models:
    """One nnsight `LanguageModel` per worker thread.

    nnsight keeps tracer state on the model object, so sharing one across threads
    interleaves graphs. Constructing another is cheap — it reads the config and
    tokenizer, never the weights, which is the whole point of a remote model.
    """

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._local = threading.local()
        self.block_tuple: bool | None = None
        self.chat_wrapped: bool | None = None

    def get(self):
        lm = getattr(self._local, "lm", None)
        if lm is None:
            from nnsight import LanguageModel

            lm = LanguageModel(self.model_id)
            tok = lm.tokenizer
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            self._local.lm = lm
            self._local.tok = tok
            self._local.blocks = remote_blocks(lm)
            self._local.embeddings = remote_embeddings(lm)
        return (
            self._local.lm,
            self._local.tok,
            self._local.blocks,
            self._local.embeddings,
        )


def stamp_remote_provenance(
    artifact: dict, deployment: dict, concurrency: int, chat_wrapped: bool
) -> dict:
    """Record what a remote fit needs to be attributable, and undo one coercion.

    `fit_steering.finalise` writes `bool(args.load_in_4bit)`, which is right for
    the local arm — the flag is only ever True or False there — but turns this
    arm's None into False. False is not neutral in that vocabulary: it is the
    name of the local bf16 ablation, so leaving it filed this fit as an ablation
    of the study's quantisation constant rather than as a run that never shared
    it. It has to be None.
    """
    fit_config = artifact.setdefault("fit_config", {})
    fit_config["load_in_4bit"] = None
    fit_config["backend"] = "ndif"
    fit_config["ndif"] = {
        "status_url": NDIF_STATUS_URL,
        "deployment": deployment,
        "nnsight_version": _nnsight_version(),
        "concurrency": concurrency,
        "chat_wrapped": chat_wrapped,
    }
    return artifact


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", default="target_lm", choices=fs.VARIANTS)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--max-length", type=int, default=768)
    p.add_argument("--min-cell-posts", type=int, default=3)
    p.add_argument("--min-audience-posts", type=int, default=10)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--limit", type=int, default=0, help="smoke run over the first N posts")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--ckpt-dir", default=os.environ.get("VECTORIAL_CKPT_DIR", ""))
    p.add_argument(
        "--allow-unwrapped",
        action="store_true",
        help="fit on a base model with no chat template; the vectors are then "
             "not comparable with the wrapped columns",
    )
    args = p.parse_args(argv)
    # `finalise` reads these off the namespace. Remote runs carry no quantisation
    # constant and no Stage A adapter, and both are recorded as such.
    args.load_in_4bit = None
    args.initial_adapter = ""

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )

    from checkpointing import resume_or_start, save_checkpoint

    manifest = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8"))
    direction = manifest.get("direction", {})
    # `State` and `finalise` read these as module globals on `fit_steering`, so
    # they are set there rather than locally.
    fs.SOURCE_PLATFORM = direction.get("source", fs.SOURCE_PLATFORM)
    fs.TARGET_PLATFORM = direction.get("target", fs.TARGET_PLATFORM)
    if {fs.SOURCE_PLATFORM, fs.TARGET_PLATFORM} != {"linkedin", "reddit"}:
        raise ValueError(f"unsupported direction: {direction}")

    resolve_api_key()
    deployment = check_model(args.model)
    log.info("ndif: %s pinned=%s", args.model, deployment.get("pinned"))

    from transformers import AutoConfig

    model_cfg = AutoConfig.from_pretrained(args.model)
    n_layers = model_cfg.num_hidden_layers + 1
    hidden = model_cfg.hidden_size

    root = Path(__file__).resolve().parents[4]
    slug = args.model.split("/")[-1]
    tag = f"{args.variant}.ndif-{slug}"
    out = args.out or (root / "runs" / "steering" / f"steering.{tag}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.ckpt_dir or str(root / "runs" / "checkpoints" / f"steering_{tag}")

    examples, provenance = fs.build_examples(args.data_dir, args.variant)
    if args.limit:
        half = max(1, args.limit // 2)
        picked = [e for e in examples if e["platform"] == fs.TARGET_PLATFORM][:half]
        picked += [e for e in examples if e["platform"] == fs.SOURCE_PLATFORM][:half]
        examples = sorted(picked, key=lambda e: e["post_id"])
        provenance["limit"] = args.limit
        provenance["n_examples"] = len(examples)
        provenance["n_target"] = sum(
            1 for e in examples if e["platform"] == fs.TARGET_PLATFORM
        )
        provenance["n_source"] = sum(
            1 for e in examples if e["platform"] == fs.SOURCE_PLATFORM
        )

    models = _Models(args.model)
    lm, tok, blocks, embeddings = models.get()
    chat_wrapped = bool(getattr(tok, "chat_template", None))
    if not chat_wrapped and not args.allow_unwrapped:
        raise ValueError(
            f"{args.model!r} has no chat template. Vectors fitted on an unwrapped "
            "prompt are not comparable with LoRA, soft prompting or the local "
            "steering arm, which all wrap. Use a pinned instruct model, or pass "
            "--allow-unwrapped to accept that."
        )
    models.chat_wrapped = chat_wrapped
    models.block_tuple = probe_block_layout(lm, blocks)
    log.info(
        "variant=%s examples=%d (target=%d source=%d) layers=%d hidden=%d "
        "chat_wrapped=%s block_output=%s out=%s",
        args.variant, len(examples), provenance["n_target"], provenance["n_source"],
        n_layers, hidden, chat_wrapped,
        "tuple" if models.block_tuple else "tensor", out,
    )

    state = fs.State(n_layers, hidden)

    def restore(ckpt):
        state.restore(ckpt.payload)

    start_step, _ = resume_or_start(ckpt_dir, restore=restore)
    state.index = start_step
    log.info("start_step=%d of %d posts", start_step, len(examples))

    def read(example: dict):
        """Return `(pooled, norms)` or `(None, reason)` for one post."""
        try:
            lm_t, tok_t, blocks_t, emb_t = models.get()
            got = remote_pooled_activations(
                lm_t, tok_t, example["prompt"], example["completion"],
                args.max_length, blocks=blocks_t, embeddings=emb_t,
                block_tuple=models.block_tuple, chat_wrapped=models.chat_wrapped,
            )
            return got if got is not None else (None, "no_completion_tokens")
        except Exception as exc:  # noqa: BLE001 - one bad post must not end the fit
            log.warning("post %s failed: %s", example["post_id"], exc)
            return (None, "forward_error")

    t0 = time.time()
    # Posts are read concurrently but accumulated in index order, so a resume
    # from a checkpoint sees exactly the prefix the checkpoint claims. The sums
    # themselves are order-independent; `state.index` is not.
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        for chunk_start in range(start_step, len(examples), args.save_every):
            chunk = examples[chunk_start : chunk_start + args.save_every]
            for example, (pooled, norms_or_reason) in zip(
                chunk, pool.map(read, chunk), strict=True
            ):
                if pooled is None:
                    state.skipped[norms_or_reason] += 1
                else:
                    state.add(
                        example["platform"], example["cell_id"], example["room"],
                        pooled, norms_or_reason,
                    )
            state.index = chunk_start + len(chunk)
            save_checkpoint(
                ckpt_dir,
                state.index,
                payload=state.payload(),
                meta={
                    "n_target": state.counts[fs.TARGET_PLATFORM],
                    "n_source": state.counts[fs.SOURCE_PLATFORM],
                    "elapsed_s": round(time.time() - t0, 1),
                },
                keep_last=2,
            )
            log.info(
                "step %d/%d target=%d source=%d %.1fs",
                state.index, len(examples), state.counts[fs.TARGET_PLATFORM],
                state.counts[fs.SOURCE_PLATFORM], time.time() - t0,
            )

    import torch

    artifact = fs.finalise(state, args, provenance)
    stamp_remote_provenance(
        artifact,
        deployment=deployment,
        concurrency=args.concurrency,
        chat_wrapped=chat_wrapped,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".pt.tmp")
    torch.save(artifact, tmp)
    os.replace(tmp, out)
    summary = {
        k: artifact[k]
        for k in ("format", "base_model", "variant", "hidden_size", "counts")
    }
    summary["diagnostics"] = {
        "relative_shift_per_layer": artifact["diagnostics"]["relative_shift_per_layer"],
    }
    summary["fit_config"] = artifact["fit_config"]
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("wrote %s (%.1f MB)", out, out.stat().st_size / 1e6)
    log.info(
        "cells with own vector: %d of %d seen",
        artifact["counts"]["n_cells_with_vector"],
        artifact["counts"]["n_cells_seen"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
