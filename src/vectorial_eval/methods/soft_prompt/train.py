#!/usr/bin/env python
"""Fit a soft prompt on the conditional target language model.

    scripts/cluster/launch.sh -j sp_target_lm -- \
        src/vectorial_eval/methods/soft_prompt/train.py \
        --variant target_lm --n-virtual-tokens 32 --lr 1e-2 --init text

Formulation
-----------
The base model is frozen in its entirety. The only trainable parameters are
`n_virtual_tokens` continuous vectors of the model's embedding width, prepended
to the embedded input of every example. Training maximises the likelihood of a
real train-split Reddit post given the shared context in
`build_training.TARGET_LM_PROMPT`, which names the audience room, the domain, the
topic and the target platform. There is no paired data anywhere in this project,
so the objective is a language-model loss over authentic target text and never a
translation loss; the source LinkedIn post enters only at inference, through the
content block of the same template.

Loss is computed on the completion tokens only. The context is identical across
every example in a cell, so scoring it would mostly measure how well the model
predicts a fixed header and would dilute the signal the soft prompt is meant to
carry.

One global prompt, not one per cell
-----------------------------------
A single soft prompt is trained over all cells. There are 337 train examples in
26 cells, a median of 7 per cell and a minimum of 2, so a per-cell prompt would
fit tens of thousands of free parameters to a handful of posts and would give the
four zero-shot held-out cells no prompt at all. The cell identity is already
supplied in text, in the context every example shares, so the conditioning on
room and topic is not lost by pooling; what the soft prompt has to encode is the
manner of the target platform, which is the part that is common across cells. A
per-cell or per-room prompt is the obvious ablation and is what `--group-by`
exists for, but the pooled prompt is the headline configuration.

Quantisation, batch size and sequence length follow
docs/09-cluster-training-config.md and are constant across all three methods.

Crash safety
------------
The GPUs are shared and jobs are killed without warning. A checkpoint is written
every `--save-every` optimiser steps and at every epoch boundary, and relaunching
the identical command resumes from the last one: the example order is a pure
function of the seed and the epoch, so a resumed run consumes exactly the
examples the killed run had not reached. Only the prompt embeddings and the
optimiser state are saved, which is a few megabytes.

Selection
---------
Dev loss is measured on `target_lm*.dev.jsonl` at every epoch boundary. The best
epoch is copied to `<ckpt_dir>/best/`, which is what the transfer function loads,
and the full curve is written to `<ckpt_dir>/summary.json`. Test and heldout files
are never opened by this script.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# `launch.sh` puts both `src/` and `scripts/cluster/` on PYTHONPATH.
from vectorial_eval.methods.soft_prompt.prompting import prompt_tail, wrap_prompt

log = logging.getLogger("soft_prompt.train")

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
VARIANTS = ("target_lm", "target_lm_paired", "target_lm_aspect")

#: Initialisation text for `--init text`. Prompt tuning is sensitive to where it
#: starts, and this is the instruction the context already implies, so the prompt
#: begins somewhere the frozen model already understands.
def init_text(target_platform: str) -> str:
    return (
        "Write the post exactly as a member of this occupational audience would "
        f"actually write it on {target_platform.title()}: their vocabulary, their length, "
        "their tone, and their formatting."
    )


@dataclass
class Config:
    variant: str = "target_lm"
    n_virtual_tokens: int = 32
    lr: float = 1e-2
    init: str = "text"  # text | sampled_vocab | random
    epochs: int = 20
    batch_size: int = 1
    grad_accum: int = 8
    max_length: int = 768
    min_completion_tokens: int = 256
    warmup_frac: float = 0.1
    weight_decay: float = 0.0
    seed: int = 0
    patience: int = 4
    save_every: int = 50
    group_by: str = "none"  # none | room | cell_id
    data_dir: str = "data/training"
    base_model: str = BASE_MODEL
    load_in_4bit: bool = True
    target_platform: str = "reddit"
    initial_adapter: str | None = None


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_split(data_dir: Path, variant: str, split: str) -> list[dict]:
    """Read one training file and verify it came from `build_training`.

    The check on the final line of the prompt is cheap insurance against training
    this method on a context the other two methods did not see, which would make
    the comparison between them meaningless without producing any error.
    """
    path = data_dir / f"{variant}.{split}.jsonl"
    records = read_jsonl(path)
    if not records:
        raise ValueError(f"{path} is empty")
    target_platform = records[0]["platform"]
    if any(rec["platform"] != target_platform for rec in records):
        raise ValueError(f"{path}: training file contains more than one target platform")
    tail = prompt_tail(target_platform)
    for rec in records:
        if not rec["prompt"].rstrip("\n").endswith(tail.rstrip("\n")):
            raise ValueError(
                f"{path}: a prompt does not end with the shared template's final "
                f"line {tail!r}; this file was not produced by build_training"
            )
    return records


def assert_train_only(records: list[dict], path_desc: str) -> None:
    """Fail loudly if anything outside the train split reached the training set.

    Every record carries the split assigned by `blake2b(post_id, seed)` in
    `build_dataset.py`. Training on a val or test post is silent leakage that no
    later stage can detect, so it is checked here rather than assumed.
    """
    bad = sorted({r["split"] for r in records} - {"train"})
    if bad:
        raise ValueError(f"{path_desc} contains non-train splits: {bad}")


# --------------------------------------------------------------------------- #
# Tokenisation
# --------------------------------------------------------------------------- #
def encode_example(tokenizer, record: dict, cfg: Config) -> dict:
    """Return input ids and labels, with the context masked out of the loss.

    The completion is capped at `max_length - len(prompt)` tokens and the prompt
    at `max_length - min_completion_tokens`. A prompt that exceeds its budget is
    truncated in the middle, which keeps the header naming the room and topic and
    the closing instruction line intact and removes text from the middle of the
    content block, the only part that is ever long enough to matter.
    """
    import torch

    text = wrap_prompt(tokenizer, record["prompt"])
    prompt_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(record["completion"], add_special_tokens=False)["input_ids"]
    completion_ids = completion_ids + [tokenizer.eos_token_id]

    prompt_budget = cfg.max_length - min(cfg.min_completion_tokens, len(completion_ids))
    truncated_prompt = False
    if len(prompt_ids) > prompt_budget:
        head = prompt_budget // 2
        tail = prompt_budget - head
        prompt_ids = prompt_ids[:head] + prompt_ids[len(prompt_ids) - tail :]
        truncated_prompt = True

    completion_ids = completion_ids[: max(1, cfg.max_length - len(prompt_ids))]
    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + list(completion_ids)
    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "attention_mask": torch.ones(1, len(input_ids), dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
        "n_completion_tokens": len(completion_ids),
        "truncated_prompt": truncated_prompt,
    }


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_model(cfg: Config):
    """Load the frozen base model and attach an untrained soft prompt."""
    import torch
    from peft import (
        PeftModel,
        PromptTuningConfig,
        PromptTuningInit,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"device_map": {"": 0}, "dtype": torch.bfloat16}
    if cfg.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    try:
        base = AutoModelForCausalLM.from_pretrained(cfg.base_model, **kwargs)
    except TypeError:  # transformers < 5 spells the dtype argument differently
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        base = AutoModelForCausalLM.from_pretrained(cfg.base_model, **kwargs)

    if cfg.load_in_4bit:
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    else:
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()
    base.config.use_cache = False

    if cfg.initial_adapter:
        initial = Path(cfg.initial_adapter)
        if not (initial / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"initial adapter {initial} has no adapter_config.json"
            )
        # The Stage A LoRA remains frozen. Prompt tuning adds one outer PEFT
        # layer whose virtual-token embedding is the only trainable tensor.
        base = PeftModel.from_pretrained(base, str(initial), is_trainable=False)

    peft_kwargs = {}
    if cfg.init == "text":
        peft_kwargs = {
            "prompt_tuning_init": PromptTuningInit.TEXT,
            "prompt_tuning_init_text": init_text(cfg.target_platform),
            "tokenizer_name_or_path": cfg.base_model,
        }
    model = get_peft_model(
        base,
        PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=cfg.n_virtual_tokens,
            **peft_kwargs,
        ),
    )

    if cfg.init == "sampled_vocab":
        _init_from_sampled_vocab(model, base, cfg)
    elif cfg.init not in ("text", "random"):
        raise ValueError(f"unknown --init {cfg.init!r}")

    model.print_trainable_parameters()
    return model, tokenizer


def _init_from_sampled_vocab(model, base, cfg: Config) -> None:
    """Initialise each virtual token from a randomly drawn vocabulary embedding.

    `peft`'s RANDOM initialisation draws from the default `nn.Embedding` normal,
    whose scale is two orders of magnitude larger than a Llama input embedding.
    Starting that far outside the region the frozen model was trained on is a
    property of the initialiser rather than of soft prompting, so the random arm
    of the sweep uses this instead: the classical initialisation from the
    5,000 most frequent vocabulary items, which is where prompt tuning was shown
    to work in the first place.
    """
    import torch

    embed = base.get_input_embeddings().weight
    rng = torch.Generator(device="cpu").manual_seed(cfg.seed)
    idx = torch.randint(0, min(5000, embed.shape[0]), (cfg.n_virtual_tokens,), generator=rng)
    with torch.no_grad():
        rows = embed[idx.to(embed.device)].detach().to("cpu")
        param = _prompt_parameter(model)
        param.copy_(rows.to(param.dtype).to(param.device))


def model_device(model):
    """Device of the model's parameters.

    `PeftModel` does not reliably expose `.device`, and a quantised base model
    holds its weights on the single visible GPU that `launch.sh` pinned.
    """
    return next(model.parameters()).device


def _prompt_parameter(model):
    """The single trainable tensor: `[n_virtual_tokens, hidden_size]`."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    if len(trainable) != 1:
        raise RuntimeError(
            f"expected exactly one trainable tensor (the soft prompt), got {len(trainable)}"
        )
    return trainable[0]


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def epoch_order(n: int, epoch: int, seed: int) -> list[int]:
    """Example order for one epoch, a pure function of (seed, epoch).

    Determinism here is what makes resuming exact: a job killed part way through
    epoch 7 resumes into the same permutation of epoch 7 and consumes precisely
    the examples it had not reached.
    """
    order = list(range(n))
    random.Random(f"{seed}:{epoch}").shuffle(order)
    return order


def evaluate(model, tokenizer, records: list[dict], cfg: Config) -> dict:
    """Mean negative log-likelihood per completion token over a split.

    Token-weighted rather than example-weighted, so a long post is not counted as
    a single observation. The example-weighted mean is reported alongside it
    because the two disagree when post length varies as much as it does here.
    """
    import torch

    model.eval()
    total_nll, total_tokens, per_example = 0.0, 0, []
    with torch.no_grad():
        for rec in records:
            batch = encode_example(tokenizer, rec, cfg)
            n_tok = batch.pop("n_completion_tokens")
            batch.pop("truncated_prompt")
            batch = {k: v.to(model_device(model)) for k, v in batch.items()}
            loss = model(**batch).loss
            if not math.isfinite(float(loss)):
                continue
            total_nll += float(loss) * n_tok
            total_tokens += n_tok
            per_example.append(float(loss))
    model.train()
    if total_tokens == 0:
        return {"loss": float("nan"), "loss_per_example": float("nan"), "n": 0}
    return {
        "loss": total_nll / total_tokens,
        "loss_per_example": sum(per_example) / len(per_example),
        "n": len(per_example),
        "n_tokens": total_tokens,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--variant", choices=VARIANTS, default="target_lm")
    p.add_argument("--n-virtual-tokens", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--init", choices=("text", "sampled_vocab", "random"), default="text")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-length", type=int, default=768)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--patience", type=int, default=4, help="epochs without dev improvement")
    p.add_argument("--save-every", type=int, default=50, help="optimiser steps")
    p.add_argument("--group-by", choices=("none", "room", "cell_id"), default="none",
                   help="ablation: fit one prompt per group instead of one globally")
    p.add_argument("--data-dir", default="data/training")
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument(
        "--initial-adapter",
        default="",
        help="frozen Stage A LoRA below the learned soft prompt (Study 4)",
    )
    p.add_argument("--no-4bit", action="store_true", help="ablation only; see docs/09")
    p.add_argument("--ckpt-dir", default=os.environ.get("VECTORIAL_CKPT_DIR", ""))
    p.add_argument("--limit", type=int, default=0, help="smoke runs only")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.group_by != "none":
        raise SystemExit(
            "--group-by is reserved for the per-group ablation and is not implemented; "
            "the headline configuration fits one global prompt."
        )

    cfg = Config(
        variant=args.variant,
        n_virtual_tokens=args.n_virtual_tokens,
        lr=args.lr,
        init=args.init,
        epochs=args.epochs,
        grad_accum=args.grad_accum,
        max_length=args.max_length,
        seed=args.seed,
        patience=args.patience,
        save_every=args.save_every,
        data_dir=args.data_dir,
        base_model=args.base_model,
        load_in_4bit=not args.no_4bit,
        initial_adapter=args.initial_adapter or None,
    )

    import torch
    from checkpointing import resume_or_start, save_checkpoint

    ckpt_dir = Path(args.ckpt_dir or f"runs/checkpoints/soft_prompt_{cfg.variant}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(cfg.data_dir)

    train = load_split(data_dir, cfg.variant, "train")
    assert_train_only(train, f"{data_dir}/{cfg.variant}.train.jsonl")
    dev = load_split(data_dir, cfg.variant, "dev")
    cfg.target_platform = train[0]["platform"]
    if any(rec["platform"] != cfg.target_platform for rec in dev):
        raise ValueError("train and dev files do not share one target platform")
    if args.limit:
        train, dev = train[: args.limit], dev[: max(2, args.limit // 4)]
    log.info(
        "variant=%s train=%d dev=%d cells=%d | %s",
        cfg.variant, len(train), len(dev), len({r["cell_id"] for r in train}), asdict(cfg),
    )

    torch.manual_seed(cfg.seed)
    model, tokenizer = build_model(cfg)
    prompt_param = _prompt_parameter(model)

    steps_per_epoch = max(1, math.ceil(len(train) / cfg.grad_accum))
    total_steps = steps_per_epoch * cfg.epochs
    optimizer = torch.optim.AdamW(
        [prompt_param], lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    from transformers import get_cosine_schedule_with_warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(cfg.warmup_frac * total_steps), total_steps
    )

    state = {
        "history": [],
        "best_dev": float("inf"),
        "best_epoch": -1,
        "epochs_without_improvement": 0,
        "stopped_early": False,
        "finished": False,
    }

    def restore(ckpt) -> None:
        payload = ckpt.payload
        with torch.no_grad():
            prompt_param.copy_(payload["prompt"].to(prompt_param.device, prompt_param.dtype))
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        state.update(payload["state"])

    # `step` is a micro step (one example). Checkpoints land only on optimiser-step
    # boundaries, so a resumed run never restarts inside a partial accumulation.
    start_step, _ = resume_or_start(ckpt_dir, restore=restore)
    log.info("start_step=%d of %d micro steps", start_step, len(train) * cfg.epochs)

    def write_checkpoint(step: int, meta: dict) -> None:
        save_checkpoint(
            ckpt_dir,
            step,
            payload={
                "prompt": prompt_param.detach().to("cpu"),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "state": state,
                "config": asdict(cfg),
            },
            meta=meta,
            keep_last=2,
        )

    def write_summary() -> None:
        (ckpt_dir / "summary.json").write_text(
            json.dumps(
                {
                    "config": asdict(cfg),
                    "n_train": len(train),
                    "n_dev": len(dev),
                    "n_cells_train": len({r["cell_id"] for r in train}),
                    "steps_per_epoch": steps_per_epoch,
                    "trainable_parameters": int(prompt_param.numel()),
                    "history": state["history"],
                    "best_dev_loss": state["best_dev"],
                    "best_epoch": state["best_epoch"],
                    "stopped_early": state["stopped_early"],
                    "base_model": cfg.base_model,
                    "initial_adapter": cfg.initial_adapter,
                    "prompt_template_tail": prompt_tail(cfg.target_platform),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def save_best() -> None:
        """Write the best soft prompt where the transfer function looks for it."""
        best = ckpt_dir / "best"
        staging = ckpt_dir / "best.tmp"
        if staging.exists():
            import shutil

            shutil.rmtree(staging)
        model.save_pretrained(staging)
        (staging / "soft_prompt_meta.json").write_text(
            json.dumps(
                {
                    "config": asdict(cfg),
                    "dev_loss": state["best_dev"],
                    "epoch": state["best_epoch"],
                    "base_model": cfg.base_model,
                    "initial_adapter": cfg.initial_adapter,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        if best.exists():
            import shutil

            shutil.rmtree(best)
        os.replace(staging, best)

    micro_total = len(train) * cfg.epochs
    already_stopped = state.get("stopped_early") or (
        state.get("epochs_without_improvement", 0) >= cfg.patience
    )
    if start_step > 0 and already_stopped:
        # A relaunch of a run that already stopped early must not train on. The
        # relaunch is how a killed job continues, and `status.sh` cannot tell a
        # killed job from a finished one, so without this guard re-issuing the
        # command trains a further epoch and can overwrite `best/`. The selected
        # artifact would then depend on how many times the command was re-issued
        # rather than on the configuration.
        log.info(
            "run already stopped early with best dev loss %.4f at epoch %d; "
            "nothing to resume",
            state["best_dev"], state["best_epoch"],
        )
        state["finished"] = True
        write_summary()
        return 0

    model.train()
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()
    running, running_n = 0.0, 0

    if start_step == 0:
        # Dev loss before a single gradient step. This is the reference every
        # later epoch is read against: without it there is no way to say whether
        # the trained prompt improved on the initialisation, and for the `text`
        # initialisation it is close to what the frozen model does unaided.
        init_metrics = evaluate(model, tokenizer, dev, cfg)
        log.info("initialisation: dev_loss %.4f (%s)", init_metrics["loss"], init_metrics)
        state["history"].append(
            {"epoch": -1, "micro_step": 0, "dev": init_metrics, "seconds": 0.0}
        )
        state["best_dev"] = init_metrics["loss"]
        state["best_epoch"] = -1
        save_best()
        write_summary()

    step = start_step
    while step < micro_total:
        epoch = step // len(train)
        order = epoch_order(len(train), epoch, cfg.seed)
        within = step % len(train)

        for j in range(within, len(train)):
            rec = train[order[j]]
            batch = encode_example(tokenizer, rec, cfg)
            batch.pop("n_completion_tokens")
            batch.pop("truncated_prompt")
            batch = {k: v.to(model_device(model)) for k, v in batch.items()}
            loss = model(**batch).loss
            (loss / cfg.grad_accum).backward()
            running += loss.detach().item()
            running_n += 1
            step += 1

            at_boundary = (step % cfg.grad_accum == 0) or (step % len(train) == 0)
            if at_boundary:
                torch.nn.utils.clip_grad_norm_([prompt_param], 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                opt_step = step // cfg.grad_accum
                if opt_step % 10 == 0:
                    log.info(
                        "epoch %d micro %d/%d train_loss %.4f lr %.2e %.1fs",
                        epoch, step, micro_total, running / max(1, running_n),
                        scheduler.get_last_lr()[0], time.time() - t0,
                    )
                    running, running_n = 0.0, 0
                if opt_step % cfg.save_every == 0 and step % len(train) != 0:
                    write_checkpoint(step, {"epoch": epoch, "kind": "interval"})

        # Epoch boundary: evaluate, checkpoint, and stop early if dev stops moving.
        dev_metrics = evaluate(model, tokenizer, dev, cfg)
        state["history"].append(
            {"epoch": epoch, "micro_step": step, "dev": dev_metrics,
             "seconds": round(time.time() - t0, 1)}
        )
        log.info("epoch %d done: dev_loss %.4f (%s)", epoch, dev_metrics["loss"], dev_metrics)
        improved = dev_metrics["loss"] < state["best_dev"] - 1e-4
        if improved:
            state["best_dev"] = dev_metrics["loss"]
            state["best_epoch"] = epoch
            state["epochs_without_improvement"] = 0
        else:
            state["epochs_without_improvement"] += 1
        # The early-stop decision is taken before the checkpoint, so that the
        # checkpoint records it. Taking it afterwards left every checkpoint
        # claiming the run was still going, and a relaunch then trained on past
        # the stop.
        if state["epochs_without_improvement"] >= cfg.patience:
            state["stopped_early"] = True
        # `best/` is written before the checkpoint that records it. A kill in
        # between leaves a `best/` the state does not yet claim, which the resumed
        # run simply rewrites; the reverse order would leave a recorded best that
        # no longer exists on disk.
        if improved:
            save_best()
        write_checkpoint(step, {"epoch": epoch, "kind": "epoch", "dev_loss": dev_metrics["loss"]})
        write_summary()
        if state["stopped_early"]:
            log.info(
                "early stop after epoch %d: %d epochs without dev improvement",
                epoch, state["epochs_without_improvement"],
            )
            break

    state["finished"] = True
    write_summary()
    log.info(
        "done: best dev loss %.4f at epoch %d in %.1fs",
        state["best_dev"], state["best_epoch"], time.time() - t0,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
