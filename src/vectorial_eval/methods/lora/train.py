#!/usr/bin/env python
"""Fit a LoRA adapter on the conditional target language model.

    scripts/cluster/launch.sh -j lora-target_lm-attn-r8 -- \
        src/vectorial_eval/methods/lora/train.py --variant target_lm --config attn-r8

What is being trained
---------------------
There is no paired data in this corpus, so this is not a translation model. The
objective is a language-model loss over **authentic train-split Reddit posts**,
conditioned on the shared context emitted by `vectorial-eval build-training`.
The adapter therefore carries the target platform's manner for an audience and a
topic; at inference the source LinkedIn post is supplied in the same context as
the content to be carried across. Loss is taken on the completion tokens only.
The prompt tokens are masked, because a loss on the context would train the
adapter to reproduce the template rather than the post.

Three conditionings exist (`--variant`): `target_lm` (room and topic only),
`target_lm_paired` (plus a retrieved same-cell source post), and
`target_lm_aspect` (plus the aspects the post foregrounds). They are trained
separately and reported separately, per docs/09.

Scale
-----
The training set is 337 posts for the first two variants and 186 for the third.
At that size overfitting is the expected outcome rather than a surprise, so this
script measures the dev loss of the *untrained* base model before the first
optimiser step and again after every epoch, keeps the adapter from the best
epoch, and stops when the dev loss has failed to improve for `--patience`
epochs. The full curve is written to `metrics.json` next to the checkpoints so a
null result can be shown rather than asserted.

Memory
------
4-bit NF4, batch size 1, sequence length 768, gradient accumulation 8, gradient
checkpointing on. These are the measured settings in docs/09; bf16 at this
sequence length peaks at 22.6 GB of 23.5 and dies as soon as another process
joins the card. Every method in the study uses the same quantisation so that it
is a constant rather than a difference between columns.

Interruption
------------
The cards are shared and jobs are killed without warning. A checkpoint is
written every `--save-every` optimiser steps and at every epoch boundary through
`scripts/cluster/checkpointing.py`, and relaunching the identical command
resumes from the last complete one. Only the adapter and the optimiser state are
saved, never a merged model: the shared disk is at 91 percent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts" / "cluster") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "cluster"))

import torch  # noqa: E402
from checkpointing import resume_or_start, save_checkpoint  # noqa: E402
from peft import (  # noqa: E402
    LoraConfig,
    PeftModel,
    get_peft_model,
    load_peft_weights,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: E402

from vectorial_eval.methods.lora.prior_prompt import (  # noqa: E402
    PRIOR_PROMPT_SCHEMA,
    PRIOR_PROMPT_TOKEN_BUDGET,
    PRIOR_SECTION_TOKEN_BUDGET,
    render_prior_sections,
)
from vectorial_eval.methods.lora.prompting import prompt_tail, wrap_prompt  # noqa: E402

log = logging.getLogger("lora.train")

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

ATTN_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP_MODULES = ["gate_proj", "up_proj", "down_proj"]


@dataclass(frozen=True)
class LoraPreset:
    """One point of the sweep.

    The axes that matter at this data scale are capacity (rank and which modules
    carry an adapter) and regularisation (dropout). The sweep is deliberately
    three points on the capacity axis rather than a grid: with 337 examples the
    question is whether any amount of added capacity helps, and a wide search
    over a set this small would select on dev noise.
    """

    key: str
    r: int
    alpha: int
    dropout: float
    modules: list[str]
    lr: float


PRESETS: dict[str, LoraPreset] = {
    # Smallest sensible adapter. If LoRA helps at this scale at all, the
    # least-capacity configuration is where it is most likely to show.
    "attn-r4": LoraPreset("attn-r4", 4, 8, 0.10, ATTN_MODULES, 1e-4),
    # The conventional default, attention only.
    "attn-r8": LoraPreset("attn-r8", 8, 16, 0.05, ATTN_MODULES, 1e-4),
    # Highest capacity in the sweep: attention and MLP. Included to make the
    # capacity/overfitting relationship measurable rather than assumed.
    "all-r16": LoraPreset("all-r16", 16, 32, 0.05, ATTN_MODULES + MLP_MODULES, 1e-4),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #


def load_records(path: Path, expect_split: str) -> list[dict]:
    """Read one conditional-LM file and verify it is the split it claims to be.

    The training file must contain train-split posts and nothing else. This is
    the invariant the whole study rests on, and it is cheap to check here, so it
    is checked here rather than assumed from the filename.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `vectorial-eval build-training`")
    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"{path} is empty")

    wrong = sorted({r["split"] for r in records} - {expect_split})
    if wrong:
        raise RuntimeError(
            f"{path} carries records from split(s) {wrong}, expected only "
            f"{expect_split!r}. Refusing to train: this would be leakage."
        )
    tail = prompt_tail(records[0]["platform"])
    off_template = [r["post_id"] for r in records if not r["prompt"].rstrip().endswith(tail.rstrip())]
    if off_template:
        raise RuntimeError(
            f"{path}: {len(off_template)} prompts do not end with the shared "
            f"template line {tail.strip()!r} (e.g. {off_template[0]}). The file was "
            "not produced by build_training and this method would not be "
            "comparable with the others."
        )
    records.sort(key=lambda r: r["post_id"])
    return records


def encode(tokenizer, record: dict, max_length: int, prompt_budget: int) -> dict:
    """Tokenise one record into ids and completion-only labels.

    Truncation policy. The completion is the authentic post and is what the loss
    is taken on, so the prompt yields first: a prompt over `prompt_budget` keeps
    its opening (the audience, domain, and topic lines) and its tail (the
    instruction and the assistant header) and drops the middle, which is the
    source post or the aspect list. The completion is then truncated from the
    right to fit, and its final token is forced to end-of-turn so that a
    truncated example still teaches the model to stop.
    """
    prompt_text = wrap_prompt(tokenizer, record["prompt"])
    p = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    if len(p) > prompt_budget:
        head = p[:48]
        p = head + p[-(prompt_budget - 48):]

    c = tokenizer(record["completion"], add_special_tokens=False)["input_ids"]
    c = c + [tokenizer.eos_token_id]
    budget = max_length - len(p)
    truncated = len(c) > budget
    if truncated:
        c = c[: max(1, budget)]
        c[-1] = tokenizer.eos_token_id

    ids = p + c
    labels = [-100] * len(p) + list(c)
    return {
        "input_ids": ids,
        "labels": labels,
        "n_prompt": len(p),
        "n_completion": len(c),
        "truncated": truncated,
        "post_id": record["post_id"],
        "cell_id": record["cell_id"],
        "heldout_cell": bool(record.get("heldout_cell")),
    }


def load_unsupervised_records(path: Path, expect_split: str) -> list[dict]:
    """Load author-grouped raw text for continued language-model training."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; build the Study 4 unsupervised corpus")
    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"{path} is empty")
    wrong = sorted({r["split"] for r in records} - {expect_split})
    if wrong:
        raise RuntimeError(f"{path} contains split(s) {wrong}, expected {expect_split!r}")
    required = {"record_id", "author_id", "platform", "audience", "text"}
    missing = sorted(required - set(records[0]))
    if missing:
        raise ValueError(f"{path} is missing fields: {missing}")
    records.sort(key=lambda r: r["record_id"])
    return records


def encode_unsupervised(
    tokenizer, record: dict, max_length: int, *, use_priors: bool = False
) -> dict:
    """Tokenise authentic text, optionally conditioned on train-only priors."""
    prompt_ids = []
    if use_priors:
        header, sections = render_prior_sections(record)
        prompt_ids = tokenizer(header, add_special_tokens=True)["input_ids"]

        # Give every available metadata family a bounded share of the context.
        # A single long profile summary or thread must not crowd out all other
        # priors. Canonical JSON also makes this representation deterministic.
        for section in sections:
            line_ids = tokenizer(section, add_special_tokens=False)["input_ids"]
            prompt_ids.extend(line_ids[:PRIOR_SECTION_TOKEN_BUDGET])
        text_marker = tokenizer("Text:\n", add_special_tokens=False)["input_ids"]
        available = max(0, PRIOR_PROMPT_TOKEN_BUDGET - len(text_marker))
        prompt_ids = prompt_ids[:available] + text_marker
    completion_ids = tokenizer(record["text"], add_special_tokens=False)["input_ids"]
    # Keep the authentic-text budget identical in the raw and prior arms. If
    # the raw arm consumed all 768 positions while the prior arm spent up to
    # 256 on conditioning, their token exposure would confound the ablation.
    budget = min(512, max(1, max_length - len(prompt_ids)))
    truncated = len(completion_ids) + 1 > budget
    completion_ids = completion_ids[: budget - 1] + [tokenizer.eos_token_id]
    ids = prompt_ids + completion_ids
    return {
        "input_ids": ids,
        "labels": [-100] * len(prompt_ids) + list(completion_ids),
        "n_prompt": len(prompt_ids),
        "n_completion": len(completion_ids),
        "truncated": truncated,
        "post_id": record["record_id"],
        "cell_id": record["audience"],
        "heldout_cell": False,
    }


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #


def build_model(
    preset: LoraPreset,
    *,
    four_bit: bool = True,
    initial_adapter: Path | None = None,
):
    quant = None
    if four_bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        quantization_config=quant,
        device_map={"": 0},
    )
    base.config.use_cache = False
    if four_bit:
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()

    if initial_adapter is not None:
        config_path = initial_adapter / "adapter_config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"initial adapter {initial_adapter} has no adapter_config.json"
            )
        initial_config = json.loads(config_path.read_text(encoding="utf-8"))
        initial_r = int(initial_config.get("r", -1))
        initial_modules = set(initial_config.get("target_modules") or [])
        if initial_r != preset.r or initial_modules != set(preset.modules):
            raise ValueError(
                f"initial adapter is r={initial_r} over {sorted(initial_modules)}, "
                f"but --config {preset.key} is r={preset.r} over "
                f"{sorted(preset.modules)}; use the matching preset so the run "
                "metadata cannot mislabel the loaded adapter"
            )
        model = PeftModel.from_pretrained(
            base, str(initial_adapter), is_trainable=True
        )
        return model

    cfg = LoraConfig(
        r=preset.r,
        lora_alpha=preset.alpha,
        lora_dropout=preset.dropout,
        target_modules=list(preset.modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, cfg)
    return model


def batch_loss(model, example: dict, device) -> tuple[torch.Tensor, int]:
    """Loss summed over the completion tokens of one example, and their count.

    Summed rather than averaged so that examples of different lengths can be
    accumulated into one gradient step and into one dev figure without short
    posts counting for more per token than long ones.
    """
    ids = torch.tensor([example["input_ids"]], device=device)
    labels = torch.tensor([example["labels"]], device=device)
    attn = torch.ones_like(ids)
    logits = model(input_ids=ids, attention_mask=attn).logits
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    loss = torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    n_tokens = int((shift_labels != -100).sum())
    return loss, n_tokens


@torch.no_grad()
def evaluate(model, examples: list[dict], device, *, tag: str) -> dict:
    """Token-weighted mean negative log likelihood over the completion tokens."""
    model.eval()
    total, tokens = 0.0, 0
    zero_shot_total, zero_shot_tokens = 0.0, 0
    for ex in examples:
        loss, n = batch_loss(model, ex, device)
        total += float(loss)
        tokens += n
        if ex["heldout_cell"]:
            zero_shot_total += float(loss)
            zero_shot_tokens += n
    model.train()
    out = {
        "split": tag,
        "nll": total / max(tokens, 1),
        "ppl": math.exp(min(total / max(tokens, 1), 20.0)),
        "n_examples": len(examples),
        "n_tokens": tokens,
    }
    if zero_shot_tokens:
        # The four zero-shot cells appear only in the test file. They contribute
        # no training signal by construction, so a test figure that mixes them
        # with the rest is not a like-for-like sample of the train distribution.
        out["zero_shot_cell_nll"] = zero_shot_total / zero_shot_tokens
        out["zero_shot_cell_tokens"] = zero_shot_tokens
    return out


def save_best(model, dest: Path) -> None:
    """Replace `dest` with the current adapter, atomically enough to survive a kill.

    Adapter only. A merged 8B model is 16 GB and the shared disk is at 91
    percent, so merging is never done here or by the transfer function.
    """
    staging = dest.with_name(dest.name + ".tmp")
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    model.save_pretrained(str(staging))
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    os.replace(staging, dest)


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="target_lm",
                    choices=["target_lm", "target_lm_paired", "target_lm_aspect",
                             "platform_lm", "platform_prior_lm"])
    ap.add_argument("--objective", default="conditional", choices=["conditional", "unsupervised"])
    ap.add_argument("--train-file", default="")
    ap.add_argument("--dev-file", default="")
    ap.add_argument(
        "--audience",
        default="",
        help="for an unsupervised objective, restrict both splits to one audience",
    )
    ap.add_argument("--config", default="attn-r8", choices=sorted(PRESETS))
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "training"))
    ap.add_argument("--ckpt-dir", default=os.environ.get("VECTORIAL_CKPT_DIR", ""))
    ap.add_argument(
        "--initial-adapter",
        default="",
        help="continue fitting an existing LoRA adapter; used for Study 4 Stage B",
    )
    ap.add_argument("--epochs", type=int, default=12, help="maximum; early stopping usually ends it sooner")
    ap.add_argument("--patience", type=int, default=3, help="epochs without dev improvement before stopping")
    ap.add_argument("--max-length", type=int, default=768)
    ap.add_argument("--prompt-budget", type=int, default=512)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=None, help="overrides the preset")
    ap.add_argument("--warmup", type=int, default=10, help="optimiser steps of linear warmup")
    ap.add_argument("--save-every", type=int, default=100, help="optimiser steps between checkpoints")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap train and dev examples. A smoke-test flag only; it is "
                         "recorded in run.json so a limited run cannot be mistaken "
                         "for a real one")
    ap.add_argument("--bf16", action="store_true",
                    help="load in bf16 instead of 4-bit NF4. An ablation only; "
                         "docs/09 measured this at 22.6 GB of 23.5 and it dies "
                         "if another process joins the card")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    preset = PRESETS[args.config]
    lr = args.lr if args.lr is not None else preset.lr
    ckpt_dir = Path(args.ckpt_dir or (REPO_ROOT / "runs" / "checkpoints" / f"lora-{args.variant}-{args.config}"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest_path = data_dir / "manifest.json"
    selection_source = "val"
    if args.objective == "unsupervised":
        if not args.train_file or not args.dev_file:
            raise ValueError("--objective unsupervised requires --train-file and --dev-file")
        train_recs = load_unsupervised_records(Path(args.train_file), "train")
        dev_recs = load_unsupervised_records(Path(args.dev_file), "dev")
        if args.audience:
            train_recs = [r for r in train_recs if r["audience"] == args.audience]
            dev_recs = [r for r in dev_recs if r["audience"] == args.audience]
            if not train_recs or not dev_recs:
                raise ValueError(
                    f"--audience {args.audience!r} has no records in one or both splits"
                )
        selection_source = "dev"
    else:
        train_recs = load_records(data_dir / f"{args.variant}.train.jsonl", "train")
        if manifest_path.exists():
            selection_source = json.loads(manifest_path.read_text()).get(
                "selection", {}
            ).get("source", "val")
        dev_recs = load_records(
            data_dir / f"{args.variant}.dev.jsonl", selection_source
        )
    if args.limit:
        train_recs = train_recs[: args.limit]
        dev_recs = dev_recs[: args.limit]
        log.warning("--limit %d: this is a smoke run, not a result", args.limit)
    id_field = "record_id" if args.objective == "unsupervised" else "post_id"
    train_ids = {r[id_field] for r in train_recs}
    dev_ids = {r[id_field] for r in dev_recs}
    if train_ids & dev_ids:
        raise RuntimeError(f"{len(train_ids & dev_ids)} post_ids appear in both train and dev")
    if args.objective == "unsupervised":
        train_authors = {r["author_id"] for r in train_recs}
        dev_authors = {r["author_id"] for r in dev_recs}
        if train_authors & dev_authors:
            raise RuntimeError(
                f"{len(train_authors & dev_authors)} author_ids appear in both train and dev"
            )

    grouping_field = "audience" if args.objective == "unsupervised" else "cell_id"
    log.info(
        "variant=%s config=%s train=%d dev=%d cells=%d device=%s",
        args.variant, args.config, len(train_recs), len(dev_recs),
        len({r[grouping_field] for r in train_recs}), device,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.objective == "unsupervised":
        use_priors = args.variant == "platform_prior_lm"
        train = [
            encode_unsupervised(tokenizer, r, args.max_length, use_priors=use_priors)
            for r in train_recs
        ]
        dev = [
            encode_unsupervised(tokenizer, r, args.max_length, use_priors=use_priors)
            for r in dev_recs
        ]
    else:
        train = [encode(tokenizer, r, args.max_length, args.prompt_budget) for r in train_recs]
        dev = [encode(tokenizer, r, args.max_length, args.prompt_budget) for r in dev_recs]
    n_trunc = sum(1 for e in train if e["truncated"])
    log.info(
        "tokenised: train %d completion tokens (%d examples truncated), dev %d",
        sum(e["n_completion"] for e in train), n_trunc,
        sum(e["n_completion"] for e in dev),
    )

    initial_adapter = Path(args.initial_adapter) if args.initial_adapter else None
    model = build_model(
        preset, four_bit=not args.bf16, initial_adapter=initial_adapter
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info("adapter parameters: %d trainable of %d (%.4f%%)", trainable, total, 100 * trainable / total)

    steps_per_epoch = max(1, math.ceil(len(train) / args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0
    )

    def lr_lambda(step: int) -> float:
        if step < args.warmup:
            return (step + 1) / max(1, args.warmup)
        prog = (step - args.warmup) / max(1, total_steps - args.warmup)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * min(prog, 1.0))))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    state = {
        "history": [],
        "best_nll": float("inf"),
        "best_epoch": -1,
        "epochs_without_improvement": 0,
        "stopped": False,
    }
    metrics_path = ckpt_dir / "metrics.json"
    best_dir = ckpt_dir / "best"

    def restore(ckpt) -> None:
        payload = ckpt.payload or {}
        adapter = ckpt.path / "adapter"
        if adapter.exists():
            # The saved weights are loaded into the live model rather than a new
            # PeftModel being built over the same base. `PeftModel.from_pretrained`
            # injects adapter layers into the module it is given, so calling it on
            # the base of an already-adapted model would nest a second adapter and
            # leave the optimiser pointing at parameters no longer in the graph.
            result = set_peft_model_state_dict(model, load_peft_weights(str(adapter)))
            if getattr(result, "unexpected_keys", None):
                raise RuntimeError(f"unexpected adapter keys on resume: {result.unexpected_keys[:3]}")
        if "optimizer" in payload:
            optim.load_state_dict(payload["optimizer"])
        if "scheduler" in payload:
            sched.load_state_dict(payload["scheduler"])
        for key in ("history", "best_nll", "best_epoch", "epochs_without_improvement", "stopped"):
            if key in payload:
                state[key] = payload[key]

    start_step, _ = resume_or_start(ckpt_dir, restore=restore)
    log.info("start_step=%d of %d (%d per epoch)", start_step, total_steps, steps_per_epoch)

    run_meta = {
        "base_model": BASE_MODEL,
        "initial_adapter": str(initial_adapter) if initial_adapter else None,
        "variant": args.variant,
        "objective": args.objective,
        "audience": args.audience or None,
        "config": asdict(preset) | {"lr": lr},
        "quantisation": "bf16" if args.bf16 else "nf4-4bit-double",
        "max_length": args.max_length,
        "prompt_budget": args.prompt_budget,
        "grad_accum": args.grad_accum,
        "per_device_batch_size": 1,
        "seed": args.seed,
        "max_epochs": args.epochs,
        "patience": args.patience,
        "steps_per_epoch": steps_per_epoch,
        "n_train": len(train),
        "n_dev": len(dev),
        "train_prompt_tokens": sum(example["n_prompt"] for example in train),
        "train_completion_tokens": sum(
            example["n_completion"] for example in train
        ),
        "dev_prompt_tokens": sum(example["n_prompt"] for example in dev),
        "dev_completion_tokens": sum(example["n_completion"] for example in dev),
        "n_train_cells": len({r[grouping_field] for r in train_recs}),
        "trainable_parameters": trainable,
        "loss": "sum over completion tokens; prompt masked with -100",
        "limit": args.limit or None,
    }
    if args.variant == "platform_prior_lm":
        run_meta["prior_prompt_schema"] = PRIOR_PROMPT_SCHEMA
    if args.objective == "unsupervised":
        run_meta["input_files"] = {
            "train": {
                "path": args.train_file,
                "sha256": sha256_file(Path(args.train_file)),
            },
            "dev": {
                "path": args.dev_file,
                "sha256": sha256_file(Path(args.dev_file)),
            },
        }
    else:
        run_meta["input_files"] = {
            split: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for split, path in {
                "train": data_dir / f"{args.variant}.train.jsonl",
                "dev": data_dir / f"{args.variant}.dev.jsonl",
            }.items()
        }
    (ckpt_dir / "run.json").write_text(json.dumps(run_meta, indent=2, sort_keys=True), encoding="utf-8")

    def write_metrics() -> None:
        metrics_path.write_text(
            json.dumps(
                {"run": run_meta, **{k: state[k] for k in state}},
                indent=2, sort_keys=True, default=str,
            ),
            encoding="utf-8",
        )

    def checkpoint(step: int, note: str) -> None:
        save_checkpoint(
            ckpt_dir,
            step,
            payload={
                "optimizer": optim.state_dict(),
                "scheduler": sched.state_dict(),
                **{k: state[k] for k in state},
            },
            meta={"note": note, "best_nll": state["best_nll"], "best_epoch": state["best_epoch"]},
            save_fn=lambda d: model.save_pretrained(str(d / "adapter")),
            keep_last=1,
        )
        write_metrics()

    # The dev loss before task-specific optimisation. In Stage B this includes
    # the Stage A adapter, so the comparison answers whether conditional fitting
    # improves over unsupervised adaptation rather than over the raw base model.
    if start_step == 0 and not state["history"]:
        t0 = time.time()
        if initial_adapter is None:
            with model.disable_adapter():
                base_dev = evaluate(model, dev, device, tag="dev")
            baseline_note = "base model, adapter disabled"
        else:
            base_dev = evaluate(model, dev, device, tag="dev")
            baseline_note = "initial Stage A adapter before conditional fitting"
        base_dev["epoch"] = 0
        base_dev["step"] = 0
        base_dev["note"] = baseline_note
        state["history"].append(base_dev)
        state["best_nll"] = base_dev["nll"]
        state["best_epoch"] = 0
        log.info("epoch 0 (base model): dev nll %.4f ppl %.2f [%.0fs]",
                 base_dev["nll"], base_dev["ppl"], time.time() - t0)
        write_metrics()

    model.train()
    step = start_step
    for epoch in range(start_step // steps_per_epoch, args.epochs):
        if state["stopped"]:
            break
        order = list(range(len(train)))
        random.Random(args.seed + epoch).shuffle(order)
        skip = max(0, step - epoch * steps_per_epoch)
        if skip:
            log.info("epoch %d: skipping %d already-completed optimiser steps", epoch + 1, skip)

        micro = 0
        run_loss, run_tokens = 0.0, 0
        epoch_loss, epoch_tokens = 0.0, 0
        for local_step in range(steps_per_epoch):
            batch = order[local_step * args.grad_accum : (local_step + 1) * args.grad_accum]
            if local_step < skip or not batch:
                continue
            optim.zero_grad(set_to_none=True)
            step_tokens = sum(train[i]["n_completion"] for i in batch)
            for i in batch:
                loss, n = batch_loss(model, train[i], device)
                (loss / max(step_tokens, 1)).backward()
                detached = float(loss.detach())
                run_loss += detached
                run_tokens += n
                epoch_loss += detached
                epoch_tokens += n
                micro += 1
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optim.step()
            sched.step()
            step += 1

            if step % 20 == 0:
                log.info(
                    "epoch %d step %d/%d train nll %.4f lr %.2e mem %.1fGB",
                    epoch + 1, step, total_steps, run_loss / max(run_tokens, 1),
                    sched.get_last_lr()[0],
                    torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0,
                )
                run_loss, run_tokens = 0.0, 0
            if args.save_every and step % args.save_every == 0:
                checkpoint(step, f"epoch {epoch + 1} step {step}")

        dev_metrics = evaluate(model, dev, device, tag="dev")
        dev_metrics["epoch"] = epoch + 1
        dev_metrics["step"] = step
        dev_metrics["train_nll"] = epoch_loss / max(epoch_tokens, 1)
        state["history"].append(dev_metrics)
        improved = dev_metrics["nll"] < state["best_nll"] - 1e-4
        log.info(
            "epoch %d: train nll %.4f dev nll %.4f ppl %.2f %s",
            epoch + 1, dev_metrics["train_nll"], dev_metrics["nll"], dev_metrics["ppl"],
            "(best)" if improved else f"(no improvement, best {state['best_nll']:.4f} "
                                      f"at epoch {state['best_epoch']})",
        )
        if improved:
            state["best_nll"] = dev_metrics["nll"]
            state["best_epoch"] = epoch + 1
            state["epochs_without_improvement"] = 0
            save_best(model, best_dir)
            (best_dir / "training_config.json").write_text(
                json.dumps(
                    run_meta | {"best_epoch": epoch + 1, "best_dev_nll": dev_metrics["nll"],
                                "history": state["history"]},
                    indent=2, sort_keys=True, default=str,
                ),
                encoding="utf-8",
            )
        else:
            state["epochs_without_improvement"] += 1
            if state["epochs_without_improvement"] >= args.patience:
                state["stopped"] = True
                log.info(
                    "early stop after epoch %d: %d epochs without dev improvement",
                    epoch + 1, state["epochs_without_improvement"],
                )
        checkpoint(step, f"end of epoch {epoch + 1}")
        if state["stopped"]:
            break

    if state["best_epoch"] == 0:
        # The honest outcome at this data scale, and the one worth reporting
        # plainly: no epoch of LoRA beat the untrained base model on dev.
        log.warning(
            "NO ADAPTER IMPROVED ON THE BASE MODEL. best dev nll %.4f is the "
            "base model's own score; %s holds no adapter.",
            state["best_nll"], best_dir,
        )
    else:
        log.info(
            "best adapter: epoch %d, dev nll %.4f (base model %.4f) -> %s",
            state["best_epoch"], state["best_nll"], state["history"][0]["nll"], best_dir,
        )
    state["base_dev_nll"] = state["history"][0]["nll"] if state["history"] else None
    state["finished"] = True
    write_metrics()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
