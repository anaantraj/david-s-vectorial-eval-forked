"""LoRA-adapted transfer — generation from an adapter fitted on real target posts.

What this function is
---------------------
`llm_rewrite` and `llm_fewshot` put everything in the prompt. This one puts the
target distribution in the weights. A LoRA adapter is fitted on
`meta-llama/Llama-3.1-8B-Instruct` over **authentic train-split Reddit posts**
conditioned on the shared context from `vectorial_eval.data.build_training`
(`vectorial_eval/methods/lora/train.py` does the fitting). At inference the same
context is rendered with the source LinkedIn post in its content block, and the
adapted model writes the post. The adapter supplies the manner of the target
platform; the prompt supplies the content to be carried across.

There is no paired data in this corpus, so nothing here was trained on a
source/target pair. That is deliberate: inventing pairs with an LLM would make
the study measure agreement with that LLM rather than with the real target
platform.

Choosing an artifact
--------------------
The adapter directory is not inferable from the split, so it is given
explicitly, either as a constructor argument or through
`VECTORIAL_LORA_ADAPTER`. The training variant the adapter was fitted on is read
back from `training_config.json` inside that directory and recorded in
`describe()` and on every output, because a LoRA number without its training
variant is not interpretable: the three variants condition on different context.

The inference context is identical for all three adapters. Only training
differed. That is what makes the variant a clean axis of comparison.

Running it
----------
The base model is 8B and this must run on a GPU, which in this project means the
cluster:

    ssh cthulhu1.ist.berkeley.edu
    cd ~/Projects/vectorial
    export VECTORIAL_LORA_ADAPTER=~/Projects/vectorial/runs/checkpoints/lora-target_lm_paired-attn-r8/best
    HF_HOME=~/.cache/huggingface PYTHONPATH=src \
      ~/micromamba/envs/vectorial/bin/python -m vectorial_eval.cli transfer \
      --fn lora --split heldout --tag paired-attn-r8

Failure policy
--------------
One `TransferOutput` per task and per draw, including on failure. A generation
that raises is retried once on its own, and a task that still fails is emitted
with `ok=False`, an empty string, and the reason in `meta`. Dropping it would
bias every distributional metric toward whatever the model happened to handle.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path

from ..data.build_training import DEV_SELECTION_CAVEAT, render_prompt
from ..data.schema import TransferOutput, TransferTask
from ..methods.lora.prompting import wrap_prompt
from .base import Corpus, TransferFunction, register

log = logging.getLogger(__name__)

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

#: Generation is sampled rather than greedy, at the same temperature the LLM
#: rewriters use, so that `--n-samples k` draws a genuine candidate pool rather
#: than k copies of one post.
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95
DEFAULT_MAX_NEW_TOKENS = 512


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    return int(raw) if raw else default


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    return float(raw) if raw else default


def _draw_seed(base_seed: int, task_id: str, sample_index: int) -> int:
    """A seed that depends on the task and the draw index.

    The draw index has to enter the seed for the same reason it has to enter the
    LLM cache key (invariant 5): without it every draw for a task is the identical
    post and the candidate pool is k copies of one output, silently.
    """
    h = hashlib.blake2b(
        f"{base_seed}:{task_id}:{sample_index}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(h, "big") % (2**31 - 1)


@register("lora")
class LoraTransfer(TransferFunction):
    name = "lora"

    def __init__(
        self,
        adapter_dir: str | os.PathLike | None = None,
        base_model: str = BASE_MODEL,
        n_samples: int | None = None,
        batch_size: int | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        four_bit: bool | None = None,
        seed: int = 17,
        name: str | None = None,
        base_only: bool = False,
        **_,
    ):
        self.base_only = bool(base_only)
        raw_adapter = adapter_dir or os.environ.get("VECTORIAL_LORA_ADAPTER", "")
        self.adapter_dir = Path(raw_adapter).expanduser() if raw_adapter else None
        if not self.base_only and self.adapter_dir is None:
            raise ValueError(
                "lora: no adapter directory. Pass adapter_dir= or set "
                "VECTORIAL_LORA_ADAPTER to a directory written by "
                "vectorial_eval/methods/lora/train.py (its `best/` subdirectory)."
            )
        # An adapter directory that holds no adapter is refused here rather than
        # at the first generation. Discovering it inside `run` would produce a
        # complete file of `ok=False` rows, which is a valid record of a failure
        # but is indistinguishable at a glance from a model that generated
        # nothing, and it costs a full pass over the split to find out.
        if not self.base_only and not (self.adapter_dir / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"{self.adapter_dir} holds no adapter (no adapter_config.json). "
                "If training reported that no epoch improved on the base model, "
                "no adapter was saved, and that null result is the finding: do "
                "not substitute the base model for it here."
            )
        self.base_model = base_model
        # An explicit argument wins over the environment. The environment is the
        # route the CLI has to use, because it passes method-specific arguments
        # only to the LLM-backed functions, but a caller that says n_samples=k
        # must not be silently overridden by a variable left in the shell.
        self.n_samples = max(
            1,
            n_samples if n_samples is not None
            else _env_int("VECTORIAL_LORA_N_SAMPLES", 1),
        )
        self.batch_size = batch_size or _env_int("VECTORIAL_LORA_BATCH", 4)
        self.max_new_tokens = max_new_tokens or _env_int(
            "VECTORIAL_LORA_MAX_NEW_TOKENS", DEFAULT_MAX_NEW_TOKENS
        )
        self.temperature = (
            temperature if temperature is not None
            else _env_float("VECTORIAL_LORA_TEMPERATURE", DEFAULT_TEMPERATURE)
        )
        self.top_p = top_p if top_p is not None else _env_float("VECTORIAL_LORA_TOP_P", DEFAULT_TOP_P)
        # 4-bit by default because that is what the adapter was trained under
        # (docs/09). Evaluating a 4-bit-trained adapter in bf16 changes the
        # numerics it was fitted to and is an ablation, not the headline run.
        self.four_bit = (
            four_bit if four_bit is not None
            else os.environ.get("VECTORIAL_LORA_BF16", "") == ""
        )
        self.seed = seed
        if name:
            self.name = name
        self.training_config = self._read_training_config()
        self._model = None
        self._tokenizer = None

    # ---------------------------------------------------------------- setup --

    def _read_training_config(self) -> dict:
        if self.base_only:
            return {"variant": "base_none"}
        path = self.adapter_dir / "training_config.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - diagnostics only
            log.warning("could not read %s: %s", path, exc)
            return {}

    @property
    def variant(self) -> str:
        return str(self.training_config.get("variant", "unknown"))

    def _load(self):
        """Load base model and adapter once, on first use."""
        if self._model is not None:
            return self._model, self._tokenizer

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not self.base_only and not (self.adapter_dir / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"{self.adapter_dir} holds no adapter (no adapter_config.json). "
                "If training reported that no epoch improved on the base model, "
                "no adapter was saved, and that null result is the finding: do "
                "not substitute the base model for it here."
            )

        quant = None
        if self.four_bit:
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        tok = AutoTokenizer.from_pretrained(self.base_model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        # Left padding: with right padding the generated continuation would start
        # after the pad tokens of the shortest prompt in the batch.
        tok.padding_side = "left"

        base = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            dtype=torch.bfloat16,
            quantization_config=quant,
            device_map={"": 0} if torch.cuda.is_available() else None,
        )
        if self.base_only:
            model = base
        else:
            from peft import PeftModel

            model = PeftModel.from_pretrained(base, str(self.adapter_dir))
        model.eval()
        self._model, self._tokenizer = model, tok
        log.info(
            "%s: loaded %s%s (variant=%s)",
            self.name,
            self.base_model,
            "" if self.base_only else f" + {self.adapter_dir}",
            self.variant,
        )
        return model, tok

    # ------------------------------------------------------------- inference --

    def _prompt(self, task: TransferTask) -> str:
        """The shared context, with the source post in its content block.

        `render_prompt` is imported rather than reimplemented so that this
        function's context is byte-identical to the one the soft-prompt and
        steering methods use. A locally formatted prompt would confound the
        adaptation mechanism with the prompt.
        """
        return render_prompt(
            room=task.room,
            topic=task.topic,
            domain=task.domain,
            target_platform=task.target_platform,
            source_text=task.source_text,
            source_platform=task.source_platform,
        )

    def _generate(self, prompts: list[str], seed: int) -> list[str]:
        import torch

        model, tok = self._load()
        texts = [wrap_prompt(tok, p) for p in prompts]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        torch.manual_seed(seed)
        with torch.no_grad():
            out = model.generate(
                **enc,
                do_sample=self.temperature > 0,
                temperature=self.temperature,
                top_p=self.top_p,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=tok.pad_token_id,
            )
        new = out[:, enc["input_ids"].shape[1]:]
        texts = [tok.decode(row, skip_special_tokens=True).strip() for row in new]
        if len(texts) != len(prompts):
            # `generate` returns one row per prompt. If a version change ever
            # breaks that, the mismatch has to raise here, where the caller
            # records one failure per affected draw, rather than silently
            # shifting texts onto the wrong tasks.
            raise RuntimeError(
                f"generate returned {len(texts)} rows for {len(prompts)} prompts"
            )
        return texts

    def run(self, tasks: list[TransferTask], corpus: Corpus) -> list[TransferOutput]:
        # Each task is expanded into `n_samples` independent draws, each with its
        # own seed, so that the candidate pool is genuinely a pool.
        jobs = [(i, k) for i in range(len(tasks)) for k in range(self.n_samples)]
        results: dict[tuple[int, int], tuple[str, bool, str | None]] = {}
        #: The seed `generate` was actually called under for each draw. In a
        #: batch that is the seed of the first draw in the batch, so it is
        #: recorded rather than recomputed per draw, which would put a number in
        #: `meta` that no generation ever used.
        seeds: dict[tuple[int, int], int] = {}

        # Rendering the context is done here, per task, and a task that cannot be
        # rendered becomes a recorded failure. Rendering inside the batch loop
        # without a guard would let one malformed task raise out of `run` and
        # take every other task's output with it, which is the one thing
        # invariant 6 forbids.
        prompts: dict[int, str] = {}
        for i, task in enumerate(tasks):
            try:
                prompts[i] = self._prompt(task)
            except Exception as exc:  # noqa: BLE001 - recorded, never dropped
                log.warning("lora: task %s could not be rendered: %s", task.task_id, exc)
                for k in range(self.n_samples):
                    results[(i, k)] = ("", False, f"prompt_error: {type(exc).__name__}: {exc}")

        runnable = [(i, k) for i, k in jobs if i in prompts]
        started = time.time()
        for start in range(0, len(runnable), self.batch_size):
            chunk = runnable[start : start + self.batch_size]
            batch_prompts = [prompts[i] for i, _ in chunk]
            seed = _draw_seed(self.seed, tasks[chunk[0][0]].task_id, chunk[0][1])
            errors: dict[tuple[int, int], str] = {}
            try:
                texts = self._generate(batch_prompts, seed)
                for job in chunk:
                    seeds[job] = seed
            except Exception as exc:  # noqa: BLE001 - retry the batch one item at a time
                log.warning("lora: batch of %d failed (%s); retrying singly", len(chunk), exc)
                texts = []
                for (i, k), prompt in zip(chunk, batch_prompts, strict=True):
                    single_seed = _draw_seed(self.seed, tasks[i].task_id, k)
                    seeds[(i, k)] = single_seed
                    try:
                        texts.append(self._generate([prompt], single_seed)[0])
                    except Exception as exc2:  # noqa: BLE001 - recorded, never dropped
                        log.warning("lora: task %s draw %d failed: %s", tasks[i].task_id, k, exc2)
                        # The reason is carried into `meta`. Recording every
                        # failure as `empty_generation` would make a model that
                        # wrote nothing indistinguishable from a card that ran
                        # out of memory.
                        errors[(i, k)] = f"{type(exc2).__name__}: {exc2}"
                        texts.append("")
            if len(texts) != len(chunk):
                # Belt and braces. `_generate` already refuses to return the
                # wrong number of rows, but if it ever did, the draws it did not
                # cover must become recorded failures rather than an exception
                # that discards every task in the run.
                log.warning(
                    "lora: %d texts for a batch of %d; the remainder are recorded as failures",
                    len(texts), len(chunk),
                )
                for job in chunk[len(texts):]:
                    errors[job] = f"generate returned {len(texts)} rows for {len(chunk)} prompts"
                texts = list(texts) + [""] * (len(chunk) - len(texts))
            for (i, k), text in zip(chunk, texts[: len(chunk)], strict=True):
                if text.strip():
                    results[(i, k)] = (text.strip(), True, None)
                else:
                    results[(i, k)] = ("", False, errors.get((i, k), "empty_generation"))
            if (start // self.batch_size) % 10 == 0:
                log.info("lora: %d/%d draws (%.0fs)", start + len(chunk), len(runnable), time.time() - started)

        outputs = []
        for i, k in jobs:
            task = tasks[i]
            text, ok, err = results.get((i, k), ("", False, "not_generated"))
            outputs.append(
                TransferOutput(
                    task_id=task.task_id,
                    cell_id=task.cell_id,
                    transfer_fn=self.name,
                    output_text=text,
                    meta={
                        "base_model": self.base_model,
                        "adapter_dir": str(self.adapter_dir) if self.adapter_dir else None,
                        "training_variant": self.variant,
                        "lora_config": self.training_config.get("config"),
                        "best_epoch": self.training_config.get("best_epoch"),
                        "sample_index": k,
                        "seed": seeds.get((i, k)),
                        "ok": ok,
                        "error": err,
                    },
                )
            )
        n_failed = sum(1 for o in outputs if not o.meta["ok"])
        if n_failed:
            log.warning("lora: %d/%d generations failed or were empty", n_failed, len(outputs))
        return outputs

    def describe(self) -> dict:
        return {
            "name": self.name,
            "class": type(self).__name__,
            "base_model": self.base_model,
            "adapter_dir": str(self.adapter_dir) if self.adapter_dir else None,
            "training_variant": self.variant,
            "training_config": self.training_config.get("config"),
            "best_epoch": self.training_config.get("best_epoch"),
            "best_dev_nll": self.training_config.get("best_dev_nll"),
            "base_dev_nll": (self.training_config.get("history") or [{}])[0].get("nll"),
            "quantisation": "nf4-4bit-double" if self.four_bit else "bf16",
            "n_samples": self.n_samples,
            "batch_size": self.batch_size,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_new_tokens": self.max_new_tokens,
            "seed": self.seed,
            "caveats": [] if self.base_only else [DEV_SELECTION_CAVEAT],
        }


@register("llama_rewrite")
class LocalLlamaRewrite(LoraTransfer):
    """Direct instruction-tuned Llama baseline under the shared task prompt."""

    name = "llama_rewrite"

    def __init__(
        self,
        n_samples: int | None = None,
        name: str | None = None,
        **kwargs,
    ):
        kwargs.pop("adapter_dir", None)
        kwargs.pop("base_only", None)
        super().__init__(
            adapter_dir=None,
            base_only=True,
            n_samples=n_samples,
            name=name,
            **kwargs,
        )
