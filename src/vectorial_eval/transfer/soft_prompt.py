"""Soft-prompt transfer: a trained continuous prompt over a frozen base model.

`vectorial_eval.methods.soft_prompt.train` fits a small set of continuous prompt
embeddings on real train-split Reddit posts, conditioned on the audience room and
the topic. Nothing else in the base model changes. This function loads those
embeddings and generates.

What the trained parameters and the context each supply
-------------------------------------------------------
There is no paired data in this project, so the soft prompt was never trained on
a source post. It was trained to produce authentic target-platform text given the
shared context in `build_training.TARGET_LM_PROMPT`. At inference the same
template is rendered with its content block filled by the source LinkedIn post,
so the context supplies the content to be carried across and the trained
parameters supply the manner of the target platform.

The `target_lm` variant therefore has a structural mismatch that is inherent to
the data rather than to this implementation: the content block is empty in
training and populated at inference, and a model trained that way may ignore the
source post, which `semantic.source_similarity` measures directly. The
`target_lm_paired` and `target_lm_aspect` variants supply a content block during
training as well. The variant is recorded in `describe()` and in every output's
`meta`, because a score without its variant is not interpretable.

Running it
----------
The base model is an 8B checkpoint, so this function runs on the cluster rather
than on the Mac:

    ssh cthulhu1.ist.berkeley.edu
    cd ~/Projects/vectorial
    CUDA_VISIBLE_DEVICES=<free gpu> HF_HOME=~/.cache/huggingface \
      VECTORIAL_SOFT_PROMPT_DIR=runs/checkpoints/soft_prompt_target_lm/best \
      ~/micromamba/envs/vectorial/bin/python -m vectorial_eval.cli transfer \
      --fn soft_prompt --split heldout --n-samples 4

Failure handling
----------------
One `TransferOutput` is emitted per task and per draw, including on failure: a
generation that raises records `ok=False` with empty text and the reason in
`meta`. A missing or unreadable artifact is a different matter and raises at
construction, because in that case every output would be empty and the run would
report a uniformly degenerate method rather than a missing file.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

from ..data.build_training import DEV_SELECTION_CAVEAT, render_prompt
from ..data.schema import TransferOutput, TransferTask
from ..methods.soft_prompt.prompting import wrap_prompt
from .base import Corpus, TransferFunction, register

log = logging.getLogger(__name__)

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

#: Where `train.py` writes the selected prompt. `--ckpt-dir` on the training
#: launch changes the parent; `best/` is fixed.
DEFAULT_CKPT_ROOT = Path("runs/checkpoints")


def default_adapter_dir(variant: str) -> Path:
    return DEFAULT_CKPT_ROOT / f"soft_prompt_{variant}" / "best"


def _draw_seed(base_seed: int, task_id: str, sample_index: int) -> int:
    """A seed that depends on the task and the draw index.

    The draw index has to enter the seed for the reason invariant 5 gives for the
    LLM cache key: without it every draw of a task repeats draw 0. The task id
    has to enter it as well, so that the draws of two different tasks are not
    taken from the same point in one random stream. This is the same function the
    LoRA method uses, so the two methods differ in their adaptation mechanism and
    not in how their candidate pools were sampled.
    """
    h = hashlib.blake2b(
        f"{base_seed}:{task_id}:{sample_index}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(h, "big") % (2**31 - 1)


@register("soft_prompt")
class SoftPromptTransfer(TransferFunction):
    """Generate with a trained soft prompt prepended to a frozen base model."""

    name = "soft_prompt"

    def __init__(
        self,
        adapter_dir: str | os.PathLike | None = None,
        variant: str = "target_lm",
        n_samples: int = 1,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_new_tokens: int = 512,
        batch_size: int = 1,
        base_model: str | None = None,
        load_in_4bit: bool = True,
        seed: int = 0,
        name: str | None = None,
        **_,
    ):
        self.variant = variant
        self.n_samples = int(n_samples)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_new_tokens = int(max_new_tokens)
        self.batch_size = max(1, int(batch_size))
        self.load_in_4bit = bool(load_in_4bit)
        self.seed = int(seed)
        if name:
            self.name = name

        env_dir = os.environ.get("VECTORIAL_SOFT_PROMPT_DIR")
        self.adapter_dir = Path(adapter_dir or env_dir or default_adapter_dir(variant))
        self.meta = self._read_meta()
        self.base_model = base_model or self.meta.get("base_model") or BASE_MODEL
        self.initial_adapter = self.meta.get("initial_adapter") or (
            self.meta.get("config") or {}
        ).get("initial_adapter")
        # The variant the artifact was actually trained on wins over the argument,
        # so a mislabelled run cannot be reported under the wrong variant.
        trained_variant = (self.meta.get("config") or {}).get("variant")
        if trained_variant and trained_variant != self.variant:
            log.warning(
                "artifact at %s was trained on variant %s, not %s; using %s",
                self.adapter_dir, trained_variant, self.variant, trained_variant,
            )
            self.variant = trained_variant
        self._model = None
        self._tokenizer = None

    # -- artifact ---------------------------------------------------------- #
    def _read_meta(self) -> dict:
        """Read the artifact's metadata, raising if it is not there.

        Failing here rather than at generation time is deliberate. An empty
        output for every task is indistinguishable in the report from a method
        that generates nothing useful, so a missing artifact must not be able to
        masquerade as a result.
        """
        if not self.adapter_dir.is_dir():
            raise FileNotFoundError(
                f"no soft-prompt artifact at {self.adapter_dir}. Train one with "
                f"src/vectorial_eval/methods/soft_prompt/train.py, or point "
                f"VECTORIAL_SOFT_PROMPT_DIR at an existing best/ directory."
            )
        meta_path = self.adapter_dir / "soft_prompt_meta.json"
        if meta_path.exists():
            return json.loads(meta_path.read_text(encoding="utf-8"))
        log.warning("%s has no soft_prompt_meta.json", self.adapter_dir)
        return {}

    def _load(self):
        """Load the frozen base model and attach the trained prompt embeddings."""
        if self._model is not None:
            return
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Generation pads on the left so that a batch's rightmost token is the
        # last real token of every sequence.
        tokenizer.padding_side = "left"

        kwargs = {"device_map": {"": 0}, "dtype": torch.bfloat16}
        if self.load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        try:
            base = AutoModelForCausalLM.from_pretrained(self.base_model, **kwargs)
        except TypeError:  # transformers < 5 spells the dtype argument differently
            kwargs["torch_dtype"] = kwargs.pop("dtype")
            base = AutoModelForCausalLM.from_pretrained(self.base_model, **kwargs)

        if self.initial_adapter:
            initial = Path(self.initial_adapter)
            if not (initial / "adapter_config.json").exists():
                raise FileNotFoundError(
                    f"soft prompt requires missing Stage A adapter {initial}"
                )
            base = PeftModel.from_pretrained(base, str(initial), is_trainable=False)
        model = PeftModel.from_pretrained(base, str(self.adapter_dir))
        model.eval()
        self._model, self._tokenizer = model, tokenizer
        log.info(
            "soft prompt loaded from %s (%s virtual tokens, dev loss %s)",
            self.adapter_dir,
            (self.meta.get("config") or {}).get("n_virtual_tokens"),
            self.meta.get("dev_loss"),
        )

    # -- generation -------------------------------------------------------- #
    def _prompt(self, task: TransferTask) -> str:
        """The shared template, with the source post in its content block."""
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

        tok, model = self._tokenizer, self._model
        texts = [wrap_prompt(tok, p) for p in prompts]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(next(model.parameters()).device) for k, v in enc.items()}
        # The seed is `_draw_seed(seed, task_id, draw)`: the draw index enters it,
        # so draw k of a task is a different sample rather than a repeat of draw
        # 0, and the task id enters it, so two tasks are not drawn from the same
        # point in one random stream.
        torch.manual_seed(seed)
        with torch.no_grad():
            out = model.generate(
                **enc,
                do_sample=self.temperature > 0,
                temperature=self.temperature,
                top_p=self.top_p,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        # A prompt-tuned model is generated from `inputs_embeds`, and whether
        # `generate` then returns the prompt tokens alongside the continuation
        # depends on the peft and transformers versions. Detecting it by
        # comparing against the input, rather than assuming either behaviour,
        # is what keeps a version bump from silently truncating every output.
        n_input = enc["input_ids"].shape[1]
        echoed = out.shape[1] > n_input and bool(
            torch.equal(out[:, :n_input], enc["input_ids"])
        )
        completions = []
        for row in out:
            ids = row[n_input:] if echoed else row
            completions.append(tok.decode(ids, skip_special_tokens=True).strip())
        return completions

    def run(self, tasks: list[TransferTask], corpus: Corpus) -> list[TransferOutput]:
        self._load()
        prompts = [self._prompt(t) for t in tasks]
        common = {
            "variant": self.variant,
            "base_model": self.base_model,
            "adapter_dir": str(self.adapter_dir),
            "initial_adapter": self.initial_adapter,
            "n_virtual_tokens": (self.meta.get("config") or {}).get("n_virtual_tokens"),
            "dev_loss": self.meta.get("dev_loss"),
            "temperature": self.temperature,
            "load_in_4bit": self.load_in_4bit,
        }

        results: dict[tuple[int, int], tuple[str, bool, str | None]] = {}
        for k in range(self.n_samples):
            for start in range(0, len(tasks), self.batch_size):
                chunk = list(range(start, min(start + self.batch_size, len(tasks))))
                seed = _draw_seed(self.seed, tasks[chunk[0]].task_id, k)
                try:
                    texts = self._generate([prompts[i] for i in chunk], seed)
                    # A batch that comes back the wrong length is a failure of the
                    # batch, not a reason to abandon the run: zipping it strictly
                    # here would raise out of `run` and drop every task, which is
                    # exactly what invariant 6 forbids.
                    if len(texts) != len(chunk):
                        raise RuntimeError(
                            f"generate returned {len(texts)} texts for {len(chunk)} prompts"
                        )
                except Exception as exc:  # noqa: BLE001 - one failure must not drop tasks
                    log.exception("soft-prompt generation failed for tasks %s", chunk)
                    for i in chunk:
                        results[(i, k)] = ("", False, f"{type(exc).__name__}: {exc}")
                    continue
                for i, text in zip(chunk, texts, strict=True):
                    ok = bool(text.strip())
                    results[(i, k)] = (text, ok, None if ok else "empty generation")

        out: list[TransferOutput] = []
        for i, task in enumerate(tasks):
            for k in range(self.n_samples):
                text, ok, error = results.get((i, k), ("", False, "not generated"))
                out.append(
                    TransferOutput(
                        task_id=task.task_id,
                        cell_id=task.cell_id,
                        transfer_fn=self.name,
                        output_text=text,
                        meta={
                            **common,
                            "sample_index": k,
                            "seed": _draw_seed(self.seed, task.task_id, k),
                            "ok": ok,
                            "error": error,
                        },
                    )
                )
        n_failed = sum(1 for o in out if not o.meta.get("ok"))
        if n_failed:
            log.warning("%s: %d/%d generations failed", self.name, n_failed, len(out))
        return out

    def describe(self) -> dict:
        return {
            "name": self.name,
            "class": type(self).__name__,
            "method": "soft prompt tuning (peft PromptTuningConfig)",
            "variant": self.variant,
            "base_model": self.base_model,
            "adapter_dir": str(self.adapter_dir),
            "initial_adapter": self.initial_adapter,
            "load_in_4bit": self.load_in_4bit,
            "n_samples": self.n_samples,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_new_tokens": self.max_new_tokens,
            "training": self.meta,
            "caveats": [DEV_SELECTION_CAVEAT],
        }
