"""Steered generation, shared by the dev sweep and the transfer function.

The model is loaded once and held; a steering configuration is applied for the
duration of a batch through a forward hook and removed afterwards, so a single
loaded model serves an entire layer-by-alpha sweep.

The shared context is placed in the base model's chat template by the same
`wrap_prompt` helper used by LoRA and soft prompting. This keeps prompt wrapping
constant across the three adaptation mechanisms.

An instruct model continuing a prompt of this shape tends to write the post and
then start a second one, so the continuation is cut at the first marker that
begins a new example.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from ..lora.prompting import wrap_prompt
from .runtime import SteeringSpec, apply_steering, resolve_vector, steering_deltas

log = logging.getLogger(__name__)

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

#: Markers at which a continuation has stopped being the requested post and has
#: begun a new example. Cutting here is presentation, not selection: nothing is
#: dropped and no output is replaced by another.
STOP_MARKERS = (
    "\nAudience:",
    "\nDomain:",
    "\nTopic:",
    "\nPlatform:",
    "\n--- content to carry across",
    "\n--- emphasise these aspects",
    "\nWrite a LinkedIn post",
    "\nWrite a Reddit post",
)

_WS = re.compile(r"[ \t]+\n")


def clean_continuation(text: str) -> str:
    cut = len(text)
    for marker in STOP_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    return _WS.sub("\n", text[:cut]).strip()


@dataclass
class GenConfig:
    model_id: str = DEFAULT_MODEL
    load_in_4bit: bool = True
    max_new_tokens: int = 320
    temperature: float = 1.0
    top_p: float = 0.95
    #: Prompts longer than this are truncated from the left. The transfer
    #: function already caps the source post by characters, so this only ever
    #: fires on a pathological input.
    max_prompt_tokens: int = 1536
    batch_size: int = 4
    seed: int = 0
    initial_adapter: str | None = None

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "load_in_4bit": self.load_in_4bit,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_prompt_tokens": self.max_prompt_tokens,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "initial_adapter": self.initial_adapter,
        }


class SteeredGenerator:
    """Holds the base model and generates under a steering specification."""

    def __init__(self, artifact: dict, cfg: GenConfig | None = None):
        self.artifact = artifact
        self.cfg = cfg or GenConfig()
        self.model = None
        self.tok = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = self.cfg.model_id or self.artifact.get("base_model") or DEFAULT_MODEL
        if self.artifact.get("base_model") and model_id != self.artifact["base_model"]:
            raise ValueError(
                f"artifact was fitted on {self.artifact['base_model']!r} but generation "
                f"was asked for {model_id!r}; the vectors do not transfer between models"
            )
        tok = AutoTokenizer.from_pretrained(model_id)
        tok.padding_side = "left"
        tok.truncation_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        kwargs: dict = {"dtype": torch.bfloat16, "device_map": {"": 0}}
        if self.cfg.load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        initial_adapter = self.cfg.initial_adapter or self.artifact.get(
            "initial_adapter"
        )
        if initial_adapter:
            from pathlib import Path

            from peft import PeftModel

            adapter = Path(initial_adapter)
            if not (adapter / "adapter_config.json").exists():
                raise FileNotFoundError(
                    f"steering requires missing Stage A adapter {adapter}"
                )
            self.model = PeftModel.from_pretrained(
                self.model, str(adapter), is_trainable=False
            )
        self.model.eval()
        self.tok = tok

    def _generate_batch(
        self, prompts: list[str], draw: int, deltas: dict | None = None
    ) -> list[str]:
        """Generate one continuation per prompt under `deltas`.

        The steering hooks are installed here rather than around the call so
        that a backend which cannot express the intervention as a forward hook —
        `ndif_backend.NDIFSteeredGenerator`, where it has to be declared inside a
        remote trace — can override this one method and inherit everything else.
        """
        import torch

        enc = self.tok(
            [wrap_prompt(self.tok, prompt) for prompt in prompts],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_prompt_tokens,
        ).to(self.model.device)
        # The draw index enters the seed, so draw k of a task is never a copy of
        # draw 0. This is the same requirement the LLM cache key carries.
        torch.manual_seed(self.cfg.seed + 1_000_003 * draw + len(prompts))
        with apply_steering(self.model, deltas or {}):
            out = self.model.generate(
                **enc,
                do_sample=self.cfg.temperature > 0,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                max_new_tokens=self.cfg.max_new_tokens,
                pad_token_id=self.tok.pad_token_id,
            )
        new = out[:, enc["input_ids"].shape[1] :]
        return [clean_continuation(t) for t in self.tok.batch_decode(new, skip_special_tokens=True)]

    def generate(
        self,
        items: list[dict],
        spec: SteeringSpec,
        draw: int = 0,
    ) -> list[dict]:
        """Generate one continuation per item.

        `items` are dicts with `prompt` and optional `cell_id`. Returns one
        result dict per item, in the input order, always including failures:
        `{text, ok, error, steering}`.
        """
        if self.model is None:
            self.load()

        results: list[dict | None] = [None] * len(items)
        groups: dict[str, list[int]] = {}
        for i, item in enumerate(items):
            if spec.scope == "cell":
                key = f"cell:{item.get('cell_id')}"
            elif spec.scope == "decomposed":
                key = f"audience:{item.get('audience_id')}"
            else:
                key = "__global__"
            groups.setdefault(str(key), []).append(i)

        for key, idxs in groups.items():
            cell_id = key.removeprefix("cell:") if key.startswith("cell:") else None
            audience_id = (
                key.removeprefix("audience:") if key.startswith("audience:") else None
            )
            # Resolving the vector and scaling it can fail on a misconfiguration,
            # for instance a layer index the artifact does not hold. That must be
            # one recorded failure per item rather than an exception that costs
            # the caller every output in the batch (invariant 6).
            try:
                vector, prov = resolve_vector(
                    self.artifact, cell_id, spec, audience_id=audience_id
                )
                deltas = steering_deltas(self.artifact, vector, spec) if spec.alpha != 0 else {}
            except Exception as exc:  # noqa: BLE001 - never drop a task
                log.error("could not resolve steering for %s: %s", key, exc)
                for i in idxs:
                    results[i] = {
                        "text": "",
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "steering": {"vector_scope": spec.scope, "vector_cell_id": cell_id},
                    }
                continue
            for start in range(0, len(idxs), self.cfg.batch_size):
                chunk = idxs[start : start + self.cfg.batch_size]
                prompts = [items[i]["prompt"] for i in chunk]
                try:
                    texts = self._generate_batch(prompts, draw, deltas)
                    for i, text in zip(chunk, texts, strict=True):
                        results[i] = {
                            "text": text,
                            "ok": bool(text.strip()),
                            "error": None if text.strip() else "empty_generation",
                            "steering": prov,
                        }
                except Exception as exc:  # noqa: BLE001 - never drop a task
                    log.warning("generation failed for %d prompts: %s", len(chunk), exc)
                    for i in chunk:
                        results[i] = {
                            "text": "",
                            "ok": False,
                            "error": f"{type(exc).__name__}: {exc}",
                            "steering": prov,
                        }
        return [
            r if r is not None else {"text": "", "ok": False, "error": "not_generated", "steering": {}}
            for r in results
        ]
