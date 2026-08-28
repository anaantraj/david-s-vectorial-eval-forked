"""Activation steering: the no-gradient arm of the adaptation comparison.

The artifact is a set of per-layer difference-in-means vectors between authentic
train-split Reddit posts and authentic train-split LinkedIn posts, fitted by
`vectorial_eval.methods.steering.fit_steering`. At inference the source LinkedIn
post is placed in the shared prompt and the vector, scaled by alpha and by the
typical residual-stream norm at that layer, is added to the residual stream while
the continuation is generated. Nothing else about the model is changed.

The reason to expect this to help is a reading of the harness's own diagnostics.
The existing rewriters report a TRM `rank_i2` of 0.068 to 0.129 against 0.261 for
`target_sample`, with `rank_i0` at 0.68 to 0.73. Candidates that sat inside the
reference cloud but too close together would show the opposite pattern, so the
error is one of location rather than of spread. Adding a constant to the residual
stream is a translation, which is the intervention that acts on location.

Two scopes are available. `global` uses one vector fitted on all 337 train-split
Reddit posts against all train-split LinkedIn posts. `cell` uses a vector fitted
within one bilateral cell, which is closer to the audience but is fitted on very
few posts; cells without enough posts on both sides fall back to the global
vector and `describe()` reports the coverage.

This module is imported by the harness at start-up on a machine that may have no
GPU and no torch, so every heavy import happens inside a method body. Loading the
model is deferred to the first call to `run`.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from ..config import PROJECT_ROOT
from ..data.build_training import DEV_SELECTION_CAVEAT, render_prompt
from ..data.schema import TransferOutput, TransferTask
from ..methods.steering.runtime import SteeringSpec, parse_layers
from .base import Corpus, TransferFunction, register

log = logging.getLogger(__name__)

DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "runs" / "steering"
DEFAULT_VARIANT = "target_lm"

#: `vectorial-eval transfer` passes only `n_samples` and `name` to a transfer
#: function, so the variant and the artifact have to reach this class some other
#: way. These environment variables are that way, and they mirror
#: `VECTORIAL_SOFT_PROMPT_DIR` in the soft-prompt method. Without them the three
#: variants required by docs/09 cannot be scored from the command line, since
#: every run would read the `target_lm` artifact.
ENV_VARIANT = "VECTORIAL_STEERING_VARIANT"
ENV_ARTIFACT = "VECTORIAL_STEERING_ARTIFACT"


def _artifact_path(variant: str, artifact: str | Path | None) -> Path:
    if artifact:
        return Path(artifact)
    return DEFAULT_ARTIFACT_DIR / f"steering.{variant}.pt"


def _chosen(path: Path) -> dict:
    """Hyperparameters selected on the dev split, if the sweep has been run."""
    import json

    chosen = path.parent / f"chosen.{path.stem}.json"
    if chosen.exists():
        try:
            return json.loads(chosen.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:  # noqa: PERF203
            log.warning("ignoring unreadable %s: %s", chosen, exc)
    return {}


@register("steering")
class SteeringTransfer(TransferFunction):
    name = "steering"

    def __init__(
        self,
        variant: str | None = None,
        artifact: str | Path | None = None,
        layer: int | str | None = None,
        alpha: float | None = None,
        scope: str | None = None,
        audience_alpha: float | None = None,
        n_samples: int = 1,
        max_new_tokens: int = 320,
        temperature: float = 1.0,
        source_chars: int | None = None,
        batch_size: int = 4,
        load_in_4bit: bool = True,
        seed: int = 0,
        name: str | None = None,
        **_ignored,
    ):
        self.variant = variant or os.environ.get(ENV_VARIANT) or DEFAULT_VARIANT
        self.artifact_path = _artifact_path(
            self.variant, artifact or os.environ.get(ENV_ARTIFACT)
        )
        selected = _chosen(self.artifact_path)
        if not selected:
            # Falling back to a layer and an alpha that were never selected on
            # dev is a silent configuration error, and the alpha that follows is
            # in the range the sweep found degrades content.
            log.warning(
                "%s: no chosen.%s.json beside %s; using unselected defaults "
                "layer=%s alpha=%s. Run methods.steering.sweep_steering first.",
                self.name, self.artifact_path.stem, self.artifact_path,
                layer if layer is not None else 16,
                alpha if alpha is not None else 1.0,
            )
        self.layers = parse_layers(
            layer if layer is not None else selected.get("layer", 16)
        )
        self.alpha = float(alpha if alpha is not None else selected.get("alpha", 1.0))
        self.scope = scope or selected.get("scope", "global")
        self.audience_alpha = float(
            audience_alpha
            if audience_alpha is not None
            else selected.get("audience_alpha", 1.0)
        )
        self.selected_on_dev = selected or None
        self.n_samples = max(1, int(n_samples))
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.source_chars = int(source_chars) if source_chars is not None else None
        self.batch_size = int(batch_size)
        self.load_in_4bit = bool(load_in_4bit)
        self.seed = int(seed)
        if name:
            self.name = name
        self._artifact: dict | None = None
        self._gen = None
        self._load_error: str | None = None

    # -- loading ---------------------------------------------------------- #

    def _spec(self) -> SteeringSpec:
        return SteeringSpec(
            layers=self.layers,
            alpha=self.alpha,
            scope=self.scope,
            audience_alpha=self.audience_alpha,
            source={
                "artifact": str(self.artifact_path),
                "variant": self.variant,
                "selected_on_dev": bool(self.selected_on_dev),
            },
        )

    def _ensure_artifact(self) -> bool:
        """Load the vectors only. Cheap, and does not touch the base model."""
        if self._artifact is not None:
            return True
        if self._load_error is not None:
            return False
        try:
            from ..methods.steering.runtime import load_artifact

            if not self.artifact_path.exists():
                raise FileNotFoundError(
                    f"{self.artifact_path} not found; fit it with "
                    "vectorial_eval.methods.steering.fit_steering"
                )
            self._artifact = load_artifact(self.artifact_path)
            # The variant the artifact was actually fitted on wins over the
            # label, so a run cannot be reported under the wrong conditioning.
            fitted = self._artifact.get("variant")
            if fitted and fitted != self.variant:
                log.warning(
                    "%s: artifact at %s was fitted on variant %s, not %s; using %s",
                    self.name, self.artifact_path, fitted, self.variant, fitted,
                )
                self.variant = fitted
            return True
        except Exception as exc:  # noqa: BLE001 - a failed load is a recorded failure
            self._load_error = f"{type(exc).__name__}: {exc}"
            log.error("%s: could not load steering artifact: %s", self.name, self._load_error)
            return False

    def _ensure_loaded(self) -> bool:
        if self._gen is not None:
            return True
        if not self._ensure_artifact():
            return False
        try:
            from ..methods.steering.generate import GenConfig, SteeredGenerator

            self._gen = SteeredGenerator(
                self._artifact,
                GenConfig(
                    model_id=self._artifact["base_model"],
                    load_in_4bit=self.load_in_4bit,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    batch_size=self.batch_size,
                    seed=self.seed,
                    initial_adapter=self._artifact.get("initial_adapter"),
                ),
            )
            self._gen.load()
            return True
        except Exception as exc:  # noqa: BLE001 - a failed load is a recorded failure
            self._load_error = f"{type(exc).__name__}: {exc}"
            log.error("%s: could not load steering artifact: %s", self.name, self._load_error)
            return False

    # -- describe --------------------------------------------------------- #

    def describe(self) -> dict:
        self._ensure_artifact()
        art = self._artifact or {}
        counts = art.get("counts", {})
        per_cell = counts.get("per_cell_posts", {})
        out = {
            "name": self.name,
            "class": type(self).__name__,
            "method": "activation steering (difference of means)",
            "variant": self.variant,
            "artifact": str(self.artifact_path),
            "artifact_sha256": art.get("sha256"),
            "base_model": art.get("base_model"),
            "initial_adapter": art.get("initial_adapter"),
            "load_in_4bit": self.load_in_4bit,
            "layers": list(self.layers),
            "alpha": self.alpha,
            "scope": self.scope,
            "audience_alpha": self.audience_alpha,
            "n_samples": self.n_samples,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "selected_on_dev": self.selected_on_dev,
            # How many posts fed each vector. A global vector fitted on hundreds
            # and a cell vector fitted on three are not the same object, so both
            # counts are recorded rather than one summary.
            "vector_posts": {
                "global": {
                    "n_target": art.get("global", {}).get("n_target"),
                    "n_source": art.get("global", {}).get("n_source"),
                },
                "per_cell": per_cell,
                "n_cells_with_vector": counts.get("n_cells_with_vector"),
                "n_cells_seen": counts.get("n_cells_seen"),
                "cells_too_thin": counts.get("cells_too_thin"),
                "min_cell_posts": counts.get("min_cell_posts"),
                "per_audience": counts.get("per_audience_posts", {}),
                "n_audiences_with_vector": counts.get("n_audiences_with_vector"),
                "n_audiences_seen": counts.get("n_audiences_seen"),
            },
            "fit_config": art.get("fit_config"),
            "source_chars": self.source_chars,
            "chat_wrapped": True,
            "caveats": [
                DEV_SELECTION_CAVEAT,
                # Stronger here than for the other two: the sweep maximises a
                # centroid cosine computed directly against the val reference
                # embeddings, rather than against a held-out loss.
                "The layer, alpha and scope were selected by maximising centroid "
                "cosine against the authentic val Reddit pool, which is part of "
                "the heldout reference pool this method is scored against.",
                "The fitted artifact and generation both apply the same chat "
                "template used by LoRA and soft prompting.",
            ],
        }
        if self._load_error:
            out["load_error"] = self._load_error
        return out

    # -- run -------------------------------------------------------------- #

    def _items(self, tasks: list[TransferTask]) -> list[dict]:
        return [
            {
                "cell_id": t.cell_id,
                "audience_id": t.room,
                "prompt": render_prompt(
                    room=t.room,
                    topic=t.topic,
                    domain=t.domain,
                    target_platform=t.target_platform,
                    source_text=(t.source_text[: self.source_chars]
                                 if self.source_chars is not None else t.source_text),
                    source_platform=t.source_platform,
                ),
            }
            for t in tasks
        ]

    def _failed(self, tasks: list[TransferTask], reason: str) -> list[TransferOutput]:
        return [
            TransferOutput(
                task_id=task.task_id,
                cell_id=task.cell_id,
                transfer_fn=self.name,
                output_text="",
                meta={
                    "ok": False,
                    "error": reason,
                    "sample_index": k,
                    "variant": self.variant,
                    "layers": list(self.layers),
                    "alpha": self.alpha,
                    "scope": self.scope,
                    "audience_alpha": self.audience_alpha,
                },
            )
            for task in tasks
            for k in range(self.n_samples)
        ]

    def run(self, tasks: list[TransferTask], corpus: Corpus) -> list[TransferOutput]:
        del corpus  # steering uses no exemplars; the vector already carries the fit
        if not self._ensure_loaded():
            # One output per task per draw, all recorded as failures. Dropping
            # them would bias every distributional metric.
            return self._failed(tasks, self._load_error or "load_failed")

        items = self._items(tasks)
        spec = self._spec()
        outputs: list[TransferOutput] = []
        log.info(
            "%s: generating %d candidates (%d tasks x %d samples), layers=%s alpha=%s scope=%s",
            self.name, len(tasks) * self.n_samples, len(tasks), self.n_samples,
            list(self.layers), self.alpha, self.scope,
        )
        # The draw index enters the sampling seed, so draw k is an independent
        # candidate rather than a copy of draw 0. A draw that raises for any
        # reason becomes one recorded failure per task rather than an exception
        # that costs the harness every output in the split (invariant 6).
        draws = []
        for k in range(self.n_samples):
            try:
                draws.append(self._gen.generate(items, spec, draw=k))
            except Exception as exc:  # noqa: BLE001 - never drop a task
                log.error("%s: draw %d failed entirely: %s", self.name, k, exc)
                draws.append(
                    [
                        {
                            "text": "",
                            "ok": False,
                            "error": f"{type(exc).__name__}: {exc}",
                            "steering": {},
                        }
                        for _ in items
                    ]
                )
        # Emitted task-major, matching the order the LLM rewriters use.
        for i, task in enumerate(tasks):
            for k in range(self.n_samples):
                res = draws[k][i]
                outputs.append(
                    TransferOutput(
                        task_id=task.task_id,
                        cell_id=task.cell_id,
                        transfer_fn=self.name,
                        output_text=res["text"] if res["ok"] else "",
                        meta={
                            "ok": res["ok"],
                            "error": res["error"],
                            "sample_index": k,
                            "variant": self.variant,
                            "layers": list(self.layers),
                            "alpha": self.alpha,
                            "scope": self.scope,
                            "audience_alpha": self.audience_alpha,
                            **res["steering"],
                        },
                    )
                )
        n_failed = sum(1 for o in outputs if not o.meta.get("ok"))
        if n_failed:
            log.warning("%s: %d/%d generations failed", self.name, n_failed, len(outputs))
        # Ordering is task-major within each draw; the harness keys on task_id.
        return outputs
