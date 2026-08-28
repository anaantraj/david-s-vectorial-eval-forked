"""Activation steering with the forward pass on NDIF: the scale arm.

`steering` answers "does adding a platform contrast to the residual stream move
generation toward the target platform?" on one 8B model, because one 23.5 GB card
is what the study has. This function asks the same question of `Llama-3.1-70B-Instruct`
and `Llama-3.1-405B-Instruct`, whose residual streams NDIF exposes for read and
write. A generation API cannot serve this arm at any price: the intervention is
inside the model.

It subclasses `SteeringTransfer` and replaces only the generator. The prompt
construction, the draw loop, the failure accounting and the output shape are the
same code, so `steering` and `steering_ndif` differ in the model and in nothing
else — which is the only way the two columns can be read against each other.

Configuration mirrors `steering`. The artifact must have been fitted against the
same NDIF model by `methods.steering.fit_steering_ndif`; an artifact fitted
locally on 8B has 4096 columns and will be refused rather than broadcast into a
model whose stream is 8192 or 16384 wide.

    VECTORIAL_STEERING_ARTIFACT=runs/steering/steering.target_lm.ndif-Llama-3.1-70B-Instruct.pt \\
    VECTORIAL_NDIF_MODEL=meta-llama/Llama-3.1-70B-Instruct \\
    NDIF_API_KEY=... .venv/bin/vectorial-eval transfer --fn steering_ndif --split heldout
"""

from __future__ import annotations

import logging
import os

from ..data.build_training import DEV_SELECTION_CAVEAT
from .base import register
from .steering import SteeringTransfer

log = logging.getLogger(__name__)

#: Which NDIF model to run against. The artifact's `base_model` wins when the two
#: disagree, because the vectors belong to the stream they were read from.
ENV_MODEL = "VECTORIAL_NDIF_MODEL"

DEFAULT_MODEL = "meta-llama/Llama-3.1-70B-Instruct"


@register("steering_ndif")
class NDIFSteeringTransfer(SteeringTransfer):
    name = "steering_ndif"

    def __init__(
        self,
        ndif_model: str | None = None,
        concurrency: int = 8,
        remote_seed: bool = True,
        require_chat_template: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ndif_model = ndif_model or os.environ.get(ENV_MODEL) or DEFAULT_MODEL
        self.concurrency = int(concurrency)
        self.remote_seed = bool(remote_seed)
        self.require_chat_template = bool(require_chat_template)

    def _ensure_loaded(self) -> bool:
        if self._gen is not None:
            return True
        if not self._ensure_artifact():
            return False
        try:
            from ..methods.steering.ndif_backend import (
                NDIFConfig,
                NDIFSteeredGenerator,
            )

            # The artifact is authoritative. Running vectors fitted on one model
            # against another is not a degraded result, it is a shape error at
            # best and silent nonsense at worst.
            fitted = self._artifact.get("base_model")
            if fitted and fitted != self.ndif_model:
                log.warning(
                    "%s: artifact was fitted on %s, not %s; using %s",
                    self.name, fitted, self.ndif_model, fitted,
                )
                self.ndif_model = fitted

            self._gen = NDIFSteeredGenerator(
                self._artifact,
                NDIFConfig(
                    model_id=self.ndif_model,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    batch_size=self.batch_size,
                    seed=self.seed,
                    remote_seed=self.remote_seed,
                    require_chat_template=self.require_chat_template,
                ),
            )
            self._gen.load()
            return True
        except Exception as exc:  # noqa: BLE001 - a failed load is a recorded failure
            self._load_error = f"{type(exc).__name__}: {exc}"
            log.error("%s: could not load the NDIF backend: %s", self.name, self._load_error)
            return False

    def describe(self) -> dict:
        out = super().describe()
        cfg = getattr(self._gen, "cfg", None)
        out.update(
            {
                "method": "activation steering (difference of means), forward pass on NDIF",
                "backend": "ndif",
                "ndif_model": self.ndif_model,
                # NDIF serves its own precision. Reporting False here would file
                # this arm under the study's 4-bit constant, which it does not share.
                "load_in_4bit": None,
                "chat_wrapped": getattr(cfg, "chat_wrapped", None),
                "remote_seed": self.remote_seed,
                "ndif_deployment": getattr(cfg, "deployment", {}),
                "nnsight_version": _nnsight_version(),
            }
        )
        out["caveats"] = [
            DEV_SELECTION_CAVEAT,
            "The layer, alpha and scope were selected by maximising centroid "
            "cosine against the authentic val Reddit pool, which is part of the "
            "heldout reference pool this method is scored against.",
            "The forward pass ran on an NDIF deployment in NDIF's own precision, "
            "not under the 4-bit NF4 quantisation every local method shares, so "
            "this column is not directly comparable with the local columns. It is "
            "comparable with the local steering column only as a scale contrast.",
            "Sampling is seeded on the NDIF worker rather than in this process. "
            "Draws are reproducible against the same deployment and not "
            "guaranteed across a redeploy.",
        ]
        return out


def _nnsight_version() -> str | None:
    try:
        import nnsight

        return getattr(nnsight, "__version__", None)
    except ImportError:
        return None
