"""Remote steering on NDIF, the scale arm of the adaptation comparison.

Why this exists
---------------
Every steering number in Studies 2-4 is an 8B number, because the local ceiling
is one 23.5 GB card (docs/09) and the 128k-vocab logits tensor, not the weights,
sets it. That makes "the platform contrast is a translation in the residual
stream" a claim about one small model rather than a claim about the mechanism.
NDIF hosts `Llama-3.1-70B` and `Llama-3.1-405B-Instruct` with the residual stream
exposed for read *and* write, which is the one thing a generation API cannot give
us. Running the same difference-in-means construction there turns the claim into
a scale-generalisation result.

What is shared with the local path
----------------------------------
Everything except the two methods that touch torch directly. `resolve_vector`,
`steering_deltas`, the scope grouping, the per-item failure handling required by
invariant 6, `clean_continuation`, and `wrap_prompt` are all inherited from
`SteeredGenerator`. This is deliberate: the local and remote arms must differ in
where the forward pass happens and in nothing else, or the comparison between
them measures the harness rather than the model.

What is *not* the same, and must be reported
--------------------------------------------
* **Sampling reproducibility.** The local path seeds `torch` in-process, so draw
  `k` is reproducible. Remotely the RNG lives on the NDIF worker. We seed it
  inside the trace, which makes draws reproducible as long as NDIF keeps serving
  the same deployment, but that is a weaker guarantee than the local one and is
  recorded as `remote_seeded` rather than assumed.
* **No quantisation.** NDIF serves these models in their own precision. The local
  study fixes 4-bit NF4 as a constant across every method; that constant does not
  hold here, so a remote number is not directly comparable to a local one and
  `load_in_4bit` is recorded as None rather than False.
* **The checkpoint is theirs.** A remote result is only reproducible against the
  deployment that produced it, so the NDIF status entry and the nnsight version
  are written into the artifact and into every output's provenance.

Access notes, current as of the probe in `docs/11-ndif-steering.md`
------------------------------------------------------------------
A standard NDIF key may only run models NDIF has *pinned*; anything else fails
with "Model is not pinned and hotswapping is not supported for this API key".
Notably `meta-llama/Llama-3.1-8B-Instruct` — the local study's base — is hot but
not pinned, while the non-instruct `meta-llama/Llama-3.1-8B` is. `check_model`
turns that into an error naming the pinned set rather than a stack trace 40
minutes into a run.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
from dataclasses import dataclass, field

from ..lora.prompting import wrap_prompt
from .generate import GenConfig, SteeredGenerator, clean_continuation

log = logging.getLogger(__name__)

#: NDIF's own API host. `ndif.dev`, which older tutorials use, no longer
#: resolves; nnsight >= 0.7 defaults to this.
NDIF_STATUS_URL = "https://api.ndif.us/status"

ENV_API_KEY = "NDIF_API_KEY"

#: Models worth steering on NDIF, largest first. Not a whitelist — any pinned
#: causal LM works — but these are the ones the scale argument is about.
SCALE_MODELS = (
    "meta-llama/Llama-3.1-405B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    "meta-llama/Llama-3.1-70B",
    "meta-llama/Llama-3.1-8B",
)

_REPO_RE = re.compile(r'"repo_id":\s*"([^"]+)"')


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #


def ndif_status(url: str = NDIF_STATUS_URL, timeout: float = 30.0) -> dict:
    """Return `{model_id: {"pinned": bool, "states": [...]}}` from NDIF.

    NDIF reports one entry per *deployment*, and a model may have several with
    different states. A model counts as pinned if any of its deployments is,
    because that is the condition a standard key is actually checked against.
    """
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode())
    deployments = payload.get("deployments", payload)
    out: dict[str, dict] = {}
    for key, value in deployments.items():
        match = _REPO_RE.search(key)
        model_id = match.group(1) if match else key
        entry = out.setdefault(model_id, {"pinned": False, "states": []})
        entry["pinned"] = entry["pinned"] or value.get("pinned") is True
        entry["states"].append(value.get("application_state"))
    return out


def pinned_models(status: dict | None = None) -> list[str]:
    status = status if status is not None else ndif_status()
    return sorted(m for m, v in status.items() if v["pinned"])


def check_model(model_id: str, status: dict | None = None) -> dict:
    """Raise unless `model_id` is runnable by a standard NDIF key.

    Failing here costs one HTTP request. Failing on the first trace instead
    costs however long the run took to reach it, and the message NDIF returns
    does not say which models *would* have worked.
    """
    try:
        status = status if status is not None else ndif_status()
    except Exception as exc:  # noqa: BLE001 - a status outage must not be fatal
        log.warning("could not reach NDIF status (%s); skipping the pinned check", exc)
        return {"checked": False, "reason": f"{type(exc).__name__}: {exc}"}
    if model_id not in status:
        raise ValueError(
            f"{model_id!r} is not deployed on NDIF. Pinned models: "
            + ", ".join(pinned_models(status))
        )
    if not status[model_id]["pinned"]:
        raise ValueError(
            f"{model_id!r} is deployed on NDIF but is not pinned, and a standard "
            f"API key cannot hotswap it. Pinned models: "
            + ", ".join(pinned_models(status))
            + f". (Note that {model_id!r} being 'hot' on the status page is not "
            "sufficient.)"
        )
    return {"checked": True, "pinned": True, "states": status[model_id]["states"]}


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #


@dataclass
class NDIFConfig(GenConfig):
    """`GenConfig` plus the remote-only knobs.

    `load_in_4bit` is inherited but meaningless here and is forced to None by
    `to_dict`, so a remote run cannot be filed as if it shared the study's
    quantisation constant.
    """

    #: Remote batch. NDIF charges wall-clock per trace rather than per token, so
    #: a larger batch is close to free until the worker's memory is the limit.
    batch_size: int = 8
    #: Seed the remote RNG inside the trace. Off makes draws non-reproducible.
    remote_seed: bool = True
    api_key: str | None = None
    status_check: bool = True
    #: Refuse to run a model whose tokenizer has no chat template. The local
    #: study wraps every method's prompt with `wrap_prompt`, and Study 3 item 1
    #: exists because steering was once compared against LoRA under different
    #: wrapping. A pinned *base* model (`Llama-3.1-8B`, `Llama-3.1-70B`) has no
    #: template, so running one silently would reintroduce that confound. Set
    #: False only for a smoke test, and read `chat_wrapped` in the provenance.
    require_chat_template: bool = True
    #: Recorded from the status endpoint at load time, for provenance.
    deployment: dict = field(default_factory=dict)
    #: Set at load time from the tokenizer, never by the caller.
    chat_wrapped: bool | None = None

    def to_dict(self) -> dict:
        out = super().to_dict()
        out.update(
            {
                "backend": "ndif",
                "load_in_4bit": None,
                "remote_seed": self.remote_seed,
                "chat_wrapped": self.chat_wrapped,
                "deployment": self.deployment,
                "nnsight_version": _nnsight_version(),
            }
        )
        return out


def _nnsight_version() -> str | None:
    try:
        import nnsight

        return getattr(nnsight, "__version__", None)
    except ImportError:
        return None


def resolve_api_key(explicit: str | None = None) -> str:
    key = explicit or os.environ.get(ENV_API_KEY)
    if not key:
        # Mirrors llm.py: the harness reads .env so a key need not be exported.
        from ...llm import load_dotenv

        load_dotenv()
        key = os.environ.get(ENV_API_KEY)
    if not key:
        raise RuntimeError(
            f"{ENV_API_KEY} is not set. Add it to .env or export it. "
            "Keys come from https://login.ndif.us."
        )
    return key


# --------------------------------------------------------------------------- #
# envoy helpers
# --------------------------------------------------------------------------- #


def remote_blocks(lm):
    """The decoder-block envoys, mirroring `runtime.decoder_layers`.

    Written against the envoy tree rather than the module tree because on a
    remote model the modules are not present locally.
    """
    for path in (("model", "layers"), ("transformer", "h"), ("model", "model", "layers")):
        node = lm
        for attr in path:
            node = getattr(node, attr, None)
            if node is None:
                break
        if node is not None:
            return node
    raise AttributeError(f"could not locate decoder blocks on {type(lm).__name__}")


def remote_embeddings(lm):
    """The envoy whose output is hidden-state index 0."""
    for path in (("model", "embed_tokens"), ("transformer", "wte"), ("model", "model", "embed_tokens")):
        node = lm
        for attr in path:
            node = getattr(node, attr, None)
            if node is None:
                break
        if node is not None:
            return node
    raise AttributeError(f"could not locate the embedding envoy on {type(lm).__name__}")


def _hidden(envoy_output, tuple_output: bool):
    """Unwrap an envoy's output to the residual-stream `[B, T, H]` tensor.

    Whether a decoder block returns `(hidden, ...)` or a bare tensor depends on
    the transformers version running *on the NDIF worker*, not on ours: under
    transformers 5 these Llama deployments return a bare tensor. The two layouts
    cannot be told apart by trying to index, because indexing a traced tensor
    proxy succeeds and silently yields row 0 of the batch — which steers only the
    first prompt of every batch and leaves the rest untouched. `probe_block_layout`
    settles it with one cheap trace instead, and the answer is passed in here.
    """
    return envoy_output[0] if tuple_output else envoy_output


def probe_block_layout(lm, blocks=None) -> bool:
    """True when a decoder block's output is a tuple. One small remote trace.

    Run once at load. The cost is a two-token forward pass; the alternative is a
    silent batch-row bug that only shows up as "some prompts were not steered".
    """
    import torch

    blocks = blocks if blocks is not None else remote_blocks(lm)
    first = blocks[0]
    input_ids = torch.tensor([[1, 2]])
    with lm.trace({"input_ids": input_ids}, remote=True):
        raw = first.output.save()
    if isinstance(raw, (tuple, list)):
        return True
    ndim = getattr(raw, "ndim", None)
    if ndim == 3:
        return False
    if ndim == 2:
        # A bare 2-D result would mean the envoy already dropped the batch axis,
        # which none of the supported layouts do. Refuse rather than guess.
        raise RuntimeError(
            f"decoder block output has an unexpected 2-D shape {tuple(raw.shape)}; "
            "the steering write cannot be placed safely"
        )
    raise RuntimeError(
        f"could not determine the decoder block output layout ({type(raw).__name__})"
    )


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #


class NDIFSteeredGenerator(SteeredGenerator):
    """`SteeredGenerator` with the forward pass on NDIF.

    Only `load` and `_generate_batch` are overridden. The scope grouping, vector
    resolution, provenance and per-item failure handling in `generate` are the
    local implementation, unchanged, so the two arms cannot drift apart.
    """

    def __init__(self, artifact: dict, cfg: NDIFConfig | None = None):
        super().__init__(artifact, cfg or NDIFConfig())
        self.status: dict = {}
        #: Set by `probe_block_layout` at load; see `_hidden`.
        self._block_tuple: bool | None = None

    def status_all(self) -> dict:
        try:
            return ndif_status()
        except Exception:  # noqa: BLE001 - only used to enrich an error message
            return {}

    def load(self) -> None:
        try:
            from nnsight import CONFIG, LanguageModel
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                'nnsight is not installed. Install the extra: pip install -e ".[ndif]"'
            ) from exc

        model_id = self.cfg.model_id or self.artifact.get("base_model")
        if not model_id:
            raise ValueError("no model_id and the artifact records no base_model")
        # The same guard the local generator applies. A vector fitted on 8B has
        # 4096 columns and 405B's residual stream has 16384; the mismatch would
        # otherwise surface as a shape error inside a remote trace.
        fitted = self.artifact.get("base_model")
        if fitted and model_id != fitted:
            raise ValueError(
                f"artifact was fitted on {fitted!r} but generation was asked for "
                f"{model_id!r}; steering vectors do not transfer between models. "
                "Fit against this model with methods.steering.fit_steering_ndif."
            )
        if self.cfg.status_check:
            self.status = check_model(model_id)
            self.cfg.deployment = {"model_id": model_id, **self.status}

        CONFIG.set_default_api_key(resolve_api_key(self.cfg.api_key))
        lm = LanguageModel(model_id)
        tok = lm.tokenizer
        tok.padding_side = "left"
        tok.truncation_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        has_template = bool(getattr(tok, "chat_template", None))
        if not has_template and self.cfg.require_chat_template:
            instruct = [m for m in pinned_models(self.status_all()) if "Instruct" in m]
            raise ValueError(
                f"{model_id!r} has no chat template, so the shared prompt cannot be "
                "wrapped the way LoRA, soft prompting and local steering wrap it. "
                "Comparing an unwrapped remote arm against those is the confound "
                "Study 3 item 1 was opened for. Use a pinned instruct model "
                f"({', '.join(instruct) or 'none available'}), or pass "
                "require_chat_template=False to accept an unwrapped run, which is "
                "recorded as chat_wrapped=False and is not comparable to the "
                "wrapped columns."
            )
        self.cfg.chat_wrapped = has_template
        self.model = lm
        self.tok = tok
        self._block_tuple = probe_block_layout(lm)
        log.info(
            "ndif: %s ready (nnsight %s, chat_wrapped=%s, block_output=%s)",
            model_id,
            _nnsight_version(),
            has_template,
            "tuple" if self._block_tuple else "tensor",
        )

    def _generate_batch(
        self, prompts: list[str], draw: int, deltas: dict | None = None
    ) -> list[str]:
        import torch

        lm = self.model
        rendered = [
            wrap_prompt(self.tok, prompt) if self.cfg.chat_wrapped else prompt
            for prompt in prompts
        ]
        enc = self.tok(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_prompt_tokens,
        )
        n_prompt = int(enc["input_ids"].shape[1])
        # Same seed expression as the local path, so draw k differs from draw 0
        # in the same way. It is applied on the worker, inside the trace.
        seed = self.cfg.seed + 1_000_003 * draw + len(prompts)
        # Everything the trace body touches is hoisted into a plain local first.
        # nnsight serialises the variables the body closes over, and a reference
        # to `self` would drag the LanguageModel, the tokenizer and the artifact
        # tensors along with it; remotely that surfaces as the singularly
        # unhelpful "'NoneType' object has no attribute '__dict__'".
        blocks = remote_blocks(lm)
        do_seed = bool(self.cfg.remote_seed)
        block_tuple = bool(self._block_tuple)
        edits = [(blocks[layer - 1], delta) for layer, delta in (deltas or {}).items()]
        gen_kwargs = {
            "max_new_tokens": self.cfg.max_new_tokens,
            "do_sample": self.cfg.temperature > 0,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "pad_token_id": self.tok.pad_token_id,
        }

        with lm.generate(enc, remote=True, **gen_kwargs) as tracer:
            if do_seed:
                torch.manual_seed(seed)
            if edits:
                # `tracer.all()` re-applies the edit at every decoding step. Without
                # it only the prefill is steered and the continuation drifts back to
                # the base model; the local forward hook gives the same formulation
                # by firing on every forward call.
                with tracer.all():
                    for envoy, delta in edits:
                        hidden = _hidden(envoy.output, block_tuple)
                        # `[B, T, H] += [H]` broadcasts over batch and position, so
                        # every prompt in the batch is steered, not just row 0.
                        hidden[:] += delta.to(hidden.device).to(hidden.dtype)
            out = lm.generator.output.save()

        new = out[:, n_prompt:]
        return [
            clean_continuation(t)
            for t in self.tok.batch_decode(new, skip_special_tokens=True)
        ]


# --------------------------------------------------------------------------- #
# fitting
# --------------------------------------------------------------------------- #


def remote_pooled_activations(
    lm,
    tok,
    prompt: str,
    completion: str,
    max_length: int,
    blocks=None,
    embeddings=None,
    block_tuple: bool | None = None,
    chat_wrapped: bool = True,
):
    """Remote equivalent of `fit_steering.pooled_activations`.

    Returns `(pooled[L+1, H] float64 cpu, norms[L+1] float64 cpu)`, or None when
    truncation leaves no completion token to pool over — the same contract, so
    the caller's `State` accumulation is untouched.

    The pooling happens *inside* the trace. Downloading the raw residual stream
    would be `[L+1, T, H]`: for 405B at 126 layers and T=200 that is about 1.6 GB
    per post, against 8 MB for the pooled result. The reduction is not an
    optimisation, it is what makes fitting on 405B possible at all.
    """
    import torch

    rendered = wrap_prompt(tok, prompt) if chat_wrapped else prompt
    ids_p = tok(rendered, add_special_tokens=True).input_ids
    ids_c = tok(completion, add_special_tokens=False).input_ids
    ids = (ids_p + ids_c)[:max_length]
    start = len(ids_p)
    if start >= len(ids):
        return None

    blocks = blocks if blocks is not None else remote_blocks(lm)
    embeddings = embeddings if embeddings is not None else remote_embeddings(lm)
    if block_tuple is None:
        block_tuple = probe_block_layout(lm, blocks)
    input_ids = torch.tensor([ids])
    # Hidden-state index 0 is the embedding output and index i is the output of
    # decoder layer i - 1, matching the HuggingFace convention the artifact
    # format is written against. The embedding module always returns a bare
    # tensor; only the blocks vary. Resolved outside the trace, because the body
    # must close over plain locals only (see `_generate_batch`).
    envoys = [(embeddings, False)] + [(block, block_tuple) for block in blocks]

    with lm.trace({"input_ids": input_ids}, remote=True):
        # Reduce on the worker and download only the result. These are the same
        # two reductions as `fit_steering.pooled_activations`: the mean over the
        # completion positions, and the mean residual norm over those positions.
        per_layer = [
            _hidden(envoy.output, tup)[0, start:, :].float() for envoy, tup in envoys
        ]
        # Reduce each layer *before* gathering. On 70B and 405B the model is
        # sharded, so layer 0 and layer 79 sit on different CUDA devices and
        # stacking them directly raises "expected all tensors to be on the same
        # device". Each reduced layer is [H] (or a scalar), so the move to host
        # memory is cheap and the cross-shard stack becomes a CPU one.
        pooled_p = torch.stack([h.mean(dim=0).cpu() for h in per_layer], dim=0).save()
        norms_p = torch.stack(
            [h.norm(dim=-1).mean().cpu() for h in per_layer], dim=0
        ).save()

    pooled = torch.as_tensor(pooled_p)
    norms = torch.as_tensor(norms_p)
    return pooled.double().cpu(), norms.double().cpu()
