"""Shared steering machinery: artifact format, vector selection, forward hooks.

This module is imported both by the fitting script, which runs on the cluster,
and by the transfer function, which the harness imports at start-up. It must
therefore remain importable without torch, so every torch reference is inside a
function body.

The artifact
------------
A steering artifact is a single file written with `torch.save`. Its contents:

    format        int, currently 1
    base_model    the model the activations were read from
    variant       which conditioning the activations were read under
    layers        list of hidden-state indices held in the vectors, always 0..L
    hidden_size   int
    global        {"vector": [L+1, H] float32, "n_target": int, "n_source": int}
    per_cell      {cell_id: {"vector": [L+1, H], "n_target": int, "n_source": int}}
    act_norm      [L+1] float32, mean residual-stream norm per layer over all
                  posts read, used to make alpha dimensionless
    counts        provenance: posts read, posts skipped, cells covered
    fit_config    every argument the fit was run with

Hidden-state indexing follows Hugging Face: index 0 is the embedding output and
index i is the output of decoder layer i - 1. Steering at index i therefore hooks
`model.model.layers[i - 1]`, and index 0 is not a valid steering site.

Scaling
-------
The stored vector is a raw difference of means, whose norm varies by an order of
magnitude across layers. Applying it directly would make a single alpha mean
something different at every layer and would render the layer sweep
uninterpretable. What is added at inference is

    delta = alpha * (v / ||v||) * act_norm[layer]

so alpha is a fraction of the typical residual-stream norm at that layer and is
comparable across layers.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ARTIFACT_FORMAT = 1

#: Steering is applied to the residual stream at these hidden-state indices.
#: Index 0 is the embedding output and is excluded: adding a constant there is
#: equivalent to editing the token embeddings and is not what is being measured.
MIN_STEER_LAYER = 1


@dataclass
class SteeringSpec:
    """A fully resolved steering configuration."""

    layers: tuple[int, ...]
    alpha: float
    scope: str = "global"  # "global" or "cell"
    #: When scope == "cell" and a cell has no vector of its own, fall back to the
    #: global vector rather than steering not at all, so that coverage does not
    #: silently change what is being compared.
    fallback: str = "global"
    #: Relative coefficient on the orthogonal audience residual when
    #: ``scope == "decomposed"``.  ``alpha`` remains the overall intervention
    #: magnitude after the composed vector is normalized.
    audience_alpha: float = 1.0
    source: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "layers": list(self.layers),
            "alpha": self.alpha,
            "scope": self.scope,
            "fallback": self.fallback,
            "audience_alpha": self.audience_alpha,
            **self.source,
        }


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_artifact(path: str | Path, map_location: str = "cpu") -> dict:
    """Load a steering artifact and check its shape."""
    import torch

    path = Path(path)
    art = torch.load(path, map_location=map_location, weights_only=False)
    if art.get("format") != ARTIFACT_FORMAT:
        raise ValueError(
            f"{path}: artifact format {art.get('format')!r}, expected {ARTIFACT_FORMAT}"
        )
    for key in ("global", "act_norm", "base_model", "variant"):
        if key not in art:
            raise ValueError(f"{path}: artifact is missing {key!r}")
    art["path"] = str(path)
    art["sha256"] = file_sha256(path)
    return art


def resolve_vector(
    artifact: dict,
    cell_id: str | None,
    spec: SteeringSpec,
    audience_id: str | None = None,
):
    """Return `(vector[L+1, H], provenance)` for one cell under `spec`.

    Provenance records which vector was used and how many posts fed it, because
    a per-cell vector fitted on three posts and the global vector fitted on 337
    are not the same object and must not be reported as one.
    """
    per_cell = artifact.get("per_cell") or {}
    per_audience = artifact.get("per_audience") or {}
    if spec.scope == "decomposed" and audience_id in per_audience:
        platform = artifact["global"]["vector"]
        audience = per_audience[audience_id]["vector"]
        platform_unit = platform / platform.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        audience_unit = audience / audience.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        combined = platform_unit + spec.audience_alpha * audience_unit
        entry = per_audience[audience_id]
        return combined, {
            "vector_scope": "decomposed",
            "vector_cell_id": None,
            "vector_audience_id": audience_id,
            "n_target_posts": int(artifact["global"]["n_target"]),
            "n_source_posts": int(artifact["global"]["n_source"]),
            "n_audience_posts": int(entry["n_target"]),
            "audience_alpha": spec.audience_alpha,
            "orthogonalised": True,
        }
    if spec.scope == "cell" and cell_id is not None and cell_id in per_cell:
        entry = per_cell[cell_id]
        return entry["vector"], {
            "vector_scope": "cell",
            "vector_cell_id": cell_id,
            "n_target_posts": int(entry["n_target"]),
            "n_source_posts": int(entry["n_source"]),
        }
    entry = artifact["global"]
    return entry["vector"], {
        "vector_scope": "global",
        "vector_cell_id": None,
        "n_target_posts": int(entry["n_target"]),
        "n_source_posts": int(entry["n_source"]),
        "fell_back": spec.scope in {"cell", "decomposed"},
    }


def steering_deltas(artifact: dict, vector, spec: SteeringSpec) -> dict[int, Any]:
    """Map layer index -> the tensor added to the residual stream at that layer."""
    import torch

    act_norm = artifact["act_norm"]
    out: dict[int, Any] = {}
    for layer in spec.layers:
        if layer < MIN_STEER_LAYER or layer >= vector.shape[0]:
            raise ValueError(
                f"layer {layer} out of range [{MIN_STEER_LAYER}, {vector.shape[0] - 1}]"
            )
        v = vector[layer].to(torch.float32)
        norm = float(v.norm())
        if norm == 0.0:
            continue
        out[layer] = spec.alpha * (v / norm) * float(act_norm[layer])
    return out


def decoder_layers(model):
    """The list of decoder blocks, across the transformers 4 and 5 layouts."""
    for getter in (
        lambda m: m.get_decoder().layers,
        lambda m: m.model.layers,
        lambda m: m.model.model.layers,
    ):
        try:
            layers = getter(model)
        except AttributeError:
            continue
        if layers is not None:
            return layers
    raise AttributeError("could not locate the decoder layers on this model")


@contextmanager
def apply_steering(model, deltas: dict[int, Any]):
    """Add `deltas[i]` to the output of the decoder layer producing hidden state i.

    The hook fires on every forward call, so during incremental decoding the
    offset is applied to each newly generated position as well as to the prompt.
    This is the usual formulation: the intervention is a property of the model
    for the duration of the generation, not of a particular position.
    """
    import torch

    layers = decoder_layers(model)
    handles = []

    def make_hook(delta):
        def hook(_module, _args, output):
            if isinstance(output, tuple):
                hidden = output[0]
                return (hidden + delta.to(hidden.dtype).to(hidden.device),) + output[1:]
            return output + delta.to(output.dtype).to(output.device)

        return hook

    try:
        for layer_idx, delta in deltas.items():
            block = layers[layer_idx - 1]
            handles.append(block.register_forward_hook(make_hook(delta)))
        with torch.no_grad():
            yield
    finally:
        for handle in handles:
            handle.remove()


def parse_layers(text: str | int | list[int]) -> tuple[int, ...]:
    """Parse a layer specification: `16`, `"16"`, `"12,16,20"`, or `"12-20:4"`."""
    if isinstance(text, int):
        return (text,)
    if isinstance(text, (list, tuple)):
        return tuple(int(x) for x in text)
    out: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            span, _, step = part.partition(":")
            lo, _, hi = span.partition("-")
            out.extend(range(int(lo), int(hi) + 1, int(step) if step else 1))
        else:
            out.append(int(part))
    return tuple(sorted(set(out)))
