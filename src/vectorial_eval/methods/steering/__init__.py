"""Activation steering: the no-gradient arm of the adaptation comparison.

The artifact is a set of per-layer difference-in-means vectors between authentic
train-split Reddit posts and authentic train-split LinkedIn posts. Nothing here
is trained by gradient descent; fitting is a single forward pass over each train
post. See `README.md` in this directory for the formulation.
"""

from .runtime import (  # noqa: F401
    ARTIFACT_FORMAT,
    SteeringSpec,
    apply_steering,
    load_artifact,
    resolve_vector,
)

# `ndif_backend` is deliberately not imported here: it pulls in nnsight, which is
# an optional extra, and this package is imported at harness start-up on machines
# with no network. Import it from the call site instead.
