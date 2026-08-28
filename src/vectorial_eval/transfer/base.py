"""Transfer-function interface and registry.

A transfer function maps a source-platform post to a prediction of how the same
audience would express that content on the target platform. The meeting listed
prompt engineering, soft prompts, LoRA/adapters, task vectors, steering vectors,
and latent-trait mediation as candidate implementations. The purpose of this
interface is that all of them present the same shape to the harness, so that a
steering-vector method may subsequently be introduced and evaluated against the
identical metric suite on the identical splits.

Contract
--------
`run(tasks, corpus)` receives every task simultaneously rather than
individually. This is deliberate: batched implementations, such as an LLM
invoked with concurrency or a GPU-resident adapter, require the full batch, and
a per-task interface would compel them to simulate one. The order of the
returned list must correspond to the input.

Implementations must return one `TransferOutput` per input task even upon
failure. Discarding failures would silently bias every distributional metric
toward whichever subset the function happened to handle successfully.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from ..data.schema import PostRecord, TransferOutput, TransferTask


class Corpus:
    """Read-only lookup over the built dataset.

    Transfer functions must resolve `exemplar_ids` into text. Mediating this
    through a read-only lookup, rather than exposing the raw frame, prevents
    access outside the split boundaries established during dataset
    construction.
    """

    def __init__(self, posts: list[PostRecord]):
        self._by_id = {p.post_id: p for p in posts}

    def text(self, post_id: str) -> str:
        return self._by_id[post_id].text

    def texts(self, post_ids: list[str]) -> list[str]:
        return [self._by_id[pid].text for pid in post_ids if pid in self._by_id]

    def post(self, post_id: str) -> PostRecord:
        return self._by_id[post_id]

    def __contains__(self, post_id: str) -> bool:
        return post_id in self._by_id


class TransferFunction(ABC):
    #: Short, stable identifier used in filenames and result tables.
    name: str = "unnamed"

    @abstractmethod
    def run(self, tasks: list[TransferTask], corpus: Corpus) -> list[TransferOutput]:
        ...

    def describe(self) -> dict:
        """Metadata recorded alongside results for reproducibility."""
        return {"name": self.name, "class": type(self).__name__}


_REGISTRY: dict[str, Callable[..., TransferFunction]] = {}


def register(name: str) -> Callable:
    def deco(factory: Callable[..., TransferFunction]) -> Callable:
        _REGISTRY[name] = factory
        return factory

    return deco


def build_transfer_fn(_kind: str, **kwargs) -> TransferFunction:
    """Instantiate a registered transfer function.

    The registry key is positional-only (`_kind`) so that callers can pass a
    `name=` kwarg to relabel the instance — used by `--tag` to run the same
    function under several models without output files colliding.
    """
    if _kind not in _REGISTRY:
        raise KeyError(f"Unknown transfer function {_kind!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[_kind](**kwargs)


def available() -> list[str]:
    return sorted(_REGISTRY)
