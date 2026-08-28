"""Checks that hold the assembled repository together.

Each method was built by a separate agent against a shared filesystem. The tests
here pin the properties that only become checkable once all of them are present:
that every function is reachable from the registry, that the three trained
methods render a byte-identical context, that the command line reaches each one
with the arguments it needs, and that the two unresolved selection dependences
travel with every run record instead of living only in a document.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from vectorial_eval.cli import _accepts, _derange_audiences, _is_llm_backed
from vectorial_eval.data.build_training import DEV_SELECTION_CAVEAT, render_prompt
from vectorial_eval.data.schema import TransferTask
from vectorial_eval.evaluate import _common_cells
from vectorial_eval.metrics.base import MetricResult
from vectorial_eval.transfer import available, build_transfer_fn
from vectorial_eval.transfer.base import Corpus

#: Everything the integration step was asked to wire, plus the three references
#: the reporting rules require in every comparison figure.
EXPECTED = {
    "identity",
    "shuffle_control",
    "target_sample",
    "llm_rewrite",
    "llm_fewshot",
    "llama_rewrite",
    "plan_then_transfer",
    "aspect_prompt",
    "soft_prompt",
    "lora",
    "steering",
}

TRAINED = ("soft_prompt", "lora", "steering")


def _task(i: int = 0) -> TransferTask:
    return TransferTask(
        task_id=f"t{i}",
        cell_id="backend_engineer::sql_query_optimization",
        room="backend_engineer",
        domain="Software Engineering",
        topic="sql query optimization",
        source_platform="linkedin",
        target_platform="reddit",
        source_post_id=f"p{i}",
        source_text=f"source post {i}",
        exemplar_ids=[],
        target_reference_ids=[],
        split="test",
        heldout_cell=False,
    )


def test_every_transfer_function_is_registered():
    missing = EXPECTED - set(available())
    assert not missing, f"not reachable from the registry: {sorted(missing)}"


def test_n_samples_reaches_every_multi_candidate_function():
    """Invariant 5, arriving through the command line rather than the cache key.

    A function that draws candidates but never receives `--n-samples` produces
    one draw per task with no warning, which both starves the group-level
    metrics and makes its column incomparable with the language-model columns
    that did get four.
    """
    for name in (*TRAINED, "aspect_prompt", "llama_rewrite", "plan_then_transfer", "llm_rewrite", "llm_fewshot",
                 "target_sample", "shuffle_control"):
        assert _is_llm_backed(name) or _accepts(name, "n_samples"), name


def test_only_llm_backed_functions_are_handed_an_llm_client():
    for name in TRAINED:
        assert not _is_llm_backed(name), (
            f"{name} runs a local model; handing it cfg.llm would record a "
            "model it never called"
        )


def test_the_three_trained_methods_render_one_shared_context(monkeypatch):
    """The comparison is only about the adaptation mechanism if the context is fixed.

    `_prompt` on the LoRA and soft-prompt functions and `_items` on the steering
    function must all reduce to the same `render_prompt` call, so that a
    difference between the three columns cannot be a difference of prompt.
    """
    from vectorial_eval.transfer import lora, soft_prompt, steering

    task = _task(0)
    expected = render_prompt(
        room=task.room,
        topic=task.topic,
        domain=task.domain,
        target_platform=task.target_platform,
        source_text=task.source_text,
        source_platform=task.source_platform,
    )

    rendered = {
        "lora": lora.LoraTransfer._prompt(_Stub(), task),
        "soft_prompt": soft_prompt.SoftPromptTransfer._prompt(_Stub(), task),
        "steering": steering.SteeringTransfer._items(_Stub(source_chars=10_000), [task])[0]["prompt"],
    }
    for name, text in rendered.items():
        assert text == expected, f"{name} does not render the shared context"


class _Stub:
    """Enough of a transfer function to call its prompt rendering unbound.

    Constructing the real objects would require an artifact on disk and, for two
    of them, a GPU. The rendering is a pure function of the task, so it is
    exercised directly.
    """

    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_wrap_prompt_agrees_between_lora_and_soft_prompt():
    """Duplicated deliberately, so it has to be checked rather than trusted."""
    from vectorial_eval.methods.lora import prompting as lora_p
    from vectorial_eval.methods.soft_prompt import prompting as sp_p

    class _Tok:
        def apply_chat_template(self, msgs, add_generation_prompt, tokenize):
            return f"<|u|>{msgs[0]['content']}<|a|>{add_generation_prompt}{tokenize}"

    tok = _Tok()
    assert lora_p.wrap_prompt(tok, "ctx") == sp_p.wrap_prompt(tok, "ctx")


def test_steering_uses_the_shared_chat_wrapper():
    """Prompt wrapping must be constant across the three trained methods."""
    from vectorial_eval.methods.steering import generate

    body = inspect.getsource(generate).split('"""', 2)[-1]
    assert "wrap_prompt" in body


@pytest.mark.parametrize("name", TRAINED)
def test_each_trained_method_carries_the_selection_caveat(name, monkeypatch):
    """The dev file is the val split and `heldout` is val union test.

    Reporting reads run records, not documents, so the dependence has to be in
    `describe()` for it to reach a reader.
    """
    from vectorial_eval.transfer import lora, soft_prompt, steering

    cls = {
        "lora": lora.LoraTransfer,
        "soft_prompt": soft_prompt.SoftPromptTransfer,
        "steering": steering.SteeringTransfer,
    }[name]
    src = inspect.getsource(cls.describe)
    assert "DEV_SELECTION_CAVEAT" in src, (
        f"{name}.describe() does not record that it selected on the val split"
    )
    assert "manifest.json" in DEV_SELECTION_CAVEAT
    assert "heldout scoring is valid only" in DEV_SELECTION_CAVEAT


def test_build_transfer_fn_refuses_an_unknown_name():
    with pytest.raises(KeyError):
        build_transfer_fn("no_such_function")


def test_local_llama_baseline_does_not_require_an_adapter(monkeypatch):
    monkeypatch.delenv("VECTORIAL_LORA_ADAPTER", raising=False)
    fn = build_transfer_fn("llama_rewrite", n_samples=4)
    assert fn.variant == "base_none"
    assert fn.adapter_dir is None
    assert fn.describe()["caveats"] == []


def test_audience_shuffle_is_a_frozen_derangement_of_conditioning_only():
    tasks = [
        _task(0).model_copy(update={"room": "zeta"}),
        _task(1).model_copy(update={"room": "alpha"}),
        _task(2).model_copy(update={"room": "middle"}),
    ]
    shuffled, mapping = _derange_audiences(tasks)
    assert mapping == {"alpha": "middle", "middle": "zeta", "zeta": "alpha"}
    assert all(
        before.room != after.room
        for before, after in zip(tasks, shuffled, strict=True)
    )
    for before, after in zip(tasks, shuffled, strict=True):
        assert before.model_dump(exclude={"room"}) == after.model_dump(exclude={"room"})


def test_audience_shuffle_requires_two_rooms():
    with pytest.raises(ValueError, match="at least two"):
        _derange_audiences([_task(0)])


def test_authentic_references_draw_from_scored_ids_not_train_exemplars():
    from vectorial_eval.transfer.baselines import TargetSampleTransfer

    task = _task(0).model_copy(
        update={"target_reference_ids": ["scored"], "exemplar_ids": ["train"]}
    )
    corpus = Corpus(
        [
            SimpleNamespace(post_id="scored", text="scored split text"),
            SimpleNamespace(post_id="train", text="train split text"),
        ]
    )
    outputs = TargetSampleTransfer(n_samples=2).run([task], corpus)
    assert len(outputs) == 2
    assert {output.output_text for output in outputs} == {"scored split text"}


def test_reference_coverage_does_not_shrink_the_common_system_base():
    results = [
        MetricResult("trm", "system_a", {"c1": {}, "c2": {}}),
        MetricResult("trm", "system_b", {"c1": {}, "c2": {}}),
        MetricResult("trm", "target_sample", {"c1": {}}),
    ]
    assert _common_cells(results)["trm"] == {"c1", "c2"}
