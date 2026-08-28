"""Tests for the soft-prompt method that do not need a GPU.

The training loop itself is exercised on the cluster (`--limit` gives a smoke
run). What is checked here is everything that can go silently wrong without one:
the context this method trains on, the guard against training on a non-train
split, the determinism of the example order that makes resuming exact, and the
transfer function's behaviour when the artifact is missing.
"""

from __future__ import annotations

import json

import pytest

from vectorial_eval.data.build_training import render_prompt
from vectorial_eval.methods.soft_prompt import prompting, train
from vectorial_eval.transfer import soft_prompt as sp_transfer


class FakeTokenizer:
    """Stands in for a chat tokenizer, recording what it was asked to render."""

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True):
        assert tokenize is False
        body = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        return body + ("<assistant>" if add_generation_prompt else "")


def test_wrap_prompt_matches_the_lora_method():
    """Both methods must wrap the shared context identically.

    They hold separate copies so that neither can break the other at import
    time. If this fails the two are no longer comparable and one of them has to
    be brought back into line before either number is reported.
    """
    lora_prompting = pytest.importorskip("vectorial_eval.methods.lora.prompting")
    tok = FakeTokenizer()
    prompt = render_prompt(
        room="backend_engineer", topic="agentic coding",
        domain="Software Engineering", target_platform="reddit",
    )
    assert prompting.wrap_prompt(tok, prompt) == lora_prompting.wrap_prompt(tok, prompt)
    assert prompting.prompt_tail() == lora_prompting.prompt_tail()


def test_study4_soft_prompt_records_the_frozen_stage_a_adapter(tmp_path):
    from vectorial_eval.transfer.soft_prompt import SoftPromptTransfer

    artifact = tmp_path / "prompt"
    artifact.mkdir()
    (artifact / "soft_prompt_meta.json").write_text(
        json.dumps(
            {
                "base_model": "base",
                "initial_adapter": "/frozen/stage-a",
                "config": {"variant": "target_lm", "n_virtual_tokens": 32},
            }
        )
    )
    transfer = SoftPromptTransfer(adapter_dir=artifact)
    assert transfer.initial_adapter == "/frozen/stage-a"
    assert transfer.describe()["initial_adapter"] == "/frozen/stage-a"


def test_prompt_tail_is_the_shared_templates_last_line():
    rendered = render_prompt(
        room="cto", topic="hiring", domain="Software Engineering",
        target_platform="reddit",
    )
    assert rendered.rstrip("\n").endswith(prompting.prompt_tail())


def _record(split="train", prompt=None, completion="a post"):
    return {
        "post_id": "t3_x", "cell_id": "c", "room": "r", "topic": "t",
        "domain": "d", "platform": "reddit", "split": split,
        "prompt": prompt if prompt is not None else render_prompt(
            room="r", topic="t", domain="d", target_platform="reddit"
        ),
        "completion": completion, "n_chars": len(completion),
    }


def _write(tmp_path, variant, split, records):
    path = tmp_path / f"{variant}.{split}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


def test_load_split_rejects_a_foreign_prompt(tmp_path):
    """A file not produced by build_training must not be trainable on.

    A method trained on a different context from the other two would still
    produce a number, and the number would not be comparable with theirs.
    """
    _write(tmp_path, "target_lm", "train", [_record(prompt="write something\n")])
    with pytest.raises(ValueError, match="shared template"):
        train.load_split(tmp_path, "target_lm", "train")


def test_load_split_accepts_the_real_training_file(tmp_path):
    _write(tmp_path, "target_lm", "train", [_record(), _record()])
    assert len(train.load_split(tmp_path, "target_lm", "train")) == 2


def test_assert_train_only_rejects_held_out_records():
    with pytest.raises(ValueError, match="non-train splits"):
        train.assert_train_only([_record(), _record(split="test")], "file")


def test_assert_train_only_passes_on_train():
    train.assert_train_only([_record(), _record()], "file")


def test_the_shipped_training_file_is_train_split_only():
    """The real artifact, not a fixture: this is the leakage check that matters."""
    from pathlib import Path

    from vectorial_eval.config import DEFAULT_DATA_DIR

    path = Path(DEFAULT_DATA_DIR) / "training" / "target_lm.train.jsonl"
    if not path.exists():
        pytest.skip("training data not built")
    records = train.load_split(path.parent, "target_lm", "train")
    train.assert_train_only(records, str(path))


def test_epoch_order_is_a_pure_function_of_seed_and_epoch():
    """Resuming is only exact if the order can be recomputed after a kill."""
    assert train.epoch_order(50, 3, 0) == train.epoch_order(50, 3, 0)
    assert train.epoch_order(50, 3, 0) != train.epoch_order(50, 4, 0)
    assert train.epoch_order(50, 3, 0) != train.epoch_order(50, 3, 1)
    assert sorted(train.epoch_order(50, 3, 0)) == list(range(50))


def test_transfer_function_is_registered():
    from vectorial_eval.transfer.base import build_transfer_fn

    with pytest.raises(FileNotFoundError):
        build_transfer_fn("soft_prompt", adapter_dir="/nonexistent/soft_prompt/best")


def test_missing_artifact_raises_rather_than_emitting_empty_outputs(tmp_path):
    """A missing artifact must not look like a method that generates nothing.

    Every output would be empty, which the report cannot distinguish from a
    degenerate method, so the failure has to surface as an exception instead.
    """
    with pytest.raises(FileNotFoundError, match="no soft-prompt artifact"):
        sp_transfer.SoftPromptTransfer(adapter_dir=tmp_path / "absent")


def _artifact(tmp_path):
    adapter = tmp_path / "best"
    adapter.mkdir()
    (adapter / "soft_prompt_meta.json").write_text(
        json.dumps({"base_model": "m", "dev_loss": 2.5,
                    "config": {"variant": "target_lm", "n_virtual_tokens": 16}}),
        encoding="utf-8",
    )
    return adapter


def _tasks(n=3):
    from vectorial_eval.data.schema import TransferTask

    return [
        TransferTask(
            task_id=f"t{i}", cell_id="c", room="r", topic="tp", domain="d",
            split="heldout", source_platform="linkedin", target_platform="reddit",
            source_post_id=f"p{i}", source_text="a source post",
            target_reference_ids=[], exemplar_ids=[], heldout_cell=False,
        )
        for i in range(n)
    ]


def _stub(tmp_path, generate, **kwargs):
    fn = sp_transfer.SoftPromptTransfer(adapter_dir=_artifact(tmp_path), **kwargs)
    fn._load = lambda: None
    fn._generate = generate
    return fn


def test_a_generation_failure_still_emits_one_output_per_task_and_draw(tmp_path):
    """Invariant 6: a failure is recorded, never dropped."""
    def boom(prompts, seed):
        raise RuntimeError("CUDA out of memory")

    tasks = _tasks()
    outputs = _stub(tmp_path, boom, n_samples=3).run(tasks, None)
    assert len(outputs) == len(tasks) * 3
    assert all(o.meta["ok"] is False and o.output_text == "" for o in outputs)
    assert all("CUDA out of memory" in o.meta["error"] for o in outputs)
    assert sorted((o.task_id, o.meta["sample_index"]) for o in outputs) == sorted(
        (t.task_id, k) for t in tasks for k in range(3)
    )


def test_a_short_batch_is_a_failed_batch_not_a_dropped_run(tmp_path):
    """A wrong-length batch must not raise out of `run` and lose every task."""
    tasks = _tasks()
    fn = _stub(
        tmp_path, lambda prompts, seed: ["a"] * (len(prompts) - 1),
        n_samples=1, batch_size=len(tasks),
    )
    outputs = fn.run(tasks, None)
    assert len(outputs) == len(tasks)
    assert all(o.meta["ok"] is False for o in outputs)


def test_an_empty_generation_is_recorded_as_a_failure(tmp_path):
    fn = _stub(tmp_path, lambda prompts, seed: ["   "] * len(prompts), n_samples=2)
    outputs = fn.run(_tasks(), None)
    assert len(outputs) == 6
    assert {o.meta["error"] for o in outputs} == {"empty generation"}


def test_every_task_and_draw_gets_its_own_seed(tmp_path):
    """Two tasks must not be sampled from the same point in one random stream."""
    seen: list[int] = []

    def record(prompts, seed):
        seen.append(seed)
        return [f"text {seed}"] * len(prompts)

    outputs = _stub(tmp_path, record, n_samples=2).run(_tasks(), None)
    assert len(seen) == len(set(seen)) == 6
    assert len({o.meta["seed"] for o in outputs}) == 6


def test_artifact_metadata_overrides_the_declared_variant(tmp_path):
    adapter = tmp_path / "best"
    adapter.mkdir()
    (adapter / "soft_prompt_meta.json").write_text(
        json.dumps({"base_model": "m", "dev_loss": 2.5,
                    "config": {"variant": "target_lm_aspect", "n_virtual_tokens": 16}}),
        encoding="utf-8",
    )
    fn = sp_transfer.SoftPromptTransfer(adapter_dir=adapter, variant="target_lm")
    assert fn.variant == "target_lm_aspect"
    assert fn.describe()["training"]["dev_loss"] == 2.5
