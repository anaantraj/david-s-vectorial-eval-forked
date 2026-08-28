"""Tests for the LoRA transfer function that do not require a GPU.

The fitting script itself cannot be exercised here: it imports torch, peft, and
bitsandbytes, which exist only in the cluster environment. What is testable
locally is the part of the method that decides what the model is asked and what
is recorded when it fails, which is where a silent error would do damage.
"""

from __future__ import annotations

import json

import pytest

from vectorial_eval.data.schema import TransferTask
from vectorial_eval.transfer.base import Corpus, build_transfer_fn
from vectorial_eval.transfer.lora import LoraTransfer, _draw_seed


def _task(i: int) -> TransferTask:
    return TransferTask(
        task_id=f"t{i}",
        cell_id="c1",
        room="backend_engineer",
        topic="agentic coding",
        domain="Software",
        split="test",
        source_platform="linkedin",
        target_platform="reddit",
        source_post_id=f"p{i}",
        source_text=f"source post {i}",
        target_reference_ids=[],
        exemplar_ids=[],
        heldout_cell=False,
    )


@pytest.fixture()
def adapter_dir(tmp_path):
    d = tmp_path / "best"
    d.mkdir()
    (d / "adapter_config.json").write_text("{}", encoding="utf-8")
    (d / "training_config.json").write_text(
        json.dumps(
            {
                "variant": "target_lm_paired",
                "config": {"key": "attn-r8", "r": 8},
                "best_epoch": 3,
                "best_dev_nll": 2.1,
                "history": [{"nll": 2.9, "epoch": 0}],
            }
        ),
        encoding="utf-8",
    )
    return d


def test_registered_and_described(adapter_dir):
    fn = build_transfer_fn("lora", adapter_dir=adapter_dir, name="lora_paired")
    d = fn.describe()
    assert d["name"] == "lora_paired"
    # The training variant must travel with the number: the three variants
    # condition on different context and are not interchangeable.
    assert d["training_variant"] == "target_lm_paired"
    assert d["base_dev_nll"] == 2.9
    assert d["best_dev_nll"] == 2.1
    assert d["quantisation"] == "nf4-4bit-double"


def test_missing_adapter_directory_is_refused(monkeypatch):
    monkeypatch.delenv("VECTORIAL_LORA_ADAPTER", raising=False)
    with pytest.raises(ValueError, match="no adapter directory"):
        build_transfer_fn("lora")


def test_prompt_carries_the_source_post_and_the_shared_template(adapter_dir):
    fn = LoraTransfer(adapter_dir=adapter_dir)
    prompt = fn._prompt(_task(0))
    assert "source post 0" in prompt
    assert prompt.startswith("Audience: backend engineer")
    assert prompt.rstrip().endswith("Write a Reddit post from this audience about this topic.")


def test_draw_seed_depends_on_the_draw_index():
    # Without the draw index in the seed every candidate for a task would be the
    # identical post and the candidate pool would be k copies of one output.
    seeds = {_draw_seed(17, "t0", k) for k in range(4)}
    assert len(seeds) == 4
    assert _draw_seed(17, "t0", 0) != _draw_seed(17, "t1", 0)
    assert _draw_seed(17, "t0", 0) == _draw_seed(17, "t0", 0)


def test_failures_are_recorded_never_dropped(adapter_dir, monkeypatch):
    fn = LoraTransfer(adapter_dir=adapter_dir, n_samples=2, batch_size=2)

    def boom(prompts, seed):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(fn, "_generate", boom)
    tasks = [_task(i) for i in range(3)]
    outputs = fn.run(tasks, Corpus([]))

    assert len(outputs) == len(tasks) * 2
    assert [o.task_id for o in outputs] == [t.task_id for t in tasks for _ in range(2)]
    assert all(o.output_text == "" for o in outputs)
    assert all(o.meta["ok"] is False for o in outputs)
    assert {o.meta["sample_index"] for o in outputs} == {0, 1}


def test_empty_generation_is_a_failure(adapter_dir, monkeypatch):
    fn = LoraTransfer(adapter_dir=adapter_dir, batch_size=2)
    monkeypatch.setattr(fn, "_generate", lambda prompts, seed: ["   " for _ in prompts])
    outputs = fn.run([_task(0), _task(1)], Corpus([]))
    assert len(outputs) == 2
    assert all(o.meta["error"] == "empty_generation" for o in outputs)
    assert all(o.meta["ok"] is False for o in outputs)
