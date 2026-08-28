"""Tests for the conditional target-LM training set.

The property pinned here is leakage. Every number the soft-prompt, LoRA, and
steering methods produce is invalid if a single validation or test post reaches
the training file, and nothing in the training loop would raise an error if one
did. The check therefore lives here rather than in a manual inspection.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vectorial_eval.config import DEFAULT_DATA_DIR, DatasetConfig
from vectorial_eval.data.build_training import (
    CONTENT_BLOCK,
    SPLIT_TO_FILE,
    TARGET_LM_PROMPT,
    build,
    read_posts,
    render_prompt,
)

DATA_DIR = DEFAULT_DATA_DIR


def _have_dataset() -> bool:
    return all(
        (DATA_DIR / f"posts.{s}.jsonl").exists() for s in ("train", "val", "test")
    )


needs_dataset = pytest.mark.skipif(
    not _have_dataset(), reason="run `vectorial-eval build` first"
)


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("training")
    manifest = build(data_dir=DATA_DIR, out_dir=out)
    files = {
        suffix: _read(out / f"target_lm.{suffix}.jsonl")
        for suffix in SPLIT_TO_FILE.values()
    }
    return manifest, files, out


# --- the test that matters --------------------------------------------------


@needs_dataset
def test_train_file_contains_only_train_split_post_ids(built):
    """The single most important property in this module.

    A post reaching the training file from val or test would be trained on and
    then scored against, and every downstream metric would be inflated with no
    error raised anywhere.
    """
    _manifest, files, _out = built

    train_split_ids = {p.post_id for p in read_posts(DATA_DIR, ("train",))}
    assert train_split_ids, "no train-split posts were read"

    train_ids = [r["post_id"] for r in files["train"]]
    assert train_ids, "the training file is empty"

    offenders = sorted(set(train_ids) - train_split_ids)
    assert offenders == [], f"non-train post_ids in the training file: {offenders}"

    # Every record must also carry the train label it was read with.
    assert {r["split"] for r in files["train"]} == {"train"}


@needs_dataset
def test_splits_are_mutually_disjoint(built):
    _manifest, files, _out = built
    ids = {k: {r["post_id"] for r in v} for k, v in files.items()}
    assert ids["train"] & ids["dev"] == set()
    assert ids["train"] & ids["test"] == set()
    assert ids["dev"] & ids["test"] == set()


@needs_dataset
def test_dev_and_test_files_carry_their_own_split_labels(built):
    _manifest, files, _out = built
    assert {r["split"] for r in files["dev"]} == {"val"}
    assert {r["split"] for r in files["test"]} == {"test"}


@needs_dataset
def test_manifest_leakage_block_is_clean(built):
    manifest, _files, _out = built
    lk = manifest["leakage_check"]
    assert lk["train_file_is_train_split_only"] is True
    assert lk["post_ids_in_more_than_one_split"] == []
    assert lk["non_train_post_ids_in_train_file"] == []
    assert lk["train_dev_overlap"] == []
    assert lk["train_test_overlap"] == []
    assert lk["dev_test_overlap"] == []


# --- construction -----------------------------------------------------------


@needs_dataset
def test_every_record_is_target_platform_and_within_char_bounds(built):
    manifest, files, _out = built
    cfg = DatasetConfig()
    for records in files.values():
        for rec in records:
            assert rec["platform"] == manifest["target_platform"]
            assert cfg.min_chars <= len(rec["completion"].strip()) <= cfg.max_chars


@needs_dataset
def test_completions_are_verbatim_post_text(built):
    _manifest, files, _out = built
    by_id = {p.post_id: p.text for p in read_posts(DATA_DIR)}
    for records in files.values():
        for rec in records:
            assert rec["completion"] == by_id[rec["post_id"]]


@needs_dataset
def test_prompts_are_rendered_from_the_shared_template(built):
    """The three adaptation methods must share one prompt exactly.

    If a method formats its context differently the comparison confounds the
    adaptation mechanism with the prompt, so the emitted prompt is required to
    be reproducible from the exported constant alone.
    """
    _manifest, files, _out = built
    for records in files.values():
        for rec in records:
            assert rec["prompt"] == render_prompt(
                room=rec["room"],
                topic=rec["topic"],
                domain=rec["domain"],
                target_platform=rec["platform"],
            )
            # Training prompts carry no source post: there is no paired data.
            assert "content to carry across" not in rec["prompt"]


@needs_dataset
def test_build_is_deterministic(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    build(data_dir=DATA_DIR, out_dir=a)
    build(data_dir=DATA_DIR, out_dir=b)
    for suffix in SPLIT_TO_FILE.values():
        name = f"target_lm.{suffix}.jsonl"
        assert (a / name).read_bytes() == (b / name).read_bytes()


@needs_dataset
def test_no_duplicate_completions_across_the_whole_set(built):
    _manifest, files, _out = built
    texts = [r["completion"] for records in files.values() for r in records]
    assert len(texts) == len(set(texts))


# --- prompt rendering, no dataset required ----------------------------------


def test_inference_rendering_adds_only_the_content_block():
    common = dict(room="backend_engineer", topic="agentic coding",
                  domain="Software Engineering", target_platform="reddit")
    train_prompt = render_prompt(**common)
    infer_prompt = render_prompt(**common, source_text="Some LinkedIn post.",
                                 source_platform="linkedin")
    block = CONTENT_BLOCK.format(source_platform="LinkedIn",
                                 source_text="Some LinkedIn post.")
    assert infer_prompt == train_prompt.replace("Platform: Reddit\n",
                                                "Platform: Reddit\n" + block)
    assert "backend engineer" in train_prompt
    assert TARGET_LM_PROMPT.count("{content_block}") == 1


def test_html_entities_are_decoded_once():
    """Scraped entities are decoded, and decoding does not repeat.

    Every post in the corpus carrying an entity is a Reddit post, so leaving
    them encoded gives the lexical platform classifier a one-sided token to key
    on. `&gt;` additionally carries Reddit's markdown quoting.
    """
    from vectorial_eval.data.build_dataset import _unescape_entities

    assert _unescape_entities("a &amp; b") == "a & b"
    assert _unescape_entities("&gt; quoted line") == "> quoted line"
    assert _unescape_entities("if a &lt; b") == "if a < b"
    assert _unescape_entities("no entities here") == "no entities here"
    # A single pass only: a double-encoded string decodes one level, so text
    # that legitimately discusses entities is not corrupted by repeated passes.
    assert _unescape_entities("&amp;gt;") == "&gt;"
