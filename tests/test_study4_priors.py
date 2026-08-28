"""Leakage and determinism checks for the Study 4 prior-corpus builders."""

from __future__ import annotations

import json

import pytest

from scripts import build_study4_linkedin_priors as linkedin
from scripts import build_study4_unsupervised as reddit
from scripts.audit_study4_priors import audit_pair, audit_scored_overlap
from scripts.backfill_study4_run_metadata import patch_artifacts
from scripts.normalize_study4_corpus import choose_record
from scripts.sanitize_study4_prior_context import sanitize
from scripts.subsample_study4_priors import nested_sample
from vectorial_eval.methods.lora.prior_prompt import render_prior_sections


def test_author_split_is_platform_stable_and_deterministic():
    author = reddit.digest("reddit:example-user")
    assert reddit.author_split(author) == reddit.author_split(author)
    assert reddit.author_split(author) in {"train", "dev"}
    # Both builders intentionally share the split rule. A future joint corpus
    # must not assign the same de-identified author differently by code path.
    assert reddit.author_split(author) == linkedin.author_split(author)


def test_reddit_evaluation_keys_cover_val_and_test(tmp_path):
    rows = [
        {
            "post_id": "t3_abc",
            "platform": "reddit",
            "url": "https://reddit.com/r/x/comments/abc/title/?utm_source=x",
        },
        {"post_id": "li", "platform": "linkedin", "url": "https://linkedin/x"},
    ]
    for split in ("val", "test"):
        (tmp_path / f"posts.{split}.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
        )
    ids, urls = reddit.evaluation_keys(tmp_path)
    assert ids == {"abc"}
    assert urls == {"https://reddit.com/r/x/comments/abc/title"}


def test_linkedin_evaluation_keys_do_not_admit_other_platforms(tmp_path):
    (tmp_path / "posts.val.jsonl").write_text(
        json.dumps(
            {
                "post_id": "urn:li:activity:1",
                "platform": "linkedin",
                "url": "https://linkedin.com/posts/x/?tracking=1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    ids, urls = linkedin.evaluation_keys(tmp_path)
    assert ids == {"urn:li:activity:1"}
    assert urls == {"https://linkedin.com/posts/x"}


def test_weak_label_flattening_is_sorted_and_ignores_numbers():
    value = {"z": [" beta ", 3], "a": {"trait": "alpha"}}
    assert reddit.label_values(value) == ["a", "trait", "alpha", "z", "beta"]
    assert linkedin.label_values(value) == ["a", "trait", "alpha", "z", "beta"]


def test_linkedin_author_identity_is_stable_across_export_profile_ids():
    profile = {"linkedin_profile_url": "https://linkedin.com/in/example/?tracking=1"}
    assert linkedin.linkedin_author_id(profile, "room-specific-a") == (
        linkedin.linkedin_author_id(profile, "room-specific-b")
    )
    assert linkedin.linkedin_author_id({}, "room-specific-a") != (
        linkedin.linkedin_author_id({}, "room-specific-b")
    )


def test_prior_audit_rejects_author_leakage_and_dev_profile_labels(tmp_path):
    base = {
        "record_id": "r1",
        "author_id": "a1",
        "platform": "reddit",
        "audience": "backend_engineer",
        "record_type": "post",
        "text": "long enough authentic text",
        "audience_prior": {"estimated_from": "train authors only"},
        "profile_weak_labels": {},
    }
    train_path, dev_path = tmp_path / "train.jsonl", tmp_path / "dev.jsonl"
    train_path.write_text(json.dumps(base | {"split": "train"}) + "\n")
    dev_path.write_text(
        json.dumps(base | {"record_id": "r2", "split": "dev"}) + "\n"
    )
    with pytest.raises(RuntimeError, match="author leakage"):
        audit_pair(train_path, dev_path)

    dev_path.write_text(
        json.dumps(
            base
            | {
                "record_id": "r2",
                "author_id": "a2",
                "split": "dev",
                "profile_weak_labels": {"summary": "leaked"},
            }
        )
        + "\n"
    )
    with pytest.raises(RuntimeError, match="retain their author's"):
        audit_pair(train_path, dev_path)


def test_exact_clone_can_retain_cross_room_author_aliases():
    base = {
        "record_id": "same-comment",
        "split": "train",
        "text": "the same exported comment",
        "audience": "backend_engineer",
        "author_id": "author-a",
    }
    resolved = choose_record(
        [
            base,
            base
            | {
                "audience": "edtech_engineering",
                "author_id": "author-b",
            },
        ]
    )
    assert resolved["audience_memberships"] == [
        "backend_engineer",
        "edtech_engineering",
    ]
    assert resolved["author_aliases"] == ["author-a", "author-b"]


def test_clone_resolution_rejects_same_id_with_different_text():
    base = {
        "record_id": "collision",
        "split": "train",
        "text": "first",
        "audience": "backend_engineer",
        "author_id": "author-a",
    }
    with pytest.raises(RuntimeError, match="2 texts"):
        choose_record([base, base | {"text": "second"}])


def test_scale_rungs_are_deterministic_and_nested():
    rows = [
        {"record_id": f"{audience}-{index}", "audience": audience}
        for audience in ("backend", "instructional")
        for index in range(20)
    ]
    small = nested_sample(rows, 5)
    large = nested_sample(list(reversed(rows)), 10)
    assert {row["record_id"] for row in small} <= {
        row["record_id"] for row in large
    }
    assert small == nested_sample(rows, 5)


def test_scale_rung_rejects_nonpositive_size():
    with pytest.raises(ValueError, match="positive"):
        nested_sample([], 0)


def test_rich_prior_prompt_uses_record_and_audience_metadata_without_summary():
    record = {
        "platform": "reddit",
        "audience": "backend_engineer",
        "record_type": "comment",
        "subreddit": "sysadmin",
        "title": "A safe parent thread title",
        "thread_context": {"title": "A migration question"},
        "engagement": {"score": 12},
        "dimensions": {"seniority": "staff"},
        "profile_weak_labels": {
            "summary": "Alice Example is a named person",
            "highlights": ["A generated prose highlight"],
            "keywords": ["distributed systems"],
        },
        "audience_prior": {
            "estimated_from": "train authors only",
            "keywords": ["reliability"],
            "dimensions": ["seniority"],
            "subreddits": ["sysadmin"],
        },
    }
    header, sections = render_prior_sections(record)
    rendered = header + "".join(sections)
    for expected in (
        "Platform: reddit",
        "Audience: backend_engineer",
        "Community: \"sysadmin\"",
        "Thread context:",
        "Engagement:",
        "Post dimensions:",
        "Train-author weak labels:",
        "Audience keywords:",
        "Audience dimensions:",
        "Audience communities:",
    ):
        assert expected in rendered
    assert "distributed systems" in rendered
    assert "A safe parent thread title" in rendered
    assert "Alice Example" not in rendered
    assert "generated prose highlight" not in rendered


def test_rich_prior_prompt_is_canonical_and_omits_empty_families():
    record = {
        "platform": "linkedin",
        "audience": "instructional_designer",
        "record_type": "post",
        "engagement": {"z": 1, "a": 2},
        "dimensions": {},
        "profile_weak_labels": {},
        "audience_prior": {"estimated_from": "train authors only"},
    }
    first = render_prior_sections(record)
    assert first == render_prior_sections(dict(reversed(list(record.items()))))
    assert first[1] == ['Engagement: {"a":2,"z":1}\n']


def test_metadata_backfill_refuses_live_run_and_updates_all_artifacts(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    best = checkpoint / "best"
    best.mkdir(parents=True)
    (checkpoint / "run.json").write_text(json.dumps({"variant": "platform_lm"}))
    (checkpoint / "metrics.json").write_text(json.dumps({"finished": False}))
    with pytest.raises(RuntimeError, match="unfinished"):
        patch_artifacts(checkpoint, {"train_completion_tokens": 10})

    (checkpoint / "metrics.json").write_text(
        json.dumps({"finished": True, "run": {"variant": "platform_lm"}})
    )
    (best / "training_config.json").write_text(
        json.dumps({"variant": "platform_lm"})
    )
    patch_artifacts(checkpoint, {"train_completion_tokens": 10})
    assert json.loads((checkpoint / "run.json").read_text())[
        "train_completion_tokens"
    ] == 10
    assert json.loads((checkpoint / "metrics.json").read_text())["run"][
        "train_completion_tokens"
    ] == 10
    assert json.loads((best / "training_config.json").read_text())[
        "train_completion_tokens"
    ] == 10


def test_scored_overlap_audit_catches_reddit_ids_and_normalized_text(tmp_path):
    scored = tmp_path / "posts.test.jsonl"
    prior = tmp_path / "prior.jsonl"
    scored.write_text(
        json.dumps(
            {
                "post_id": "t3_leaked",
                "platform": "reddit",
                "text": "A sufficiently long scored post with &amp; normalized spacing.",
            }
        )
        + "\n"
    )
    prior.write_text(
        json.dumps(
            {
                "record_id": "reddit:post:leaked",
                "record_type": "post",
                "platform": "reddit",
                "text": "A sufficiently long scored post with & normalized   spacing.",
            }
        )
        + "\n"
    )
    with pytest.raises(RuntimeError, match="scored-post leakage"):
        audit_scored_overlap([prior], [scored])


def test_scored_overlap_audit_accepts_disjoint_corpora(tmp_path):
    scored = tmp_path / "posts.val.jsonl"
    prior = tmp_path / "prior.jsonl"
    scored.write_text(
        json.dumps(
            {
                "post_id": "urn:li:activity:1",
                "platform": "linkedin",
                "text": "This is a long scored LinkedIn post that must remain held out.",
            }
        )
        + "\n"
    )
    prior.write_text(
        json.dumps(
            {
                "record_id": "linkedin:post:hash",
                "record_type": "post",
                "platform": "linkedin",
                "text": "This is unrelated training material from a different author.",
            }
        )
        + "\n"
    )
    result = audit_scored_overlap([prior], [scored])
    assert result["normalized_text_overlaps"] == 0


def test_scored_overlap_audit_rejects_comment_title_used_by_v3_prompt(tmp_path):
    scored = tmp_path / "posts.test.jsonl"
    prior = tmp_path / "prior.jsonl"
    shared = "A scored parent-thread title long enough for the exact overlap guard."
    scored.write_text(
        json.dumps({"post_id": "t3_parent", "platform": "reddit", "text": shared})
        + "\n"
    )
    prior.write_text(
        json.dumps(
            {
                "record_id": "reddit:comment:child",
                "record_type": "comment",
                "platform": "reddit",
                "title": shared,
                "text": "A distinct comment completion that does not copy its parent title.",
            }
        )
        + "\n"
    )
    with pytest.raises(RuntimeError, match="scored-post leakage"):
        audit_scored_overlap([prior], [scored])


def test_context_sanitizer_blanks_scored_metadata_without_changing_completion(tmp_path):
    scored = tmp_path / "posts.test.jsonl"
    source = tmp_path / "prior.jsonl"
    output = tmp_path / "prior.safe.jsonl"
    shared = "A held-out thread body that must not condition prior training."
    scored.write_text(
        json.dumps({"post_id": "t3_parent", "platform": "reddit", "text": shared})
        + "\n"
    )
    source.write_text(
        json.dumps(
            {
                "record_id": "reddit:comment:child",
                "record_type": "comment",
                "platform": "reddit",
                "text": "A distinct authentic comment remains the completion.",
                "thread_context": {"body": f"Quoted parent: {shared} End quote."},
                "title": shared,
            }
        )
        + "\n"
    )
    result = sanitize(source, output, [scored])
    row = json.loads(output.read_text())
    assert row["text"] == "A distinct authentic comment remains the completion."
    assert row["thread_context"] == {"body": ""}
    assert row["title"] == ""
    assert result["removed_exact_metadata_strings"] == {
        "thread_context": 1,
        "title": 1,
    }
