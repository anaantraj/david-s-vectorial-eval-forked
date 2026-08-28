"""Deterministic, privacy-bounded rendering of Study 4 metadata priors."""

from __future__ import annotations

import json

PRIOR_PROMPT_SCHEMA = "study4-rich-priors-v3"
PRIOR_PROMPT_TOKEN_BUDGET = 128
PRIOR_SECTION_TOKEN_BUDGET = 12


def render_prior_sections(record: dict) -> tuple[str, list[str]]:
    """Return the fixed header and independently truncatable metadata lines."""
    prior = record.get("audience_prior") or {}
    header = "\n".join(
        (
            f"Platform: {record['platform']}",
            f"Audience: {record['audience']}",
            f"Record type: {record.get('record_type', 'post')}",
        )
    ) + "\n"
    profile = dict(record.get("profile_weak_labels") or {})
    profile.pop("summary", None)
    profile.pop("highlights", None)
    thread = {}
    if record.get("title"):
        thread["title"] = record["title"]
    context = record.get("thread_context") or record.get("context_summary")
    if context:
        thread["context"] = context
    fields = (
        ("Community", record.get("subreddit") or record.get("community")),
        (
            "Thread context",
            thread,
        ),
        ("Engagement", record.get("engagement")),
        ("Post dimensions", record.get("dimensions")),
        ("Train-author weak labels", profile),
        ("Audience keywords", prior.get("keywords")),
        ("Audience dimensions", prior.get("dimensions")),
        (
            "Audience communities",
            prior.get("subreddits") or prior.get("communities"),
        ),
    )
    sections = []
    for label, value in fields:
        if value in (None, "", [], {}):
            continue
        rendered = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        sections.append(f"{label}: {rendered}\n")
    return header, sections
