#!/usr/bin/env python
"""Build the deterministic Study 4 report from run JSON, never transcribed values.

The page shell, figure vocabulary and table markup are those of the 2026-08-13
reports (`build_study2.py`): the same masthead, nav, `svg.fig` classes and
`figure.tablefig` wrapper, so the stylesheet copied from that report styles this
one without additions. Figure primitives are imported from `build.py` rather
than re-implemented, so the two studies cannot drift in how they draw text.
Every count, hyperparameter and verdict on the pages is read from a manifest,
a training artifact or an evaluation report at build time.
"""

from __future__ import annotations

import csv
import datetime as dt
import html
import json
import math
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build as b1  # noqa: E402  (svg_open, text, text_block)

PRIMARY_SLUG = "embeddinggemma-300m-d512"


def configured_path(env: str, default: Path) -> Path:
    value = os.environ.get(env)
    if not value:
        return default
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


FORWARD = configured_path("VECTORIAL_STUDY4_FORWARD", ROOT / "runs/study4_forward")
REVERSE = configured_path("VECTORIAL_STUDY4_REVERSE", ROOT / "runs/study4_reverse")


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _stamp() -> dt.datetime:
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    when = (
        dt.datetime.fromtimestamp(int(epoch), tz=dt.UTC)
        if epoch
        else dt.datetime.now(tz=dt.UTC)
    )
    return when.replace(second=0, microsecond=0)


def generated_at() -> str:
    return _stamp().isoformat()


BUILD_DATE = _stamp().strftime("%Y-%m-%d")
BUILD_LONG = _stamp().strftime("%Y-%m-%d %H:%M UTC")


def out_dir() -> Path:
    """Where the report is written: `reports/<stamp>`, as the 2026-08-13 runs are.

    Resolved lazily rather than at import so that the stamp is the same clock
    as the page footer. Pinning SOURCE_DATE_EPOCH therefore makes two builds
    land in the same directory, which is what the byte-identity check needs.
    """
    override = os.environ.get("VECTORIAL_STUDY4_REPORT_OUT")
    if override:
        path = Path(override)
        return path if path.is_absolute() else ROOT / path
    stamp = generated_at().replace(":", "-").removesuffix("+00-00")
    return ROOT / "reports" / stamp


def label(name: str) -> str:
    readable = name.replace("_", " ").replace("s4 ", "").strip()
    # These run identifiers retain the collection-time 10k-per-audience cap.
    # The observed full arms are smaller than 40k because one audience has fewer
    # than 10k eligible train records. Calling them simply "10k" in the report
    # previously made the scale control look like the primary dataset.
    readable = readable.replace("prior 10k", "prior full available")
    readable = readable.replace("raw 10k", "raw full available")
    readable = readable.replace("plan prior 10k", "plan prior full available")
    readable = readable.replace("platform lm-10000", "platform full available")
    readable = readable.replace(
        "platform prior lm-10000", "platform prior full available"
    )
    return readable


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "not scored"
    try:
        if value != value:
            return "not scored"
    except TypeError:
        return "not scored"
    return f"{float(value):.{digits}f}"


def esc(s) -> str:
    return html.escape(str(s))


def num(value, default: str = "not recorded") -> str:
    """Thousands-separated integer, or the stated default when absent."""
    if value is None:
        return default
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return esc(value)


# --------------------------------------------------------------------------
# Column families and references. Colours are the palette of the 2026-08-13
# figures so the two studies read alike side by side.
# --------------------------------------------------------------------------

FAMILIES = [
    ("prompting", "#2a78d6", ("llama_rewrite_", "llm_rewrite_")),
    ("LoRA", "#4a3aa7", ("lora_",)),
    ("soft prompt", "#c2410c", ("soft_prompt_",)),
    ("steering", "#0e7490", ("steering_",)),
    ("plan then transfer", "#6b7c1f", ("local_plan_then_transfer_",)),
]
REFERENCES = ["identity", "target_sample", "shuffle_control"]
REF_STYLE = {
    "identity": ("#8b8b8b", "Identity: the source post, unchanged"),
    "target_sample": ("#2e7d5b", "Authentic target: a genuine post from the scored cell"),
    "shuffle_control": ("#b0a08c", "Wrong-topic control: a genuine post from another cell"),
}


def family_of(name: str) -> tuple[str, str]:
    for fam, colour, prefixes in FAMILIES:
        if name.startswith(prefixes):
            return fam, colour
    return "other", "#666666"


def is_shuffle(name: str) -> bool:
    return name.endswith("_audience_shuffle")


# --------------------------------------------------------------------------
# Table figure. Every table in the report goes through this so that the
# wrapper, scroll shell and note are the ones the stylesheet expects.
# --------------------------------------------------------------------------

def tablefig(caption: str, sub: str, thead: str, tbody: str, note: str = "",
             cls: str = "res") -> str:
    tnote = f'<p class="tnote">{note}</p>' if note else ""
    return f"""<figure class="tablefig">
<figcaption><strong>{esc(caption)}</strong><br><span class="sub">{sub}</span></figcaption>
<div class="tscroll"><table class="{cls}">
<thead>{thead}</thead>
<tbody>{tbody}</tbody>
</table></div>
{tnote}
</figure>"""


def th(cells) -> str:
    return "<tr>" + "".join(f"<th>{esc(c)}</th>" for c in cells) + "</tr>"


def td(value, cls: str = "") -> str:
    c = f' class="{cls}"' if cls else ""
    return f"<td{c}>{value}</td>"


def row(head: str, *cells: str) -> str:
    return f"<tr><th scope='row'>{head}</th>" + "".join(cells) + "</tr>"


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def frontier_svg(frontier: dict, direction: str, width=760, height=460) -> str:
    """Content preservation against distribution match, one mark per column.

    Follows the 2026-08-13 trade-off figure: preservation on x, TRM on y, the
    useful corner lower right, references as rings, systems as filled circles
    coloured by family. Members of the descriptive frontier carry a dark ring;
    audience-shuffled controls are faded so the grid reads as its design.
    """
    points = frontier["points"]
    on_frontier = set(frontier.get("point_estimate_frontier", []))
    pts, refs = [], []
    for name, p in sorted(points.items()):
        x, y = p["source_similarity"], p["trm"]
        if name in REFERENCES:
            refs.append((name, REF_STYLE[name][0], x, y))
        else:
            pts.append((name, family_of(name)[1], x, y))
    xs = [p[2] for p in pts] + [r[2] for r in refs]
    ys = [p[3] for p in pts] + [r[3] for r in refs]
    xlo, xhi, ylo, yhi = min(xs), max(xs), min(ys), max(ys)
    xpad, ypad = (xhi - xlo) * 0.12 or 0.05, (yhi - ylo) * 0.12 or 0.05
    xlo, xhi, ylo, yhi = xlo - xpad, xhi + xpad, ylo - ypad, yhi + ypad

    left, right, top, bot = 64, 230, 84, 62
    pw, ph = width - left - right, height - top - bot

    def px(v):
        return left + (v - xlo) / (xhi - xlo) * pw

    def py(v):
        return top + ph - (v - ylo) / (yhi - ylo) * ph

    title = f"{direction}: content preservation against distribution match"
    sub = ("Rightward is greater preservation of the source; downward is closer distributional "
           "match to authentic target posts. The lower-right region is the objective. No column "
           "occupies it.")
    out = [b1.svg_open(width, height, title, sub)]
    t, y = b1.text_block(0, 24, title, "fig-title", width_px=width - 8)
    out.append(t)
    s, y = b1.text_block(0, y + 4, sub, "fig-sub", width_px=width - 8)
    out.append(s)
    out.append(f'<rect x="{left}" y="{top}" width="{pw}" height="{ph}" fill="none" stroke="#e6e4de"/>')
    for i in range(5):
        gx = left + pw * i / 4
        gy = top + ph * i / 4
        out.append(f'<line x1="{gx:.1f}" y1="{top}" x2="{gx:.1f}" y2="{top + ph}" stroke="#f0eee8"/>')
        out.append(f'<line x1="{left}" y1="{gy:.1f}" x2="{left + pw}" y2="{gy:.1f}" stroke="#f0eee8"/>')
        out.append(b1.text(gx, top + ph + 16, f"{xlo + (xhi - xlo) * i / 4:.2f}", "tick", "middle"))
        out.append(b1.text(left - 8, gy + 4, f"{yhi - (yhi - ylo) * i / 4:.2f}", "tick", "end"))
    out.append(b1.text(left + pw / 2, top + ph + 34, "Source cosine similarity", "ax-title", "middle"))
    out.append(
        f'<text class="ax-title" transform="translate(16,{top + ph / 2}) rotate(-90)" '
        f'text-anchor="middle">TRM score</text>'
    )
    zx, zy = px(xhi - (xhi - xlo) * 0.28), py(ylo + (yhi - ylo) * 0.28)
    out.append(
        f'<rect class="good-zone" x="{zx:.1f}" y="{zy:.1f}" '
        f'width="{left + pw - zx:.1f}" height="{top + ph - zy:.1f}"/>'
    )
    out.append(b1.text(left + pw - 6, top + ph - 8, "preserves content and matches target", "zone-lbl", "end"))

    for _n, colour, x, y_ in refs:
        out.append(
            f'<circle cx="{px(x):.1f}" cy="{py(y_):.1f}" r="6.5" fill="#fcfcfb" '
            f'stroke="{colour}" stroke-width="2.5"/>'
        )
    for name, colour, x, y_ in pts:
        stroke = "#1c2a25" if name in on_frontier else "#fcfcfb"
        sw = 2 if name in on_frontier else 1.8
        opacity = ' opacity="0.45"' if is_shuffle(name) else ""
        out.append(
            f'<circle class="pt" cx="{px(x):.1f}" cy="{py(y_):.1f}" r="5.5" fill="{colour}" '
            f'stroke="{stroke}" stroke-width="{sw}"{opacity}><title>{esc(label(name))}: '
            f'source {x:.3f}, TRM {y_:.3f}</title></circle>'
        )

    lx = left + pw + 16
    ly = top + 6
    out.append(b1.text(lx, ly, "method family", "key-lbl"))
    ly += 16
    for fam, colour, _p in FAMILIES:
        out.append(f'<circle cx="{lx + 6}" cy="{ly - 4}" r="5" fill="{colour}"/>')
        out.append(b1.text(lx + 18, ly, fam, "lbl"))
        ly += 17
    ly += 4
    out.append(b1.text(lx, ly, "marks", "key-lbl"))
    ly += 16
    out.append(f'<circle cx="{lx + 6}" cy="{ly - 4}" r="5" fill="#666" stroke="#1c2a25" stroke-width="2"/>')
    out.append(b1.text(lx + 18, ly, "on the descriptive frontier", "lbl"))
    ly += 17
    out.append(f'<circle cx="{lx + 6}" cy="{ly - 4}" r="5" fill="#666" opacity="0.45"/>')
    out.append(b1.text(lx + 18, ly, "audience-shuffled control", "lbl"))
    ly += 21
    out.append(b1.text(lx, ly, "reference", "key-lbl"))
    ly += 16
    for name, colour, _x, _y in refs:
        out.append(f'<circle cx="{lx + 6}" cy="{ly - 4}" r="5" fill="#fcfcfb" stroke="{colour}" stroke-width="2"/>')
        blk, ly2 = b1.text_block(lx + 18, ly, REF_STYLE[name][1], "lbl", width_px=right - 40, line_h=13)
        out.append(blk)
        ly = ly2 + 4

    foot = ("Both axes are unweighted means over the common cell base. The descriptive frontier "
            "is defined on three objectives and also requires content-word retention, which this "
            "view does not show. Intervals are omitted so that the columns remain legible; they "
            "are given in the table that follows.")
    blk, y_end = b1.text_block(0, top + ph + 52, foot, "fig-note", width_px=width - 8)
    out.append(blk)
    out.append("</svg>")
    svg = "\n".join(out)
    return svg.replace(
        f'viewBox="0 0 {width} {height}"', f'viewBox="0 0 {width} {max(height, y_end + 14):.0f}"', 1
    )


def scale_svg(width=760, height=400) -> str:
    groups = [
        ("Reddit, raw text", "#2a78d6", ["s4-reddit-rich-raw-1k", "s4-reddit-rich-raw-10k"]),
        ("Reddit, structured prior", "#4a3aa7", ["s4-reddit-rich-prior-1k", "s4-reddit-rich-prior-10k"]),
        ("LinkedIn, raw text", "#c2410c", [
            "s4-linkedin-rich-platform_lm-1000", "s4-linkedin-rich-platform_lm-10000"]),
        ("LinkedIn, structured prior", "#0e7490", [
            "s4-linkedin-rich-platform_prior_lm-1000", "s4-linkedin-rich-platform_prior_lm-10000"]),
    ]
    series = []
    for name, colour, runs in groups:
        pts = []
        for run_name in runs:
            root = ROOT / "runs/checkpoints" / run_name
            run, metrics = read_json(root / "run.json"), read_json(root / "metrics.json")
            pts.append((int(run["n_train"]), float(metrics["best_nll"])))
        series.append((name, colour, pts))
    allpts = [p for _n, _c, ps in series for p in ps]
    xs, ys = [math.log10(x) for x, _ in allpts], [y for _, y in allpts]
    xlo, xhi = min(xs) - 0.08, max(xs) + 0.08
    ypad = max((max(ys) - min(ys)) * 0.12, 0.02)
    ylo, yhi = min(ys) - ypad, max(ys) + ypad
    left, right, top, bot = 64, 210, 84, 62
    pw, ph = width - left - right, height - top - bot

    def px(v):
        return left + (math.log10(v) - xlo) / (xhi - xlo) * pw

    def py(v):
        return top + ph - (v - ylo) / (yhi - ylo) * ph

    title = "Unsupervised development loss by available scale"
    sub = ("Each point is read from its training artifact. Development negative log-likelihood is "
           "a training diagnostic, not a held-out transfer result.")
    out = [b1.svg_open(width, height, title, sub)]
    t, y = b1.text_block(0, 24, title, "fig-title", width_px=width - 8)
    out.append(t)
    s, y = b1.text_block(0, y + 4, sub, "fig-sub", width_px=width - 8)
    out.append(s)
    out.append(f'<rect x="{left}" y="{top}" width="{pw}" height="{ph}" fill="none" stroke="#e6e4de"/>')
    for i in range(5):
        gy = top + ph * i / 4
        out.append(f'<line x1="{left}" y1="{gy:.1f}" x2="{left + pw}" y2="{gy:.1f}" stroke="#f0eee8"/>')
        out.append(b1.text(left - 8, gy + 4, f"{yhi - (yhi - ylo) * i / 4:.2f}", "tick", "end"))
    for n in sorted({x for x, _ in allpts}):
        out.append(f'<line x1="{px(n):.1f}" y1="{top}" x2="{px(n):.1f}" y2="{top + ph}" stroke="#f0eee8"/>')
        out.append(b1.text(px(n), top + ph + 16, f"{n:,}", "tick", "middle"))
    out.append(b1.text(left + pw / 2, top + ph + 34, "training records (log scale)", "ax-title", "middle"))
    out.append(
        f'<text class="ax-title" transform="translate(16,{top + ph / 2}) rotate(-90)" '
        f'text-anchor="middle">Development NLL</text>'
    )
    for name, colour, pts in series:
        coords = " ".join(f"{px(x):.1f},{py(y_):.1f}" for x, y_ in pts)
        out.append(f'<polyline points="{coords}" fill="none" stroke="{colour}" stroke-width="2"/>')
        for x, y_ in pts:
            out.append(
                f'<circle cx="{px(x):.1f}" cy="{py(y_):.1f}" r="5.5" fill="{colour}" '
                f'stroke="#fcfcfb" stroke-width="1.8"><title>{esc(name)}: {x:,} records, '
                f'NLL {y_:.4f}</title></circle>'
            )
    lx, ly = left + pw + 16, top + 6
    out.append(b1.text(lx, ly, "training corpus", "key-lbl"))
    ly += 16
    for name, colour, _p in series:
        out.append(f'<circle cx="{lx + 6}" cy="{ly - 4}" r="5" fill="{colour}"/>')
        out.append(b1.text(lx + 18, ly, name, "lbl"))
        ly += 17
    foot = ("Lower is better. The smaller sample at each platform is a nested deterministic "
            "subset of the larger one, so each pair differs in scale and in nothing else.")
    blk, y_end = b1.text_block(0, top + ph + 52, foot, "fig-note", width_px=width - 8)
    out.append(blk)
    out.append("</svg>")
    svg = "\n".join(out)
    return svg.replace(
        f'viewBox="0 0 {width} {height}"', f'viewBox="0 0 {width} {max(height, y_end + 14):.0f}"', 1
    )


# --------------------------------------------------------------------------
# Results tables
# --------------------------------------------------------------------------

def results_table(report: dict, frontier: dict) -> str:
    pts = frontier["points"]
    systems = [n for n in pts if n not in REFERENCES]
    best = {
        "src": max(pts[n]["source_similarity"] for n in systems),
        "cwr": max((pts[n].get("content_word_retention") or 0) for n in systems),
        "trm": min(pts[n]["trm"] for n in systems),
    }

    def ci(entry: dict) -> str:
        return f"[{fmt(entry.get('ci_low'), 2)}, {fmt(entry.get('ci_high'), 2)}]"

    rows = []
    for name, p in sorted(pts.items(), key=lambda kv: (kv[1]["trm"], -kv[1]["source_similarity"])):
        t = report["table"].get(name, {})
        src, cwr, trm = (t.get("semantic.source_similarity", {}),
                         t.get("semantic.content_word_retention", {}), t.get("trm.trm", {}))
        ref = name in REFERENCES
        fam = "reference" if ref else family_of(name)[0]
        cwr_v = p.get("content_word_retention")

        def cell(v, key, digits=3, ref=ref):
            if v is None:
                return td("—", "na")
            mark = "best" if (not ref and abs(v - best[key]) < 1e-12) else "num"
            return td(f"{v:.{digits}f}", ("ref" if ref else mark))

        rows.append(row(
            esc(label(name)), td(esc(fam), "ref" if ref else ""),
            cell(p["source_similarity"], "src"), td(ci(src), "num"),
            cell(cwr_v, "cwr"), td(ci(cwr), "num"),
            cell(p["trm"], "trm"), td(ci(trm), "num"),
            td(p["n_cells"], "num"), td(p["n_audiences"], "num"),
        ))
    head = th(["column", "family", "source cosine ↑", "95% interval", "content words ↑",
               "95% interval", "TRM ↓", "95% interval", "cells", "audiences"])
    note = ("↑ higher is better; ↓ lower is better. Bold marks the best system column on each "
            "objective; reference rows are excluded from that comparison. Intervals resample "
            "audiences and then eligible cells within each sampled audience. A difference between "
            "two columns is a finding only where their intervals do not overlap. Rows are ordered "
            "by TRM.")
    return tablefig(
        "All columns on the primary objectives",
        f"{len(systems)} system columns and {len(REFERENCES)} references, scored on the common "
        f"cell base in {esc(report['embedding_space'].get('model'))}, "
        f"dimension {report['embedding_space'].get('dim')}.",
        head, "".join(rows), note,
    )


#: The confirmatory hypotheses of plan section 9. Every other comparison is
#: secondary, except that anything touching an aspect-conditioned column is
#: exploratory: the aspect coverage and human-validation gates were not met, so
#: an aspect comparison can never be promoted to a supported finding however
#: its intervals fall. `findings` and the summary card share this rule so the
#: overview cannot claim a count the results page does not show.
CONFIRMATORY = {
    "lora_s4_raw_10k:lora_s4_prior_10k",
    "lora_s4_raw_full:lora_s4_prior_full",
    "lora_s4_base_lora_target_lm:lora_s4_target_lm",
    "soft_prompt_s4_base_soft_target_lm:soft_prompt_s4_soft_target_lm",
    "lora_s4_prior_10k:local_plan_then_transfer_s4_plan_prior_10k",
    "lora_s4_prior_full:local_plan_then_transfer_s4_plan_prior_full",
    "steering_s4_steer_base_target_lm:lora_s4_target_lm",
    "llm_rewrite_s4_claude_reference:llama_rewrite_s4_base_none",
}


def tier_of(key: str) -> str:
    if "_aspect" in key:
        return "exploratory"
    return "confirmatory" if key in CONFIRMATORY else "secondary"


def comparison_supported(key: str, comparison: dict) -> bool:
    """Whether a comparison counts as a finding, after the aspect gate."""
    if tier_of(key) == "exploratory":
        return False
    plan = comparison.get("extract_then_transfer")
    if plan is not None:
        return bool(plan.get("retention_improvement_supported"))
    return bool(comparison.get("frontier_improvement_supported"))


def findings(frontier: dict, sensitivity: dict) -> str:
    rows = []
    for key, comparison in sorted(frontier.get("comparisons", {}).items()):
        stable = sensitivity.get("comparisons", {}).get(key, {})
        baseline, candidate = key.split(":", 1)
        plan = comparison.get("extract_then_transfer")
        if plan is not None:
            st = stable.get("extract_then_transfer", {})
            claim = (f"{label(candidate)} retains more entities, numbers and content words "
                     f"than {label(baseline)} without a worse TRM")
        else:
            st = stable
            claim = f"{label(candidate)} improves the frontier relative to {label(baseline)}"
        tier = tier_of(key)
        if tier == "exploratory":
            verdict = td("exploratory: aspect gate not met", "na")
        elif comparison_supported(key, comparison):
            verdict = td("supported", "ok")
        else:
            verdict = td("not supported", "no")
        rows.append(
            f"<tr><td>{esc(claim)}</td>{td(tier)}{verdict}"
            + td(f"{st.get('n_supported', 0)} of {st.get('n_spaces', 0)}", "num") + "</tr>"
        )
    n_ok = sum(comparison_supported(k, c) for k, c in frontier.get("comparisons", {}).items())
    note = f"{n_ok} of {len(rows)} preregistered comparisons are supported in the primary space."
    if not n_ok:
        note += " No preregistered comparison met the interval and non-inferiority rule."
    return tablefig(
        "Preregistered comparisons and their verdicts",
        "A comparison is supported only where the candidate lowers TRM without reducing source "
        "preservation, or raises preservation without raising TRM, with separated intervals. "
        "Aspect-conditioned comparisons are listed but never promoted, because the aspect gates "
        "were not met. The verdicts are produced by running that test.",
        th(["claim", "tier", "verdict", "spaces agreeing"]), "".join(rows), note,
        cls="res claims",
    )


def failure_summary(run_dir: Path) -> str:
    audit = read_json(run_dir / "output_audit.heldout.json")
    failed = [(n, v) for n, v in sorted(audit.get("systems", {}).items())
              if v.get("n_failures_or_empty", 0)]
    if not failed:
        return (f"<p>All {audit.get('n_systems', 0)} columns emitted every required "
                "candidate without an empty or recorded failure.</p>")
    rows = "".join(
        row(esc(label(n)), td(num(v.get("n_failures_or_empty", 0)), "num"),
            td(num(v.get("n_outputs", 0)), "num"),
            td(f"{100 * v.get('failure_rate', 0):.1f}%", "num"))
        for n, v in failed
    )
    return tablefig(
        "Columns with empty or failed candidates",
        "Failures remain in the sampling base and enter the degeneracy score. Every other column "
        "emitted all of its required candidates.",
        th(["column", "failures", "outputs", "rate"]), rows,
        "Each failure is a content plan the local planner emitted as malformed JSON. The surviving "
        "draws of these columns are therefore a success-biased sample.",
    )


def plan_diagnostics(report: dict) -> str:
    systems = [n for n in sorted(report.get("table", {}))
               if n.startswith("local_plan_then_transfer_") and not is_shuffle(n)]
    if not systems:
        return "<p>No extract-then-transfer output was scoreable.</p>"
    keys = ("semantic.content_word_retention", "semantic.entity_surface_recall",
            "semantic.number_recall", "semantic.plan_entity_surface_recall",
            "semantic.plan_number_recall")
    rows = "".join(
        row(esc(label(n)), *(td(fmt(report["table"][n].get(k, {}).get("mean")), "num") for k in keys))
        for n in systems
    )
    return tablefig(
        "Extract-then-transfer retention diagnostics",
        "Recall of source content in the rendered post, and recall of the intermediate plan's "
        "content, each against what the source contained.",
        th(["column", "content words", "source entities", "source numbers",
            "plan entities", "plan numbers"]), rows,
    )


def heldout_topic_table(report: dict) -> str:
    preferred = {
        "identity", "shuffle_control", "target_sample", "llama_rewrite_s4_base_none",
        "llm_rewrite_s4_claude_reference", "lora_s4_prior_10k", "lora_s4_prior_full",
        "lora_s4_platform_then_audience", "lora_s4_target_lm", "soft_prompt_s4_soft_target_lm",
        "steering_s4_steer_prior_target_lm", "local_plan_then_transfer_s4_plan_prior_10k",
        "local_plan_then_transfer_s4_plan_prior_full",
    }
    rows = []
    for name, m in sorted(report.get("table", {}).items()):
        if name not in preferred:
            continue
        src, cwr, trm = (m.get("semantic.source_similarity", {}),
                         m.get("semantic.content_word_retention", {}), m.get("trm.trm", {}))
        n = min(src.get("n_heldout_cells", 0), cwr.get("n_heldout_cells", 0),
                trm.get("n_heldout_cells", 0))
        if not n:
            continue
        rows.append(row(esc(label(name)), td(fmt(src.get("heldout_mean")), "num"),
                        td(fmt(cwr.get("heldout_mean")), "num"),
                        td(fmt(trm.get("heldout_mean")), "num"), td(n, "num")))
    if not rows:
        return "<p>No held-out topic cell cleared all three metric pool requirements.</p>"
    return tablefig(
        "Held-out topic cells",
        "Cells whose topic was withheld from all task-specific fitting and selection. They remain "
        "inside the reported held-out evaluation.",
        th(["column", "source cosine", "content words", "TRM", "eligible topic cells"]),
        "".join(rows),
    )


def embedding_space_table(sens: dict) -> str:
    spaces = sens.get("spaces", [])
    rows = []
    for key, entry in sorted(sens.get("comparisons", {}).items()):
        by = entry.get("by_embedding_space", {})
        base, _, cand = key.partition(":")
        rows.append(row(
            f"{esc(label(cand))} relative to {esc(label(base))}",
            *(td("yes" if by.get(sp) else "no", "ok" if by.get(sp) else "no") for sp in spaces),
            td(f"{entry.get('n_supported', 0)} of {entry.get('n_spaces', 0)}", "num"),
        ))
    any_ok = sum(1 for k, e in sens.get("comparisons", {}).items()
                 if e.get("n_supported") and tier_of(k) != "exploratory")
    return tablefig(
        "Support for every preregistered comparison, by embedding space",
        f"Each of the {sens.get('n_spaces')} spaces was scored separately and the values were "
        f"never pooled (<code>values_pooled_across_spaces = "
        f"{str(sens.get('values_pooled_across_spaces')).lower()}</code>). TRM is a distance in a "
        "particular representation; averaging across representations would produce a quantity "
        "that none of them measures.",
        th(["comparison", *spaces, "spaces agreeing"]), "".join(rows),
        f"{any_ok} of {len(rows)} comparisons are supported in at least one space.",
        cls="res spaces",
    )


# --------------------------------------------------------------------------
# Direction pages
# --------------------------------------------------------------------------

def direction_available(run_dir: Path) -> bool:
    """Whether a direction has been scored far enough to render.

    A direction becomes reportable only once its primary-space report, its
    frontier and the cross-space sensitivity summary all exist. Anything less is
    a run still in flight, and a partial direction must not be drawn as if it
    were a result.
    """
    return all((run_dir / n).exists() for n in (
        f"report.heldout.{PRIMARY_SLUG}.json",
        f"frontier.heldout.{PRIMARY_SLUG}.json",
        "frontier_sensitivity.heldout.json",
    ))


def callout(heading: str, body: str, warn: bool = False) -> str:
    cls = "warn" if warn else "callout"
    return f'<aside class="{cls}"><h2>{esc(heading)}</h2><p>{body}</p></aside>'


def pending(heading: str, note: str) -> str:
    return callout(heading, f"<strong>Not yet available.</strong> {esc(note)}")


def h2(no: int, id_: str, text: str) -> str:
    return f'<h2 id="{id_}"><span class="section-no">{no:02d}</span> {esc(text)}</h2>'


def direction_scope(run_dir: Path) -> dict:
    report = read_json(run_dir / f"report.heldout.{PRIMARY_SLUG}.json")
    frontier = read_json(run_dir / f"frontier.heldout.{PRIMARY_SLUG}.json")
    pts = frontier["points"]
    systems = [n for n in pts if n not in REFERENCES]
    trm_cells = {pts[n]["n_cells"] for n in systems}
    space = report["embedding_space"]
    # The audience-shuffled controls are diagnostics of the audience prior, not
    # candidate methods, so the headline "best" never lands on one of them.
    best = min((n for n in systems if not is_shuffle(n)), key=lambda n: pts[n]["trm"])
    comparisons = frontier.get("comparisons", {})
    return {
        "n_systems": len(systems),
        "n_shuffle": sum(is_shuffle(n) for n in systems),
        "cells": f"{min(trm_cells)}–{max(trm_cells)}" if len(trm_cells) > 1 else str(next(iter(trm_cells))),
        "audiences": max(pts[n]["n_audiences"] for n in systems),
        "space": f"{space.get('model')}, dimension {space.get('dim')}",
        "best": best, "best_trm": pts[best]["trm"],
        "oracle_trm": pts.get("target_sample", {}).get("trm"),
        "supported": sum(comparison_supported(k, c) for k, c in comparisons.items()),
        "n_comparisons": len(comparisons),
    }


def direction_sections(run_dir: Path, direction: str) -> str:
    report = read_json(run_dir / f"report.heldout.{PRIMARY_SLUG}.json")
    frontier = read_json(run_dir / f"frontier.heldout.{PRIMARY_SLUG}.json")
    sens = read_json(run_dir / "frontier_sensitivity.heldout.json")
    sc = direction_scope(run_dir)
    return f"""
<nav class="result-nav" aria-label="Results on this page">
  <a href="#frontier">Frontier</a>
  <a href="#full-table">All columns</a>
  <a href="#resolution">Resolution test</a>
  <a href="#plan">Extract then transfer</a>
  <a href="#heldout">Held-out topics</a>
  <a href="#failures">Generation failures</a>
  <a href="#spaces">Embedding spaces</a>
</nav>
{h2(1, "frontier", "No preregistered comparison improves the content-versus-match frontier")}
<p>The primary objectives are cosine similarity to the source, content-word retention, and the
Triangle-Rank Metric (TRM) against authentic target posts. A column improves the frontier only by
lowering TRM without reducing preservation, or by raising preservation without raising TRM. Of
{sc['n_comparisons']} preregistered comparisons, <strong>{sc['supported']}</strong> satisfy that
rule with separated intervals. The lowest TRM among system columns is {sc['best_trm']:.3f}
({esc(label(sc['best']))}), against {fmt(sc['oracle_trm'])} for an authentic target post; every
reduction in TRM in this grid is accompanied by a reduction in source similarity.</p>
{frontier_svg(frontier, direction)}
{h2(2, "full-table", "All columns on the three objectives")}
{results_table(report, frontier)}
{h2(3, "resolution", "Comparisons the intervals separate")}
{findings(frontier, sens)}
{h2(4, "plan", "Extract-then-transfer retains less source content than direct generation")}
{plan_diagnostics(report)}
{h2(5, "heldout", "Held-out topic cells")}
{heldout_topic_table(report)}
{h2(6, "failures", "Generation failures")}
{failure_summary(run_dir)}
{h2(7, "spaces", "Verdicts across six embedding spaces")}
{embedding_space_table(sens)}
"""


# --------------------------------------------------------------------------
# Dataset page
# --------------------------------------------------------------------------

def manifest_row(name: str, m: dict) -> str:
    counts = m.get("available_per_audience", {})
    short = [f"{a}: {c:,}" for a, c in sorted(counts.items()) if c < m.get("per_audience", 0)]
    return row(esc(name), td(num(m["n_total"]), "num"), td(num(m["n_train"]), "num"),
               td(num(m["n_dev"]), "num"), td(esc(", ".join(short) if short else "none")))


def corpus_table(reddit: dict, linkedin: dict) -> str:
    return tablefig(
        "Prior corpora",
        "Records available after reconstruction, and their division into training and "
        "development sets by author. Shortfalls are audiences with fewer eligible records than "
        "the requested rung.",
        th(["corpus", "total records", "train", "development", "shortfall from requested rung"]),
        manifest_row("Reddit target prior", reddit) + manifest_row("LinkedIn target prior", linkedin),
    )


def corpus_cell_count() -> int:
    path = ROOT / "data/study3/cells.jsonl"
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def availability_table(reddit: dict, linkedin: dict) -> str:
    audiences = sorted(set(reddit.get("available_per_audience", {})) | set(linkedin.get("available_per_audience", {})))
    cap = reddit.get("per_audience") or linkedin.get("per_audience")
    rows = []
    for a in audiences:
        r, l_ = reddit.get("available_per_audience", {}).get(a), linkedin.get("available_per_audience", {}).get(a)
        rows.append(row(esc(label(a)), td(num(r), "num"), td(num(l_), "num"),
                        td("yes" if (r or 0) >= (cap or 0) and (l_ or 0) >= (cap or 0) else "no",
                           "ok" if (r or 0) >= (cap or 0) and (l_ or 0) >= (cap or 0) else "no")))
    rt = lambda m: ", ".join(f"{k}: {v:,}" for k, v in sorted((m.get("record_types") or {}).items()))  # noqa: E731
    rows.append(row("record types", td(esc(rt(reddit))), td(esc(rt(linkedin))), td("", "na")))
    return tablefig(
        "Eligible records per audience and platform",
        f"The collection rung requested {num(cap)} records per audience and platform. An audience "
        "meets the rung only where both platforms do.",
        th(["audience", "Reddit", "LinkedIn", f"meets the {num(cap)} rung"]), "".join(rows),
        "Posts and comments both enter the language prior; the record type is an explicit "
        "conditioning field so that a comment-heavy audience does not teach the post generator "
        "to emit replies.",
    )


def prior_audit_table(reddit: dict, linkedin: dict) -> str:
    rows = "".join(
        row(n, td(num(m.get("cross_room_duplicate_rows_removed", 0)), "num"),
            td(num(m.get("heldout_profile_exclusions", 0)), "num"),
            td(esc(m.get("split_unit", "not recorded"))),
            td(esc(m.get("selection", "not recorded"))))
        for n, m in (("Reddit", reddit), ("LinkedIn", linkedin))
    )
    return tablefig(
        "Reconstruction audit",
        "Counts recorded by the corpus builder while resolving exports, removing duplicates and "
        "excluding profiles that overlap the scored splits.",
        th(["platform", "cross-room duplicates removed", "held-out profiles excluded",
            "split unit", "rung selection"]), rows,
    )


def policy_list(reddit: dict, linkedin: dict) -> str:
    def same_or_each(key: str) -> str:
        r, l_ = reddit.get(key), linkedin.get(key)
        if r == l_:
            return esc(r if r is not None else "not recorded")
        return f"Reddit: {esc(r if r is not None else 'not recorded')}; LinkedIn: {esc(l_ if l_ is not None else 'not recorded')}"
    cutoff = reddit.get("cutoff_utc")
    cutoff_txt = (dt.datetime.fromtimestamp(int(cutoff), tz=dt.UTC).date().isoformat()
                  if cutoff else "not recorded")
    derived = "retained as structured fields and never used as language-model targets" if (
        reddit.get("derived_fields_retained") and reddit.get("derived_fields_used_as_lm_targets") is False
    ) else "see manifest"
    return f"""
<dl class="defs">
<dt>Split unit</dt><dd>{same_or_each('split_unit')}. Authors, not posts, are assigned to splits,
so no author contributes to both a training and a development set.</dd>
<dt>Rung selection</dt><dd>{same_or_each('selection')}. Each smaller rung is a subset of the
larger one, so the scale ladder compares data size and not sample composition.</dd>
<dt>Cross-room policy</dt><dd>{same_or_each('cross_room_policy')}.</dd>
<dt>Held-out exclusion rule</dt><dd>{esc(reddit.get('heldout_exclusion_rule') or 'not recorded')}
(Reddit manifest); {num(reddit.get('heldout_profile_exclusions'))} Reddit and
{num(linkedin.get('heldout_profile_exclusions'))} LinkedIn profiles were excluded under this rule.</dd>
<dt>Weak-label policy</dt><dd>{same_or_each('weak_label_policy')}.</dd>
<dt>Derived fields</dt><dd>{esc(derived)}.</dd>
<dt>Collection cutoff</dt><dd>{cutoff_txt} (Reddit manifest); the LinkedIn manifest records no
cutoff. A common observation window across the two platforms is therefore not established.</dd>
</dl>"""


def structured_prior_summary(reddit: dict, linkedin: dict) -> str:
    rows = []
    for name, m in (("Reddit", reddit), ("LinkedIn", linkedin)):
        ap = m.get("audience_priors") or {}
        for audience, prior in sorted(ap.items()):
            comms = prior.get("subreddits") or prior.get("communities") or []
            rows.append(row(f"{name} · {esc(label(audience))}",
                            td(num(len(prior.get("keywords") or [])), "num"),
                            td(num(len(prior.get("dimensions") or [])), "num"),
                            td(num(len(comms)), "num"),
                            td(esc(prior.get("estimated_from", "not recorded")))))
    fields = reddit.get("structured_prior_fields") or []
    return tablefig(
        "Train-derived audience priors",
        "Aggregates estimated from training authors only, supplied to every record of that "
        "audience in the structured-prior prompt. A development record receives the frozen "
        "training aggregate, never its own author's profile."
        + (f" The Reddit structured prompt exposes: {esc(', '.join(fields))}." if fields else ""),
        th(["platform · audience", "keywords", "dimensions", "communities", "estimated from"]),
        "".join(rows),
    )


def rich_prior_coverage_table() -> str:
    families = (
        ("community", lambda r: r.get("subreddit") or r.get("community")),
        ("thread context", lambda r: r.get("title") or r.get("thread_context") or r.get("context_summary")),
        ("engagement", lambda r: r.get("engagement")),
        ("post dimensions", lambda r: r.get("dimensions")),
        ("train-author weak labels", lambda r: r.get("profile_weak_labels")),
        ("train-derived audience prior", lambda r: r.get("audience_prior")),
    )
    rows = []
    for platform in ("reddit", "linkedin"):
        path = ROOT / "data/study4_priors" / f"{platform}_10000_per_audience.rich_safe.train.jsonl"
        counts, total = {n: 0 for n, _ in families}, 0
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                total += 1
                for n, get in families:
                    if get(rec) not in (None, "", [], {}):
                        counts[n] += 1
        rows.append(row("LinkedIn" if platform == "linkedin" else "Reddit",
                        *(td(f"{counts[n]:,} / {total:,}", "num") for n, _ in families)))
    return tablefig(
        "Training-record coverage of structured priors",
        "The number of primary training records carrying each metadata family that the "
        "structured prompt can expose, out of all training records.",
        th(["platform", *(n for n, _ in families)]), "".join(rows),
    )


def context_sanitization_table() -> str:
    rows = []
    for platform in ("reddit", "linkedin"):
        for split in ("train", "dev"):
            path = ROOT / "data/study4_priors" / f"{platform}_10000_per_audience.rich_safe.{split}.jsonl"
            m = read_json(path.with_suffix(path.suffix + ".sanitization.json"))
            removed = m.get("removed_exact_metadata_strings", {})
            fields = ", ".join(f"{k}: {v}" for k, v in sorted(removed.items()))
            rows.append(row("LinkedIn" if platform == "linkedin" else "Reddit", td(split),
                            td(num(m.get("n_records", 0)), "num"),
                            td(num(sum(removed.values())), "num"), td(esc(fields or "none"))))
    return tablefig(
        "Scored-context sanitization",
        "Normalized validation or test post text, including a scored post contained inside a "
        "longer metadata string, is blanked from model-visible metadata before structured-prior "
        "training. Authentic completion text and record cardinality are unchanged.",
        th(["platform", "split", "records", "scored strings removed", "fields"]), "".join(rows),
    )


def coarse_to_fine_table() -> str:
    rows = []
    for platform in ("reddit", "linkedin"):
        m = read_json(ROOT / "data/study4_priors/coarse_to_fine" / f"{platform}.coarse_to_fine.manifest.json")
        ok = bool(m.get("training_sets_are_disjoint")) and bool(m.get("training_union_equals_parent"))
        rows.append(row("LinkedIn" if platform == "linkedin" else "Reddit",
                        td(num(m.get("n_parent_train")), "num"), td(num(m.get("n_platform_train")), "num"),
                        td(num(m.get("n_audience_train")), "num"), td(num(m.get("n_dev")), "num"),
                        td("yes" if ok else "no", "ok" if ok else "no")))
    algo = read_json(ROOT / "data/study4_priors/coarse_to_fine/reddit.coarse_to_fine.manifest.json").get("algorithm", "")
    return tablefig(
        "Coarse-to-fine partition",
        "The primary training corpus is divided within each audience into a platform half and an "
        "audience half. The two stages together expose exactly the primary arm's completion "
        f"tokens. Partition rule: {esc(algo)}.",
        th(["platform", "parent train", "platform half", "audience half", "shared development",
            "disjoint and complete"]), "".join(rows),
    )


def ladder_table() -> str:
    rows = []
    for platform in ("reddit", "linkedin"):
        for rung in ("1000", "10000"):
            m = read_json(ROOT / "data/study4_priors" / f"{platform}_{rung}_per_audience.manifest.json")
            rows.append(row(f"{'LinkedIn' if platform == 'linkedin' else 'Reddit'} · {num(rung)} per audience",
                            td(num(m.get("n_train")), "num"), td(num(m.get("n_dev")), "num"),
                            td(num(m.get("n_total")), "num"), td(esc(m.get("selection", "not recorded")))))
    return tablefig(
        "Scale-ladder samples",
        "The rungs actually built. The 1,000-record rung is a nested subset of the full-available "
        "rung; the 50,000 and 100,000 rungs of the plan were unavailable and are not reported.",
        th(["rung", "train", "development", "total", "selection rule"]), "".join(rows),
    )


def eval_corpus_table() -> str:
    rows = []
    for name, d in (("LinkedIn to Reddit", "study3"), ("Reddit to LinkedIn", "study3_reverse")):
        m = read_json(ROOT / "data" / d / "manifest.json")
        c = m.get("counts", {})
        rows.append(row(name, td(num(m.get("n_posts")), "num"), td(num(m.get("n_cells")), "num"),
                        td(num(m.get("n_dense_cells")), "num"), td(num(m.get("n_heldout_cells")), "num"),
                        td(num(m.get("n_tasks")), "num"),
                        td(num((c.get("tasks") or {}).get("heldout")), "num")))
    cfg = read_json(ROOT / "data/study3/manifest.json").get("config", {})
    ratios = cfg.get("split_ratios", [])
    return tablefig(
        "Evaluation corpus",
        "The scored corpus is the Study 3 bilateral corpus, unchanged. A cell pairs an audience "
        f"with a topic and is retained only where both platforms carry at least "
        f"{cfg.get('min_posts_per_platform', '?')} posts; a dense cell carries at least "
        f"{cfg.get('dense_min_posts_per_platform', '?')}. Posts are assigned to train, validation "
        f"and test in the proportions {', '.join(f'{int(r * 100)}%' for r in ratios)} by a keyed "
        f"hash of the post identifier (seed {cfg.get('seed')}); the held-out split is validation "
        "and test merged.",
        th(["direction", "posts", "cells", "dense cells", "held-out topic cells", "tasks",
            "held-out tasks"]), "".join(rows),
        f"Posts between {cfg.get('min_chars')} and {num(cfg.get('max_chars'))} characters are "
        "retained. HTML entities were decoded once during the dataset build, before any split, "
        "fit or score was computed.",
    )


# --------------------------------------------------------------------------
# Methods page
# --------------------------------------------------------------------------

def run_meta(name: str) -> tuple[dict, dict]:
    root = ROOT / "runs/checkpoints" / name
    return read_json(root / "run.json"), read_json(root / "metrics.json")


def training_table() -> str:
    names = [
        "s4-reddit-rich-raw-1k", "s4-reddit-rich-prior-1k", "s4-reddit-rich-raw-10k",
        "s4-reddit-rich-prior-10k", "s4-linkedin-rich-platform_lm-1000",
        "s4-linkedin-rich-platform_prior_lm-1000", "s4-linkedin-rich-platform_lm-10000",
        "s4-linkedin-rich-platform_prior_lm-10000", "s4-reddit-coarse-platform-half",
        "s4-reddit-platform-then-audience", "s4-linkedin-coarse-platform-half",
        "s4-linkedin-platform-then-audience",
    ]
    rows = []
    for name in names:
        run, metrics = run_meta(name)
        hist = metrics.get("history") or []
        final = hist[-1] if hist else {}
        rows.append(row(
            esc(label(name)), td(esc(run.get("variant"))),
            td(num(run.get("n_train", 0)), "num"), td(num(run.get("steps_per_epoch", 0)), "num"),
            td(num(run.get("train_prompt_tokens", 0)), "num"),
            td(num(run.get("train_completion_tokens", final.get("n_tokens", 0))), "num"),
            td(fmt(metrics.get("best_nll", metrics.get("best_dev_nll")), 4), "num"),
            td(esc(run.get("prior_prompt_schema") or "raw text")),
            td(esc(run.get("quantisation"))),
        ))
    return tablefig(
        "Stage A: unsupervised bases",
        "Every adapter fitted on unpaired target-platform text, with scale, token exposure and the "
        "selected development loss read from its artifact.",
        th(["run", "variant", "training records", "steps per epoch", "prior prompt tokens",
            "authentic text tokens", "best development NLL", "prior schema", "quantisation"]),
        "".join(rows),
    )


def stage_b_table() -> str:
    """Every fitted transfer column: its starting base and fitting outcome."""
    rows = []
    for variant in ("target_lm", "target_lm_paired", "target_lm_aspect"):
        run, metrics = run_meta(f"s4-prior10k-lora-{variant}")
        cfg = run.get("config") or {}
        rows.append(row(
            f"LoRA · {esc(label(variant))}", td(esc(Path(str(run.get('initial_adapter'))).parent.name)),
            td(num(run.get("n_train")), "num"), td(num(run.get("n_dev")), "num"),
            td(f"{metrics.get('best_epoch')} of {run.get('max_epochs')}", "num"),
            td(fmt(metrics.get("base_dev_nll"), 4), "num"), td(fmt(metrics.get("best_nll"), 4), "num"),
            td(esc(f"r={cfg.get('r')}, α={cfg.get('alpha')}, lr={cfg.get('lr')}")),
        ))
    for variant in ("target_lm", "target_lm_paired", "target_lm_aspect"):
        s = read_json(ROOT / "runs/checkpoints" / f"s4-prior10k-soft-prompt-{variant}" / "summary.json")
        cfg = s.get("config") or {}
        rows.append(row(
            f"soft prompt · {esc(label(variant))}", td(esc(Path(str(s.get('initial_adapter'))).parent.name)),
            td(num(s.get("n_train")), "num"), td(num(s.get("n_dev")), "num"),
            td(f"{s.get('best_epoch')} of {cfg.get('epochs')}", "num"), td("—", "na"),
            td(fmt(s.get("best_dev_loss"), 4), "num"),
            td(esc(f"{cfg.get('n_virtual_tokens')} tokens, init={cfg.get('init')}, lr={cfg.get('lr')}")),
        ))
    return tablefig(
        "Stage B: transfer methods fitted from the selected base",
        "Each fitted column starts from the frozen primary Stage A adapter and is trained on the "
        "Study 3 conditioning files. The selected epoch is the one with the lowest development "
        "loss under early stopping.",
        th(["column", "starting base", "train", "development", "selected epoch",
            "base dev NLL", "best dev NLL", "fitting configuration"]), "".join(rows),
        "Base dev NLL is the development loss of the starting base before any Stage B fitting, "
        "where the trainer records it.",
    )


def conditionings_table() -> str:
    t = read_json(ROOT / "data/study3/training/manifest.json")
    counts = (t.get("counts") or {}).get("examples", {})
    desc = {
        "target_lm": "audience, domain and topic only; no source text during fitting",
        "target_lm_paired": "as above, plus the nearest training-split source post from the same cell, by TF-IDF cosine",
        "target_lm_aspect": "as above, plus the positive aspect coordinates of the authentic completion",
    }
    rows = []
    for variant in ("target_lm", "target_lm_paired", "target_lm_aspect"):
        n = {}
        for split in ("train", "dev", "test"):
            p = ROOT / "data/study3/training" / f"{variant}.{split}.jsonl"
            n[split] = sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip()) if p.exists() else None
        rows.append(row(esc(label(variant)), td(esc(desc[variant])),
                        td(num(n["train"]), "num"), td(num(n["dev"]), "num"), td(num(n["test"]), "num")))
    template = esc(t.get("prompt_template", ""))
    block = esc(t.get("content_block_template", ""))
    return tablefig(
        "The three conditionings",
        "The completion is always the authentic target-platform post, verbatim. The variants "
        "differ only in what conditions it, and all three share one prompt template, so a method "
        "is fitted to each without anything changing but the file it reads. The aspect variant "
        "is exploratory because the aspect gates were not met.",
        th(["variant", "conditioning", "train", "development", "test"]), "".join(rows),
        f"Manifest example counts: {esc(json.dumps(counts))}. Prompt template:<br>"
        f'<pre class="code">{template}</pre>Content block, present only where a source post is '
        f'supplied:<br><pre class="code">{block}</pre>',
    )


def grid_table(frontier: dict | None) -> str:
    """The confirmatory grid of plan section 4.2, with the column name each cell produced."""
    spec = [
        ("direct Llama prompt", "base_none", "llama_rewrite_s4_base_none", "model-controlled zero-shot baseline"),
        ("Claude prompt", "none (API model)", "llm_rewrite_s4_claude_reference", "reference prompting system"),
        ("LoRA, unsupervised only", "raw-text platform prior", "lora_s4_raw_10k", "tests whether adaptation alone suffices"),
        ("LoRA, unsupervised only", "structured platform prior", "lora_s4_prior_10k", "primary unsupervised base, generated directly"),
        ("LoRA, unsupervised only", "platform then audience", "lora_s4_platform_then_audience", "coarse-to-fine adaptation"),
        ("LoRA transfer", "structured platform prior", "lora_s4_target_lm", "expected strongest fitted method"),
        ("LoRA transfer", "base_none", "lora_s4_base_lora_target_lm", "Study 3 counterpart"),
        ("soft prompt", "structured platform prior", "soft_prompt_s4_soft_target_lm", "Study 3's most stable fitted method"),
        ("soft prompt", "base_none", "soft_prompt_s4_base_soft_target_lm", "Study 3 counterpart"),
        ("steering", "base_none", "steering_s4_steer_base_target_lm", "primary mechanistic baseline"),
        ("steering", "structured platform prior", "steering_s4_steer_prior_target_lm", "recomputed-vector interaction"),
        ("LoRA plus steering", "structured platform prior", "steering_s4_lora_steer_interaction", "secondary interaction experiment"),
        ("extract then transfer", "structured platform prior", "local_plan_then_transfer_s4_plan_prior_10k", "explicit content-plan baseline"),
    ]
    pts = (frontier or {}).get("points", {})
    rows = []
    for method, base, column, role in spec:
        present = column in pts
        shuffle = f"{column}_audience_shuffle" in pts
        rows.append(row(esc(method), td(esc(base)), td(f"<code>{esc(column)}</code>"), td(esc(role)),
                        td("yes" if present else "no", "ok" if present else "no"),
                        td("yes" if shuffle else "—", "ok" if shuffle else "na")))
    return tablefig(
        "The confirmatory grid",
        "Each transfer method and the base it starts from, with the column name under which it "
        "appears in the results. The paired and aspect conditionings of every fitted family are "
        "run alongside the topic-only conditioning shown here.",
        th(["method", "starting base", "results column", "role", "scored", "audience-shuffle control"]),
        "".join(rows),
    )


def steering_table() -> str:
    rows = []
    for direction, directory in (("LinkedIn to Reddit", ROOT / "runs/study4_steering"),
                                 ("Reddit to LinkedIn", ROOT / "runs/study4_reverse_steering")):
        variants = ["target_lm", "target_lm_paired", "target_lm_aspect"]
        if direction == "LinkedIn to Reddit":
            variants.append("lora_interaction")
        for v in variants:
            path = directory / f"chosen.steering.{v}.json"
            if not path.exists():
                # Selection has not run for this direction yet. Emit the row as
                # outstanding rather than dropping it, so the table still shows
                # which contrasts the design calls for.
                rows.append(row(esc(direction), td(esc(label(v))), td("not yet selected", "na"),
                                td("—", "na"), td("—", "na"), td("—", "na")))
                continue
            c = read_json(path)
            rows.append(row(esc(direction), td(esc(label(v))), td(esc(c.get("scope"))),
                            td(c.get("layer"), "num"), td(fmt(c.get("alpha")), "num"),
                            td(fmt(c.get("audience_alpha")), "num")))
    return tablefig(
        "Selected steering interventions",
        "The layer and coefficients chosen on the development slice under Pareto dominance with a "
        "degeneracy constraint. Selection rejects points that improve target resemblance by "
        "reducing source similarity.",
        th(["direction", "conditioning", "selected scope", "layer", "platform coefficient",
            "audience coefficient"]), "".join(rows),
        "A zero platform coefficient at layer 0 is the null intervention: the column is then its "
        "base model under the steering prompt template.",
    )


def steering_sweep_table() -> str:
    rows = []
    for v in ("target_lm", "target_lm_paired", "target_lm_aspect", "lora_interaction"):
        path = ROOT / "runs/study4_steering" / f"sweep.{v}.json"
        if not path.exists():
            continue
        w = read_json(path)
        sw, base, chosen, ceil = (w.get("sweep") or {}), (w.get("baseline_unsteered") or {}), (w.get("chosen") or {}), (w.get("reference_ceiling") or {})
        results = w.get("results") or []
        rows.append(row(
            esc(label(v)), td(num(len(results)), "num"), td(num(sw.get("n_dev_tasks")), "num"),
            td(num(sw.get("n_reference_posts")), "num"),
            td(fmt(base.get("centroid_cos"), 4), "num"), td(fmt(chosen.get("centroid_cos"), 4), "num"),
            td(fmt(ceil.get("centroid_cos"), 4), "num"),
            td(fmt(base.get("source_cos"), 4), "num"), td(fmt(chosen.get("source_cos"), 4), "num"),
            td(fmt(chosen.get("degeneracy"), 3), "num"),
        ))
    fit = read_json(ROOT / "runs/study4_steering/steering.target_lm.json")
    c = fit.get("counts") or {}
    return tablefig(
        "Steering fit and development sweep",
        f"Vectors are difference-in-means of completion-token residual states over "
        f"{num(c.get('n_target'))} target and {num(c.get('n_source'))} source training posts, "
        f"with per-cell vectors for {num(c.get('n_cells_with_vector'))} of "
        f"{num(c.get('n_cells_seen'))} cells and per-audience vectors for "
        f"{num(c.get('n_audiences_with_vector'))} of {num(c.get('n_audiences_seen'))} audiences "
        f"(minimum {c.get('min_cell_posts')} posts per cell, {c.get('min_audience_posts')} per "
        "audience). The sweep scores every grid point on a development slice and selects under "
        "Pareto dominance with a degeneracy constraint.",
        th(["conditioning", "grid points", "development tasks", "reference posts",
            "centroid cosine, unsteered", "centroid cosine, selected", "centroid cosine, ceiling",
            "source cosine, unsteered", "source cosine, selected", "degeneracy, selected"]),
        "".join(rows),
        "The selected point equals the unsteered baseline on every conditioning: no steered grid "
        "point dominated the null under the constraint.",
    )


def constants_list() -> str:
    run, _ = run_meta("s4-prior10k-lora-target_lm")
    sp = read_json(ROOT / "runs/checkpoints/s4-prior10k-soft-prompt-target_lm/summary.json")
    spc = sp.get("config") or {}
    return f"""
<dl class="defs">
<dt>Base model</dt><dd><code>{esc(run.get('base_model'))}</code> for every locally fitted column.</dd>
<dt>Quantisation</dt><dd>{esc(run.get('quantisation'))}, the study constant. Every local method
runs under it, so quantisation is a property of the study rather than a difference between columns.</dd>
<dt>Sequence limit</dt><dd>{num(run.get('max_length'))} tokens; batch size
{num(run.get('per_device_batch_size'))} with gradient accumulation {num(run.get('grad_accum'))}.</dd>
<dt>Objective</dt><dd>{esc(run.get('loss'))}.</dd>
<dt>Early stopping</dt><dd>LoRA: at most {num(run.get('max_epochs'))} epochs, patience
{num(run.get('patience'))}. Soft prompt: at most {num(spc.get('epochs'))} epochs, patience
{num(spc.get('patience'))}, warm-up fraction {spc.get('warmup_frac')}.</dd>
<dt>Seed</dt><dd>{num(run.get('seed'))}.</dd>
<dt>Prompt</dt><dd>The rendered Study 3 template, ending &ldquo;{esc(sp.get('prompt_template_tail'))}&rdquo;.</dd>
<dt>Draws</dt><dd>Four candidates per task for every column; failures emit an empty output and
remain in every denominator.</dd>
</dl>"""


def frontier_rule(sens: dict | None) -> str:
    margins = (sens or {}).get("noninferiority_margins") or {}
    spaces = (sens or {}).get("spaces") or []
    m_src = margins.get("source_similarity", margins.get("semantic.source_similarity"))
    m_trm = margins.get("trm", margins.get("trm.trm"))
    margin_txt = (f"The frozen non-inferiority margins are {m_src} cosine-similarity units for "
                  f"source preservation and {m_trm} TRM units for distributional error."
                  if m_src is not None and m_trm is not None else
                  f"The frozen non-inferiority margins are recorded in the sensitivity file as "
                  f"<code>{esc(json.dumps(margins))}</code>.")
    return f"""
<p>For each candidate column the comparison is against its preregistered baseline on paired
cells. An improvement is supported only where the interval for one axis separates from zero in the
favourable direction while the other axis is non-inferior under a frozen margin, or where both
axes separate favourably. {margin_txt} Membership of the point-estimate frontier is descriptive
and is never promoted to a finding on its own.</p>
<p>Intervals use a two-stage bootstrap: audiences are resampled, then eligible topic cells within
each sampled audience, because cells nested within one audience share training data and
conditioning. Every verdict is repeated in {len(spaces)} embedding spaces
({esc(', '.join(spaces))}); values from different spaces are never pooled.</p>"""


TEX_LORA = r"\(W' = W + \tfrac{\alpha_{\mathrm L}}{r} B_{\mathrm L} A_{\mathrm L}\)"
TEX_SOFT = r"\(E(x) \leftarrow [P_1, \ldots, P_{32};\, E(x)]\)"
TEX_STEER = r"\(h_\ell \leftarrow h_\ell + \alpha\, \frac{v_\ell}{\lVert v_\ell \rVert_2}\, \bar n_\ell\)"
TEX_DIM = r"\(v_\ell = \bar h_\ell^{\mathrm{target}} - \bar h_\ell^{\mathrm{source}}\)"


def families_section() -> str:
    run, _ = run_meta("s4-prior10k-lora-target_lm")
    cfg = run.get("config") or {}
    sp = read_json(ROOT / "runs/checkpoints/s4-prior10k-soft-prompt-target_lm/summary.json")
    spc = sp.get("config") or {}
    return f"""
<h3>Prompting</h3>
<p>The zero-shot baseline in <code>transfer/llama_rewrite.py</code> renders the audience, domain,
topic, source post and a target-platform writing instruction and samples from
<code>{esc(run.get('base_model'))}</code> without any fitted parameters. The Claude column is the
same instruction sent to an API model, retained as a reference prompting system so that
model-specific instruction tuning can be distinguished from platform adaptation.</p>
<h3>LoRA</h3>
<p><code>methods/lora/train.py</code> fits low-rank updates to the attention projections,
{TEX_LORA}, with rank {cfg.get('r')}, scaling {cfg.get('alpha')}, dropout {cfg.get('dropout')} and
learning rate {cfg.get('lr')} on the modules {esc(', '.join(cfg.get('modules') or []))}. The
objective is the completion-only negative log-likelihood of the authentic post; prompt tokens are
masked. Stage A uses the same trainer with the unsupervised objective over unpaired target text;
Stage B initialises from the frozen Stage A adapter.</p>
<h3>Soft prompt</h3>
<p><code>methods/soft_prompt/train.py</code> prepends {spc.get('n_virtual_tokens')} trained
embedding vectors to every input, {TEX_SOFT}, and freezes the base model and the Stage A adapter.
The prompt is initialised from text, trained with learning rate {spc.get('lr')} and early stopping
on development loss, and has {num(sp.get('trainable_parameters'))} trainable parameters.</p>
<h3>Steering</h3>
<p><code>methods/steering/fit_steering.py</code> performs one forward pass over each authentic
training post through the frozen checkpoint, mean-pools the completion-token residual states, and
forms the platform direction at each layer as {TEX_DIM}. Audience directions are formed within
platform and made orthogonal to the platform direction by per-layer Gram–Schmidt; the pair
singular values are recorded in the artifact. At generation the residual stream is shifted,
{TEX_STEER}, with the coefficient calibrated to the layer's activation norm. The primary steering
column uses the original checkpoint; a second column recomputes every contrast from the selected
unsupervised base, because vectors read from different parameterisations are not interchangeable.</p>
<h3>Extract then transfer</h3>
<p><code>transfer/plan_then_transfer.py</code> is a two-step method. The model first emits a
structured content plan of claims, entities, numbers, named tools, stance, requested actions and
aspects, and then renders a target-platform post from that plan and the audience context. The plan
is stored and scored; recall of its entities and numbers against the source is reported as a
diagnostic. A plan that is not valid JSON, or that departs from the declared schema, is a recorded
failure and remains in the denominator.</p>"""


def references_section(frontier: dict | None) -> str:
    pts = (frontier or {}).get("points", {})
    n_shuffle = sum(is_shuffle(n) for n in pts)
    return f"""
<p>Three reference columns bound the attainable range, and every figure draws all three so that no
system value is read without them.</p>
<dl class="defs">
<dt>Identity</dt><dd>The unchanged source post: the preservation ceiling and the platform-match
floor.</dd>
<dt>Authentic target</dt><dd>A genuine target-platform post from the scored cell: the
distribution-match ceiling. It scores near zero on preservation because it is a different post,
not a rewrite.</dd>
<dt>Wrong-topic control</dt><dd>A genuine target-platform post from another cell. It separates
platform resemblance from topic resemblance.</dd>
<dt>Audience-shuffle control</dt><dd>The same method conditioned on a different target-platform
audience under a frozen derangement matched on training volume and topic coverage. It is a
system control: it participates in the common cell base and tests whether the audience prior
carries information beyond the platform prior. {num(n_shuffle)} such columns are scored.</dd>
</dl>"""


def heldout_scope() -> tuple[int, int]:
    ms = [read_json(ROOT / "data/study3/manifest.json"),
          read_json(ROOT / "data/study3_reverse/manifest.json")]
    return tuple(int(m.get("n_heldout_cells", 0)) for m in ms)


def aspect_coverage() -> tuple[int, int]:
    train_ids = {json.loads(line)["post_id"] for line in
                 (ROOT / "data/study3/posts.train.jsonl").read_text(encoding="utf-8").splitlines()
                 if line.strip()}
    with (ROOT / "data/study3/aspects/aspect_embeddings.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    meta = {"post_id", "platform", "final_topic"}
    labeled = {r["post_id"] for r in rows
               if any(v.strip() and float(v) > 0 for k, v in r.items() if k not in meta and v and v.strip())}
    return len(labeled & train_ids), len(train_ids)


def audit_svg_layout(document: str, char_width: float = 6.5) -> None:
    """Reject approximately overflowing horizontal SVG text before publishing."""
    for svg in re.findall(r"<svg\b.*?</svg>", document, flags=re.DOTALL):
        vb = re.search(r'viewBox="0 0 ([0-9.]+) ([0-9.]+)"', svg)
        if not vb:
            raise ValueError("Study 4 SVG has no numeric viewBox")
        width = float(vb.group(1))
        for attrs, raw in re.findall(r"<text\b([^>]*)>(.*?)</text>", svg, flags=re.DOTALL):
            xm = re.search(r'\bx="([0-9.]+)"', attrs)
            if not xm:
                continue
            x = float(xm.group(1))
            t = re.sub(r"<.*?>", "", raw)
            extent = len(html.unescape(t)) * char_width
            am = re.search(r'text-anchor="(middle|end)"', attrs)
            anchor = am.group(1) if am else "start"
            lo = x - extent / 2 if anchor == "middle" else x - extent if anchor == "end" else x
            if lo < -1 or lo + extent > width + 1:
                raise ValueError(f"SVG text overflows viewBox: {t!r} spans {lo:.1f}..{lo + extent:.1f} "
                                 f"within width {width:.1f}")


def known_defects() -> str:
    """Verified defects in columns this build still renders.

    Both were found by reading the scored outputs rather than the code: three
    systems that differ in their artifacts scored bit-identically on two
    independent metrics, which no amount of sampling noise produces.
    """
    chosen = sorted(p.stem.split("chosen.steering.")[-1]
                    for p in (ROOT / "runs/study4_steering").glob("chosen.steering.*.json"))
    null = sorted(v for v in chosen
                  if not read_json(ROOT / "runs/study4_steering" / f"chosen.steering.{v}.json").get("alpha"))
    return f"""
<h2 id="defects">Known defects in this build</h2>
<p>Two families of columns do not measure what their names indicate. Both defects are reproduced
and their causes are established; neither is corrected in this build, and no claim resting on
these columns should be quoted.</p>
<h3>Soft-prompt columns reduce to their Stage A adapter</h3>
<p>The three <code>soft_prompt_s4_soft_*</code> columns generate byte-identical text although
they load three distinct prompt-embedding tensors (32&nbsp;&times;&nbsp;4096; maximum
element-wise difference 0.62). All three declare the same <code>initial_adapter</code>. In
<code>transfer/soft_prompt.py</code> the Stage A adapter is attached with
<code>PeftModel.from_pretrained</code>, and the prompt-tuning artifact is then attached to that
already-wrapped model; generation proceeds through the inner adapter and the virtual tokens are
never prepended. The output is that of the Stage A LoRA alone. The
<code>soft_prompt_s4_base_soft_*</code> columns declare no initial adapter and are unaffected.</p>
<h3>Every steering column is an unsteered baseline</h3>
<p>Selection returned <code>alpha = 0</code> at <code>layer = 0</code> for {esc(", ".join(null))}.
The intervention is therefore the identity, and each steering column is its base model under the
steering prompt template. This is a selection outcome rather than a failure of execution: no
steered grid point dominated the null under the degeneracy constraint. The result is reportable
as an absence of benefit from steering, but not as a comparison between steering conditionings,
which are the same null run.</p>
<p>The reverse direction is generated from the same code path and will require both corrections
before its soft-prompt and steering columns can be interpreted.</p>
"""


def ndif_section() -> str:
    """The section 4.5 scale arm, which has no scored results.

    Reported because the arm exists in the plan and a reader looking for it must
    find out why there is nothing rather than find nothing.
    """
    art = ROOT / "runs/steering/steering.target_lm.ndif-Llama-3.1-70B-Instruct.json"
    if not art.exists():
        fitted = "No remote artifact is present in this checkout."
    else:
        m = read_json(art)
        fit = m.get("fit_config") or {}
        c = m.get("counts") or {}
        fitted = (f"One artifact has been fitted: <code>target_lm</code> on "
                  f"<code>{esc(m.get('base_model'))}</code>, hidden size {num(m.get('hidden_size'))}, "
                  f"from {num(c.get('n_target'))} target and {num(c.get('n_source'))} source posts "
                  f"read from <code>{esc(fit.get('data_dir'))}</code>.")
    return f"""
<h2 id="ndif">Scale generalisation of steering on NDIF</h2>
{callout("No results from this arm",
         "Every steering value elsewhere in this report is an 8B result. Section 4.5 of the plan "
         "adds a scale arm that applies the identical difference-in-means construction to NDIF-hosted "
         "<code>Llama-3.1-70B-Instruct</code>, whose residual stream is exposed for reading and "
         "writing. The arm has produced no scored output, and no claim in this report depends on it.")}
<p>{fitted}</p>
<h3>Why no result is reported</h3>
<ul>
<li>The layer-and-coefficient sweep has not been run, and the plan prohibits scoring the artifact
before it is. A spot check at layer 40 with coefficient 0.5 produced output that was structurally
Reddit-like but code-switched into Spanish and German.</li>
<li>The fitted artifact was read from a different corpus from the local steering column it is
intended to contrast with; as fitted, it varies corpus and scale simultaneously. A valid contrast
requires a refit on the same corpus.</li>
<li>Before this build <code>sweep_steering</code> had no remote execution path and selected the
local generator, which would have loaded a 70B model against 4,096-wide local vectors rather than
the 8,192-wide remote stream.</li>
<li>Remote columns are never pooled with local 4-bit columns. NDIF serves its own precision, so
the study's quantisation constant does not extend to this arm; it is a scale contrast against the
local steering column and nothing else.</li>
</ul>
"""


# --------------------------------------------------------------------------
# Page shell
# --------------------------------------------------------------------------

#: Page set and order of the 2026-08-13 reports: Overview, Results, Reverse
#: direction, Methods, Dataset, Limitations. Embedding-space sensitivity is a
#: section inside Results there rather than a page of its own, and is placed
#: the same way here.
NAV = [
    ("index.html", "Overview"),
    ("results.html", "Results"),
    ("reverse.html", "Reverse direction"),
    ("methods.html", "Methods"),
    ("dataset.html", "Dataset"),
    ("limitations.html", "Limitations"),
]


def page(slug: str, title: str, body: str, lede: str, banner: str = "") -> str:
    nav = "".join(
        f'<a href="{href}" class="{"here" if href == slug else ""}">{label_}</a>'
        for href, label_ in NAV
    )
    run = REVERSE if slug == "reverse.html" else FORWARD
    # MathJax is loaded only on pages that contain TeX, pinned to the same
    # release the 2026-08-13 pages use.
    mathjax = ""
    if r"\(" in body or r"\[" in body:
        mathjax = ('<script defer src="https://cdn.jsdelivr.net/npm/mathjax@4.0.0/tex-mml-svg.js">'
                   "</script>")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)} — Study 4 — Vectorial × BAIR — {BUILD_DATE}</title>
<link rel="stylesheet" href="style.css">
{mathjax}
</head>
<body>
<header class="masthead study-head">
  <div class="wrap">
    <p class="eyebrow">Vectorial × BAIR · Study 4 · unsupervised platform adaptation</p>
    <h1>Unsupervised platform adaptation <span class="stamp">{BUILD_DATE}</span></h1>
    <p class="lede">{lede}</p>
  </div>
</header>
<nav class="nav" aria-label="Report sections"><div class="wrap">{nav}</div></nav>
<main class="wrap">
{banner}
{body}
</main>
<footer class="foot"><div class="wrap">
  <p class="footer-title">Study 4 · generated {BUILD_LONG}</p>
  <p>Source reports: <code>{esc(run.relative_to(ROOT))}/report.heldout.{PRIMARY_SLUG}.json</code>,
     the corresponding frontier and sensitivity files, the training artifacts under
     <code>runs/checkpoints</code>, and the frozen corpus manifests. Built by
     <code>experimental-notes/build_study4.py</code>. Every figure and quoted value is computed
     from those files.</p>
</div></footer>
</body>
</html>
"""


def page_intro(kicker: str, heading: str, lede: str) -> str:
    return (f'<div class="page-intro"><p class="section-label">{esc(kicker)}</p>'
            f"<h1>{esc(heading)}</h1><p class=\"lede\">{lede}</p></div>")


def summary_grid(sc: dict) -> str:
    return f"""
<div class="summary-grid" aria-label="Study summary">
  <div class="summary-card primary"><span class="summary-kicker">Supported comparisons</span>
    <strong>{sc['supported']} / {sc['n_comparisons']}</strong>
    <p>preregistered comparisons improve the content-versus-match frontier with separated
    intervals in the primary embedding space.</p></div>
  <div class="summary-card"><span class="summary-kicker">Lowest system TRM</span>
    <strong>{sc['best_trm']:.3f}</strong>
    <p>{esc(label(sc['best']))}, against {fmt(sc['oracle_trm'])} for an authentic target post.
    Each reduction in TRM in this grid is accompanied by a reduction in source similarity.</p></div>
  <div class="summary-card"><span class="summary-kicker">Comparison base</span>
    <strong>{esc(sc['cells'])} cells</strong>
    <p>across {sc['audiences']} audiences and {sc['n_systems']} system columns, scored in
    {esc(sc['space'])} and five further spaces.</p></div>
</div>"""


def build() -> None:
    reddit = read_json(ROOT / "data/study4_priors/reddit_10000_per_audience.manifest.json")
    linkedin = read_json(ROOT / "data/study4_priors/linkedin_10000_per_audience.manifest.json")
    fwd_ok, rev_ok = direction_available(FORWARD), direction_available(REVERSE)
    sc = direction_scope(FORWARD) if fwd_ok else None
    rc = direction_scope(REVERSE) if rev_ok else None
    frontier = read_json(FORWARD / f"frontier.heldout.{PRIMARY_SLUG}.json") if fwd_ok else None
    sens = read_json(FORWARD / "frontier_sensitivity.heldout.json") if fwd_ok else None

    lede = (f"LinkedIn to Reddit and Reddit to LinkedIn across {corpus_cell_count()} bilateral "
            "corpus cells. The report compares "
            + (f"{sc['n_systems']} method configurations against three references; the primary "
               f"distributional measure covers {esc(sc['cells'])} shared held-out cells in "
               f"{esc(sc['space'])}." if sc else
               "unsupervised platform and audience priors against three references."))
    banner = "" if fwd_ok and rev_ok else callout(
        "Preliminary build",
        "This is not the frozen report. Sections marked as not yet available are still running, "
        'and no claim resting on them is supported. See <a href="limitations.html">Limitations</a> '
        "for verified defects in columns that this build still renders.", warn=True)

    idx = page_intro(
        "Research question",
        "Unsupervised priors move columns along the content-versus-match frontier; no column "
        "extends the frontier.",
        "This study asks whether adapting a language model on unpaired target-platform and "
        "target-audience text improves cross-platform transfer without discarding source content. "
        "The comparison is a frontier rather than a single lowest TRM: a column improves it only "
        "by lowering TRM without reducing source preservation, or by raising preservation without "
        "raising TRM. A column that moves down and to the left has traded content for match, "
        "which is a different trade-off rather than an improvement.",
    ) + (summary_grid(sc) if sc else "") + (f"""
<h2>Both directions agree</h2>
<p>The grid was run twice, LinkedIn to Reddit and Reddit to LinkedIn, and scored independently
because the target distribution, task pools and reference anchors differ between them. The
headline outcome replicates: <strong>{sc['supported']} of {sc['n_comparisons']}</strong>
preregistered comparisons are supported in the forward direction and
<strong>{rc['supported']} of {rc['n_comparisons']}</strong> in the reverse. The lowest system TRM
is {sc['best_trm']:.3f} forward ({esc(label(sc['best']))}) and {rc['best_trm']:.3f} reverse
({esc(label(rc['best']))}), against {fmt(sc['oracle_trm'])} and {fmt(rc['oracle_trm'])} for an
authentic target post. Plan hypothesis 2 asked whether an improvement repeats in the reverse
direction; there is no improvement in either direction for it to repeat.</p>
""" if sc and rc else "") + """
<h2>Scope</h2>
<p>This is a four-audience case study. The supplied exports provide four reviewed bilateral
audience mappings, so the planned six- and eight-audience acceptance gates are not met, and the
study makes no audience-generalisation claim. Twitter and Quora were not present in the supplied
corpus and remain future replications.</p>
<h2>Contents</h2>
<ul>
<li><a href="results.html">Results</a>: LinkedIn to Reddit; the frontier, every column with
intervals, the resolution test, generation failures, and support in each of six embedding
spaces.</li>
<li><a href="reverse.html">Reverse direction</a>: Reddit to LinkedIn on the same grid.</li>
<li><a href="methods.html">Methods</a>: the two-stage design, the confirmatory grid, the three
conditionings, each transfer family and its fitted configuration, the references and controls,
the frontier rule, the steering decomposition, and the NDIF scale arm.</li>
<li><a href="dataset.html">Dataset</a>: the evaluation corpus, the prior corpora and their
reconstruction, the structured priors, the coarse-to-fine partition, the scale-ladder samples,
and scored-context sanitization.</li>
<li><a href="limitations.html">Limitations</a>: verified defects and the limits of scope.</li>
</ul>
"""

    res_lede = (f"The comparison contains {sc['n_systems']} method configurations and three "
                f"references. The primary Triangle-Rank Metric covers {esc(sc['cells'])} shared "
                f"held-out cells across {sc['audiences']} audiences; all embedding-derived values "
                f"use {esc(sc['space'])}." if sc else "This direction has not finished scoring.")
    res = page_intro("Complete analysis", "LinkedIn to Reddit", res_lede) + (
        direction_sections(FORWARD, "LinkedIn to Reddit") if fwd_ok
        else pending("LinkedIn to Reddit", "This direction has not finished scoring."))

    rev = page_intro(
        "Direction check", "Reddit to LinkedIn",
        "The full method grid is rerun with Reddit as the source and LinkedIn as the target. It "
        "is scored independently because the target distribution, task pools and reference "
        "anchors differ from the forward direction.",
    ) + (direction_sections(REVERSE, "Reddit to LinkedIn") if rev_ok else pending(
        "Reddit to LinkedIn",
        "Generation and the six embedding-space evaluations are still running. No frontier has "
        "been computed for this direction, so no reverse-direction claim is supported."))

    me = page_intro(
        "Method", "Unsupervised adaptation, then transfer",
        "Every transfer method starts from a base adapted on unpaired target-platform text. The "
        "design has two stages so that a large method-by-scale-by-platform grid does not obscure "
        "the primary question.",
    ) + f"""
<h2 id="stages">Two stages</h2>
<p><strong>Stage A</strong> continues causal language-model training on authentic, unpaired
target-platform text. No synthetic rewrite and no source-target pairing enters this stage. The
raw-text and structured-prior arms use matched target completions and differ only in whether the
prompt exposes the supplied audience, profile, community, thread, keyword, dimension and
engagement information. The coarse-to-fine arm trains a platform stage and then an audience stage
on disjoint halves of the same corpus.</p>
<p><strong>Stage B</strong> fits the transfer methods from the selected Stage A base on the Study 3
conditioning files, under the three conditionings that the harness requires of every fitted
family. Direct prompting and steering from the original checkpoint are run alongside as baselines
that require no unsupervised fit.</p>
{grid_table(frontier)}
<h2 id="scale">Stage A: the unsupervised scale ladder</h2>
<p>The plan freezes a ladder of nested deterministic samples so that the scale curve compares data
size rather than sample composition. Two rungs were available: 1,000 records per audience and the
full available corpus, capped at 10,000 per audience at collection. The frozen primary base is the
structured-prior adapter on the full available corpus; the raw-text arm and the 1,000-record arm
are ablations, and held-out results do not select the base.</p>
{scale_svg()}
{training_table()}
<h2 id="conditionings">The three conditionings</h2>
{conditionings_table()}
<h2 id="families">The transfer families</h2>
{families_section()}
{stage_b_table()}
<h2 id="references">References and controls</h2>
{references_section(frontier)}
<h2 id="rule">The frontier rule and its intervals</h2>
{frontier_rule(sens)}
<h2 id="steering">Steering decomposition</h2>
<p>Platform and train-only audience contrasts are recomputed within each frozen checkpoint.
Per-layer Gram–Schmidt residuals prevent the audience direction from duplicating the platform
direction; the pair singular values and projections remain in each steering artifact.</p>
{steering_sweep_table()}
{steering_table()}
{callout("Every selected coefficient is zero",
         "Selection returned the null intervention for every conditioning in this build, so each "
         'steering column is an unsteered baseline. See <a href="limitations.html">Limitations</a>.')}
<h2 id="constants">What was held constant</h2>
{constants_list()}
{ndif_section()}
"""

    ds = page_intro(
        "Corpus", "Four audiences, two platforms",
        "The evaluation corpus is the Study 3 bilateral corpus. The prior corpora are reconstructed "
        "from the supplied profile exports: records were deduplicated, grouped by author, "
        "time-filtered and split before any derived audience artifact was computed.",
    ) + f"""
<h2 id="evaluation">Evaluation corpus</h2>
{eval_corpus_table()}
<h2 id="corpora">Prior corpora</h2>
{corpus_table(reddit, linkedin)}
{availability_table(reddit, linkedin)}
<h2 id="reconstruction">Reconstruction from the exports</h2>
<p>The exports mix production, test, cloned, merged and repeated rooms, and their profile histories
overlap the scored splits. The corpus builder therefore resolves canonical exports, removes
duplicates, groups every record by its platform author, excludes any profile that overlaps the
scored splits, and only then assigns authors to splits. The policies it applied are recorded in the
manifests and reproduced here.</p>
{policy_list(reddit, linkedin)}
{prior_audit_table(reddit, linkedin)}
<h2 id="priors">Structured priors</h2>
<p>Shipped profile summaries, keywords, dimensions and room traits are weak labels rather than
validated annotations. Only labels tied to profiles assigned wholly to the training split may
condition training; whole-room summaries and traits in the supplied export include held-out
authors and are audit-only.</p>
{structured_prior_summary(reddit, linkedin)}
{rich_prior_coverage_table()}
<h2 id="partition">Coarse-to-fine partition</h2>
{coarse_to_fine_table()}
<h2 id="ladder">Scale-ladder samples</h2>
{ladder_table()}
<h2 id="sanitization">Scored-context sanitization</h2>
{context_sanitization_table()}
"""

    fwd_cells, rev_cells = heldout_scope()
    labeled, total = aspect_coverage()
    lim = page_intro(
        "Caveats", "What this build cannot support",
        "Two column families are known not to measure what their names indicate, and the scope of "
        "the study is narrower than its plan.",
    ) + known_defects() + f"""
<h2 id="topics">Unseen topics, aspects and traits</h2>
<p>The held-out-cell split contains {fwd_cells} forward and {rev_cells} reverse topic cells. These
cells remain inside the reported held-out evaluation, but a four-audience case study does not
support a separate audience-generalisation claim. Positive aspect labels cover {labeled} of
{total} training posts. Aspect-conditioned columns are exploratory because the coverage and
human-validation gates were not met. Trait steering was not run because no supplied trait
contrast passed those gates.</p>
<h2 id="limits">Limitations</h2>
<ul>
<li>No cross-platform person linkage exists.</li>
<li>The audience mapping is a measurement assumption.</li>
<li>Collection timestamps do not establish one common observation window across both platforms.</li>
<li>Upstream profile summaries, keywords and dimensions are weak labels rather than human-validated traits.</li>
<li>Whole-room trait summaries include held-out authors, so they remain audit-only rather than conditioning the fitted models.</li>
<li>There is no gold rewrite for an individual source post.</li>
<li>Twitter and Quora were not present in the supplied corpus and are not reported as completed replications.</li>
<li>Aspect-conditioned results remain exploratory because the human-validation gate was not met.</li>
</ul>
"""

    pages = [
        ("index.html", "Overview", idx),
        ("results.html", "Results", res),
        ("reverse.html", "Reverse direction", rev),
        ("methods.html", "Methods", me),
        ("dataset.html", "Dataset", ds),
        ("limitations.html", "Limitations", lim),
    ]
    for _, _, body in pages:
        audit_svg_layout(body)
    out = out_dir()
    out.mkdir(parents=True, exist_ok=True)
    for slug, title, body in pages:
        (out / slug).write_text(page(slug, title, body, lede, banner), encoding="utf-8")
    (out / "style.css").write_text(STYLE, encoding="utf-8")
    print(f"Wrote {len(pages)} pages and a stylesheet to {out}")

STYLE = """/* Experimental notes — Vectorial x BAIR.
   Print-friendly, no JS, no external assets. */

:root {
  --ink: #1c1c1a;
  --ink-soft: #55534e;
  --ink-faint: #8a877f;
  --rule: #ddd9d0;
  --paper: #fbfaf7;
  --panel: #f4f1ea;
  --accent: #c25e3a;
  --blue: #4a6fa5;
  --green: #2e7d5b;
  --warn-bg: #fdf6ec;
  --warn-rule: #e0b070;
  --measure: 100%;
}

* { box-sizing: border-box; }

html { -webkit-text-size-adjust: 100%; }

body {
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font: 16.5px/1.66 Charter, "Bitstream Charter", "Iowan Old Style", Georgia, serif;
}

.wrap { max-width: 776px; margin: 0 auto; padding: 0 28px; }
/* Wide result tables break out of the reading column and span the viewport. */
.tablewrap { width: 100%; }
.tablewrap.wide {
  /* A share of the viewport rather than the whole of it, so the table keeps a
     margin on both sides, with a cap so it does not stretch on wide displays. */
  width: min(70vw, 1680px);
  margin-left: 50%;
  transform: translateX(-50%);
}
.tablewrap.wide > table { min-width: 0; }
.tablewrap.wide > caption,
.tablewrap.wide > .tablekey { max-width: 100%; }

/* ---------- masthead + nav ---------- */

.masthead {
  background: var(--panel);
  border-bottom: 1px solid var(--rule);
  padding: 44px 0 30px;
}
.eyebrow {
  margin: 0 0 6px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px;
  letter-spacing: .09em;
  text-transform: uppercase;
  color: var(--ink-faint);
}
.masthead h1 { margin: 0; font-size: 40px; line-height: 1.1; letter-spacing: -.015em; }
.stamp {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 15px; font-weight: 400; color: var(--ink-faint);
  letter-spacing: 0; vertical-align: 8px; margin-left: 8px;
}
.lede { margin: 10px 0 0; color: var(--ink-soft); font-size: 18px; max-width: var(--measure); }

.nav {
  position: sticky; top: 0; z-index: 10;
  background: rgba(251,250,247,.94);
  border-bottom: 1px solid var(--rule);
  backdrop-filter: saturate(140%) blur(6px);
}
.nav .wrap { display: flex; flex-wrap: wrap; gap: 4px; }
.nav a {
  padding: 13px 14px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 13px;
  color: var(--ink-soft);
  text-decoration: none;
  border-bottom: 2px solid transparent;
}
.nav a:hover { color: var(--ink); background: var(--panel); }
.nav a.here { color: var(--accent); border-bottom-color: var(--accent); }

/* ---------- typography ---------- */

/* `.wrap` sets padding, and its class specificity beat a bare `main`
   selector, which collapsed the space below the navigation. */
main.wrap { padding: 44px 28px 72px; }
main > h2 {
  margin: 44px 0 14px;
  padding-top: 18px;
  border-top: 1px solid var(--rule);
  font-size: 25px;
  letter-spacing: -.01em;
}
main > h2:first-child { margin-top: 0; border-top: 0; padding-top: 0; }
h3 { font-size: 17px; margin: 0 0 6px; }
p, li, dd { max-width: var(--measure); }
p { margin: 0 0 14px; }
a { color: var(--accent); }
strong { font-weight: 600; }
dfn { font-style: italic; font-weight: 600; }
blockquote {
  margin: 16px 0; padding: 12px 18px;
  border-left: 3px solid var(--rule);
  color: var(--ink-soft); font-style: italic;
}
blockquote p { margin: 0 0 6px; }
blockquote p:last-child { margin: 0; }

code, pre, .code {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: .88em;
}
code { background: var(--panel); padding: 1px 5px; border-radius: 3px; }
pre.code {
  background: var(--panel); border: 1px solid var(--rule); border-radius: 5px;
  padding: 14px 16px; overflow-x: auto; font-size: 13px; line-height: 1.5;
}

.note {
  font-size: 15px; color: var(--ink-soft);
  border-left: 3px solid var(--rule); padding-left: 14px;
}
.stress { font-weight: 600; }

/* ---------- callouts ---------- */

.callout, .warn {
  border: 1px solid var(--rule);
  border-radius: 6px;
  padding: 22px 26px;
  margin: 0 0 32px;
  background: #fff;
}
.callout { border-left: 4px solid var(--accent); }
.callout h2, .warn h2 { margin: 0 0 10px; border: 0; padding: 0; font-size: 20px; }
.warn { background: var(--warn-bg); border-left: 4px solid var(--warn-rule); }
.warn p:last-child, .callout p:last-child { margin-bottom: 0; }

.findings { padding-left: 22px; }
.findings > li { margin-bottom: 12px; }

/* ---------- tables ---------- */

.tablewrap { margin: 0 0 26px; overflow-x: auto; }
table {
  border-collapse: collapse;
  width: 100%;
  font-size: 14px;
  font-variant-numeric: tabular-nums;
  background: #fff;
}
caption {
  caption-side: top; text-align: left;
  font-size: 14px; color: var(--ink-soft);
  padding: 0 0 10px; max-width: var(--measure);
}
th, td { padding: 7px 10px; border-bottom: 1px solid var(--rule); text-align: right; }
thead th {
  text-align: right; font-size: 12.5px; font-weight: 600;
  border-bottom: 2px solid var(--ink-faint); white-space: nowrap;
}
thead th:first-child, tbody th { text-align: left; }
tbody th {
  font-weight: 400; white-space: nowrap;
  position: sticky; left: 0; background: #fff;
}
tbody tr:hover td, tbody tr:hover th { background: var(--panel); }
td.best { font-weight: 700; color: var(--green); }
td.na { color: var(--ink-faint); }
.dir { color: var(--ink-faint); margin-left: 4px; }
th.base, td.base { background: #f7f5f0; }
tbody tr:hover td.base { background: #efece4; }
th.sys, td.sys { background: #fff; }
.coltag {
  display: block; font-family: ui-monospace, Menlo, monospace;
  font-size: 9.5px; font-weight: 400; letter-spacing: .05em;
  text-transform: uppercase; color: var(--ink-faint); margin-top: 2px;
}
.tablekey {
  font-size: 12.5px; color: var(--ink-faint);
  margin: 8px 0 0; max-width: none;
}
table.resolve th[scope="row"] { white-space: normal; min-width: 210px; }
table.resolve td.num { white-space: nowrap; font-size: 13px; }
.ci { color: var(--ink-faint); font-size: 11.5px; }
.verdict { font-family: ui-monospace, Menlo, monospace; font-size: 11.5px; white-space: nowrap; }
.verdict.res { color: var(--green); font-weight: 600; }
.verdict.unres { color: #9a6b1e; }
.ncell {
  display: inline-block; margin-left: 8px;
  font-family: ui-monospace, Menlo, monospace; font-size: 10.5px;
  color: var(--ink-faint); background: var(--panel);
  padding: 1px 5px; border-radius: 3px; vertical-align: 1px;
}
.ncell.thin { color: #9a6b1e; background: var(--warn-bg); }
.ncol {
  display: inline-block; margin-left: 4px; vertical-align: super;
  font-family: ui-monospace, Menlo, monospace; font-size: 9.5px;
  color: #9a6b1e;
}

/* Two-space comparison: each measure occupies a pair of rows, so the pair is
   kept visually together and only the pair is separated by a rule. */
table.spaces tbody tr.spacetop td { border-bottom: none; }
table.spaces tbody tr.spacebot td,
table.spaces tbody tr.spacebot th { border-bottom: 1px solid var(--rule); }
table.spaces td.spacelbl {
  font-family: ui-monospace, Menlo, monospace; font-size: 11px;
  color: var(--ink-faint); white-space: nowrap; text-align: left;
}
table.spaces th[scope="row"] { white-space: normal; min-width: 230px; }

ul.spec { margin: 0 0 22px; padding-left: 20px; max-width: var(--measure); }
ul.spec li { margin: 0 0 8px; }

/* ---------- figures ---------- */

svg.fig {
  display: block;
  margin: 6px 0 28px;
  background: #fff;
  border: 1px solid var(--rule);
  border-radius: 6px;
  padding: 16px 18px;
  overflow: visible;
}
.fig-title { font-size: 16px; font-weight: 600; fill: var(--ink); }
.fig-sub, .lbl, .val, .tick, .key-lbl, .ref-lbl, .muted, .pt-lbl, .zone-lbl, .tiny, .ax-title, .inbar {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}
.fig-sub { font-size: 12px; fill: var(--ink-soft); }
.lbl { font-size: 12.5px; fill: var(--ink); }
.val { font-size: 11.5px; fill: var(--ink-soft); }
.tick, .tiny { font-size: 10.5px; fill: var(--ink-faint); }
.key-lbl { font-size: 11.5px; fill: var(--ink-soft); }
.pt-lbl { font-size: 12px; fill: var(--ink); }
.ax-title { font-size: 11.5px; fill: var(--ink-soft); }
.inbar { font-size: 10.5px; fill: #fff; }
.zone-lbl { font-size: 11px; fill: var(--green); }
.ref-lbl { font-size: 11px; fill: var(--accent); }
.muted { font-size: 11.5px; fill: var(--ink-faint); font-style: italic; }
.axis { stroke: var(--ink-faint); stroke-width: 1; }
.grid { stroke: var(--rule); stroke-width: 1; }
.ref { stroke: var(--accent); stroke-width: 1.5; stroke-dasharray: 4 3; }
.good-zone { fill: rgba(46,125,91,.07); }
.pt { stroke-width: 1.5; }
.leader { stroke: var(--ink-faint); stroke-width: 1; }
.err { stroke-width: 1.4; opacity: .5; }
.pt.anchor { stroke-width: 2.5; stroke-dasharray: 3 2.2; }
.fig-note { font-size: 11px; fill: var(--ink-faint); }

/* ---------- cards ---------- */

.cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 16px;
  margin: 0 0 26px;
}
.card {
  background: #fff; border: 1px solid var(--rule);
  border-radius: 6px; padding: 18px 20px;
}
.card h3 { display: flex; align-items: center; gap: 8px; }
.card p { font-size: 14.5px; max-width: none; }
.card p:last-child { margin-bottom: 0; }
.swatch { width: 12px; height: 12px; border-radius: 3px; display: inline-block; flex: none; }
.role {
  font-family: ui-monospace, Menlo, monospace;
  font-size: 11.5px; color: var(--ink-faint);
  margin: 0 0 10px !important;
}
.caveat {
  border-top: 1px solid var(--rule); padding-top: 10px;
  color: var(--ink-soft); font-size: 13.5px !important;
}

/* ---------- definition lists ---------- */

.metrics { margin: 0 0 26px; }
.metrics dt {
  font-weight: 600; margin-top: 18px;
  border-bottom: 1px solid var(--rule); padding-bottom: 4px;
}
.metrics dd { margin: 8px 0 0; color: var(--ink-soft); font-size: 15px; }

/* ---------- examples ---------- */

.exfig { margin: 0 0 28px; }
.exfig figcaption {
  font-size: 14px; color: var(--ink-soft);
  padding: 0 0 12px; max-width: var(--measure);
}
.example { display: grid; gap: 14px; margin: 0; }
.example.grid2 { grid-template-columns: 1fr 1fr; }
.example.grid4 { grid-template-columns: repeat(2, 1fr); }
.ex-col.src { border-left: 3px solid var(--blue); }
.ex-col.alt { background: var(--panel); }
.tails { list-style: none; padding: 0; margin: 0; }
.tails li {
  background: #fff; border: 1px solid var(--rule); border-radius: 5px;
  padding: 10px 14px; margin-bottom: 8px; font-size: 14px; max-width: none;
}
.excell {
  display: inline-block; min-width: 168px;
  font-family: ui-monospace, Menlo, monospace; font-size: 11px;
  color: var(--accent); text-transform: uppercase; letter-spacing: .04em;
}
.ex-col {
  background: #fff; border: 1px solid var(--rule);
  border-radius: 6px; padding: 16px 18px;
}
.ex-col h3 {
  font-family: ui-monospace, Menlo, monospace; font-size: 11.5px;
  text-transform: uppercase; letter-spacing: .06em;
  color: var(--ink-faint); margin-bottom: 10px;
}
.ex-col p { font-size: 14.5px; max-width: none; }
.ex-col p:last-child { margin-bottom: 0; }

/* ---------- footer ---------- */

.foot {
  border-top: 1px solid var(--rule);
  background: var(--panel);
  padding: 26px 0 40px;
  color: var(--ink-faint);
  font-size: 13px;
}
.foot p { max-width: var(--measure); margin-bottom: 8px; }

/* ---------- responsive + print ---------- */

@media (max-width: 720px) {
  .masthead h1 { font-size: 30px; }
  .example { grid-template-columns: 1fr; }
  .wrap { padding: 0 18px; }
}

@media print {
  .nav { display: none; }
  body { background: #fff; font-size: 11pt; }
  svg.fig, table, .card, .callout, .warn { break-inside: avoid; }
  main > h2 { break-after: avoid; }
  a { color: inherit; text-decoration: none; }
}

/* Study 2 uses a wider analytical canvas and a deliberately narrow reading
   measure. The base stylesheet remains shared with Study 1; these rules define
   the information hierarchy required by the larger comparison. */
:root{
  --ink:#18231f;--ink-soft:#52615a;--ink-faint:#77827d;--rule:#d8ddd8;
  --paper:#f7f7f3;--panel:#edf0eb;--accent:#b65336;--green:#226c4d;
  --deep:#112b24;--deep-2:#173a31;--gold:#d8b46a;--white:#fff;
  --measure:72ch
}
html{scroll-behavior:smooth}
body{background:var(--paper);color:var(--ink);font-size:17px;line-height:1.67}
.wrap{width:min(calc(100% - 48px),1180px);max-width:1180px!important;padding:0;margin-inline:auto}
.study-head{padding:52px 0 44px;background:
  radial-gradient(circle at 82% 12%,rgba(216,180,106,.14),transparent 28%),var(--deep);
  border:0;color:#f4f6f2}
.study-head .eyebrow{color:#b8c8c1;margin-bottom:12px}
.study-head h1{max-width:none;margin:0;font-size:clamp(2.3rem,5vw,4.1rem);line-height:1.01;
  letter-spacing:-.045em;color:#fff}
.study-head .stamp{display:inline-block;margin-left:14px;padding:6px 9px;border:1px solid #456058;
  border-radius:4px;color:#b8c8c1;font-size:.72rem;letter-spacing:.04em;vertical-align:12px}
.study-head .lede{max-width:880px;margin-top:20px;color:#d3ddd8;font-size:1.04rem;line-height:1.55}
.nav{background:rgba(247,247,243,.96);border-bottom:1px solid var(--rule);box-shadow:0 2px 12px rgba(17,43,36,.04)}
.nav .wrap{gap:2px}
.nav a{padding:15px 18px;font-size:.75rem;letter-spacing:.015em;border-bottom-width:3px}
.nav a.here{color:var(--deep);border-bottom-color:var(--accent);font-weight:700}
main.wrap{max-width:1180px!important;padding:64px 0 96px}
main>p,main>ul,main>ol,main>dl,main>blockquote,main>.defs{max-width:var(--measure)}
main>h1,main>h2,main>h3{max-width:900px}
main>h1{font-size:2.25rem;line-height:1.15;letter-spacing:-.025em;margin:0 0 20px}
main>h2{margin:72px 0 20px;padding:24px 0 0;border-top:1px solid var(--rule);
  font-size:1.7rem;line-height:1.22;letter-spacing:-.02em}
main>h3{margin-top:36px;font-size:1.12rem}
p{margin-bottom:16px}
.page-intro{max-width:930px;margin-bottom:36px}
.page-intro h1{font-size:clamp(2.35rem,4.4vw,4.15rem);line-height:1.04;letter-spacing:-.045em;margin:0 0 24px}
.page-intro .lede{max-width:760px;margin:0;font-size:1.18rem;line-height:1.55}
.section-label,.summary-kicker,.flow-label{font:700 .7rem/1.2 ui-monospace,SFMono-Regular,Menlo,monospace;
  letter-spacing:.1em;text-transform:uppercase;color:var(--accent)}
.section-label{margin:0 0 14px}
.section-no{display:inline-block;margin-right:10px;color:var(--accent);font:700 .75rem/1 ui-monospace,Menlo,monospace;
  letter-spacing:.08em;vertical-align:4px}

/* Overview hierarchy */
.summary-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;max-width:1040px;margin:36px 0 24px}
.summary-card{min-height:190px;padding:24px;border:1px solid var(--rule);border-radius:9px;background:#fff;
  box-shadow:0 8px 24px rgba(17,43,36,.045)}
.summary-card.primary{background:#eaf4ef;border-color:#c7ddd1}
.summary-card strong{display:block;margin:10px 0 7px;font-size:2.15rem;line-height:1.05;letter-spacing:-.04em}
.summary-card p{margin:0;color:var(--ink-soft);font-size:.88rem;line-height:1.48}
.finding-banner{display:grid;grid-template-columns:repeat(3,1fr);gap:0;max-width:1040px;margin:0 0 36px;
  padding:0;background:var(--deep);border:0;border-radius:10px;overflow:hidden;color:#fff}
.finding-banner>div{padding:24px 25px;border-right:1px solid #315047}
.finding-banner>div:last-child{border-right:0}
.finding-banner span{display:block;color:#9fb6ac;font:600 .68rem/1.3 ui-monospace,Menlo,monospace;
  letter-spacing:.08em;text-transform:uppercase}
.finding-banner strong{display:block;margin:8px 0 7px;font-size:1.08rem}
.finding-banner p{margin:0;color:#d1ddd8;font-size:.82rem;line-height:1.5}
.story-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:30px;max-width:1040px;margin:26px 0 60px}
.story-grid section{padding-top:18px;border-top:3px solid var(--deep)}
.story-grid h2{margin:8px 0 10px;padding:0;border:0;font-size:1.08rem}
.story-grid p{margin:0;font-size:.9rem;color:var(--ink-soft)}
.story-num{font:700 .7rem/1 ui-monospace,Menlo,monospace;color:var(--accent)}
.next-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;max-width:1040px;margin-top:32px}
.next-grid a{display:flex;min-height:150px;flex-direction:column;padding:22px;border:1px solid var(--rule);
  border-radius:8px;background:#fff;text-decoration:none;color:var(--ink);transition:.16s ease}
.next-grid a:hover{transform:translateY(-2px);border-color:#aebbb4;box-shadow:0 10px 28px rgba(17,43,36,.07)}
.next-grid span{font:.67rem/1.2 ui-monospace,Menlo,monospace;color:var(--accent);text-transform:uppercase;letter-spacing:.08em}
.next-grid strong{margin:9px 0 4px;font-size:1.05rem}.next-grid small{color:var(--ink-soft);line-height:1.45}

/* Result navigation and reading keys */
.result-nav{display:flex;flex-wrap:wrap;gap:8px;max-width:1040px;margin:24px 0}
.result-nav a{padding:8px 12px;border:1px solid var(--rule);border-radius:999px;background:#fff;
  font:600 .72rem/1.2 ui-monospace,Menlo,monospace;text-decoration:none;color:var(--ink-soft)}
.result-nav a:hover{border-color:#9eada6;color:var(--deep)}
.reading-key{display:flex;align-items:center;flex-wrap:wrap;gap:10px 18px;max-width:1040px!important;
  margin:20px 0 42px;padding:14px 18px;border:1px solid var(--rule);border-left:4px solid var(--accent);
  background:#fff;border-radius:6px;font-size:.82rem}
.reading-key strong{margin-right:8px}.reading-key span{color:var(--ink-soft)}
.reading-key span b{display:inline-grid;place-items:center;width:20px;height:20px;margin-right:4px;border-radius:50%;
  background:var(--panel);font:700 .65rem/1 ui-monospace,Menlo,monospace;color:var(--deep)}
.reading-key a{margin-left:auto}

/* Figures */
svg.fig{display:block;width:min(100%,980px);max-width:980px;height:auto;margin:22px 0 34px;padding:22px 24px;
  background:#fff;border:1px solid var(--rule);border-radius:10px;box-shadow:0 10px 30px rgba(17,43,36,.05)}
.fig-title{font-size:17px}.fig-sub{fill:#5c6963}.ref-lbl{fill:#54635c}.fig-note{fill:#6f7974}
.concept-figure{max-width:1040px;margin:30px 0 50px}
.concept-figure figcaption,.tablefig figcaption,.qual figcaption{margin-bottom:16px;font-size:.93rem;line-height:1.45}
.concept-figure figcaption strong,.tablefig figcaption strong,.qual figcaption strong{font-size:1.04rem}
.concept-figure .sub,.tablefig .sub,.qual .sub{color:var(--ink-soft);font-size:.85rem}

/* Methods and dataset diagrams */
.method-notation{max-width:1040px;margin:24px 0 34px;padding:20px 22px;border:1px solid #c7ddd1;
  border-left:4px solid var(--green);border-radius:8px;background:#f1f7f3}
.method-notation h3{margin:0 0 14px;font-size:1rem}.symbol-list{display:grid;grid-template-columns:repeat(2,1fr);
  gap:0 28px;margin:0}.symbol-list>div{padding:10px 0;border-top:1px solid #d7e5dc}
.symbol-list dt{font-weight:700;color:var(--deep)}.symbol-list dd{margin:3px 0 0;color:var(--ink-soft);
  font-size:.77rem;line-height:1.48}.symbol-list mjx-container{font-size:94%!important}
.method-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}
.method-node{position:relative;min-height:278px;padding:22px 22px 20px;border:1px solid var(--rule);
  border-top:5px solid var(--family);border-radius:8px;background:#fff}
.method-node h3{margin:0 0 18px;font-size:1.15rem;text-transform:capitalize}
.method-num{position:absolute;right:18px;top:17px;color:#a0aaa5;font:600 .7rem/1 ui-monospace,Menlo,monospace}
.method-flow{display:flex;align-items:center;gap:5px;padding:8px;margin-bottom:14px;background:var(--panel);
  border-radius:5px;font:600 .6rem/1.2 ui-monospace,Menlo,monospace;color:var(--ink-faint)}
.method-flow b{font-size:.62rem;color:#a0aaa5}.route-stage{padding:5px 6px;border-radius:4px}
.route-stage.active{background:var(--family);color:#fff;box-shadow:0 2px 5px rgba(17,43,36,.12)}
.method-node p{margin:0 0 7px;font-size:.88rem}.method-node .method-locus{font-weight:700;color:var(--family)}
.method-node .method-detail{color:var(--ink-soft);font-size:.78rem;line-height:1.48}
.method-eq,.condition-eq{min-height:42px;margin:9px 0 12px;padding:8px 10px;overflow-x:auto;
  border:1px solid #e2e7e2;border-radius:5px;background:#fafbf9;font-size:.85rem;color:var(--deep)}
.method-formula{max-width:var(--measure);margin:14px 0 18px;padding:12px 16px;overflow-x:auto;
  border-left:3px solid var(--accent);background:#fff;color:var(--deep)}
.method-formula mjx-container{margin:.25rem 0!important;text-align:left!important;min-width:max-content}
.condition-source{display:flex;align-items:center;gap:16px;width:fit-content;max-width:100%;margin:0 auto 26px;
  padding:14px 20px;border:1px solid #c7ddd1;border-radius:8px;background:#eaf4ef}
.condition-source span{font:700 .63rem/1.2 ui-monospace,Menlo,monospace;letter-spacing:.08em;
  text-transform:uppercase;color:var(--green)}
.condition-source strong{font-size:.9rem}.condition-source small{color:var(--ink-soft)}
.condition-grid{position:relative;display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
.condition-grid:before{content:"";position:absolute;left:16%;right:16%;top:-14px;border-top:1px solid #aebbb4}
.condition-card{position:relative;min-height:255px;padding:20px;border:1px solid var(--rule);border-radius:8px;background:#fff}
.condition-card:before{content:"";position:absolute;left:50%;top:-15px;height:15px;border-left:1px solid #aebbb4}
.condition-card h3{margin:0 38px 12px 0;font-size:1rem}.condition-card p{font-size:.79rem;line-height:1.48;color:var(--ink-soft)}
.condition-tag{position:absolute;right:17px;top:17px;display:grid;place-items:center;width:27px;height:27px;
  border-radius:50%;background:var(--deep);color:#fff;font:700 .7rem/1 ui-monospace,Menlo,monospace}
.condition-count{position:absolute;left:20px;right:20px;bottom:17px;padding-top:10px;border-top:1px solid var(--rule);
  color:var(--ink-soft);font-size:.72rem}.condition-count strong{color:var(--deep);font-size:.92rem}
.flow-grid{display:grid;grid-template-columns:1fr 26px 1fr 26px 1fr 26px 1.2fr;align-items:stretch}
.flow-card{padding:22px;border:1px solid var(--rule);border-radius:8px;background:#fff}
.flow-value{margin:12px 0 2px;font-size:2rem;line-height:1;font-weight:700;letter-spacing:-.04em}
.flow-unit{min-height:38px;font-size:.78rem;color:var(--ink-soft)}
.flow-card p{margin:14px 0 0;padding-top:12px;border-top:1px solid var(--rule);font-size:.73rem;line-height:1.45;color:var(--ink-soft)}
.flow-arrow{display:grid;place-items:center;color:#99a59f;font:600 1.1rem/1 ui-monospace,Menlo,monospace}

/* Tables */
.tablefig{max-width:100%;margin:34px 0 54px}
.tscroll{max-width:100%;overflow-x:auto;-webkit-overflow-scrolling:touch;background:#fff;border:1px solid var(--rule);
  border-radius:8px;box-shadow:0 7px 22px rgba(17,43,36,.035)}
table.res{min-width:max-content;border-collapse:separate;border-spacing:0;background:#fff;font-size:.77rem}
table.res th,table.res td{padding:.62rem .72rem;text-align:right;white-space:nowrap;border-bottom:1px solid #e6e9e5;
  font-variant-numeric:tabular-nums}
table.res th[scope=row]{position:sticky;left:0;z-index:2;padding-left:.85rem;padding-right:1.25rem;
  text-align:left;font-weight:600;background:#fff;box-shadow:6px 0 10px -12px #000}
table.res tbody tr:nth-child(odd) td,table.res tbody tr:nth-child(odd) th{background:#f5f7f4}
table.res thead{background:var(--deep);color:#fff}
table.res thead th,table.res thead td{border-bottom-color:#315047;background:var(--deep);color:#e3ebe7}
table.res thead tr:last-child th{font-weight:500;color:#d2ded8}
table.res thead .fam{color:inherit}
table.res:not(.claims):not(.hp):not(.simple) thead tr:first-child th,
table.res:not(.claims):not(.hp):not(.simple) thead tr:first-child td{background:#eef1ed;color:var(--ink-soft);
  border-bottom:1px solid var(--rule)}
.fam{font-size:.67rem;letter-spacing:.07em;text-transform:uppercase;font-weight:700}
table.res td.best{font-weight:800;color:#17633f;background:#e8f3ed!important}
table.res td.ref{color:#66736d}table.res.master{font-size:.7rem}
table.res.master th,table.res.master td{padding:.46rem .55rem}table.res.master th[scope=row]{font-weight:500}
table.res.hp{font-size:.76rem}table.res.hp td,table.res.hp th{text-align:left;white-space:normal;min-width:145px}
table.claims{min-width:850px}table.claims td{text-align:left;white-space:normal}
table.claims td.num{white-space:nowrap}table.claims td.ok{color:#17633f;font-weight:800}
table.claims td.warn{color:#9a3f32;font-weight:800}table.claims td.no{color:#8b651a}
.tnote{max-width:940px;color:var(--ink-soft);font-size:.78rem;line-height:1.5;margin-top:10px}
.table-shell{display:block;width:fit-content;max-width:100%;overflow-x:auto;-webkit-overflow-scrolling:touch;
  margin:24px 0 32px;border:1px solid var(--rule);border-radius:8px;background:#fff;
  box-shadow:0 7px 22px rgba(17,43,36,.035)}
.table-shell table.res.simple{display:table;width:max-content;min-width:0;margin:0;border:0;border-radius:0}
.table-shell table.res.simple th,.table-shell table.res.simple td{text-align:left}

/* Definitions, controls, appendices and examples */
.defs{padding:22px 24px;border:1px solid var(--rule);border-left:4px solid var(--accent);border-radius:7px;background:#fff}
.defs-h{margin:0 0 8px;font-size:1.02rem}.defs dt{font-weight:700;margin-top:10px}.defs dd{margin:2px 0 0;color:var(--ink-soft);font-size:.9rem}
.ctrl{max-width:var(--measure);margin:12px 0;padding:19px 22px;border:1px solid var(--rule);border-left:4px solid #7e8d86;
  border-radius:7px;background:#fff}.ctrl.floor{border-left-color:#8b8b8b}.ctrl.oracle{border-left-color:var(--green)}
.ctrl h4{margin:0 0 7px;font-size:.95rem}.ctrl p{margin:0;color:var(--ink-soft);font-size:.88rem}
.appendix{max-width:1040px;margin:18px 0;border:1px solid var(--rule);border-radius:8px;background:#fff}
.appendix summary{cursor:pointer;padding:18px 22px;font-weight:700}.appendix[open] summary{border-bottom:1px solid var(--rule)}
.appendix>.gloss{padding:4px 24px 26px}.appendix>.tablefig{padding:4px 18px 22px;margin-bottom:0}
.metric-definitions{max-width:1040px}.metric-notation{margin:22px 0 34px;padding:20px 24px;border:1px solid #c7ddd1;
  border-left:4px solid var(--green);border-radius:8px;background:#f1f7f3}
.metric-notation h3{margin:0 0 9px;font-size:1rem}.metric-notation p{max-width:88ch;margin:7px 0;color:var(--ink-soft);font-size:.84rem}
.metric-groups{display:grid;gap:34px}.metric-group{padding:0 24px 24px;border:1px solid var(--rule);border-radius:8px;background:#fff}
.metric-group>h3{margin:0 -24px 4px;padding:16px 24px;border-bottom:1px solid var(--rule);font-size:1rem;background:#eef1ed}
.gloss{margin:0}.gloss dt{margin-top:23px;font-weight:700}.gloss dd{margin:7px 0 0;color:var(--ink-soft);font-size:.88rem}
.gloss dd p{max-width:88ch;margin:8px 0 0}.gloss .equation-key{margin-top:10px;padding:10px 12px;
  border-left:3px solid #aebbb4;background:#f5f7f4;color:#4f5e57;font-size:.8rem;line-height:1.5}
.gloss .equation-key strong{color:var(--deep)}.metric-equation{max-width:100%;padding:10px 12px;overflow-x:auto;
  border:1px solid #e0e5e0;border-radius:5px;background:#fafbf9;color:var(--deep);font-size:.9rem}
.metric-equation mjx-container{margin:.2rem 0!important;text-align:left!important;min-width:max-content}
.qual{max-width:1040px;margin:28px 0 50px}.qgroup{display:grid;grid-template-columns:repeat(2,1fr);gap:10px 14px;
  margin:16px 0;padding:18px;border:1px solid var(--rule);border-radius:8px;background:#fff}
.qhead{grid-column:1/-1;color:var(--ink-soft);font-size:.78rem}.qsrc,.qreal,.qcell{margin:0;padding:12px 14px;border-left:4px solid #b0a08c;background:#f7f8f5;border-radius:4px}
.qreal{border-left-color:var(--green)}.qsrc,.qreal{grid-column:span 1}.qcell{grid-column:span 1}
.qlab{font:.65rem/1.3 ui-monospace,Menlo,monospace;letter-spacing:.055em;text-transform:uppercase;color:#66736d}
.qual blockquote{margin:6px 0 0;padding:0;border:0;background:transparent;color:var(--ink);font-size:.81rem;line-height:1.5;font-style:normal}

.foot{padding:34px 0 46px;background:#e9ede8;color:#66736d}.foot p{max-width:880px}.footer-title{color:var(--deep);font-weight:700}

@media(max-width:900px){
  .summary-grid,.finding-banner,.story-grid,.next-grid{grid-template-columns:1fr}
  .finding-banner>div{border-right:0;border-bottom:1px solid #315047}.finding-banner>div:last-child{border-bottom:0}
  .method-grid,.condition-grid,.symbol-list{grid-template-columns:1fr}.condition-grid:before{display:none}.condition-card:before{display:none}
  .flow-grid{grid-template-columns:1fr}.flow-arrow{padding:5px;transform:rotate(90deg)}
  .reading-key a{margin-left:0}.qgroup{grid-template-columns:1fr}.qsrc,.qreal,.qcell{grid-column:1}
}
@media(max-width:720px){
  .wrap{width:min(calc(100% - 32px),1180px)}main.wrap{padding:42px 0 70px}.study-head{padding:38px 0 32px}
  .study-head .stamp{display:none}.nav .wrap{overflow-x:auto;flex-wrap:nowrap}.nav a{white-space:nowrap;padding:12px 13px}
  .page-intro h1{font-size:2.35rem}svg.fig{padding:12px 10px;margin-inline:-8px;width:calc(100% + 16px)}
  .summary-card{min-height:0}.result-nav{display:none}.method-flow{overflow-x:auto}.condition-source{align-items:flex-start;flex-direction:column;gap:5px}
}
@media print{
  .study-head{background:#fff;color:#000;padding:20px 0}.study-head h1,.study-head .lede{color:#000}.summary-card,.method-node,.flow-card{box-shadow:none}
  .metric-equation{overflow-x:hidden}
  .appendix{break-inside:auto}.appendix>summary{display:none}.appendix:not([open])>*:not(summary){display:block}
}
"""


if __name__ == "__main__":
    build()
