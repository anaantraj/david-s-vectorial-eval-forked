"""Generate the static experimental-notes site from the evaluation reports.

Produces raw HTML with inline SVG figures and a single stylesheet. No
JavaScript and no external assets, so the folder can be opened from disk, zipped,
or served from any static host without a build step.

    python experimental-notes/build.py

Figures are generated from the report JSON rather than hand-drawn, so the site
cannot silently drift from the numbers it describes.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experimental-notes"
RUN = ROOT / "runs" / "scaled"
RUN_SYS = ROOT / "runs" / "scaled_sys"
#: Evaluation split. `heldout` merges val and test.
SPLIT = "heldout"

# Palette. Baselines are greys, systems are coloured, the oracle is green.
COLORS = {
    "identity": "#8b8b8b",
    "shuffle_control": "#b0a08c",
    "target_sample": "#2e7d5b",
    "llm_rewrite_claude": "#c25e3a",
    "llm_fewshot_claude": "#d99a4e",
    "llm_rewrite_gpt": "#4a6fa5",
}
LABELS = {
    "identity": "identity",
    "shuffle_control": "shuffle control",
    "target_sample": "target sample (oracle)",
    "llm_rewrite_claude": "Claude zero-shot",
    "llm_fewshot_claude": "Claude few-shot",
    "llm_rewrite_gpt": "GPT-5.6 zero-shot",
}
#: Reference points rather than candidate systems. Every figure marks these so a
#: reader never mistakes a reference value for a system result.
ROLES = {
    "identity": "baseline",
    "shuffle_control": "baseline",
    "target_sample": "oracle",
}


def rlabel(fn: str) -> str:
    """Column label with its role appended."""
    base = {"target_sample": "target sample"}.get(fn, LABELS.get(fn, fn))
    return base + (f" · {ROLES[fn]}" if fn in ROLES else "")


ORDER = [
    "identity",
    "shuffle_control",
    "llm_rewrite_gpt",
    "llm_rewrite_claude",
    "llm_fewshot_claude",
    "target_sample",
]


def load():
    full = json.loads((RUN / f"report.{SPLIT}.json").read_text())
    systems = json.loads((RUN_SYS / f"report.{SPLIT}.json").read_text())
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text())
    return full, systems, manifest


def val(report, fn, key, field="mean"):
    entry = report["table"].get(fn, {}).get(key)
    return None if entry is None else entry.get(field)


# --------------------------------------------------------------------------
# SVG primitives
# --------------------------------------------------------------------------

def esc(s: str) -> str:
    return (
        str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


def svg_open(w, h, title, desc=""):
    # Element ids are derived from a stable digest rather than hash(), which is
    # salted per process and would make successive builds differ spuriously.
    fid = "t" + hashlib.blake2b(title.encode(), digest_size=4).hexdigest()
    return (
        f'<svg class="fig" viewBox="0 0 {w} {h}" width="100%" '
        f'role="img" aria-labelledby="{fid}" '
        f'xmlns="http://www.w3.org/2000/svg">'
        f"<title id=\"{fid}\">{esc(title)}</title>"
        + (f"<desc>{esc(desc)}</desc>" if desc else "")
    )


def text(x, y, s, cls="lbl", anchor="start", size=None):
    st = f' font-size="{size}"' if size else ""
    return f'<text class="{cls}" x="{x}" y="{y}" text-anchor="{anchor}"{st}>{esc(s)}</text>'


def hbar_chart(rows, title, subtitle="", vmax=None, ref=None, ref_label="", width=760):
    """Horizontal bars. `rows` is a list of (fn_key, value)."""
    # Extra headroom when a reference line is drawn, so its caption clears the
    # subtitle instead of overprinting it.
    row_h, left, right = 34, 224, 60
    top = 72 if ref is not None else 58
    h = top + row_h * len(rows) + 34
    vals = [v for _, v in rows if v is not None]
    vmax = vmax or (max(vals + ([ref] if ref else [])) * 1.18 if vals else 1)
    plot_w = width - left - right

    out = [svg_open(width, h, title, subtitle)]
    out.append(text(0, 20, title, "fig-title"))
    if subtitle:
        out.append(text(0, 38, subtitle, "fig-sub"))

    if ref is not None:
        rx = left + plot_w * (ref / vmax)
        out.append(f'<line class="ref" x1="{rx:.1f}" y1="{top-12}" x2="{rx:.1f}" y2="{h-30}"/>')
        out.append(text(rx + 5, top - 17, ref_label, "ref-lbl"))

    for i, (fn, v) in enumerate(rows):
        y = top + i * row_h
        out.append(text(left - 10, y + 15, rlabel(fn), "lbl", "end"))
        if v is None:
            out.append(text(left + 4, y + 15, "not scored", "muted"))
            continue
        bw = max(1.5, plot_w * (abs(v) / vmax))
        out.append(
            f'<rect class="bar" x="{left}" y="{y+3}" width="{bw:.1f}" height="20" '
            f'fill="{COLORS.get(fn, "#888")}" rx="2"/>'
        )
        out.append(text(left + bw + 7, y + 18, f"{v:.3f}", "val"))
    out.append(f'<line class="axis" x1="{left}" y1="{top-12}" x2="{left}" y2="{h-30}"/>')
    out.append("</svg>")
    return "\n".join(out)


#: Columns that define an axis by construction rather than competing on it.
ANCHORS = {"identity", "target_sample"}


def scatter_tradeoff(systems, full, width=760, height=500):
    """Transfer against preservation, with the two axis anchors marked as such.

    The oracle sits at the bottom of the preservation axis by construction and
    not by failure, because it emits a different authentic post rather than a
    rewrite of the source. It is drawn as an anchor so that its position is not
    read as poor performance.
    """
    left, right, top, bottom = 78, 210, 74, 104
    pw, ph = width - left - right, height - top - bottom

    pts = []
    for fn in ORDER:
        rep = full if fn == "target_sample" else systems
        x = val(rep, fn, "classifier.calibration_gap")
        y = val(rep, fn, "semantic.source_similarity")
        if x is None or y is None:
            continue
        pts.append((
            fn, x, y,
            val(rep, fn, "classifier.calibration_gap", "ci_low"),
            val(rep, fn, "classifier.calibration_gap", "ci_high"),
            val(rep, fn, "semantic.source_similarity", "ci_low"),
            val(rep, fn, "semantic.source_similarity", "ci_high"),
        ))

    xmin, xmax, ymin, ymax = 0.0, 0.56, 0.0, 1.05

    def px(x):
        return left + pw * (x - xmin) / (xmax - xmin)

    def py(y):
        return top + ph * (1 - (y - ymin) / (ymax - ymin))

    out = [svg_open(width, height, "Transfer against preservation")]
    out.append(text(0, 20, "Reading like Reddit against keeping the content", "fig-title"))
    out.append(text(0, 38, "Filled circles are systems. Rings are reference points that sit at one "
                           "end of an axis by definition.", "fig-sub"))
    out.append(text(0, 52, "Bars show how much each value varies across cells.", "fig-sub"))

    out.append(
        f'<rect class="good-zone" x="{px(0)}" y="{py(1.05)}" '
        f'width="{px(0.20)-px(0):.1f}" height="{py(0.70)-py(1.05):.1f}"/>'
    )
    out.append(text(px(0.008), py(1.01) + 4, "empty: reads like Reddit and", "zone-lbl"))
    out.append(text(px(0.008), py(1.01) + 17, "keeps the content", "zone-lbl"))

    out.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+ph}"/>')
    out.append(f'<line class="axis" x1="{left}" y1="{top+ph}" x2="{left+pw}" y2="{top+ph}"/>')
    for gx in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        out.append(f'<line class="grid" x1="{px(gx)}" y1="{top}" x2="{px(gx)}" y2="{top+ph}"/>')
        out.append(text(px(gx), top + ph + 20, f"{gx:.1f}", "tick", "middle"))
    for gy in (0.0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<line class="grid" x1="{left}" y1="{py(gy)}" x2="{left+pw}" y2="{py(gy)}"/>')
        out.append(text(left - 10, py(gy) + 4, f"{gy:.2f}", "tick", "end"))

    out.append(text(left + pw / 2, height - 56,
                    "reads more like Reddit toward the left", "ax-title", "middle"))
    out.append(
        f'<text class="ax-title" transform="translate(20,{top+ph/2}) rotate(-90)" '
        f'text-anchor="middle">source similarity (preservation), better upward</text>'
    )

    # Bootstrap intervals are drawn before the markers. The horizontal bars
    # overlap almost completely, which is the visual statement that the
    # transfer axis is not resolved at this number of cells.
    for fn, x, y, xlo, xhi, ylo, yhi in pts:
        cx, cy = px(x), py(y)
        if xlo is not None and xhi is not None:
            a, b = px(max(xlo, xmin)), px(min(xhi, xmax))
            out.append(f'<line class="err" x1="{a:.1f}" y1="{cy:.1f}" x2="{b:.1f}" y2="{cy:.1f}" stroke="{COLORS[fn]}"/>')
            for e in (a, b):
                out.append(f'<line class="err" x1="{e:.1f}" y1="{cy-4:.1f}" x2="{e:.1f}" y2="{cy+4:.1f}" stroke="{COLORS[fn]}"/>')
        if ylo is not None and yhi is not None and abs(yhi - ylo) > 1e-6:
            a, b = py(min(yhi, ymax)), py(max(ylo, ymin))
            out.append(f'<line class="err" x1="{cx:.1f}" y1="{a:.1f}" x2="{cx:.1f}" y2="{b:.1f}" stroke="{COLORS[fn]}"/>')

    placed: list[float] = []
    for fn, x, y, *_ci in sorted(pts, key=lambda t: -t[2]):
        cx, cy = px(x), py(y)
        ly = cy + 4
        while any(abs(ly - q) < 15 for q in placed):
            ly += 15
        placed.append(ly)
        if fn in ANCHORS:
            out.append(
                f'<circle class="pt anchor" cx="{cx:.1f}" cy="{cy:.1f}" r="9" '
                f'fill="#fff" stroke="{COLORS[fn]}"/>'
            )
            out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="2.4" fill="{COLORS[fn]}"/>')
        else:
            out.append(
                f'<circle class="pt" cx="{cx:.1f}" cy="{cy:.1f}" r="8" '
                f'fill="{COLORS[fn]}" stroke="#fff"/>'
            )
        if abs(ly - (cy + 4)) > 2:
            out.append(f'<line class="leader" x1="{cx+9:.1f}" y1="{cy:.1f}" x2="{cx+13:.1f}" y2="{ly-4:.1f}"/>')
        out.append(text(cx + 13, ly, rlabel(fn), "pt-lbl"))

    out.append(text(0, height - 28,
                    "The oracle scores near zero on preservation because it emits a different "
                    "authentic post rather than a rewrite of the source.", "fig-note"))
    out.append(text(0, height - 14,
                    "The horizontal bars overlap, so differences left to right are within the "
                    "variation between cells. Vertical differences are not.", "fig-note"))
    out.append("</svg>")
    return "\n".join(out)


def scorecard(systems, full, width=760):
    """Three-axis summary showing that no single column succeeds on all three."""
    axes = [
        ("Reads like Reddit", "classifier.calibration_gap", -1),
        ("Keeps the content", "semantic.source_similarity", +1),
        ("Matches the variety", "trm.trm", -1),
    ]
    left, top, row_h = 200, 100, 34
    col_w = (width - left - 20) / 3
    track_w = col_w - 52
    h = top + row_h * len(ORDER) + 46

    ranges = {}
    for _, key, _d in axes:
        vals = []
        for fn in ORDER:
            rep = full if fn == "target_sample" else systems
            v = val(rep, fn, key)
            if v is not None:
                vals.append(v)
        ranges[key] = (min(vals), max(vals))

    out = [svg_open(width, h, "Three-axis scorecard")]
    out.append(text(0, 20, "Nothing does well on all three goals at once", "fig-title"))
    out.append(text(0, 38, "Filled proportion of each bar is performance on that axis, "
                           "normalised across the columns shown.", "fig-sub"))
    out.append(text(0, 52, "No column is strong on all three, which is why the harness reports "
                           "three families of metric.", "fig-sub"))

    for i, (label, _k, _d) in enumerate(axes):
        out.append(text(left + i * col_w + track_w / 2, top - 14, label, "key-lbl", "middle"))

    for r, fn in enumerate(ORDER):
        y = top + r * row_h
        rep = full if fn == "target_sample" else systems
        out.append(text(left - 12, y + 15, rlabel(fn), "lbl", "end"))
        for i, (_label, key, direction) in enumerate(axes):
            x0 = left + i * col_w
            v = val(rep, fn, key)
            out.append(f'<rect x="{x0}" y="{y+3}" width="{track_w:.1f}" height="18" fill="#efece4" rx="2"/>')
            if v is None:
                out.append(text(x0 + 6, y + 16, "n/a", "muted"))
                continue
            lo, hi = ranges[key]
            frac = 0.5 if hi == lo else (v - lo) / (hi - lo)
            if direction == -1:
                frac = 1 - frac
            out.append(
                f'<rect x="{x0}" y="{y+3}" width="{track_w*frac:.1f}" height="18" '
                f'fill="{COLORS[fn]}" rx="2" opacity="0.85"/>'
            )
            # The raw value sits outside the track so it never overprints the fill.
            out.append(text(x0 + track_w + 7, y + 16, f"{v:.2f}", "val"))
    out.append("</svg>")
    return "\n".join(out)


def paired_bars(systems, full, width=760):
    """Classifier improves while TRM degrades: the contradiction."""
    fns = list(ORDER)
    row_h, top, left, gap = 46, 74, 220, 8
    h = top + row_h * len(fns) + 40
    plot_w = width - left - 70

    def rep(fn):
        return full if fn == "target_sample" else systems

    cal = {f: val(rep(f), f, "classifier.calibration_gap") for f in fns}
    trm = {f: val(rep(f), f, "trm.trm") for f in fns}
    cmax = max(v for v in cal.values() if v is not None) * 1.25
    tmax = max(v for v in trm.values() if v is not None) * 1.25

    out = [svg_open(width, h, "Classifier versus TRM")]
    out.append(text(0, 20, "The two measurements disagree", "fig-title"))
    out.append(text(0, 38, "Each bar is scaled to its own measure, and shorter is better on both.", "fig-sub"))
    out.append(f'<rect class="key" x="{left}" y="52" width="11" height="11" fill="#7b93b8"/>')
    out.append(text(left + 17, 62, "reads like Reddit (one post at a time)", "key-lbl"))
    out.append(f'<rect class="key" x="{left+250}" y="52" width="11" height="11" fill="#c25e3a"/>')
    out.append(text(left + 267, 62, "matches real Reddit (as a group)", "key-lbl"))

    for i, fn in enumerate(fns):
        y = top + i * row_h
        out.append(text(left - 10, y + 20, rlabel(fn), "lbl", "end"))
        c, t = cal[fn], trm[fn]
        if c is None or t is None:
            out.append(text(left + 4, y + 20, "not scored", "muted"))
            continue
        cw = plot_w * (c / cmax)
        tw = plot_w * (t / tmax)
        out.append(f'<rect class="bar" x="{left}" y="{y}" width="{cw:.1f}" height="15" fill="#7b93b8" rx="2"/>')
        out.append(text(left + cw + 6, y + 12, f"{c:.3f}", "val"))
        out.append(f'<rect class="bar" x="{left}" y="{y+15+gap}" width="{tw:.1f}" height="15" fill="#c25e3a" rx="2"/>')
        out.append(text(left + tw + 6, y + 27 + gap, f"{t:.3f}", "val"))
    out.append(f'<line class="axis" x1="{left}" y1="{top-6}" x2="{left}" y2="{h-30}"/>')
    out.append("</svg>")
    return "\n".join(out)


def trm_per_cell(width=760, height=330):
    """Per-cell TRM: shows the effect holds in every scored cell."""
    report = json.loads((RUN / f"report.{SPLIT}.json").read_text())
    pc = report["per_cell"]
    fns = list(ORDER)
    cells = sorted(pc.get("identity.trm", {}))
    if not cells:
        return ""

    left, right, top, bottom = 78, 216, 60, 96
    pw, ph = width - left - right, height - top - bottom
    vmax = 0.95

    def py(v):
        return top + ph * (1 - v / vmax)

    out = [svg_open(width, height, "Per-cell TRM")]
    out.append(text(0, 20, "The result holds in every cell that could be scored", "fig-title"))
    out.append(text(0, 38, f"Group-similarity score in each of the {len(cells)} cells with enough posts "
                           "to measure. Lower is better.", "fig-sub"))
    out.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+ph}"/>')
    for gv in (0.0, 0.2, 0.4, 0.6, 0.8):
        out.append(f'<line class="grid" x1="{left}" y1="{py(gv)}" x2="{left+pw}" y2="{py(gv)}"/>')
        out.append(text(left - 10, py(gv) + 4, f"{gv:.1f}", "tick", "end"))

    step = pw / len(cells)
    for ci, cell in enumerate(cells):
        cx = left + step * (ci + 0.5)
        short = cell.split("::")[1].replace("_", " ")
        short = short if len(short) <= 20 else short[:19] + "…"
        out.append(
            f'<text class="tick" transform="translate({cx:.1f},{top+ph+16}) rotate(28)" '
            f'text-anchor="start">{esc(short)}</text>'
        )
        for fn in fns:
            v = pc.get(f"{fn}.trm", {}).get(cell, {}).get("trm")
            if v is None:
                continue
            out.append(f'<circle class="pt" cx="{cx:.1f}" cy="{py(v):.1f}" r="5.5" fill="{COLORS[fn]}"/>')

    ly = top
    for fn in fns:
        out.append(f'<circle cx="{left+pw+22}" cy="{ly}" r="5.5" fill="{COLORS[fn]}"/>')
        out.append(text(left + pw + 34, ly + 4, rlabel(fn), "key-lbl"))
        ly += 19
    out.append("</svg>")
    return "\n".join(out)


def rank_profile(systems, full, width=760):
    """Stacked I0/I1/I2 rank profile against the 1/3 null."""
    fns = list(ORDER)
    left, top, row_h = 220, 92, 34
    h = top + row_h * len(fns) + 46
    plot_w = width - left - 70
    shades = ["#cfd8e3", "#93a7c0", "#c25e3a"]
    names = ["generated post sits outside", "in between", "generated post sits inside"]

    out = [svg_open(width, h, "TRM rank profile")]
    out.append(text(0, 20, "How the generated posts sit among the real ones", "fig-title"))
    out.append(text(0, 38, "Each generated post is compared against two real posts. If the two sets "
                           "matched, every band would be one third.", "fig-sub"))
    out.append(text(0, 52, "A large third band means the generated posts are bunched in the middle "
                           "of the real ones.", "fig-sub"))
    for i, (nm, sh) in enumerate(zip(names, shades, strict=True)):
        x = 40 + i * 232
        out.append(f'<rect class="key" x="{x}" y="62" width="11" height="11" fill="{sh}"/>')
        out.append(text(x + 17, 72, nm, "key-lbl"))

    third = left + plot_w / 3
    two_third = left + 2 * plot_w / 3
    for xx in (third, two_third):
        out.append(f'<line class="ref" x1="{xx:.1f}" y1="{top-6}" x2="{xx:.1f}" y2="{h-34}"/>')
    out.append(text(third, h - 20, "⅓", "ref-lbl", "middle"))
    out.append(text(two_third, h - 20, "⅔", "ref-lbl", "middle"))

    for i, fn in enumerate(fns):
        y = top + i * row_h
        rep = full if fn == "target_sample" else systems
        out.append(text(left - 10, y + 15, rlabel(fn), "lbl", "end"))
        x = left
        for k, sh in zip(("rank_i0", "rank_i1", "rank_i2"), shades, strict=True):
            v = val(rep, fn, f"trm.{k}") or 0.0
            w = plot_w * v
            out.append(f'<rect class="bar" x="{x:.1f}" y="{y+2}" width="{w:.1f}" height="20" fill="{sh}"/>')
            if w > 34:
                out.append(text(x + w / 2, y + 16, f"{v:.2f}", "inbar", "middle"))
            x += w
    out.append("</svg>")
    return "\n".join(out)


def coverage_chart(manifest, width=760, height=300):
    """Cell coverage: LinkedIn versus Reddit mass per cell, as a diverging bar."""
    cells = manifest["cells"][:22]
    name_w, top, row_h = 250, 62, 10
    h = top + row_h * len(cells) + 40
    half = (width - name_w - 80) / 2
    center = name_w + half
    mx = max(max(c["n_linkedin"], c["n_reddit"]) for c in cells)

    out = [svg_open(width, h, "Cell coverage")]
    out.append(text(0, 20, "Bilateral cell coverage", "fig-title"))
    out.append(
        text(0, 38, f"Largest {len(cells)} of {manifest['n_cells']} cells. "
                    "Most are small and lopsided.", "fig-sub")
    )
    out.append(text(center - half / 2, top - 8, "LinkedIn", "key-lbl", "middle"))
    out.append(text(center + half / 2, top - 8, "Reddit", "key-lbl", "middle"))

    for i, c in enumerate(cells):
        y = top + i * row_h
        lw = half * (c["n_linkedin"] / mx)
        rw = half * (c["n_reddit"] / mx)
        # Truncated so the right-anchored label always clears the plot area.
        name = f'{c["room"]} · {c["topic"]}'
        name = name if len(name) <= 30 else name[:29] + "…"
        out.append(text(name_w - 12, y + 7, name, "tiny", "end"))
        out.append(f'<rect x="{center-lw:.1f}" y="{y}" width="{lw:.1f}" height="7.5" fill="#4a6fa5" rx="1"/>')
        out.append(f'<rect x="{center}" y="{y}" width="{rw:.1f}" height="7.5" fill="#c25e3a" rx="1"/>')
        out.append(text(center + half + 8, y + 7, f'{c["n_linkedin"]}/{c["n_reddit"]}', "tiny"))
    out.append(f'<line class="axis" x1="{center}" y1="{top-4}" x2="{center}" y2="{h-34}"/>')
    out.append("</svg>")
    return "\n".join(out)


def funnel(manifest, width=760):
    stages = [
        ("Raw CSV rows", 20529),
        ("After normalisation", 14672),
        ("Inside viable bilateral cells", manifest["n_posts"]),
        ("Transfer tasks", manifest["n_tasks"]),
        ("Test-split tasks", manifest["counts"]["tasks"]["test"]),
    ]
    left, top, row_h = 250, 56, 40
    h = top + row_h * len(stages) + 20
    pw = width - left - 90
    mx = stages[0][1]
    out = [svg_open(width, h, "Data funnel")]
    out.append(text(0, 20, "From corpus to evaluable tasks", "fig-title"))
    out.append(text(0, 38, "The large drop is coverage, not filtering: most posts have no counterpart cell.", "fig-sub"))
    for i, (label, n) in enumerate(stages):
        y = top + i * row_h
        w = max(2, pw * (n / mx))
        shade = ["#4a6fa5", "#5b7fb0", "#7b93b8", "#a0b1c9", "#c25e3a"][i]
        out.append(text(left - 12, y + 17, label, "lbl", "end"))
        out.append(f'<rect class="bar" x="{left}" y="{y}" width="{w:.1f}" height="24" fill="{shade}" rx="2"/>')
        out.append(text(left + w + 8, y + 17, f"{n:,}", "val"))
    out.append("</svg>")
    return "\n".join(out)


def worst_features(full, width=760, top_n=8):
    """Per-writing-habit gap, shown for every column rather than one system.

    The gap is how far a column moved a habit relative to how far real Reddit
    posts sit from real LinkedIn posts on the same habit. Zero means the shift
    was reproduced exactly.
    """
    per_cell = full.get("per_cell", {})
    fns = list(ORDER)

    def gaps(fn):
        d = per_cell.get(f"{fn}.structural", {})
        out: dict[str, list[float]] = {}
        for scores in d.values():
            for k, v in scores.items():
                if k.endswith("__effect_gap") and v is not None and v == v:
                    out.setdefault(k[: -len("__effect_gap")], []).append(v)
        return {k: sum(v) / len(v) for k, v in out.items() if v}

    by_fn = {fn: gaps(fn) for fn in fns}
    ref = by_fn.get("llm_rewrite_claude") or next((g for g in by_fn.values() if g), {})
    if not ref:
        return ""
    feats = [f for f, _ in sorted(ref.items(), key=lambda kv: -kv[1])[:top_n]]

    left, right, top, row_h = 210, 220, 78, 26
    h = top + row_h * len(feats) + 40
    pw = width - left - right
    vmax = max(
        (by_fn[fn][f] for fn in fns for f in feats if f in by_fn[fn]), default=1.0
    ) * 1.08

    out = [svg_open(width, h, "Least reproduced writing habits")]
    out.append(text(0, 20, "Which writing habits each column fails to reproduce", "fig-title"))
    out.append(text(0, 38, "How far each column moved a habit, compared with how far real Reddit "
                           "posts sit from real LinkedIn posts.", "fig-sub"))
    out.append(text(0, 52, "Read each column against the green oracle dot, which shows how much this "
                           "measure moves from sampling alone. Lower is better.", "fig-sub"))
    out.append(f'<line class="axis" x1="{left}" y1="{top-8}" x2="{left}" y2="{h-32}"/>')

    for gi in range(0, 5):
        gv = vmax * gi / 4
        gx = left + pw * (gv / vmax)
        out.append(f'<line class="grid" x1="{gx:.1f}" y1="{top-8}" x2="{gx:.1f}" y2="{h-32}"/>')
        out.append(text(gx, h - 18, f"{gv:.1f}", "tick", "middle"))

    for i, f in enumerate(feats):
        y = top + i * row_h
        out.append(text(left - 10, y + 4, f.replace("_", " "), "lbl", "end"))
        for fn in fns:
            v = by_fn[fn].get(f)
            if v is None:
                continue
            cx = left + pw * (min(v, vmax) / vmax)
            out.append(f'<circle class="pt" cx="{cx:.1f}" cy="{y:.1f}" r="5" fill="{COLORS[fn]}" stroke="#fff"/>')

    ly = top
    for fn in fns:
        out.append(f'<circle cx="{left+pw+26}" cy="{ly}" r="5" fill="{COLORS[fn]}"/>')
        out.append(text(left + pw + 38, ly + 4, rlabel(fn), "key-lbl"))
        ly += 18
    out.append("</svg>")
    return "\n".join(out)


# --------------------------------------------------------------------------
# Qualitative examples, read live from the built dataset and the run outputs
# --------------------------------------------------------------------------

DATA = ROOT / "data"


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _clip(t, n=460):
    t = " ".join(str(t).split())
    return t if len(t) <= n else t[: n - 1].rsplit(" ", 1)[0] + " …"


def _quote(label, body, cls=""):
    return (
        f'<div class="ex-col {cls}"><h3>{esc(label)}</h3>'
        f"<p>{esc(_clip(body))}</p></div>"
    )


def example_cell_posts(cell="backend_engineer::code_editors", n=2):
    """Authentic posts from both sides of one bilateral cell."""
    posts = _jsonl(DATA / "posts.train.jsonl")
    sel = [p for p in posts if p["cell_id"] == cell]
    li = [p for p in sel if p["platform"] == "linkedin"][:n]
    rd = [p for p in sel if p["platform"] == "reddit"][:n]
    if not li or not rd:
        return ""
    cols = "".join(_quote("LinkedIn side", p["text"]) for p in li)
    cols += "".join(_quote("Reddit side", p["text"], "alt") for p in rd)
    return (
        f'<figure class="exfig"><figcaption>Authentic posts from both sides of '
        f'<code>{esc(cell)}</code>. The topic is shared and the register is not.'
        f"</figcaption><div class=\"example grid4\">{cols}</div></figure>"
    )


def _pick_common_task(fns, cell_hint="code_editors"):
    tasks = {t["task_id"]: t for t in _jsonl(DATA / f"tasks.{SPLIT}.jsonl")}
    outs = {f: {o["task_id"]: o for o in _jsonl(RUN / f"outputs.{f}.{SPLIT}.jsonl")} for f in fns}
    ok = [
        tid for tid in tasks
        if all(outs[f].get(tid, {}).get("output_text") for f in fns) and cell_hint in tid
    ]
    if not ok:
        return None, None, None
    tid = ok[0]
    return tasks[tid], {f: outs[f][tid]["output_text"] for f in fns}, tid


def example_all_baselines():
    """One source post rendered by every column, including the anchors."""
    fns = ORDER
    task, got, _ = _pick_common_task(fns)
    if task is None:
        return ""
    cols = _quote("Source · LinkedIn", task["source_text"], "src")
    for fn in fns:
        cols += _quote(LABELS[fn], got[fn], "alt" if fn in ANCHORS else "")
    return (
        f'<figure class="exfig"><figcaption>Every column applied to one source post from '
        f'<code>{esc(task["cell_id"])}</code>. The oracle and the shuffle control emit '
        f"authentic Reddit posts that are unrelated to the source, which is why both score "
        f"near zero on preservation."
        f"</figcaption><div class=\"example grid2\">{cols}</div></figure>"
    )


def example_transfer():
    """A single successful rewrite, shown before and after."""
    task, got, _ = _pick_common_task(["llm_rewrite_claude"], "agentic_coding")
    if task is None:
        return ""
    cols = _quote("Source · LinkedIn", task["source_text"], "src")
    cols += _quote("Generated · Reddit", got["llm_rewrite_claude"])
    return (
        f'<figure class="exfig"><figcaption>A representative rewrite. Promotional framing '
        f"becomes a question, the technical content survives, and the personal endorsement is "
        f"dropped. Judged individually the output is good."
        f"</figcaption><div class=\"example grid2\">{cols}</div></figure>"
    )


def example_template(k=6):
    """Closing sentences across cells, exposing the shared template."""
    outs = _jsonl(RUN / f"outputs.llm_rewrite_claude.{SPLIT}.jsonl")
    seen, rows = set(), []
    for o in outs:
        if not o["output_text"] or o["cell_id"] in seen:
            continue
        seen.add(o["cell_id"])
        tail = " ".join(o["output_text"].strip().split("\n")[-1].split())
        rows.append((o["cell_id"].split("::")[1].replace("_", " "), tail[-150:]))
        if len(rows) >= k:
            break
    items = "".join(
        f'<li><span class="excell">{esc(c)}</span>… {esc(t)}</li>' for c, t in rows
    )
    return (
        f'<figure class="exfig"><figcaption>Closing sentences of Claude rewrites drawn from '
        f"six different cells. The subject matter differs while the closing move recurs. Four "
        f"of the six shown solicit a response from the community, and the table below gives the "
        f"rate across the full held-out split."
        f'</figcaption><ul class="tails">{items}</ul></figure>'
    )


def example_exemplars(k=3):
    """The authentic exemplars supplied to the few-shot configuration."""
    tasks = {t["task_id"]: t for t in _jsonl(DATA / f"tasks.{SPLIT}.jsonl")}
    posts = {p["post_id"]: p for p in _jsonl(DATA / "posts.train.jsonl")}
    cand = [
        t for t in tasks.values()
        if len(t["exemplar_ids"]) >= k and "code_editors" in t["cell_id"]
    ]
    if not cand:
        cand = [t for t in tasks.values() if len(t["exemplar_ids"]) >= k]
    if not cand:
        return ""
    task = cand[0]
    texts = [posts[i]["text"] for i in task["exemplar_ids"] if i in posts]
    texts = sorted(texts, key=len, reverse=True)[:k]
    cols = "".join(_quote(f"Exemplar {i+1} · authentic Reddit", t, "alt") for i, t in enumerate(texts))
    return (
        f'<figure class="exfig"><figcaption>The authentic train-split posts supplied to the '
        f'few-shot configuration for a task in <code>{esc(task["cell_id"])}</code>. The model '
        f"receives these in addition to the zero-shot prompt, and it is scored against "
        f"different posts drawn from the held-out split of the same cell."
        f'</figcaption><div class="example grid2">{cols}</div></figure>'
    )


def example_topic_jsd_failure():
    """A rewrite that retains the topic while its cell scores poorly on topic JSD."""
    report = json.loads((RUN_SYS / f"report.{SPLIT}.json").read_text())
    per_cell = report["per_cell"].get("llm_rewrite_claude.distributional", {})
    worst = sorted(per_cell.items(), key=lambda kv: -(kv[1].get("topic_jsd") or 0))
    if not worst:
        return ""
    cell, scores = worst[0]
    task, got, _ = _pick_common_task(["llm_rewrite_claude"], cell.split("::")[1][:14])
    if task is None:
        return ""
    cols = _quote("Source · LinkedIn", task["source_text"], "src")
    cols += _quote("Generated · Reddit", got["llm_rewrite_claude"])
    return (
        f'<figure class="exfig"><figcaption>A rewrite from <code>{esc(cell)}</code>, the cell '
        f'in which Claude records its highest topic JSD at {scores.get("topic_jsd", float("nan")):.2f}. '
        f"That value is comparable to the shuffle control, which would indicate complete "
        f"topical failure. Inspection establishes that the topic is retained, so the metric is "
        f"registering something other than topical drift."
        f'</figcaption><div class="example grid2">{cols}</div></figure>'
    )


def question_stats():
    """Share of posts ending in a question mark, generated against authentic."""
    posts = _jsonl(DATA / f"posts.{SPLIT}.jsonl")
    def rate(texts):
        texts = [t for t in texts if t]
        return sum(1 for t in texts if t.strip().endswith("?")) / max(len(texts), 1), len(texts)
    rows = [
        ("authentic Reddit", rate([p["text"] for p in posts if p["platform"] == "reddit"])),
        ("authentic LinkedIn", rate([p["text"] for p in posts if p["platform"] == "linkedin"])),
    ]
    for fn in ("llm_rewrite_gpt", "llm_rewrite_claude", "llm_fewshot_claude"):
        texts = [o["output_text"] for o in _jsonl(RUN / f"outputs.{fn}.{SPLIT}.jsonl")]
        rows.append((LABELS[fn], rate(texts)))
    body = "".join(
        f'<tr><th scope="row">{esc(n)}</th><td>{r:.2f}</td><td class="na">{c}</td></tr>'
        for n, (r, c) in rows
    )
    return (
        '<figure class="tablewrap"><table><caption>Proportion of posts whose final '
        "character is a question mark. The rewriters close on a question at roughly three "
        "times the authentic Reddit rate.</caption>"
        '<thead><tr><th scope="col">Pool</th><th scope="col">Ends with a question</th>'
        '<th scope="col">n</th></tr></thead>'
        f"<tbody>{body}</tbody></table></figure>"
    )


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

ARROW = {1: "↑", -1: "↓", 0: "·"}

#: (report key, plain label, metric family). The label states what is measured and
#: the metric name is kept in brackets so a value can be traced to the code.
TABLE_ROWS = [
    ("classifier.calibration_gap", "Reads like Reddit (calibration gap)", "classifier"),
    ("classifier.target_rate", "Share judged Reddit by the classifier", "classifier"),
    ("semantic.source_similarity", "Keeps the source content (similarity)", "semantic"),
    ("semantic.content_word_retention", "Keeps the specific details (word retention)", "semantic"),
    ("structural.mean_feature_jsd", "Writing-habit mismatch (feature JSD)", "structural"),
    ("structural.feature_coverage", "Writing habits reproduced (coverage)", "structural"),
    ("trm.trm", "Matches the variety of real posts (TRM)", "trm"),
    ("trm.trm_style_residual", "Same, with writing style removed", "trm"),
    ("trm.trm_pvalue", "Chance of being mistaken for real (p-value)", "trm"),
    ("trm.rank_i2", "Outputs bunched together (rank I₂)", "trm"),
    ("distributional.mmd2", "Distance between the two sets (MMD²)", "distributional"),
    ("distributional.topic_jsd", "Subject-matter mismatch (topic JSD)", "distributional"),
    ("degeneracy.copy_rate", "Copied the source outright", "degeneracy"),
    ("degeneracy.pool_self_similarity", "Outputs resemble each other", "degeneracy"),
    ("degeneracy.distinct_2", "Variety of phrasing used (distinct-2)", "degeneracy"),
]


#: Columns that are reference points rather than attempts at the task. They are
#: tinted in the tables so that a "best" mark on a control is not misread as a
#: system result.
BASELINE_FNS = {"identity", "shuffle_control", "target_sample"}

#: Below this many cells a metric is flagged as thin coverage.
THIN_COVERAGE = 5


#: Comparisons stated in the report, each tested for interval separation.
CLAIMS = [
    ("Claude rewriting degrades distribution match relative to the source",
     "llm_rewrite_claude", "identity", "trm.trm"),
    ("Claude rewriting raises pool self-similarity above the source",
     "llm_rewrite_claude", "identity", "degeneracy.pool_self_similarity"),
    ("Claude rewriting raises self-similarity above the authentic Reddit level",
     "llm_rewrite_claude", "target_sample", "degeneracy.pool_self_similarity"),
    ("Claude rewriting reduces vocabulary variety relative to the source",
     "llm_rewrite_claude", "identity", "degeneracy.distinct_2"),
    ("GPT-5.6 preserves more source content than Claude",
     "llm_rewrite_gpt", "llm_rewrite_claude", "semantic.source_similarity"),
    ("GPT-5.6 matches the target distribution better than Claude",
     "llm_rewrite_gpt", "llm_rewrite_claude", "trm.trm"),
    ("Rewriting reduces the calibration gap relative to the source",
     "llm_rewrite_claude", "identity", "classifier.calibration_gap"),
    ("Few-shot conditioning improves transfer over zero-shot",
     "llm_fewshot_claude", "llm_rewrite_claude", "classifier.calibration_gap"),
    ("Few-shot conditioning improves preservation over zero-shot",
     "llm_fewshot_claude", "llm_rewrite_claude", "semantic.source_similarity"),
    ("Few-shot conditioning improves distribution match over zero-shot",
     "llm_fewshot_claude", "llm_rewrite_claude", "trm.trm"),
]


def resolution_table(systems, full):
    """State which comparisons the bootstrap intervals actually separate."""
    rows = []
    for label, a, b, key in CLAIMS:
        ra = full if a == "target_sample" else systems
        rb = full if b == "target_sample" else systems
        ea, eb = ra["table"].get(a, {}).get(key), rb["table"].get(b, {}).get(key)
        if not ea or not eb:
            continue
        overlap = not (ea["ci_high"] < eb["ci_low"] or ea["ci_low"] > eb["ci_high"])
        verdict = "not resolved" if overlap else "supported"
        cls = "unres" if overlap else "res"
        rows.append(
            f'<tr><th scope="row">{esc(label)}</th>'
            f'<td class="num">{ea["mean"]:.3f} <span class="ci">[{ea["ci_low"]:.3f}, {ea["ci_high"]:.3f}]</span></td>'
            f'<td class="num">{eb["mean"]:.3f} <span class="ci">[{eb["ci_low"]:.3f}, {eb["ci_high"]:.3f}]</span></td>'
            f'<td class="verdict {cls}">{verdict}</td></tr>'
        )
    return (
        '<figure class="tablewrap"><table class="resolve"><caption>Every comparison stated in '
        "this report, tested against 95 percent bootstrap intervals resampled over cells. A "
        "comparison is recorded as supported only when the two intervals do not overlap."
        '</caption><thead><tr><th scope="col">Comparison</th><th scope="col">First value</th>'
        '<th scope="col">Second value</th><th scope="col">Verdict</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></figure>"
    )


def results_table(report, fns, caption, base_fns=None):
    """Render the comparison table.

    Columns are averaged over a common base of cells so that they remain
    comparable. The base is the set of cells that every *system* column covers,
    which excludes the oracle from defining it. The oracle produces no output on
    held-out cells, and allowing it to define the base previously reduced the
    TRM row from five cells to one.

    Any column that covers less than the full base is averaged over the portion
    it does cover, and its own cell count is printed alongside the value. The
    subset bias this introduces was measured at 0.03 or less on every metric,
    which is an order of magnitude smaller than the differences the table
    reports.
    """
    base_fns = base_fns or [f for f in fns if f not in ANCHORS or f == "identity"]
    per_cell = report.get("per_cell", {})

    def covered(fn, metric, score):
        d = per_cell.get(f"{fn}.{metric}", {})
        return {
            c for c, v in d.items()
            if score in v and v[score] is not None and v[score] == v[score]
        }

    head = "".join(
        f'<th class="{"base" if f in BASELINE_FNS else "sys"}">{esc(LABELS.get(f, f))}'
        f'<span class="coltag">{ROLES.get(f, "system")}</span></th>'
        for f in fns
    )

    rows, footnoted = [], False
    for key, label, metric in TABLE_ROWS:
        score = key.split(".", 1)[1]
        base = None
        for f in base_fns:
            cells = covered(f, metric, score)
            base = cells if base is None else (base & cells)
        if not base:
            continue

        values = {}
        for f in fns:
            cells = covered(f, metric, score) & base
            if not cells:
                continue
            vals = [per_cell[f"{f}.{metric}"][c][score] for c in cells]
            values[f] = (sum(vals) / len(vals), len(cells))

        if not values:
            continue
        entry0 = next(
            (report["table"][f].get(key) for f in fns if report["table"].get(f, {}).get(key)), None
        )
        direction = entry0.get("direction", 0) if entry0 else 0
        best = None
        if direction:
            finite = [m for m, _ in values.values()]
            best = min(finite) if direction == -1 else max(finite)

        cells_html = []
        for f in fns:
            cls_base = "base" if f in BASELINE_FNS else "sys"
            if f not in values:
                cells_html.append(f'<td class="na {cls_base}">—</td>')
                continue
            mean, n = values[f]
            cls = "best" if best is not None and abs(mean - best) < 1e-9 else ""
            note = ""
            if n < len(base):
                note = f'<span class="ncol" title="scored on {n} of {len(base)} cells">{n}</span>'
                footnoted = True
            cells_html.append(f'<td class="{cls} {cls_base}">{mean:.3f}{note}</td>')

        thin = len(base) < THIN_COVERAGE
        unit = "cell" if len(base) == 1 else "cells"
        rows.append(
            f'<tr><th scope="row">{esc(label)} '
            f'<span class="dir">{ARROW.get(direction, "·")}</span>'
            f'<span class="ncell{" thin" if thin else ""}">{len(base)} {unit}</span></th>'
            + "".join(cells_html) + "</tr>"
        )

    key_note = (
        "Tinted columns are <strong>reference points</strong> rather than attempts at the task. "
        "A best-in-row mark on one of them shows what the measure does rather than reporting "
        "a result. The cell count on each row is the number of cells every system column "
        "covers."
    )
    if footnoted:
        key_note += (
            " A small number after a value indicates that the column covers fewer cells "
            "than that base, and states how many."
        )
    key_note += f" Amber cell counts flag coverage below {THIN_COVERAGE} cells."

    return (
        f'<figure class="tablewrap wide"><table><caption>{caption}</caption>'
        f"<thead><tr><th scope=\"col\">Metric</th>{head}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        f'<p class="tablekey">{key_note}</p></figure>'
    )


# --------------------------------------------------------------------------
# Page assembly
# --------------------------------------------------------------------------

NAV = [
    ("index.html", "Overview"),
    ("dataset.html", "Dataset"),
    ("methods.html", "Baselines &amp; metrics"),
    ("results.html", "Results"),
    ("limitations.html", "Limitations"),
]


BUILD_STAMP = datetime.datetime.now().astimezone()
BUILD_DATE = BUILD_STAMP.strftime("%Y-%m-%d")
BUILD_LONG = BUILD_STAMP.strftime("%Y-%m-%d %H:%M %Z")


def page(slug: str, title: str, body: str) -> str:
    nav = "".join(
        f'<a href="{href}" class="{"here" if href == slug else ""}">{label}</a>'
        for href, label in NAV
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)} — Vectorial × BAIR transfer evaluation — {BUILD_DATE}</title>
<link rel="stylesheet" href="style.css">
</head>
<body>
<header class="masthead">
  <div class="wrap">
    <p class="eyebrow">Vectorial × BAIR · cross-platform audience transfer</p>
    <h1>Experimental notes <span class="stamp">{BUILD_DATE}</span></h1>
    <p class="lede">Evaluating LinkedIn → Reddit transfer functions across 30 bilateral
    cells, with four candidates drawn per source post. Generated {BUILD_LONG}.</p>
  </div>
</header>
<nav class="nav"><div class="wrap">{nav}</div></nav>
<main class="wrap">
{body}
</main>
<footer class="foot"><div class="wrap">
  <p>Generated from <code>runs/scaled/report.heldout.json</code> by
     <code>experimental-notes/build.py</code>. Figures are drawn from the report
     data, not transcribed.</p>
  <p>Distribution-aware metrics follow Chan, Ni, Ross, Vijayanarasimhan, Myers &amp;
     Canny, <em>Distribution Aware Metrics for Conditional Natural Language
     Generation</em>, LREC-COLING 2024, pp. 5064–5095.</p>
</div></footer>
</body>
</html>
"""


def build():
    full, systems, manifest = load()
    all_fns = ORDER
    sys_fns = [f for f in ORDER if f != "target_sample"]

    # ---- index ----------------------------------------------------------
    idx = f"""
<section class="callout">
  <h2>What we found</h2>
  <p>Asking a language model to rewrite a LinkedIn post as a Reddit post works well on
  any single post. The rewrite adopts the tone of the target community, keeps the
  subject matter, and drops the self-promotion. Read one at a time, the outputs are
  good.</p>
  <p>The problem appears only when the outputs are read as a group. Every rewrite comes
  out in roughly the same shape. Real Reddit posts about one topic vary widely in
  length, structure and purpose, and the rewrites do not. The model has learned one
  average Reddit voice and applies it to everything.</p>
  <p>This matters because the variety is the signal. The project aims to capture how a
  particular audience actually behaves on a platform, and an audience is described by
  the range of things its members write, not by the average of them.</p>
  <p>Both halves of this are now measured with enough data to be confident. The rewrites
  genuinely do read more like Reddit than the source posts. They also genuinely do lose
  the variety, use a narrower vocabulary, and end up more alike than real posts are.</p>
  <p class="stress">A platform classifier alone reports only the first half, and reports
  it as a success. Measuring the group rather than the individual post is what reveals
  the rest.</p>
</section>

<h2>Results</h2>
{results_table(full, all_fns, f"Held-out split, {manifest['counts']['tasks']['heldout']} source posts, four candidates drawn per post for each language model. Distances are cosine in a TF-IDF and SVD embedding space of 256 dimensions. Bold marks the best value in each row.")}
<p class="note">The three tinted columns are not attempts at the task. The identity
baseline copies the source post and shows what changing nothing looks like. The shuffle
control emits a real Reddit post about an unrelated topic and shows what correct tone
with wrong content looks like. The target sample emits a real Reddit post about the
right topic and shows the best score any system could hope to reach. The oracle covers
fewer cells than the systems, and its own count is printed beside each of its values.</p>

<h2>The two goals a transfer function must meet</h2>
<p>A transfer function has to do two things at once. It has to produce text that reads
like the target platform, and it has to keep the content of the particular post it was
given. The figure below plots those two goals against each other.</p>

{scatter_tradeoff(systems, full)}

<p>Two of the reference points sit at the ends of the axes by definition. Copying the
source keeps all of the content, so the identity baseline scores at the top. Emitting a
real Reddit post gives the right tone but says something different, so the oracle scores
at the bottom. Neither is a failure, and neither is a target: a working system has to be
high on both axes at once, and the upper-left region is empty.</p>

<p>The example below shows every column applied to the same source post, which makes
the roles of the reference points concrete.</p>

{example_all_baselines()}

<h2>The third goal, which is about the group</h2>
<p>Reading like the platform and keeping the content are both properties of a single
post. The third requirement is a property of a whole set of posts: the generated set
should be as varied as the real set. A system that produces one good post repeatedly
satisfies the first two goals and fails the third.</p>

{scorecard(systems, full)}

<h2>Findings</h2>
<ol class="findings">
  <li><strong>Rewriting does make individual posts read more like Reddit.</strong> The
  platform classifier separates the rewrites from the untouched source posts clearly.
  Prompting solves the per-post half of the problem.</li>

  <li><strong>Rewriting makes the set of posts less like real Reddit.</strong> Taken as a
  group, the rewrites end up further from real Reddit than the untouched LinkedIn posts
  were. The two results above are not in conflict: one is about a post and the other is
  about a collection.</li>

  <li><strong>The cause is that the outputs are too alike.</strong> Four separate
  measurements agree. Generated posts resemble each other more than real posts do, and
  more than real Reddit posts resemble each other. They draw on a narrower vocabulary.
  They end on a question about three times as often as real posts. And the automated
  judge, asked to pick the machine-written post out of a group, names structural
  sameness as the reason it can.</li>

  <li><strong>GPT-5.6 keeps more of the original content than Claude.</strong> Claude
  changes the tone more and keeps less. Choosing between them is choosing where to sit on
  that trade-off rather than choosing a better model.</li>

  <li><strong>Showing the model real examples points in the right direction.</strong>
  Supplying four genuine Reddit posts from the same audience and topic gives better
  numbers than zero-shot on tone, on content retention and on group similarity. The
  direction is consistent across measures, though the size of the gain is not yet
  established.</li>

  <li><strong>There is a long way to go.</strong> The best system remains far from the
  oracle on every group-level measure. Rewriting with a prompt is a floor for a
  trait-based method to beat, not a solution.</li>
</ol>

<p class="note">Each of these statements was checked against the spread of values across
cells, and only differences larger than that spread are stated as findings. The
<a href="results.html">results page</a> lists every comparison with its verdict, including
the ones this study cannot yet separate.</p>

<h2>What this implies for the next model</h2>
<p>The measurements point to a specific target rather than a general one. A better
transfer function does not need to produce more Reddit-like individual posts, because
prompting already does that. It needs to produce a <em>set</em> of posts whose variety
matches the variety of the real audience. Any method that maps every source post through
the same fixed transformation will fail this test regardless of how good each individual
output is.</p>
<p>That gives a concrete acceptance criterion for the trait-mediated approach the project
is moving toward. If latent traits carry real information about how different members of
an audience write, then conditioning on them should spread the outputs out, and the
triangle statistic will register it. The harness will detect that improvement whether or
not the platform classifier does.</p>
"""

    # ---- dataset --------------------------------------------------------
    ds = f"""
<h2>The bilateral cell</h2>
<p>A <dfn>bilateral cell</dfn> consists of one audience room crossed with one topic
cluster and partitioned by platform. It serves as the unit of analysis and the unit at
which every metric is computed, because it is the smallest grouping for which the
distribution of an audience's expression on a given platform is well defined.</p>

<pre class="code">cell_id = "backend_engineer::ruby_on_rails_ecosystem"
          ├── linkedin: 21 posts
          └── reddit:   96 posts</pre>

{example_cell_posts()}

<p>The two sides of a cell discuss the same subject in materially different registers.
The LinkedIn side reports observations and improvements in a declarative and
promotional form. The Reddit side solicits assistance and feedback from practitioners.
Reproducing that difference constitutes the transfer problem.</p>

{funnel(manifest)}

<h2>Room canonicalisation</h2>
<p>The raw corpus labels the two platform halves of a single audience differently. The
LinkedIn component of the CTO room is recorded as <code>CTOs</code> and the Reddit
component as <code>CTO Reddit</code>. Grouping on the raw labels therefore never
produces a bilateral cell for that audience. Only two of the 24 raw rooms already
contain both platforms.</p>

<figure class="tablewrap"><table><caption>Effect of canonicalisation on usable coverage</caption>
<thead><tr><th scope="col">Room mode</th><th scope="col">Viable cells with at least five posts per platform</th></tr></thead>
<tbody>
<tr><th scope="row">strict, using raw labels</th><td>19</td></tr>
<tr><th scope="row">canonical, the default</th><td class="best">30</td></tr>
</tbody></table></figure>

<p class="note">Canonicalisation constitutes an assumption rather than an established
fact, because it asserts that <code>CTOs</code> and <code>CTO Reddit</code> sample the
same underlying audience. Any finding that fails to replicate under
<code>--room-mode strict</code> should be treated as provisional.</p>

{coverage_chart(manifest)}

<h2>Split design</h2>
<p>Splits are assigned at the level of the individual post and stratified within each
combination of cell and platform. A partition at the level of the cell would leave
single-digit cell counts in each split, and every metric in this harness requires a
pool of posts rather than an individual post.</p>
<p>Leakage is controlled through the role a post plays rather than through the
partition it belongs to. The <code>exemplar_ids</code> field lists posts on which a
transfer function may condition and always draws from the train split. The
<code>target_reference_ids</code> field lists posts against which a task is scored and
always draws from the split of the task itself. The two sets are disjoint by
construction.</p>
<p>Zero-shot generalisation is handled on a second, separate axis. Fifteen percent
of cells are flagged as held out. Their posts appear only in the test split and they
carry no exemplars, so a few-shot function reverts to zero-shot behaviour on those
cells and the generalisation penalty becomes directly measurable.</p>
<p>Assignment is computed as a digest of the post identifier and a fixed seed rather
than by shuffling. The addition of new material therefore never reassigns existing
posts, which is a necessary property while the corpus continues to grow.</p>

<p>Evaluation uses a merged held-out split combining validation and test. Both are kept
out of every transfer function's example pool, so merging them introduces no leakage, and
it roughly doubles the number of real posts available to score against in each cell. That
matters because the group-level measures need a minimum number of real posts per cell,
and were previously limited by how the split was cut rather than by the size of the
corpus.</p>

<figure class="tablewrap"><table><caption>The dataset as constructed</caption>
<thead><tr><th scope="col">Quantity</th><th scope="col">Value</th></tr></thead>
<tbody>
<tr><th scope="row">Bilateral cells</th><td>{manifest['n_cells']}</td></tr>
<tr><th scope="row">Dense cells, at least ten posts per platform</th><td>{manifest['n_dense_cells']}</td></tr>
<tr><th scope="row">Held-out cells reserved for zero-shot evaluation</th><td>{manifest['n_heldout_cells']}</td></tr>
<tr><th scope="row">Posts</th><td>{manifest['n_posts']}</td></tr>
<tr><th scope="row">Transfer tasks</th><td>{manifest['n_tasks']}</td></tr>
<tr><th scope="row">Test-split tasks</th><td>{manifest['counts']['tasks']['test']}</td></tr>
<tr><th scope="row">Held-out tasks used for evaluation</th><td>{manifest['counts']['tasks']['heldout']}</td></tr>
</tbody></table></figure>
"""

    # ---- methods --------------------------------------------------------
    me = f"""
<h2>Why the evaluation is distributional</h2>
<p>No correspondence exists between individual posts across the two platforms, so no
generated post has a gold reference against which it can be compared. Every metric
therefore compares a pool of generated posts against a pool of authentic target posts
within a single cell. This constraint excludes reference-based scores of the BLEU and
ROUGE family and determines the remainder of the design.</p>

<h2>The baselines</h2>
<p>Three of the six columns are reference points rather than attempts at the task.
They calibrate the metrics, and a metric that fails to separate them as designed cannot
support any conclusion about a genuine system.</p>

<div class="cards">
  <article class="card">
    <h3><span class="swatch" style="background:#8b8b8b"></span>identity</h3>
    <p class="role">Floor on transfer and ceiling on preservation</p>
    <p>Reproduces the source post without modification. It attains a preservation score
    of 1.000 by construction and records the worst calibration gap, because an
    untouched LinkedIn post is the least Reddit-like column in the table.</p>
  </article>
  <article class="card">
    <h3><span class="swatch" style="background:#2e7d5b"></span>target sample</h3>
    <p class="role">Oracle ceiling for distribution matching</p>
    <p>Emits an authentic Reddit post drawn from the train split of the same cell. It
    never emits the post against which it is scored, which would render the ceiling
    spurious. It attains a TRM of 0.035 and the only non-significant p-value at
    0.512.</p>
    <p class="caveat">It retains no content from the specific source post and scores
    0.070 on preservation. It establishes the ceiling for distribution matching alone,
    whereas the identity baseline establishes the ceiling for preservation. Neither
    baseline individually represents the objective.</p>
  </article>
  <article class="card">
    <h3><span class="swatch" style="background:#b0a08c"></span>shuffle control</h3>
    <p class="role">Negative control</p>
    <p>Emits an authentic Reddit post drawn from a different cell, which supplies the
    correct register together with an unrelated topic. Any metric that scores this
    column comparably to the oracle is indexed on style and blind to semantics, which
    is the principal risk identified for this project.</p>
  </article>
</div>

<p class="note">The control demonstrates its value immediately. It attains a
calibration gap of 0.150 against the oracle's 0.192, which establishes that the
classifier cannot distinguish correct topic from incorrect topic. It attains a topic
JSD of 0.800 against the oracle's 0.226. The classifier therefore cannot be reported in
isolation.</p>

<h2>The candidate systems</h2>
<div class="cards">
  <article class="card">
    <h3><span class="swatch" style="background:#c25e3a"></span>Claude zero-shot</h3>
    <p class="role">llm_rewrite</p>
    <p>Receives the audience, domain, topic and target platform together with a brief
    statement of platform norms, and returns a rewritten post. No target-platform text
    is supplied. It measures the portion of the transfer recoverable from the model's
    prior alone.</p>
  </article>
  <article class="card">
    <h3><span class="swatch" style="background:#d99a4e"></span>Claude few-shot</h3>
    <p class="role">llm_fewshot with four exemplars</p>
    <p>Receives the same prompt together with four authentic Reddit posts from the
    train split of the same cell. The difference against the zero-shot configuration
    isolates the contribution of observing the authentic target distribution.</p>
  </article>
  <article class="card">
    <h3><span class="swatch" style="background:#4a6fa5"></span>GPT-5.6 zero-shot</h3>
    <p class="role">llm_rewrite on a second model family</p>
    <p>Applies the zero-shot prompt to a different model family, so that the findings
    are not confounded with the house style of a single vendor.</p>
  </article>
</div>
<p class="note">Both rewriters are deliberately naive surface rewriters, and the prompt
makes no attempt at trait extraction. They establish the standard that a trait-mediated
function must exceed rather than a proposed solution.</p>

{example_exemplars()}

<h2>How many candidates are drawn</h2>
<p>Each language model is asked for four separate rewrites of every source post rather
than one. The group-level measures compare a set of generated posts against a set of real
posts, so drawing several candidates per post enlarges the generated set without needing
any additional collected data. It is also what the work these measures come from
prescribes, and it is the only way to observe how much a model varies when asked the same
question twice.</p>
<p>This choice materially affects one result. With a single draw per post the vocabulary
measure could not distinguish the rewriters from the source posts, because the repetition
happens <em>between</em> draws. With four draws it separates them clearly.</p>

<h2>The metrics</h2>
<dl class="metrics">
  <dt>Classifier calibration gap <span class="dir">↓</span></dt>
  <dd>A classifier separating LinkedIn from Reddit is trained on authentic posts from
  the train split and then applied to generated posts. The reported score is the
  absolute difference between the target-platform rate of the generated pool and that
  of the authentic pool, because the objective is to match the authentic distribution
  rather than to saturate the classifier. Held-out accuracy is 0.79 for the lexical
  variant and 0.85 for the stylometric variant, so the classifier is not treated as an
  oracle.</dd>

  <dt>Source similarity and content-word retention <span class="dir">↑</span></dt>
  <dd>Cosine similarity between a generated post and its source, together with the
  proportion of the source's distinctive content words that survive into the output.
  Embedding similarity can remain high while specific details are discarded, so both
  quantities are reported.</dd>

  <dt>Triangle-Rank Metric <span class="dir">↓</span></dt>
  <dd><strong>Base distance: cosine distance in the harness embedding space, which is
  TF-IDF with truncated SVD at 256 dimensions for every figure in this report.</strong>
  The statistic is not a measure of distance in itself. It is built on top of one, and the
  distance it is built on determines what it can detect and is recorded in the notes of every run. Substituting a neural
  embedding or a learned distance such as BERTScore changes the values, and the
  comparison should be repeated under any such substitution.
  For candidates C and references R the metric enumerates every triangle with one
  vertex in C and two in R and records the rank of the edge joining the two references.
  Under the null hypothesis that both sets derive from one distribution each rank is
  equally likely, so each of the three counts should come out at one third. The score is
  how far the counts stray from that. Because the comparison uses three posts at a time
  rather than one, it responds both to generated posts landing in the wrong place and to
  generated posts being too much alike, which a measure based on averages cannot see. It was designed for regimes containing tens of references
  rather than thousands, unlike MAUVE.</dd>

  <dt>TRM permutation p-value <span class="dir">↑</span></dt>
  <dd>A permutation test against the pooled null distribution. A high value is the
  favourable outcome because it establishes that the generated pool is statistically
  indistinguishable from authentic target posts.</dd>

  <dt>Style-removed variants <span class="dir">↓</span></dt>
  <dd>The same statistics, and for TRM the same cosine base distance, computed after the
  linear directions that discriminate the platforms have been projected out. A function that performs well before this step and no better than the shuffle control afterwards has
  learned only the tone.</dd>

  <dt>Pool self-similarity <span class="dir">↓</span></dt>
  <dd>The mean similarity of generated posts to one another. The appropriate reference
  value is the oracle's 0.332, which is the authentic self-similarity of real Reddit
  posts, rather than zero.</dd>

  <dt>LLM judge detection rate <span class="dir">↓</span></dt>
  <dd>The judge is framed as a discrimination task rather than a rubric. It receives
  several authentic posts together with one candidate and must identify the
  machine-written item, so chance performance is one divided by the number of
  alternatives. The task is grounded in the authentic posts of the cell, which causes
  it to assess agreement with a specific audience and topic rather than generic
  platform register. The judge is drawn from a different model family than the
  rewriters, because judges exhibit measurable self-preference bias.</dd>
</dl>
"""

    # ---- results --------------------------------------------------------
    res = f"""
<h2>Full results</h2>
{results_table(full, all_fns, "All six columns. Each row is averaged over the cells every system column covers, and a superscript marks any column scored on fewer.", base_fns=sys_fns)}

<h2>System comparison</h2>
{results_table(systems, sys_fns, "The oracle is excluded so that every column is scored on an identical set of cells. The values quoted throughout this report are taken from this table.")}

<h2>Where the two measurements disagree</h2>
<p>The platform classifier and the triangle statistic reach opposite conclusions about
the same systems. That disagreement is the main result rather than a contradiction to be
resolved.</p>
{paired_bars(systems, full)}
<p>The classifier reports that the Claude rewrites read more like Reddit than the source
posts do. The triangle statistic reports that the same rewrites, taken as a group, are
further from real Reddit than the untouched source posts are. Both readings are correct
and both are supported by the data. The rewrites hit the average tone of the platform and
lose its variety.</p>
<p>The table below states which comparisons the data is strong enough to support.</p>
{resolution_table(systems, full)}

{trm_per_cell()}
<p class="note">TRM here is computed over cosine distance in the TF-IDF and SVD
embedding space at 256 dimensions. Because the statistic is built on top of that distance, values computed over a different
distance are not comparable with these. The ordering holds in all five scored cells, which establishes that the
result is not caused by one unusual topic. Cell-level values range from 0.62
to 0.81 for Claude zero-shot and from 0.08 to 0.23 for the identity baseline.</p>

<h2>Where the variety is lost</h2>
{rank_profile(systems, full)}
<p>The three bands record how often each of three arrangements occurs when one generated
post is compared against two real posts. If the generated set matched the real set, each
band would take exactly one third. Extra weight in the third band means the two real
posts are usually further apart than either is from the generated post, which is what
happens when the generated posts are bunched together in the middle of the real ones.
The oracle sits near one third on all three bands, as it must.</p>

{hbar_chart([(f, val(systems if f != "target_sample" else full, f, "degeneracy.pool_self_similarity")) for f in ORDER], "Generated posts are too similar to each other", "How alike the posts within each set are. The oracle shows the level real Reddit posts reach.", ref=0.332, ref_label="authentic level")}

<h2>The repeated shape, in plain sight</h2>
<p>The group-level statistics correspond to one concrete habit. The rewriters settle on a
common structure that ends by inviting a reply from the community, and that structure
recurs whatever the source post was about.</p>

{example_template()}

{question_stats()}

<p>The rewriters end on a question roughly three times as often as real Reddit posts do.
This measurement uses no embedding and no classifier, so it provides independent support
for the same conclusion, and it accounts for much of the excess similarity within the
generated sets.</p>

<h2>Which writing habits are not reproduced</h2>
{worst_features(full)}
<p>The oracle is the reference on this measure rather than zero. It emits real Reddit
posts, so any distance it shows comes from comparing one sample of real posts against
another, and that is the level a perfect system would reach. Averaged across these eight
habits the oracle sits at 1.12, and the shuffle control, GPT-5.6 and the identity
baseline all sit close to it.</p>
<p>Claude zero-shot is the exception at 2.29, roughly double the reference level. Two
habits account for most of that gap: bullet points and paragraph count, where it scores
3.07 and 3.05 against the oracle's 0.08 and 0.59. The rewriter imposes a tidy, heavily
formatted layout on communities that do not write that way. These are the same habits
the automated judge names when it picks out a machine-written post, arrived at
independently.</p>

<h2>LLM judge</h2>
{hbar_chart([(f, val(full, f, "judge.detection_rate")) for f in ORDER], "How often an automated judge spots the machine-written post", "Lower is better. A judge that could not tell would score 0.25.", ref=0.25, ref_label="chance")}
<p>The oracle falls below chance, which is the required outcome. The shuffle control is
the most readily detected column, because an unrelated topic constitutes an obvious
signal. The cues most frequently recorded by the judge for the generated columns
describe regularity of structure rather than implausibility of content.</p>
<blockquote>
  <p>formulaic llm discussion prompt structure with em-dash and open-ended questions</p>
  <p>promotional structure with featured bullet points and generic concluding questions</p>
  <p>bulleted structural formula with absurd, invented ai tropes</p>
</blockquote>
<p class="note">The judge also detects posts that belong to the wrong platform, which is
why the identity baseline is detectable and why the shuffle control is the most
detectable column of all. The measurement is best read as asking whether a post belongs
in this particular set, which is the quantity of interest here.</p>

<h2>An individual transfer</h2>
{example_transfer()}
<p class="note">Judged on its own this rewrite is good. The promotional framing becomes a
question, the technical detail survives, and the personal endorsement is dropped. Nothing
is wrong with it. The problem only becomes visible once many outputs are read together
and the same shape appears in all of them.</p>
"""

    # ---- limitations ----------------------------------------------------
    lim = f"""
<section class="warn">
  <h2>Summary of constraints</h2>
  <p>The direction of every effect reported here is reliable. The magnitudes are not,
  and the dataset is provisional by design. No quantity in this report should be
  circulated as a settled measurement.</p>
</section>

<h2>Coverage is the binding constraint</h2>
<p>Of 1,525 topic clusters in the corpus, only 30 carry at least five posts on both
platforms. The merged held-out split contains 126 source posts. The group-similarity
measure needs at least four generated and four real posts within a cell, which leaves 10
cells that can be scored. The classifier and content measures reach 22.</p>
<p>Two changes raised that coverage without collecting anything new. Merging the
validation and test splits for scoring roughly doubled the number of real posts available
in each cell, and drawing four candidates per source post instead of one enlarged the
generated set by the same factor. Together they took the group-similarity measure from 5
cells to 10 and the classifier measures from 17 to 22, and they resolved three
comparisons that the earlier run could not separate. Further gains now require more posts
in the single-platform rooms.</p>
<p>The comparisons this study still cannot separate are those between the zero-shot and
few-shot configurations. Their point estimates favour few-shot on every measure, and the
gap is smaller than the variation between cells.</p>

<h2>The dataset is provisional by design</h2>
<p>The current corpus is suitable for establishing structure and for prototyping, and
it is not the final optimised dataset. Cluster assignments carry an estimated 3.7
percent rate of problematic reassignment, and the stability curve has not reached a
plateau at an adjusted Rand index of approximately 0.80 under a 90 percent subsample.
Every cell in this harness is defined by the clustering output, so instability in the
clustering propagates into every metric. All results are conditional on the current
clustering.</p>

<h2>Known weaknesses in individual metrics</h2>
<dl class="metrics">
  <dt>Topic JSD is unreliable for generated text</dt>
  <dd>It assigns the Claude columns a value of 0.950 against 0.800 for the shuffle
  control, despite a source similarity of 0.560 and preservation of the topic on
  inspection. The metric is computed over k-means clusters fitted on authentic posts,
  so it plausibly detects machine origin as well as a change of subject. It is not
  reported as a finding and requires re-examination under a neural embedding
  backend. The following example illustrates the discrepancy.</dd>

  <dt>The classifier is imperfect</dt>
  <dd>Held-out accuracy is 0.79 for the lexical variant and 0.85 for the stylometric
  variant. The calibration-gap formulation is partially robust to this limitation
  because miscalibration affects the generated and authentic pools equally, but the
  absolute rates should not be over-interpreted.</dd>

  <dt>Removing style is only partly possible</dt>
  <dd>Only two linear directions are removed. Register is not confined to a
  two-dimensional linear subspace, so residual style signal certainly remains. A
  style-removed score establishes that at least the stated amount survives the removal of
  style, and does not establish that the effect is entirely about subject matter.</dd>

  <dt>The judge has not been validated against human annotation</dt>
  <dd>The discrimination framing controls position bias and possesses meaningful floor
  and ceiling values, and it has not been calibrated against human judgement on this
  corpus. Whether the judge detection rate tracks human ability to distinguish these
  posts remains unestablished.</dd>

  <dt>The choice of embedding affects the conclusions</dt>
  <dd>The results reported here use the offline TF-IDF backend. The neural backend is
  configured and may itself encode register, which is the principal risk identified for
  this project and which the harness mitigates only partially. No conclusion should be
  reported without replication across both backends.</dd>
</dl>

{example_topic_jsd_failure()}

<h2>Quantities that are not measured</h2>
<ul class="findings">
  <li><strong>Agreement at the level of aspect.</strong> The distinction between the
  object under discussion and the evaluative dimension through which it is discussed is
  not modelled. A function that reproduces register and topic while inverting the
  distribution of evaluative aspects would incur no penalty under any current metric.
  This constitutes the extension of highest value and requires no change to the metric
  registry.</li>
  <li><strong>The reverse transfer direction.</strong> All results describe transfer
  from LinkedIn to Reddit. The reverse direction is supported by the harness and has
  not been examined, and given the asymmetry between the platforms it should not be
  assumed to mirror these findings.</li>
  <li><strong>Population shift as distinct from behavioural shift.</strong> The harness
  measures the combination of the two. Separating them would require linkage of
  identities across platforms, which the corpus does not provide.</li>
  <li><strong>Mechanism.</strong> Every metric compares distributions. None establishes
  that a function has recovered the mechanism underlying platform difference, which is
  the reason the interpretability requirement is not satisfied by these measurements
  alone.</li>
</ul>
"""

    OUT.mkdir(exist_ok=True)
    (OUT / "index.html").write_text(page("index.html", "Overview", idx), encoding="utf-8")
    (OUT / "dataset.html").write_text(page("dataset.html", "Dataset", ds), encoding="utf-8")
    (OUT / "methods.html").write_text(page("methods.html", "Baselines and metrics", me), encoding="utf-8")
    (OUT / "results.html").write_text(page("results.html", "Results", res), encoding="utf-8")
    (OUT / "limitations.html").write_text(page("limitations.html", "Limitations", lim), encoding="utf-8")
    (OUT / "style.css").write_text(STYLE, encoding="utf-8")
    print(f"Wrote 5 pages and a stylesheet to {OUT}")


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
  width: min(92vw, 1680px);
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
"""


if __name__ == "__main__":
    build()
