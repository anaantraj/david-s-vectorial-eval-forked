"""Build the second study's report: adaptation methods against prompting.

Why this is a separate builder
------------------------------
`build.py` describes the first study, whose shape it encodes throughout: six
columns, one comparison per figure, and a replication across six embedding
spaces that occupies three of its pages. The second study has a different shape.
It scores fifteen system columns arranged as four method families crossed with
three training variants, and it scores them in one embedding space. Rendering it
through the first study's figures produced a bar chart with fifteen unlabelled
rows and a results table too dense to read, which is a form problem rather than a
styling one.

The SVG primitives, the page shell and the stylesheet are imported from
`build.py` so the two studies remain one site with one visual language. What is
redefined here is the choice of figure, and the prose.

Figure choices
--------------
Every figure carries an interval, so the mark is a dot with a whisker rather than
a bar: a bar's length invites reading the value as a magnitude from zero, which
is wrong for a statistic whose ideal is a middle value, and it leaves nowhere to
draw the interval. Rows are grouped by method family, because the question the
study asks is whether a family beats prompting, not whether one run beats
another. Reference columns are drawn as vertical rules across every panel rather
than as rows, so a reader compares each system against the floor, the control and
the oracle without tracking three extra rows.

Colour identifies the family and nothing else. It is assigned in a fixed order
and is never reused for a variant, because the variant is already carried by
position within the group and by the row label.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import build as b1  # SVG primitives, page shell, esc/text helpers

ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------
# Columns
# --------------------------------------------------------------------------

REFERENCES = ["identity", "shuffle_control", "target_sample"]

#: Method families in reading order. The key is the family, the value is the
#: ordered (column, variant label) pairs within it.
FAMILIES: list[tuple[str, str, list[tuple[str, str]]]] = [
    (
        "prompting",
        "#2a78d6",
        [
            ("llm_rewrite_gpt", "GPT-5.6 zero-shot"),
            ("llm_rewrite_claude", "Claude zero-shot"),
            ("llm_fewshot_claude", "Claude few-shot"),
            ("aspect_prompt_claude", "Claude aspect-aware"),
        ],
    ),
    (
        "steering",
        "#eb6834",
        [
            ("steering_steer_target_lm", "topic only"),
            ("steering_steer_target_lm_paired", "retrieved source"),
            ("steering_steer_target_lm_aspect", "aspects"),
        ],
    ),
    (
        "soft prompt",
        "#1baf7a",
        [
            ("soft_prompt_sp_target_lm", "topic only"),
            ("soft_prompt_sp_target_lm_paired", "retrieved source"),
            ("soft_prompt_sp_target_lm_aspect", "aspects"),
        ],
    ),
    (
        "LoRA",
        "#4a3aa7",
        [
            ("lora_lora_target_lm", "topic only"),
            ("lora_lora_target_lm_paired", "retrieved source"),
            ("lora_lora_target_lm_aspect", "aspects"),
        ],
    ),
]

REF_STYLE = {
    "identity": ("#8b8b8b", "Identity (copy floor)"),
    "shuffle_control": ("#b0a08c", "Shuffle-Control (wrong topic)"),
    "target_sample": ("#2e7d5b", "Oracle (authentic posts)"),
}

#: Identity and the wrong-topic control must be read from the systems-only
#: report, because that report evaluates them on the same common cells as the
#: methods. Only the oracle comes from the full report. Reading every reference
#: from the full report once collapsed all three to the oracle's one-cell base
#: and made the copy floor appear next to the oracle in the primary figure.
REF_REPORT_FULL: dict = {}
REF_REPORT_SYSTEMS: dict = {}


def ref_entry(key: str, fn: str) -> dict | None:
    rep = REF_REPORT_FULL if fn == "target_sample" else REF_REPORT_SYSTEMS
    e = (rep.get("table", {}) or {}).get(fn, {}).get(key)
    return e if isinstance(e, dict) and e.get("mean") is not None else None


def ref_mean(key: str, fn: str):
    e = ref_entry(key, fn)
    return None if e is None else e["mean"]


def ref_label(key: str, fn: str) -> str:
    """Compact plot label with the reference's actual comparison base.

    The tables retain the longer formal names in ``REF_STYLE``.  SVG labels use
    the shorter role names below so the annotation remains legible at report and
    mobile widths.
    """
    label = {
        "identity": "Copy floor",
        "shuffle_control": "Wrong-topic control",
        "target_sample": "Authentic oracle",
    }[fn]
    e = ref_entry(key, fn)
    return label if not e or e.get("n_cells") is None else f"{label} · n={e['n_cells']}"


def ref_lines(key, px, y_top, y_bot, width_right_edge, out, *, label_y=None):
    """Draw all three reference columns as named vertical rules.

    Labels are spread vertically when two references fall close together, because
    two overlapping labels are worse than none: the reader cannot tell which rule
    is which and may attribute the wrong role to a value.
    """
    present = []
    for fn in REFERENCES:
        m = ref_mean(key, fn)
        if m is not None:
            present.append((fn, m))
    present.sort(key=lambda t: t[1])
    placed: list[tuple[float, float, float]] = []
    for fn, m in present:
        colour, _label = REF_STYLE[fn]
        label = ref_label(key, fn)
        x = px(m)
        dash = "" if fn == "target_sample" else ' stroke-dasharray="3 3"'
        out.append(
            f'<line x1="{x:.1f}" y1="{y_top - 6:.1f}" x2="{x:.1f}" y2="{y_bot:.1f}" '
            f'stroke="{colour}" stroke-width="2"{dash} opacity="0.9"/>'
        )
        ly = (label_y if label_y is not None else y_top - 12)
        anchor = "middle"
        # ref-lbl is 11 px monospace in the shared figure stylesheet; 6.7 px is
        # its measured average glyph advance in the rendered report.
        half = len(label) * 6.7 / 2
        if x - half < 2:
            anchor, x_lab, x_lo, x_hi = "start", 2, 2, 2 + 2 * half
        elif x + half > width_right_edge:
            anchor, x_lab = "end", width_right_edge
            x_lo, x_hi = width_right_edge - 2 * half, width_right_edge
        else:
            x_lab, x_lo, x_hi = x, x - half, x + half
        # Use another annotation row only when the glyph spans would actually
        # intersect.  The former y-only test stacked even distant labels and
        # could push the third one into a multi-line subtitle.
        while any(abs(ly - py) < 12 and x_lo < phi + 5 and x_hi > plo - 5
                  for py, plo, phi in placed):
            ly -= 13
        placed.append((ly, x_lo, x_hi))
        out.append(b1.text(x_lab, ly, label, "ref-lbl", anchor))
    return present

TRAINED = [c for _, _, cols in FAMILIES[1:] for c, _ in cols]
PROMPTED = [c for c, _ in FAMILIES[0][2]]


def _run_dir(env: str, default: Path) -> Path:
    raw = os.environ.get(env)
    if not raw:
        return default
    p = Path(raw)
    return p if p.is_absolute() else (ROOT / p)


RUN = _run_dir("VECTORIAL_REPORT_RUN", ROOT / "runs" / "methods")
RUN_SYS = _run_dir("VECTORIAL_REPORT_RUN_SYS", ROOT / "runs" / "methods_sys")
OUT = _run_dir("VECTORIAL_REPORT_OUT", ROOT / "reports" / "study2")
SPLIT = os.environ.get("VECTORIAL_REPORT_SPLIT", "test")
REPORT_TAG = os.environ.get("VECTORIAL_REPORT_TAG")
DATA = _run_dir("VECTORIAL_REPORT_DATA", ROOT / "data")
STUDY_LABEL = os.environ.get("VECTORIAL_REPORT_STUDY", "Study 2")
ASPECT_REPORT = Path(os.environ.get(
    "VECTORIAL_REPORT_ASPECT",
    ROOT / "runs" / "study3" / "report.heldout.aspect-systems.json",
))
ASPECT_REPORT_FULL = Path(os.environ.get(
    "VECTORIAL_REPORT_ASPECT_FULL",
    ROOT / "runs" / "study3" / "report.heldout.aspect.json",
))
REVERSE_RUN = _run_dir("VECTORIAL_REPORT_REVERSE_RUN", ROOT / "runs" / "study3_reverse")
REVERSE_RUN_SYS = _run_dir(
    "VECTORIAL_REPORT_REVERSE_RUN_SYS", ROOT / "runs" / "study3_reverse_sys"
)
EMBEDDING_RUN = _run_dir(
    "VECTORIAL_REPORT_EMBEDDING_RUN", ROOT / "runs" / "study3_sys"
)

STUDY_NAV = [
    ("index.html", "Overview"),
    ("results.html", "Results"),
    ("methods.html", "Methods"),
    ("dataset.html", "Dataset"),
    ("limitations.html", "Limitations"),
]
_reverse_name = (
    f"report.{SPLIT}.{REPORT_TAG}.json" if REPORT_TAG else f"report.{SPLIT}.json"
)
if (STUDY_LABEL == "Study 3" and (REVERSE_RUN / _reverse_name).exists()
        and (REVERSE_RUN_SYS / _reverse_name).exists()):
    STUDY_NAV.insert(2, ("reverse.html", "Reverse direction"))


def _report_path(run: Path) -> Path:
    if REPORT_TAG:
        tagged = run / f"report.{SPLIT}.{REPORT_TAG}.json"
        if not tagged.exists():
            raise RuntimeError(f"{run}: requested report does not exist: {tagged.name}")
        return tagged
    plain = run / f"report.{SPLIT}.json"
    if plain.exists():
        return plain
    slugged = sorted(run.glob(f"report.{SPLIT}.*.json"))
    if len(slugged) != 1:
        raise RuntimeError(f"{run}: expected one report for {SPLIT}, found {len(slugged)}")
    return slugged[0]


def _report(run: Path) -> dict:
    return json.loads(_report_path(run).read_text())


def study_page(slug: str, title: str, body: str, *, corpus_cells: int,
               primary_cells: int, n_methods: int, space_name: str) -> str:
    """Study-specific shell with accurate scope and provenance."""
    nav = "".join(
        f'<a href="{href}" class="{"here" if href == slug else ""}">{label}</a>'
        for href, label in STUDY_NAV
    )
    page_run = REVERSE_RUN if slug == "reverse.html" else RUN
    page_run_sys = REVERSE_RUN_SYS if slug == "reverse.html" else RUN_SYS
    full_path = _report_path(page_run).relative_to(ROOT)
    systems_path = _report_path(page_run_sys).relative_to(ROOT)
    direction = "Reddit to LinkedIn" if slug == "reverse.html" else "LinkedIn to Reddit"
    # MathJax is loaded only on pages that contain TeX. Pinning the major and
    # minor release keeps the report reproducible while using the current v4
    # component API documented by MathJax.
    mathjax = ""
    if r"\(" in body or r"\[" in body:
        mathjax = (
            '<script defer '
            'src="https://cdn.jsdelivr.net/npm/mathjax@4.0.0/tex-mml-svg.js">'
            '</script>'
        )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{b1.esc(title)} — {STUDY_LABEL} — Vectorial × BAIR — {b1.BUILD_DATE}</title>
<link rel="stylesheet" href="style.css">
{mathjax}
</head>
<body>
<header class="masthead study-head">
  <div class="wrap">
    <p class="eyebrow">Vectorial × BAIR · {STUDY_LABEL} · audience transfer</p>
    <h1>Adaptation versus prompting <span class="stamp">{b1.BUILD_DATE}</span></h1>
    <p class="lede">{direction} across {corpus_cells} bilateral corpus cells. The report
    compares {n_methods} method configurations; the primary distributional measure covers
    {primary_cells} shared test cells in {b1.esc(space_name)}.</p>
  </div>
</header>
<nav class="nav" aria-label="Report sections"><div class="wrap">{nav}</div></nav>
<main class="wrap">
{body}
</main>
<footer class="foot"><div class="wrap">
  <p class="footer-title">{STUDY_LABEL} · generated {b1.BUILD_LONG}</p>
  <p>Source reports: <code>{b1.esc(str(full_path))}</code> and
     <code>{b1.esc(str(systems_path))}</code>. Built by
     <code>experimental-notes/build_study2.py</code>. Every figure and quoted value is computed
     from those reports.</p>
  <p>Distribution-aware metrics follow Chan, Ni, Ross, Vijayanarasimhan, Myers &amp;
     Canny, <em>Distribution Aware Metrics for Conditional Natural Language Generation</em>,
     LREC-COLING 2024, pp. 5064–5095.</p>
</div></footer>
</body>
</html>
"""


def entry(rep: dict, fn: str, key: str) -> dict | None:
    e = rep["table"].get(fn, {}).get(key)
    return e if isinstance(e, dict) and e.get("mean") is not None else None


def mean(rep: dict, fn: str, key: str):
    e = entry(rep, fn, key)
    return None if e is None else e["mean"]


# --------------------------------------------------------------------------
# Metric registry
# --------------------------------------------------------------------------
#
# Every quantity is referred to by its formal name throughout the report. The
# code identifier appears only in the master table, as a traceability column, so
# that a figure can be tied back to the row of the report JSON that produced it
# without the prose reading as a list of variable names.
#
# `space` marks a quantity computed in the embedding space, whose value is not
# comparable with the same quantity computed in another space. Those names carry
# the space explicitly, because a Fréchet distance without its space is not a
# number anyone can act on.

SPACE_SHORT = "Gemma-512"

METRICS: dict[str, dict] = {
    # --- distribution match ---
    "trm.trm": dict(
        name="Triangle-Rank Metric (symmetrised)", short="Triangle-Rank Metric",
        dirn=-1, ideal=0.0, space=True,
        defn="Chan et al. (LREC-COLING 2024). Over every triangle with one vertex in the "
             "generated pool and two in the authentic pool, the rank taken by the "
             "authentic-authentic edge is recorded; under the null that both pools are drawn "
             "from one distribution each rank is equiprobable. The statistic is the summed "
             "squared deviation from that uniform profile, symmetrised over both directions. "
             "Zero denotes indistinguishability."),
    "trm.rank_i2": dict(
        name="Triangle in-edge-longest rate", short="In-edge-longest rate",
        dirn=0, ideal=1/3, space=True,
        defn="Proportion of triangles in which the authentic-authentic edge is the longest of "
             "the three. One third is the value under a perfect match. Values above it are "
             "consistent with generated points concentrated inside the authentic cloud and are "
             "the under-dispersion signature. Values below it are not interpretable alone: the "
             "in-edge-shortest rate distinguishes outward displacement from over-dispersion."),
    "trm.rank_i0": dict(
        name="Triangle in-edge-shortest rate", short="In-edge-shortest rate",
        dirn=0, ideal=1/3, space=True,
        defn="Proportion of triangles in which the authentic-authentic edge is shortest. "
             "Elevated values indicate the generated pool lies outside the authentic cloud, "
             "which is an error of location rather than of spread."),
    "trm.rank_i1": dict(
        name="Triangle in-edge-intermediate rate", short="In-edge-intermediate rate",
        dirn=0, ideal=1/3, space=True, defn="The remaining rank; the three sum to one."),
    "trm.frechet": dict(
        name=f"{SPACE_SHORT} embedding Fréchet distance", short="Fréchet distance",
        dirn=-1, ideal=0.0, space=True,
        defn="Fréchet distance between Gaussians fitted to the generated and authentic pools in "
             "the embedding space. The kernel-free companion to the Triangle-Rank Metric; "
             "sensitive to both mean and covariance, and to neither robustly at these pool sizes."),
    "trm.trm_pvalue": dict(
        name="Triangle-Rank permutation p-value", short="Permutation p-value",
        dirn=1, ideal=None, space=True,
        defn="Proportion of label permutations whose statistic is at least the observed one, "
             "combined across cells by the harmonic mean p-value of Wilson (2019), which remains "
             "valid under arbitrary dependence. A large value is the favourable outcome."),
    "trm.trm_style_residual": dict(
        name="Triangle-Rank Metric, style directions removed", short="TRM, style removed",
        dirn=-1, ideal=0.0, space=True,
        defn="The same statistic after the platform-classifier boundary and the between-platform "
             "difference of means are projected out. Read as a lower bound on what survives the "
             "removal of register, never as a purely semantic quantity."),
    "trm.q_candidate_ref": dict(name="Directed Q, generated against authentic", short="Q (gen→auth)",
        dirn=-1, ideal=0.0, space=True, defn="One direction of the symmetrised statistic."),
    "trm.q_ref_candidate": dict(name="Directed Q, authentic against generated", short="Q (auth→gen)",
        dirn=-1, ideal=0.0, space=True, defn="The other direction."),
    "distributional.centroid_distance": dict(
        name=f"{SPACE_SHORT} centroid cosine distance", short="Centroid distance",
        dirn=-1, ideal=0.0, space=True,
        defn="Cosine distance between the mean embedding of the generated pool and that of the "
             "authentic pool. Sensitive to location only; a pool with the right centre and no "
             "spread scores perfectly here, which is why it is reported beside the Triangle-Rank "
             "Metric rather than instead of it."),
    "distributional.centroid_distance_style_residual": dict(
        name=f"{SPACE_SHORT} centroid distance, style removed", short="Centroid distance, style removed",
        dirn=-1, ideal=0.0, space=True, defn="As above, after the platform directions are projected out."),
    "distributional.mmd2": dict(
        name="Unbiased squared MMD (RBF, median heuristic)", short="Unbiased MMD²",
        dirn=-1, ideal=0.0, space=True,
        defn="The U-statistic estimator of squared maximum mean discrepancy, with diagonal terms "
             "removed to eliminate the O(1/n) bias. Because the diagonal is removed the statistic "
             "is no longer a squared norm and is not constrained to be non-negative: it takes "
             "negative values when the true discrepancy is near zero. Suppressed below a pool of "
             "five, which leaves it scored on too few cells here to support a comparison."),
    "distributional.mmd2_style_residual": dict(
        name="Unbiased squared MMD, style removed", short="Unbiased MMD², style removed",
        dirn=-1, ideal=0.0, space=True, defn="As above, after the platform directions are projected out."),
    "distributional.mmd2_biased": dict(
        name="Biased squared MMD (RBF, median heuristic)", short="Non-negative MMD²",
        dirn=-1, ideal=0.0, space=True,
        defn="The V-statistic estimator retaining diagonal terms. It is non-negative by "
             "construction but positively biased at small pool sizes, so it is reported "
             "beside rather than instead of the unbiased estimator."),
    "distributional.mmd2_biased_style_residual": dict(
        name="Biased squared MMD, style removed", short="Non-negative MMD², style removed",
        dirn=-1, ideal=0.0, space=True,
        defn="As above, after the platform directions are projected out."),
    "distributional.topic_jsd": dict(
        name=f"{SPACE_SHORT} topic Jensen–Shannon divergence", short="Topic JSD",
        dirn=-1, ideal=0.0, space=True,
        defn="Jensen–Shannon divergence between the two pools' distributions over a shared k-means "
             "partition of the embedding space. The vocabulary contains twenty-four clusters, "
             "is fitted once on authentic posts, and is shared by every method."),
    # --- platform match ---
    "classifier.calibration_gap": dict(
        name="Lexical platform-classifier calibration gap", short="Calibration gap (lexical)",
        dirn=-1, ideal=0.0, space=False,
        defn="Absolute difference between the rate at which a TF-IDF logistic classifier, trained "
             "on authentic train-split posts, labels the generated pool as target-platform and the "
             "rate at which it so labels the authentic target pool. The objective is to match the "
             "authentic rate, not to saturate the classifier, which is why the gap rather than the "
             "rate is the score."),
    "classifier.stylometric_calibration_gap": dict(
        name="Stylometric platform-classifier calibration gap", short="Calibration gap (stylometric)",
        dirn=-1, ideal=0.0, space=False,
        defn="The same quantity for a classifier restricted to twenty-one interpretable surface "
             "features, which cannot exploit topic vocabulary. Agreement between the two "
             "classifiers is evidence that a transfer is stylistic rather than lexical."),
    "classifier.target_rate": dict(name="Target-platform classification rate (lexical)",
        short="Target rate (lexical)", dirn=0, ideal=None, space=False,
        defn="Proportion of generated posts labelled target-platform. Diagnostic; its target is the "
             "rate authentic posts achieve, reported alongside."),
    "classifier.real_target_rate": dict(name="Authentic target-platform classification rate",
        short="Authentic target rate", dirn=0, ideal=None, space=False,
        defn="The rate authentic target posts achieve, which is the reference for the row above."),
    "classifier.stylometric_target_rate": dict(name="Target-platform classification rate (stylometric)",
        short="Target rate (stylometric)", dirn=0, ideal=None, space=False, defn="As above, surface features only."),
    "classifier.mean_target_prob": dict(name="Mean target-platform probability",
        short="Mean target probability", dirn=0, ideal=None, space=False,
        defn="Mean predicted probability, more sensitive than the hard rate when outputs cross the "
             "decision boundary only marginally."),
    "classifier.prob_gap_vs_real": dict(name="Mean-probability gap against authentic posts",
        short="Probability gap", dirn=0, ideal=0.0, space=False, defn="Difference of the above against the authentic pool."),
    "structural.feature_coverage": dict(
        name="Writing-habit shift coverage", short="Habit coverage",
        dirn=1, ideal=1.0, space=False,
        defn="Proportion of twenty-one surface features whose authentic source-to-target shift the "
             "generated pool reproduced to within half its magnitude. A stricter criterion than "
             "moving a feature in the correct direction."),
    "structural.mean_feature_jsd": dict(name="Mean per-feature Jensen–Shannon divergence",
        short="Mean feature JSD", dirn=-1, ideal=0.0, space=False,
        defn="Mean divergence between binned generated and authentic distributions across the "
             "twenty-one surface features. Each feature uses ten bins whose edges span the "
             "generated, authentic and source pools jointly."),
    "structural.mean_effect_gap": dict(name="Mean per-feature effect-size gap",
        short="Mean effect gap", dirn=-1, ideal=0.0, space=False,
        defn="Mean absolute difference between the generated and authentic Cohen's d relative to "
             "the source pool, per feature."),
    # --- content and well-formedness ---
    "semantic.source_similarity": dict(
        name=f"{SPACE_SHORT} cosine similarity to the source post", short="Similarity to source",
        dirn=1, ideal=None, space=True,
        defn="Cosine between a generated post and the source post it was given. The copy floor "
             "attains one by construction and the wrong-topic control sits near the corpus floor, "
             "so this row is bounded on both sides by reference columns."),
    "semantic.target_pool_similarity": dict(
        name=f"{SPACE_SHORT} mean cosine to the authentic target pool", short="Similarity to target pool",
        dirn=1, ideal=None, space=True, defn="Mean cosine between a generated post and the authentic posts of its cell."),
    "semantic.content_word_retention": dict(
        name="Content-word retention", short="Content-word retention",
        dirn=1, ideal=None, space=False,
        defn="Mean proportion of the source post's unique lower-cased, non-stopword terms longer "
             "than three characters that also occur in the output. Reported beside embedding "
             "similarity because the latter can stay high while specific terms are discarded."),
    "degeneracy.length_ratio_vs_real": dict(
        name="Output length ratio against authentic posts", short="Length ratio",
        dirn=0, ideal=1.0, space=False,
        defn="Mean character length of the generated pool divided by that of the authentic pool in "
             "the same cell. One is the target; both directions are failures."),
    "degeneracy.distinct_2": dict(
        name="Distinct-2 (distinct bigram ratio)", short="Distinct-2",
        dirn=1, ideal=None, space=False,
        defn="Number of distinct token bigrams divided by the total number of bigrams across the "
             "pool. Falls when a pool repeats phrasing within or across outputs. It is a ratio, so "
             "it is depressed by long outputs as well as by repetitive ones, and should be read "
             "beside the length ratio rather than alone."),
    "degeneracy.pool_self_similarity": dict(
        name=f"{SPACE_SHORT} mean pairwise similarity within the generated pool",
        short="Pool self-similarity", dirn=0, ideal=None, space=True,
        defn="Mean cosine between distinct generated posts in a cell. Its reference is the value "
             "the authentic pool attains, reported in the oracle column; materially above it "
             "indicates outputs more alike one another than real posts are."),
    "degeneracy.copy_rate": dict(name="Near-verbatim source copy rate", short="Copy rate",
        dirn=-1, ideal=0.0, space=False,
        defn="Proportion of outputs reproducing the source post near-verbatim, which is the absence "
             "of transfer. The copy floor attains one by construction."),
    "degeneracy.empty_rate": dict(name="Empty-output rate", short="Empty rate", dirn=-1, ideal=0.0,
        space=False, defn="Proportion of outputs that are empty."),
    "degeneracy.failure_rate": dict(name="Generation failure rate", short="Failure rate", dirn=-1,
        ideal=0.0, space=False,
        defn="Proportion of tasks on which generation failed and was recorded rather than dropped."),
}

# Display equations mirror the implementations in vectorial_eval.metrics. They
# deliberately describe the finite-sample estimator that produced the report,
# not only the corresponding population quantity. G is a generated pool, R the
# authentic target pool, S the supplied source pool, and e(.) the report's
# embedding function; the notation block rendered above the glossary defines
# the remaining symbols.
METRIC_EQUATIONS: dict[str, str] = {
    "trm.trm": (
        r"\operatorname{TRM}(G,R)=Q(G,R)+Q(R,G),\qquad "
        r"Q(G,R)=\sum_{k=0}^{2}\left(p_k(G,R)-\frac13\right)^2"
    ),
    "trm.rank_i2": (
        r"p_2(G,R)=\mathbb E\!\left[\mathbf 1\{\max(a,b)\le u\}\,"
        r"\mathbf 1\{u>\min(a,b)\}\right]"
    ),
    "trm.rank_i0": (
        r"p_0(G,R)=\mathbb E\!\left[\mathbf 1\{u\le\min(a,b)\}\right]"
    ),
    "trm.frechet": (
        r"D_F(G,R)=\lVert\mu_G-\mu_R\rVert_2^2+\operatorname{tr}\!\left("
        r"\Sigma_G+\Sigma_R-2(\Sigma_G\Sigma_R)^{1/2}\right)"
    ),
    "distributional.centroid_distance": (
        r"D_{\mathrm{cent}}(G,R)=1-\frac{\bar e_G^{\mathsf T}\bar e_R}"
        r"{\lVert\bar e_G\rVert_2\lVert\bar e_R\rVert_2},\qquad "
        r"\bar e_G=\frac1{|G|}\sum_{g\in G}e(g)"
    ),
    "distributional.topic_jsd": (
        r"\operatorname{JSD}(P_G,P_R)=\frac12D_{\mathrm{KL},2}(P_G\Vert M)"
        r"+\frac12D_{\mathrm{KL},2}(P_R\Vert M),\qquad M=\frac12(P_G+P_R)"
    ),
    "classifier.calibration_gap": (
        r"\Delta_{\mathrm{cal}}=\left|\frac1{|G|}\sum_{g\in G}"
        r"\mathbf 1\{p_{\mathrm{lex}}(g)>0.5\}-\frac1{|R|}\sum_{r\in R}"
        r"\mathbf 1\{p_{\mathrm{lex}}(r)>0.5\}\right|"
    ),
    "classifier.stylometric_calibration_gap": (
        r"\Delta_{\mathrm{sty}}=\left|\frac1{|G|}\sum_{g\in G}"
        r"\mathbf 1\{p_{\mathrm{sty}}(g)>0.5\}-\frac1{|R|}\sum_{r\in R}"
        r"\mathbf 1\{p_{\mathrm{sty}}(r)>0.5\}\right|"
    ),
    "classifier.target_rate": (
        r"\operatorname{TargetRate}(G)=\frac1{|G|}\sum_{g\in G}"
        r"\mathbf 1\{p_{\mathrm{lex}}(g)>0.5\}"
    ),
    "structural.feature_coverage": (
        r"\begin{aligned}\operatorname{Coverage}&=\frac1{21}\sum_{f=1}^{21}"
        r"\mathbf 1\!\left\{\Delta_f\le\tau_f\right\},\\"
        r"\tau_f&=\max\!\left(0.5|d_f(R,S)|,0.1\right),\qquad "
        r"\Delta_f=|d_f(G,S)-d_f(R,S)|\end{aligned}"
    ),
    "structural.mean_feature_jsd": (
        r"\overline{\operatorname{JSD}}_{\mathrm{feature}}="
        r"\frac1{21}\sum_{f=1}^{21}\operatorname{JSD}"
        r"\!\left(H_f(G),H_f(R)\right)"
    ),
    "structural.mean_effect_gap": (
        r"\overline\Delta_{\mathrm{effect}}=\frac1{|F^*|}\sum_{f\in F^*}"
        r"\left|d_f(G,S)-d_f(R,S)\right|"
    ),
    "semantic.source_similarity": (
        r"\operatorname{SourceSim}=\frac1N\sum_{i=1}^{N}"
        r"\frac{e(g_i)^{\mathsf T}e(s_i)}{\lVert e(g_i)\rVert_2\lVert e(s_i)\rVert_2}"
    ),
    "semantic.content_word_retention": (
        r"\operatorname{Retention}=\frac1{|I^*|}\sum_{i\in I^*}"
        r"\frac{|W(s_i)\cap W(g_i)|}{|W(s_i)|}"
    ),
    "semantic.target_pool_similarity": (
        r"\operatorname{TargetPoolSim}=\frac1{|G||R|}\sum_{g\in G}\sum_{r\in R}"
        r"\frac{e(g)^{\mathsf T}e(r)}{\lVert e(g)\rVert_2\lVert e(r)\rVert_2}"
    ),
    "degeneracy.length_ratio_vs_real": (
        r"\operatorname{LengthRatio}=\frac{|G|^{-1}\sum_{g\in G}\operatorname{chars}(g)}"
        r"{|R|^{-1}\sum_{r\in R}\operatorname{chars}(r)}"
    ),
    "degeneracy.distinct_2": (
        r"\operatorname{Distinct}_2(G)=\frac{|\bigcup_{g\in G}B_2(g)|}"
        r"{\sum_{g\in G}|B_2(g)|}"
    ),
    "degeneracy.pool_self_similarity": (
        r"\operatorname{SelfSim}(G)=\frac1{|G|(|G|-1)}\sum_{i\ne j}"
        r"\frac{e(g_i)^{\mathsf T}e(g_j)}{\lVert e(g_i)\rVert_2\lVert e(g_j)\rVert_2}"
    ),
    "degeneracy.copy_rate": (
        r"\operatorname{CopyRate}=\frac1N\sum_{i=1}^{N}\mathbf 1\{J_i>0.85\}"
    ),
    "degeneracy.failure_rate": (
        r"\operatorname{FailureRate}=\frac1N\sum_{i=1}^{N}"
        r"\mathbf 1\{\operatorname{ok}_i=\mathrm{false}\}"
    ),
}

# Every equation has a local key as well as the shared notation rendered above
# the glossary. Keeping this adjacent to METRIC_EQUATIONS makes an undefined
# symbol visible in code review rather than leaving it to prose elsewhere.
METRIC_SYMBOLS: dict[str, str] = {
    "trm.trm": (
        r"For pools \(A,B\), \(p_k(A,B)\) is the frequency with which the within-\(B\) edge has "
        r"rank \(k\in\{0,1,2\}\), from shortest to longest, in triangles containing one point "
        r"from \(A\) and two distinct points from \(B\). Thus \(p_k(G,R)\) uses the authentic–authentic "
        r"edge and \(p_k(R,G)\) uses the generated–generated edge. \(p_1=1-p_0-p_2\). \(Q\) is the directed "
        r"squared deviation of this rank profile from \((1/3,1/3,1/3)\); TRM adds both directions."
    ),
    "trm.rank_i2": (
        r"\(g\) is sampled uniformly from \(G\), and \((r_j,r_\ell)\) uniformly from unordered "
        r"distinct pairs in \(R\). The edge lengths are \(a=\rho(g,r_j)\), "
        r"\(b=\rho(g,r_\ell)\), and \(u=\rho(r_j,r_\ell)\). "
        r"\(\rho(x,z)=1-\cos(e(x),e(z))\), where "
        r"\(\cos(v,w)=v^{\mathsf T}w/(\lVert v\rVert_2\lVert w\rVert_2)\). "
        r"\(\min,\max\) select the smaller and larger argument. \(\mathbb E\) averages "
        r"over those triangles and \(\mathbf 1\{\cdot\}\) is the indicator function. The second "
        r"indicator implements the code's tie rule: an all-edge tie belongs to \(p_0\), not \(p_2\)."
    ),
    "trm.rank_i0": (
        r"\(g,(r_j,r_\ell),a,b,u,\rho,\mathbb E\) and \(\mathbf 1\) are defined as for \(p_2\). "
        r"The event \(u\le\min(a,b)\) assigns every shortest-edge tie to \(p_0\)."
    ),
    "trm.frechet": (
        r"For pool \(A\), \(\mu_A\) is the mean of \(\{e(x):x\in A\}\) and \(\Sigma_A\) is its "
        r"sample covariance plus \(10^{-6}I\), with \(I\) the embedding-space identity matrix. "
        r"\(\lVert\cdot\rVert_2\) is the Euclidean norm, "
        r"\(\operatorname{tr}\) the matrix trace, and \((\Sigma_G\Sigma_R)^{1/2}\) denotes the "
        r"matrix square-root term whose trace is computed from the product's eigenvalues."
    ),
    "distributional.centroid_distance": (
        r"\(\bar e_A=|A|^{-1}\sum_{x\in A}e(x)\) is pool \(A\)'s mean embedding. Superscript "
        r"\(\mathsf T\) is transpose and \(\lVert\cdot\rVert_2\) is the Euclidean norm; their "
        r"quotient is cosine similarity between the two centroids."
    ),
    "distributional.topic_jsd": (
        r"\(P_A\) is the smoothed, normalised histogram obtained by assigning every \(e(x)\), "
        r"\(x\in A\), to its nearest one of 24 shared k-means centres, adding \(10^{-9}\) to "
        r"each count. \(M=(P_G+P_R)/2\). For categorical distributions \(P,Q\), "
        r"\(D_{\mathrm{KL},2}(P\Vert Q)=\sum_k P_k\log_2(P_k/Q_k)\), where \(k\) indexes a "
        r"cluster and \(\log_2\) is the base-2 logarithm."
    ),
    "classifier.calibration_gap": (
        r"\(p_{\mathrm{lex}}(x)\) is the target-platform probability from the train-only TF–IDF "
        r"logistic classifier. \(G\) and \(R\) are the generated and authentic target pools; "
        r"\(\mathbf 1\{p>0.5\}\) converts a probability to its hard target-platform decision."
    ),
    "classifier.stylometric_calibration_gap": (
        r"\(p_{\mathrm{sty}}(x)\) is the target-platform probability from the train-only logistic "
        r"classifier over 21 stylometric features. \(G,R\) and \(\mathbf 1\) have the meanings "
        r"given for the lexical calibration gap."
    ),
    "classifier.target_rate": (
        r"\(p_{\mathrm{lex}}(g)\) is the lexical classifier's target-platform probability for "
        r"generated post \(g\); \(\mathbf 1\{p_{\mathrm{lex}}(g)>0.5\}\) is one exactly when the "
        r"post is classified as target-platform."
    ),
    "structural.feature_coverage": (
        r"\(f\) indexes the 21 stylometric features. \(d_f(A,S)\) is Cohen's \(d\) between feature "
        r"\(f\)'s values in pool \(A\) and the authentic source pool \(S\): "
        r"\(d_f(A,S)=(\bar f_A-\bar f_S)/s_{p,f}(A,S)\), where \(\bar f_A\) is the sample mean "
        r"and \(s_{p,f}\) the pooled sample standard deviation. \(\Delta_f\) is the "
        r"generated-versus-authentic effect-size gap, \(\tau_f\) is its implemented tolerance, and "
        r"\(\mathbf 1\) records whether it meets the implemented tolerance."
    ),
    "structural.mean_feature_jsd": (
        r"\(H_f(A)\) is the normalised ten-bin histogram of feature \(f\) in pool \(A\), with "
        r"\(10^{-9}\) added to every count; "
        r"the bin edges span \(G,R,S\) jointly. \(\operatorname{JSD}\) is the base-2 "
        r"Jensen–Shannon divergence defined in the topic-JSD entry."
    ),
    "structural.mean_effect_gap": (
        r"\(F^*\) is the subset of the 21 features for which both Cohen's-\(d\) values are finite. "
        r"\(d_f\) and \(S\) are defined in the coverage entry, and the overbar denotes the "
        r"arithmetic mean over \(F^*\)."
    ),
    "semantic.source_similarity": (
        r"\(N\) is the number of non-empty generated outputs with a recorded supplied source; \(g_i\) and "
        r"\(s_i\) are output \(i\) and its paired source. \(e,\mathsf T\), and "
        r"\(\lVert\cdot\rVert_2\) denote the embedding, transpose, and Euclidean norm."
    ),
    "semantic.content_word_retention": (
        r"\(W(x)\) is the set of unique lower-cased tokens matching "
        r"\(\texttt{[A-Za-z][A-Za-z'-]+}\), longer than three characters, and absent from the "
        r"implementation's stopword set. \(I^*=\{i:|W(s_i)|>0\}\) contains the paired tasks whose source "
        r"has at least one such term; \(g_i,s_i\) are the generated post and supplied source."
    ),
    "semantic.target_pool_similarity": (
        r"The two sums range over every generated–authentic pair in the cell. \(e,\mathsf T\), "
        r"and \(\lVert\cdot\rVert_2\) denote the embedding, transpose, and Euclidean norm; "
        r"\(|G||R|\) is the number of cross-pool pairs."
    ),
    "degeneracy.length_ratio_vs_real": (
        r"\(\operatorname{chars}(x)\) is Python's character count \(\operatorname{len}(x)\). "
        r"The numerator and denominator are the mean character counts in \(G\) and \(R\)."
    ),
    "degeneracy.distinct_2": (
        r"\(B_2(x)\) is the multiset of adjacent lower-cased tokens matching "
        r"\(\texttt{[A-Za-z][A-Za-z'-]+}\) in \(x\). The union "
        r"in the numerator discards duplicates; the denominator sums all bigram occurrences."
    ),
    "degeneracy.pool_self_similarity": (
        r"Indices \(i,j\in\{1,\ldots,|G|\}\) identify generated posts, and the sum is over ordered "
        r"pairs \(i\ne j\), matching the off-diagonal "
        r"mean of the cosine-similarity matrix. \(e,\mathsf T\), and "
        r"\(\lVert\cdot\rVert_2\) have their usual definitions above."
    ),
    "degeneracy.copy_rate": (
        r"\(N\) counts all attempted outputs in the cell. \(W\) is the content-word set defined "
        r"for retention. \(J_i=|W(s_i)\cap W(g_i)|/|W(s_i)\cup W(g_i)|\) when the source, output, "
        r"and both word sets are non-empty; \(J_i=0\) otherwise. Thus \(J_i\) is the implemented "
        r"content-word Jaccard similarity with skipped cases contributing zero."
    ),
    "degeneracy.failure_rate": (
        r"\(N\) is the number of attempted outputs in the cell. \(\operatorname{ok}_i\) is the "
        r"Boolean success flag stored in output \(i\)'s metadata; failed attempts remain in the "
        r"denominator."
    ),
}

#: Pooled scalars rather than per-cell means. They carry no interval and are
#: reported only in the master table.
POOLED_PREFIXES = (".overall.",)


def mname(key: str, short: bool = False) -> str:
    m = METRICS.get(key)
    if not m:
        return key
    return m["short"] if short else m["name"]


def mdefn(key: str) -> str:
    return (METRICS.get(key) or {}).get("defn", "")


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def _aggregation_note(rep, key, cols, extra=""):
    """State how the plotted quantity was aggregated.

    Every figure and table in this report shows a value averaged over cells. The
    average is unweighted and the interval is a percentile bootstrap resampled
    over cells rather than over posts, and both of those choices change the
    number, so neither may be left to the reader to assume. The cell count is
    read from the report rather than typed, because it differs by metric: a
    statistic with a minimum pool size scores fewer cells than one without.
    """
    ns = sorted({(entry(rep, fn, key) or {}).get("n_cells") for fn, _ in cols} - {None})
    n_txt = str(ns[0]) if len(ns) == 1 else f"{min(ns)} to {max(ns)}"
    boot = (rep.get("config", {}).get("eval", {}) or {}).get("n_bootstrap", 500)
    return (
        f"Aggregation: each point is the unweighted mean of the per-cell score over {n_txt} "
        f"cells. The mean is unweighted because the largest cell holds roughly fifteen times "
        f"the posts of the median, so a size-weighted mean would principally report one topic. "
        f"The interval is the 2.5 and 97.5 percentiles of {boot} bootstrap resamples drawn over "
        f"cells, cells being the independent unit rather than posts." + (" " + extra if extra else "")
    )


def dotplot(rep, key, title, subtitle, *, axis_label="", ideal=None, ideal_label="",
            lo=None, hi=None, better="lower", width=760, note=""):
    """Dot-and-interval plot, rows grouped by method family.

    Reference columns are vertical rules rather than rows. `ideal` draws a
    target line for a diagnostic whose best value is neither extreme.
    """
    rows: list[tuple[str, str, str, dict]] = []
    for fam, colour, cols in FAMILIES:
        for i, (fn, label) in enumerate(cols):
            e = entry(rep, fn, key)
            if e:
                rows.append((fam if i == 0 else "", label, colour, e | {"_fn": fn}))
    if not rows:
        return ""

    vals = [r[3]["mean"] for r in rows] + [r[3]["ci_low"] for r in rows] + [r[3]["ci_high"] for r in rows]
    for ref in REFERENCES:
        m = ref_mean(key, ref)
        if m is not None:
            vals.append(m)
    if ideal is not None:
        vals.append(ideal)
    vlo = lo if lo is not None else min(vals)
    vhi = hi if hi is not None else max(vals)
    pad = (vhi - vlo) * 0.08 or 0.1
    vlo, vhi = vlo - pad, vhi + pad

    left, right = 210, 74
    pw = width - left - right
    ttl, title_end = b1.text_block(0, 24, title, "fig-title", width_px=width - 8)
    sub, subtitle_end = b1.text_block(
        0, title_end + 4, subtitle, "fig-sub", width_px=width - 8
    )
    # Reserve a real annotation band after however many subtitle lines were
    # needed. Reference labels may use up to three rows within this band.
    top = max(104, subtitle_end + 48)
    row_h = 21
    gap = 9  # between family groups
    n_groups = len({r[0] for r in rows if r[0]})
    height = top + len(rows) * row_h + n_groups * gap + 78

    def px(v):
        return left + (v - vlo) / (vhi - vlo) * pw

    out = [b1.svg_open(width, height, title, subtitle), ttl, sub]

    # axis
    y0 = top
    y1 = top + len(rows) * row_h + n_groups * gap
    for t in range(5):
        v = vlo + (vhi - vlo) * t / 4
        x = px(v)
        out.append(f'<line x1="{x:.1f}" y1="{y0:.1f}" x2="{x:.1f}" y2="{y1:.1f}" stroke="#e6e4de" stroke-width="1"/>')
        out.append(b1.text(x, y1 + 15, f"{v:.2f}", "tick", "middle"))

    ref_lines(key, px, y0, y1, width - 4, out)

    if ideal is not None and vlo <= ideal <= vhi:
        x = px(ideal)
        out.append(
            f'<line x1="{x:.1f}" y1="{y0 - 8:.1f}" x2="{x:.1f}" y2="{y1:.1f}" '
            f'stroke="#0b0b0b" stroke-width="1.5" stroke-dasharray="1 3"/>'
        )
        out.append(b1.text(x, y1 + 29, ideal_label, "ref-lbl", "middle"))

    # rows
    yy = top + row_h * 0.72
    for fam, label, colour, e in rows:
        if fam:
            yy += gap
            out.append(b1.text(6, yy - 1, fam, "key-lbl"))
        out.append(b1.text(left - 12, yy, label, "lbl", "end"))
        a, c = px(e["ci_low"]), px(e["ci_high"])
        out.append(
            f'<line x1="{a:.1f}" y1="{yy - 4:.1f}" x2="{a:.1f}" y2="{yy:.1f}" stroke="{colour}" stroke-width="1.5" opacity="0.75"/>'
        )
        out.append(
            f'<line x1="{c:.1f}" y1="{yy - 4:.1f}" x2="{c:.1f}" y2="{yy:.1f}" stroke="{colour}" stroke-width="1.5" opacity="0.75"/>'
        )
        out.append(
            f'<line x1="{a:.1f}" y1="{yy - 2:.1f}" x2="{c:.1f}" y2="{yy - 2:.1f}" stroke="{colour}" stroke-width="2" opacity="0.55"/>'
        )
        cx = px(e["mean"])
        out.append(f'<circle cx="{cx:.1f}" cy="{yy - 2:.1f}" r="4.5" fill="{colour}" stroke="#fcfcfb" stroke-width="2"/>')
        out.append(b1.text(width - right + 10, yy, f"{e['mean']:.3f}", "val"))
        yy += row_h

    # Axis title names the metric. A figure whose axis reads only as a number
    # cannot be interpreted away from the paragraph that introduced it.
    axis = axis_label or key
    axis_y = y1 + (48 if ideal is not None else 32)
    out.append(b1.text(left + pw / 2, axis_y, axis, "ax-title", "middle"))

    direction = (f"Closer to {ideal:.3f} is better. " if ideal is not None
                 else f"{'Lower' if better == 'lower' else 'Higher'} is better. ")
    foot = direction + _aggregation_note(
        rep, key, [(fn, lb) for _f, _c, cs in FAMILIES for fn, lb in cs], note
    )
    foot_y = y1 + (68 if ideal is not None else 52)
    blk, y_end = b1.text_block(0, foot_y, foot, "fig-note", width_px=width - 8)
    out.append(blk)
    out.append("</svg>")
    svg = "\n".join(out)
    # The footnote wraps to a variable number of lines, so the height declared in
    # the viewBox is corrected here rather than guessed above.
    return svg.replace(
        f'viewBox="0 0 {width} {height}"', f'viewBox="0 0 {width} {max(height, y_end + 14):.0f}"', 1
    )


def family_overview(rep, key="trm.trm", width=860):
    """Compact family-level reading of the primary result.

    Every method configuration remains visible as a dot, while the row and its
    span make the family comparison legible before the detailed interval plot.
    The span is explicitly not an uncertainty interval.
    """
    series = []
    for fam, colour, cols in FAMILIES:
        vals = [(label, mean(rep, fn, key)) for fn, label in cols]
        vals = [(label, value) for label, value in vals if value is not None]
        if vals:
            series.append((fam, colour, vals))
    if not series:
        return ""

    allv = [value for _fam, _colour, vals in series for _label, value in vals]
    allv += [value for fn in REFERENCES if (value := ref_mean(key, fn)) is not None]
    lo, hi = min(allv), max(allv)
    pad = (hi - lo) * 0.09 or 0.05
    lo, hi = lo - pad, hi + pad
    left, right, top = 126, 92, 112
    pw = width - left - right
    row_h = 48
    y1 = top + len(series) * row_h
    height = y1 + 88

    def px(value):
        return left + (value - lo) / (hi - lo) * pw

    title = "The primary result, by method family"
    subtitle = ("Each dot is one evaluated configuration. The line joins the lowest and highest "
                "observed value within the family; it is not an uncertainty interval.")
    out = [b1.svg_open(width, height, title, subtitle)]
    block, y = b1.text_block(0, 24, title, "fig-title", width_px=width - 8)
    out.append(block)
    block, y = b1.text_block(0, y + 4, subtitle, "fig-sub", width_px=width - 8)
    out.append(block)
    for i in range(5):
        value = lo + (hi - lo) * i / 4
        x = px(value)
        out.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{y1}" stroke="#e8e6df"/>')
        out.append(b1.text(x, y1 + 18, f"{value:.2f}", "tick", "middle"))
    ref_lines(key, px, top, y1, width - 4, out, label_y=top - 13)
    for i, (fam, colour, vals) in enumerate(series):
        yy = top + i * row_h + row_h / 2
        values = [value for _label, value in vals]
        out.append(b1.text(left - 14, yy + 4, fam, "lbl", "end"))
        out.append(
            f'<line x1="{px(min(values)):.1f}" y1="{yy:.1f}" '
            f'x2="{px(max(values)):.1f}" y2="{yy:.1f}" stroke="{colour}" '
            'stroke-width="5" stroke-linecap="round" opacity="0.28"/>'
        )
        for _label, value in vals:
            out.append(
                f'<circle cx="{px(value):.1f}" cy="{yy:.1f}" r="5.2" fill="{colour}" '
                'stroke="#fff" stroke-width="1.8"/>'
            )
        best = min(values)
        out.append(b1.text(width - right + 10, yy + 4, f"best {best:.3f}", "val"))
    out.append(b1.text(left + pw / 2, y1 + 38,
                       "TRM score · lower is better; target = 0",
                       "ax-title", "middle"))
    foot = ("Reference rules state their own cell bases. Method dots use the systems-only common-cell "
            "base. Detailed bootstrap intervals and the generated overlap test appear on the Results page.")
    block, y_end = b1.text_block(0, y1 + 58, foot, "fig-note", width_px=width - 8)
    out.append(block)
    out.append("</svg>")
    return "\n".join(out).replace(
        f'viewBox="0 0 {width} {height}"', f'viewBox="0 0 {width} {max(height, y_end + 14):.0f}"', 1
    )


def tradeoff(rep, width=760, height=440):
    """Content preservation against distribution match.

    The two axes are the study's tension: a column can match the target pool by
    discarding what it was asked to carry across. Reading them together is the
    only way to see that, which is why they share one figure rather than two.
    """
    pts = []
    for fam, colour, cols in FAMILIES:
        for fn, label in cols:
            x = mean(rep, fn, "semantic.source_similarity")
            y = mean(rep, fn, "trm.trm")
            if x is not None and y is not None:
                pts.append((fn, f"{fam} · {label}", colour, x, y))
    refs = []
    for ref in REFERENCES:
        x = ref_mean("semantic.source_similarity", ref)
        y = ref_mean("trm.trm", ref)
        if x is not None and y is not None:
            refs.append((ref, ref_label("trm.trm", ref), REF_STYLE[ref][0], x, y))
    if not pts:
        return ""

    xs = [p[3] for p in pts] + [r[3] for r in refs]
    ys = [p[4] for p in pts] + [r[4] for r in refs]
    xlo, xhi = min(xs), max(xs)
    ylo, yhi = min(ys), max(ys)
    xpad, ypad = (xhi - xlo) * 0.12 or 0.05, (yhi - ylo) * 0.12 or 0.05
    xlo, xhi, ylo, yhi = xlo - xpad, xhi + xpad, ylo - ypad, yhi + ypad

    left, right, top, bot = 64, 210, 84, 62
    pw, ph = width - left - right, height - top - bot

    def px(v):
        return left + (v - xlo) / (xhi - xlo) * pw

    def py(v):
        return top + ph - (v - ylo) / (yhi - ylo) * ph

    title = "Keeping the content while matching the target"
    sub = ("Right is better content preservation. Down is better distribution match. "
           "The useful corner is the lower right.")
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
    out.append(b1.text(left + pw / 2, top + ph + 34,
                       "Source cosine similarity", "ax-title", "middle"))
    out.append(
        f'<text class="ax-title" transform="translate(16,{top + ph / 2}) rotate(-90)" text-anchor="middle">'
        f"TRM score</text>"
    )

    # references drawn as rings so they are never read as systems
    for _fn, label, colour, x, y_ in refs:
        out.append(
            f'<circle cx="{px(x):.1f}" cy="{py(y_):.1f}" r="6.5" fill="#fcfcfb" stroke="{colour}" stroke-width="2.5"/>'
        )
    for _fn, label, colour, x, y_ in pts:
        out.append(
            f'<circle cx="{px(x):.1f}" cy="{py(y_):.1f}" r="5.5" fill="{colour}" stroke="#fcfcfb" stroke-width="1.8"/>'
        )

    # legend, one entry per family, to the right of the panel
    ly = top + 6
    out.append(b1.text(left + pw + 16, ly, "method family", "key-lbl"))
    ly += 16
    for fam, colour, _cols in FAMILIES:
        out.append(f'<circle cx="{left + pw + 22}" cy="{ly - 4}" r="5" fill="{colour}"/>')
        out.append(b1.text(left + pw + 34, ly, fam, "lbl"))
        ly += 17
    ly += 4
    out.append(b1.text(left + pw + 16, ly, "reference", "key-lbl"))
    ly += 16
    for _fn, label, colour, _x, _y in refs:
        out.append(
            f'<circle cx="{left + pw + 22}" cy="{ly - 4}" r="5" fill="#fcfcfb" stroke="{colour}" stroke-width="2"/>'
        )
        blk, ly2 = b1.text_block(left + pw + 34, ly, label, "lbl", width_px=right - 44, line_h=13)
        out.append(blk)
        ly = ly2 + 4

    cols = [(fn, lb) for _f, _c, cs in FAMILIES for fn, lb in cs]
    foot = _aggregation_note(
        rep, "trm.trm", cols,
        "Both axes are aggregated the same way. Intervals are omitted here so that fifteen "
        "columns remain legible; they are given in the tables and the resolution test below.",
    )
    blk, y_end = b1.text_block(0, top + ph + 52, foot, "fig-note", width_px=width - 8)
    out.append(blk)
    out.append("</svg>")
    svg = "\n".join(out)
    return svg.replace(
        f'viewBox="0 0 {width} {height}"', f'viewBox="0 0 {width} {max(height, y_end + 14):.0f}"', 1
    )


def length_tradeoff(rep, width=760, height=470):
    """Output length against the primary distributional measure."""
    import math

    points = []
    for fam, colour, cols in FAMILIES:
        vals = []
        for fn, label in cols:
            x = mean(rep, fn, "degeneracy.length_ratio_vs_real")
            y = mean(rep, fn, "trm.trm")
            if x is not None and x > 0 and y is not None:
                vals.append((fn, label, x, y))
        if vals:
            points.append((fam, colour, vals))
    refs = []
    for fn in REFERENCES:
        x = ref_mean("degeneracy.length_ratio_vs_real", fn)
        y = ref_mean("trm.trm", fn)
        if x is not None and x > 0 and y is not None:
            refs.append((fn, x, y))
    if not points:
        return ""

    xs = [x for _fam, _colour, vals in points for _fn, _label, x, _y in vals]
    xs += [x for _fn, x, _y in refs]
    ys = [y for _fam, _colour, vals in points for _fn, _label, _x, y in vals]
    ys += [y for _fn, _x, y in refs]
    log_lo, log_hi = min(math.log10(x) for x in xs + [1.0]), max(math.log10(x) for x in xs + [1.0])
    log_pad = (log_hi - log_lo) * 0.08 or 0.05
    log_lo, log_hi = log_lo - log_pad, log_hi + log_pad
    ylo, yhi = 0.0, max(ys) * 1.09
    left, right, top, bot = 72, 208, 90, 78
    pw, ph = width - left - right, height - top - bot

    def px(value):
        return left + (math.log10(value) - log_lo) / (log_hi - log_lo) * pw

    def py(value):
        return top + ph - (value - ylo) / (yhi - ylo) * ph

    title = "Low triangle scores arise at very different output lengths"
    subtitle = ("Comparable triangle scores can occur at different output lengths; length alone "
                "does not determine the distributional result.")
    out = [b1.svg_open(width, height, title, subtitle)]
    block, y = b1.text_block(0, 24, title, "fig-title", width_px=width - 8)
    out.append(block)
    block, y = b1.text_block(0, y + 4, subtitle, "fig-sub", width_px=width - 8)
    out.append(block)
    out.append(f'<rect x="{left}" y="{top}" width="{pw}" height="{ph}" fill="#fff" stroke="#dedbd2"/>')
    for i in range(5):
        lx = log_lo + (log_hi - log_lo) * i / 4
        gx = left + pw * i / 4
        gy = top + ph * i / 4
        value = 10 ** lx
        label = f"{value:.1f}×" if value < 10 else f"{value:.0f}×"
        out.append(f'<line x1="{gx:.1f}" y1="{top}" x2="{gx:.1f}" y2="{top + ph}" stroke="#efede7"/>')
        out.append(f'<line x1="{left}" y1="{gy:.1f}" x2="{left + pw}" y2="{gy:.1f}" stroke="#efede7"/>')
        out.append(b1.text(gx, top + ph + 17, label, "tick", "middle"))
        out.append(b1.text(left - 8, gy + 4, f"{yhi - (yhi - ylo) * i / 4:.2f}", "tick", "end"))
    if log_lo <= 0 <= log_hi:
        out.append(f'<line x1="{px(1):.1f}" y1="{top}" x2="{px(1):.1f}" y2="{top + ph}" '
                   'stroke="#2e7d5b" stroke-width="2" stroke-dasharray="4 3"/>')
        out.append(b1.text(px(1), top - 8, "authentic length · 1×", "zone-lbl", "middle"))
    for fam, colour, vals in points:
        if fam != "prompting" and len(vals) > 1:
            path = " ".join(
                f"{'M' if i == 0 else 'L'}{px(x):.1f},{py(yv):.1f}"
                for i, (_fn, _label, x, yv) in enumerate(vals)
            )
            out.append(f'<path d="{path}" fill="none" stroke="{colour}" stroke-width="1.5" opacity="0.38"/>')
        for _fn, _label, x, yv in vals:
            out.append(f'<circle cx="{px(x):.1f}" cy="{py(yv):.1f}" r="5.2" fill="{colour}" '
                       'stroke="#fff" stroke-width="1.8"/>')
    for fn, x, yv in refs:
        colour, _label = REF_STYLE[fn]
        out.append(f'<circle cx="{px(x):.1f}" cy="{py(yv):.1f}" r="6.4" fill="#fff" '
                   f'stroke="{colour}" stroke-width="2.4"/>')
    out.append(b1.text(left + pw / 2, top + ph + 38,
                       "Length ratio (log scale)",
                       "ax-title", "middle"))
    out.append(f'<text class="ax-title" transform="translate(15,{top + ph / 2}) rotate(-90)" '
               'text-anchor="middle">TRM score</text>')
    ly = top + 8
    out.append(b1.text(left + pw + 18, ly, "method family", "key-lbl"))
    ly += 18
    for fam, colour, _vals in points:
        out.append(f'<circle cx="{left + pw + 24}" cy="{ly - 4}" r="5" fill="{colour}"/>')
        out.append(b1.text(left + pw + 37, ly, fam, "lbl"))
        ly += 18
    ly += 8
    out.append(b1.text(left + pw + 18, ly, "reference rings", "key-lbl"))
    ly += 18
    for fn, _x, _yv in refs:
        colour, _label = REF_STYLE[fn]
        out.append(f'<circle cx="{left + pw + 24}" cy="{ly - 4}" r="5" fill="#fff" '
                   f'stroke="{colour}" stroke-width="2"/>')
        block, ly2 = b1.text_block(left + pw + 37, ly, ref_label("trm.trm", fn), "lbl",
                                  width_px=right - 50, line_h=13)
        out.append(block)
        ly = ly2 + 5
    foot = (_aggregation_note(rep, "trm.trm",
                              [(fn, label) for _fam, _colour, cols in FAMILIES for fn, label in cols])
            + " Lines join the three training variants within a family and are not fitted trends.")
    block, y_end = b1.text_block(0, top + ph + 57, foot, "fig-note", width_px=width - 8)
    out.append(block)
    out.append("</svg>")
    return "\n".join(out).replace(
        f'viewBox="0 0 {width} {height}"', f'viewBox="0 0 {width} {max(height, y_end + 14):.0f}"', 1
    )


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

TABLE_GROUPS = [
    (
        "Distribution match",
        "Whether the pool of generated posts is distributed like the pool of authentic ones. This is "
        "the study's primary question, and the Triangle-Rank Metric is its primary measure.",
        ["trm.trm", "trm.rank_i2", "trm.rank_i0", "trm.frechet",
         "distributional.centroid_distance", "distributional.topic_jsd"],
    ),
    (
        "Platform match",
        "Whether the output reads as the target platform. The calibration gaps compare the rate at "
        "which a classifier labels generated posts as target-platform against the rate at which it "
        "so labels authentic ones, so zero is the objective and saturating the classifier is not.",
        ["classifier.calibration_gap", "classifier.stylometric_calibration_gap",
         "classifier.target_rate", "structural.feature_coverage",
         "structural.mean_feature_jsd", "structural.mean_effect_gap"],
    ),
    (
        "Content preservation and well-formedness",
        "Whether the post the method was given survives into the output, and whether the output is "
        "well formed. A column can match the target distribution by discarding its input; these "
        "rows make that visible.",
        ["semantic.source_similarity", "semantic.content_word_retention",
         "semantic.target_pool_similarity", "degeneracy.length_ratio_vs_real",
         "degeneracy.distinct_2", "degeneracy.pool_self_similarity",
         "degeneracy.copy_rate", "degeneracy.failure_rate"],
    ),
]


def table(rep, rows, caption, subtitle):
    """One focused table. Columns are grouped by family with a spanning header."""
    cols = [(fn, lbl) for _f, _c, cs in FAMILIES for fn, lbl in cs]
    head_span = "".join(
        f'<th colspan="{len(cs)}" class="fam" style="color:{c}">{b1.esc(f)}</th>'
        for f, c, cs in FAMILIES
    )
    head = "".join(f"<th>{b1.esc(lbl)}</th>" for _fn, lbl in cols)
    refhead = "".join(f"<th>{b1.esc(REF_STYLE[r][1].split(' · ')[0])}</th>" for r in REFERENCES)

    body = []
    for key in rows:
        meta = METRICS.get(key, {})
        label = meta.get("short", key)
        direction = meta.get("dirn", 0)
        target = meta.get("ideal")
        vals = {fn: mean(rep, fn, key) for fn, _ in cols}
        present = [v for v in vals.values() if v is not None]
        if not present:
            continue
        if target is not None:
            best = min(present, key=lambda v: abs(v - target))
        elif direction > 0:
            best = max(present)
        elif direction < 0:
            best = min(present)
        else:
            best = None
        arrow = {1: " ↑", -1: " ↓", 0: " ·"}[direction]
        cells = []
        for fn, _ in cols:
            v = vals[fn]
            if v is None:
                cells.append("<td>—</td>")
            else:
                mark = ' class="best"' if best is not None and v == best else ""
                cells.append(f"<td{mark}>{v:.3f}</td>")
        # A reference column may cover fewer cells than the systems, in which case
        # it is averaged over the portion it covers and must print that count
        # rather than pass silently for a like-for-like value.
        sys_n = max((entry(rep, fn, key) or {}).get("n_cells", 0) for fn, _ in cols)
        refcells = []
        for r in REFERENCES:
            e = ref_entry(key, r)
            if e is None:
                refcells.append("<td class='ref'>—</td>")
                continue
            sup = f"<sup>{e['n_cells']}</sup>" if e.get("n_cells") != sys_n else ""
            refcells.append(f"<td class='ref'>{e['mean']:.3f}{sup}</td>")
        body.append(
            f"<tr><th scope='row'>{b1.esc(label)}{arrow}</th>" + "".join(cells) + "".join(refcells) + "</tr>"
        )

    return f"""<figure class="tablefig">
<figcaption><strong>{b1.esc(caption)}</strong><br><span class="sub">{b1.esc(subtitle)}</span></figcaption>
<div class="tscroll"><table class="res">
<thead>
<tr><td></td>{head_span}<th colspan="{len(REFERENCES)}" class="fam">reference</th></tr>
<tr><td></td>{head}{refhead}</tr>
</thead>
<tbody>{"".join(body)}</tbody>
</table></div>
<p class="tnote">↑ higher is better · ↓ lower is better · · diagnostic, best value marked against its target.
Bold is the best system column in the row; reference columns are excluded from that comparison.
A superscript on a reference value is the number of cells it was averaged over, printed whenever
that differs from the system columns' base.<br>
{b1.esc(_aggregation_note(rep, rows[0], cols))} Intervals for these values are given in the
resolution test below, and a difference should not be read from the point estimates alone.</p>
</figure>"""





def strip_by_cell(rep_full, key, title, subtitle, axis_label, width=760, better="lower"):
    """Per-cell values behind an aggregate, one strip per column.

    An aggregate over five cells can be produced by five similar values or by one
    outlier and four ties, and the two support very different claims. This figure
    shows the per-cell scores the mean was taken over, so a reader can see which
    of those the number is. The mean is marked with a rule.
    """
    per = rep_full.get("per_cell", {})
    raw = []
    cell_sets = []
    for fam, colour, cols in FAMILIES:
        for fn, label in cols:
            metric = key.split(".")[0]
            cells = per.get(f"{fn}.{metric}", {})
            sub = key.split(".", 1)[1]
            valid = {cell_id: values[sub] for cell_id, values in cells.items()
                     if values.get(sub) is not None and values[sub] == values[sub]}
            if valid:
                raw.append((fam, label, colour, valid))
                cell_sets.append(set(valid))
    # Identity and the wrong-topic control participate in the systems-only
    # common-cell intersection even though they are drawn as reference rules
    # rather than rows. Include their valid cell sets when reconstructing the
    # exact aggregate base from the per-cell payload.
    metric = key.split(".")[0]
    sub = key.split(".", 1)[1]
    for fn in ("identity", "shuffle_control"):
        cells = per.get(f"{fn}.{metric}", {})
        valid = {cell_id for cell_id, values in cells.items()
                 if values.get(sub) is not None and values[sub] == values[sub]}
        if valid:
            cell_sets.append(valid)
    common = set.intersection(*cell_sets) if cell_sets else set()
    series = [
        (fam, label, colour, [values[cell_id] for cell_id in sorted(common)])
        for fam, label, colour, values in raw
        if common
    ]
    if not series:
        return ""
    allv = [v for _f, _l, _c, vs in series for v in vs]
    allv += [m for fn in REFERENCES if (m := ref_mean(key, fn)) is not None]
    lo, hi = min(allv), max(allv)
    pad = (hi - lo) * 0.08 or 0.05
    lo, hi = lo - pad, hi + pad

    left, right = 210, 60
    title_block, title_end = b1.text_block(
        0, 24, title, "fig-title", width_px=width - 8
    )
    subtitle_block, subtitle_end = b1.text_block(
        0, title_end + 4, subtitle, "fig-sub", width_px=width - 8
    )
    top = max(104, subtitle_end + 48)
    pw = width - left - right
    row_h, gap = 19, 8
    n_groups = len({f for f, _l, _c, _v in series})
    height = top + len(series) * row_h + n_groups * gap + 80

    def px(v):
        return left + (v - lo) / (hi - lo) * pw

    out = [b1.svg_open(width, height, title, subtitle), title_block, subtitle_block]
    y0 = top
    y1 = top + len(series) * row_h + n_groups * gap
    for i in range(5):
        v = lo + (hi - lo) * i / 4
        out.append(f'<line x1="{px(v):.1f}" y1="{y0}" x2="{px(v):.1f}" y2="{y1}" stroke="#f0eee8"/>')
        out.append(b1.text(px(v), y1 + 15, f"{v:.2f}", "tick", "middle"))
    ref_lines(key, px, y0, y1, width - 4, out)
    seen = set()
    yy = top + row_h * 0.7
    for fam, label, colour, vals in series:
        if fam not in seen:
            seen.add(fam); yy += gap
            out.append(b1.text(6, yy - 1, fam, "key-lbl"))
        out.append(b1.text(left - 12, yy, label, "lbl", "end"))
        for v in vals:
            out.append(f'<circle cx="{px(v):.1f}" cy="{yy - 3:.1f}" r="3" fill="{colour}" opacity="0.5"/>')
        m = sum(vals) / len(vals)
        out.append(f'<line x1="{px(m):.1f}" y1="{yy - 10:.1f}" x2="{px(m):.1f}" y2="{yy + 4:.1f}" stroke="{colour}" stroke-width="2.2"/>')
        out.append(b1.text(width - right + 8, yy, f"n={len(vals)}", "tiny"))
        yy += row_h
    out.append(b1.text(left + pw / 2, y1 + 32, axis_label, "ax-title", "middle"))
    foot = (f"{'Lower' if better == 'lower' else 'Higher'} is better. Each dot is one cell; the rule is "
            f"the unweighted mean over the same {len(common)} shared cells used by every method "
            f"column in this figure. This is the value plotted in the dot-and-interval figure.")
    blk, y_end = b1.text_block(0, y1 + 50, foot, "fig-note", width_px=width - 8)
    out.append(blk); out.append("</svg>")
    return "\n".join(out).replace(f'viewBox="0 0 {width} {height}"',
                                  f'viewBox="0 0 {width} {max(height, y_end + 14):.0f}"', 1)


def variant_slopes(rep, key, title, subtitle, axis_label, width=760, better="lower"):
    """Effect of the training variant within each family.

    The variant axis is crossed with the family axis, so the question of whether a
    variant helps is separate from the question of which family wins. A slope
    chart puts the three variants of one family on one line, which makes a
    consistent variant effect visible as parallel slopes and an inconsistent one
    visible as crossing.
    """
    variants = ["topic only", "retrieved source", "aspects"]
    lines = []
    for fam, colour, cols in FAMILIES[1:]:
        pts = []
        for i, (fn, label) in enumerate(cols):
            m = mean(rep, fn, key)
            if m is not None:
                pts.append((i, m))
        if len(pts) > 1:
            lines.append((fam, colour, pts))
    if not lines:
        return ""
    allv = [v for _f, _c, ps in lines for _i, v in ps]
    refs = [(fn, m) for fn in REFERENCES if (m := ref_mean(key, fn)) is not None]
    allv += [m for _fn, m in refs]
    lo, hi = min(allv), max(allv)
    pad = (hi - lo) * 0.18 or 0.05
    lo, hi = lo - pad, hi + pad
    left, right, top, bot = 62, 210, 78, 78
    pw, ph = width - left - right, 250

    def px(i):
        return left + pw * i / (len(variants) - 1)

    def py(v):
        return top + ph - (v - lo) / (hi - lo) * ph

    out = [b1.svg_open(width, top + ph + bot, title, subtitle)]
    t, y = b1.text_block(0, 24, title, "fig-title", width_px=width - 8); out.append(t)
    sb, y = b1.text_block(0, y + 4, subtitle, "fig-sub", width_px=width - 8); out.append(sb)
    for i, vlabel in enumerate(variants):
        out.append(f'<line x1="{px(i):.1f}" y1="{top}" x2="{px(i):.1f}" y2="{top + ph}" stroke="#eeece6"/>')
        out.append(b1.text(px(i), top + ph + 18, vlabel, "tick", "middle"))
    for i in range(4):
        v = lo + (hi - lo) * i / 3
        out.append(b1.text(left - 8, py(v) + 4, f"{v:.2f}", "tick", "end"))
    # Reference columns are horizontal rules here, since the x axis is the
    # variant and a reference has no variant. Reference and family names share
    # one collision-aware label lane in the right margin.
    right_labels: list[tuple[float, str, str, str]] = []
    for fn, m in sorted(refs, key=lambda t: -t[1]):
        colour, _label = REF_STYLE[fn]
        label = ref_label(key, fn)
        yv = py(m)
        dash = "" if fn == "target_sample" else ' stroke-dasharray="3 3"'
        out.append(f'<line x1="{left}" y1="{yv:.1f}" x2="{left + pw:.1f}" y2="{yv:.1f}" '
                   f'stroke="{colour}" stroke-width="1.8"{dash} opacity="0.9"/>')
        right_labels.append((yv + 4, label, "ref-lbl", colour))
    for fam, colour, pts in lines:
        d = " ".join(f"{'M' if k == 0 else 'L'}{px(i):.1f},{py(v):.1f}" for k, (i, v) in enumerate(pts))
        out.append(f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="2.2"/>')
        for i, v in pts:
            out.append(f'<circle cx="{px(i):.1f}" cy="{py(v):.1f}" r="4.5" fill="{colour}" stroke="#fcfcfb" stroke-width="1.8"/>')
        li, lv = pts[-1]
        right_labels.append((py(lv) + 4, fam, "lbl", colour))

    ordered = sorted(right_labels, key=lambda item: item[0])
    label_ys: list[float] = []
    for desired, _label, _cls, _colour in ordered:
        label_ys.append(max(top + 10, desired, (label_ys[-1] + 14) if label_ys else top + 10))
    if label_ys and label_ys[-1] > top + ph - 2:
        shift = label_ys[-1] - (top + ph - 2)
        label_ys = [y - shift for y in label_ys]
        for i in range(len(label_ys) - 2, -1, -1):
            label_ys[i] = min(label_ys[i], label_ys[i + 1] - 14)
    for (desired, label, cls, colour), ly in zip(ordered, label_ys):
        out.append(f'<path d="M{left + pw:.1f},{desired - 4:.1f} L{left + pw + 7:.1f},{ly - 4:.1f}" '
                   f'stroke="{colour}" stroke-width="1" opacity="0.55"/>')
        out.append(b1.text(left + pw + 10, ly, label, cls))
    out.append(b1.text(left + pw / 2, top + ph + 38, "training variant", "ax-title", "middle"))
    out.append(f'<text class="ax-title" transform="translate(14,{top + ph / 2}) rotate(-90)" text-anchor="middle">{b1.esc(axis_label)}</text>')
    foot = (f"{'Lower' if better == 'lower' else 'Higher'} is better. Parallel lines indicate a variant "
            f"effect that is consistent across families; crossing lines indicate that the best variant "
            f"depends on the family, in which case the variant cannot be chosen once for all of them. "
            f"Intervals are omitted so that three families remain legible and are given in the tables.")
    blk, y_end = b1.text_block(0, top + ph + 52, foot, "fig-note", width_px=width - 8)
    out.append(blk); out.append("</svg>")
    return "\n".join(out).replace(f'viewBox="0 0 {width} {top + ph + bot}"',
                                  f'viewBox="0 0 {width} {max(top + ph + bot, y_end + 14):.0f}"', 1)


def overview_heatmap(rep, width=760):
    """Every column against every headline measure, normalised to the oracle.

    Each cell is the column's distance from the oracle on that measure, divided by
    the largest such distance in the row, so the rows are comparable despite being
    on different scales. This is a summary rather than evidence: the underlying
    values are in the tables, and no interval is shown, so a difference should not
    be read from a shade.
    """
    keys = ["trm.trm", "trm.rank_i2", "distributional.centroid_distance",
            "classifier.calibration_gap", "structural.feature_coverage",
            "semantic.source_similarity", "degeneracy.length_ratio_vs_real",
            "degeneracy.distinct_2"]
    cols = [(fn, lbl, colour) for _f, colour, cs in FAMILIES for fn, lbl in cs]
    cols += [(fn, REF_STYLE[fn][1].split(" (")[0], REF_STYLE[fn][0]) for fn in REFERENCES]
    rows = []
    for k in keys:
        oracle = ref_mean(k, "target_sample")
        if oracle is None:
            oracle = (METRICS.get(k) or {}).get("ideal")
        if oracle is None:
            continue
        vals = {fn: (mean(rep, fn, k) if fn not in REFERENCES else ref_mean(k, fn))
                for fn, _l, _c in cols}
        devs = {fn: (abs(v - oracle) if v is not None else None) for fn, v in vals.items()}
        mx = max([d for d in devs.values() if d is not None] or [1]) or 1
        rows.append((k, oracle, {fn: (None if d is None else d / mx) for fn, d in devs.items()}, vals))
    if not rows:
        return ""
    left, top = 268, 108
    cw = (width - left - 16) / len(cols)
    rh = 26
    height = top + len(rows) * rh + 96
    title = "Every column against every headline measure"
    sub = ("Shade is the column's distance from the authentic-post oracle on that measure, scaled "
           "within the row. Pale is closer to the oracle. Rows use different scales, so shades are "
           "comparable within a row and not between rows.")
    out = [b1.svg_open(width, height, title, sub)]
    t, y = b1.text_block(0, 24, title, "fig-title", width_px=width - 8); out.append(t)
    sb, y = b1.text_block(0, y + 4, sub, "fig-sub", width_px=width - 8); out.append(sb)
    for j, (fn, lbl, colour) in enumerate(cols):
        x = left + j * cw + cw / 2
        out.append(f'<text class="tiny" transform="translate({x:.1f},{top - 8:.1f}) rotate(-55)" text-anchor="start" fill="{colour}">{b1.esc(lbl)}</text>')
    for i, (k, oracle, norm, vals) in enumerate(rows):
        yy = top + i * rh
        out.append(b1.text(left - 10, yy + rh * 0.66, mname(k, short=True), "lbl", "end"))
        for j, (fn, _lbl, colour) in enumerate(cols):
            d = norm.get(fn)
            x = left + j * cw
            if d is None:
                out.append(f'<rect x="{x + 1:.1f}" y="{yy + 2:.1f}" width="{cw - 2:.1f}" height="{rh - 4:.1f}" fill="#f4f2ec"/>')
                continue
            op = 0.10 + 0.80 * d
            out.append(f'<rect x="{x + 1:.1f}" y="{yy + 2:.1f}" width="{cw - 2:.1f}" height="{rh - 4:.1f}" fill="{colour}" opacity="{op:.2f}"/>')
    yb = top + len(rows) * rh
    key_blk, _ = b1.text_block(
        0, yb + 20,
        "pale = closer to the authentic-post oracle · grey = not scored on this measure",
        "fig-note", width_px=width - 8)
    out.append(key_blk)
    foot = ("This figure is an overview and not evidence. It shows no interval, and the normalisation "
            "is per row, so a darker cell in one row is not a larger shortfall than a lighter cell in "
            "another. Where the oracle produces no value the target for the measure is used instead.")
    blk, y_end = b1.text_block(0, yb + 40, foot, "fig-note", width_px=width - 8)
    out.append(blk); out.append("</svg>")
    return "\n".join(out).replace(f'viewBox="0 0 {width} {height}"',
                                  f'viewBox="0 0 {width} {max(height, y_end + 14):.0f}"', 1)


def metric_glossary(groups):
    """Code-grounded definitions and finite-sample equations for report rows."""
    sections = []
    for title, _subtitle, keys in groups:
        items = []
        for k in keys:
            m = METRICS.get(k)
            if not m:
                continue
            d = {1: "higher is better", -1: "lower is better", 0: "diagnostic"}[m["dirn"]]
            tgt = "" if m.get("ideal") is None else f" Target {m['ideal']:.4g}."
            equation = METRIC_EQUATIONS.get(k, "")
            symbols = METRIC_SYMBOLS.get(k, "")
            if not equation or not symbols:
                raise RuntimeError(f"{k}: metric glossary requires both an equation and symbol key")
            eq_html = (
                f'<div class="metric-equation" aria-label="Equation for {b1.esc(m["name"])}">'
                f"\\[{equation}\\]</div>"
            )
            items.append(
                f"<dt>{b1.esc(m['name'])} <span class='dirn'>({d}){b1.esc(tgt)}</span></dt>"
                f'<dd>{eq_html}<p class="equation-key"><strong>Symbols and functions.</strong> '
                f"{symbols}</p><p>{b1.esc(m['defn'])}</p></dd>"
            )
        sections.append(
            f'<section class="metric-group"><h3>{b1.esc(title)}</h3>'
            f'<dl class="gloss">{"".join(items)}</dl></section>'
        )
    notation = r"""
<aside class="metric-notation">
<h3>Shared notation and aggregation</h3>
<p>Within bilateral cell \(c\), \(G_c=\{g_i\}\), \(R_c=\{r_j\}\), and
\(S_c=\{s_i\}\) denote the generated, authentic Reddit, and authentic LinkedIn
pools. A cell-level equation suppresses subscript \(c\) and writes \(G,R,S\).
\(|A|\) is the cardinality of set or multiset \(A\); \(e(x)\) is the report's
Gemma-512 embedding of text \(x\).</p>
<p>\(\mathbf 1\{B\}\) equals one when proposition \(B\) is true and zero
otherwise; \(\mathbb E\) is an explicitly identified finite average, not a
model expectation. \(\sum\) is a finite sum; \(\min\) and \(\max\) select the
smaller and larger real argument. For a scalar \(z\), \(|z|\) is absolute
value, whereas \(|A|\) is set or multiset cardinality. \(\lVert z\rVert_2\),
\(z^{\mathsf T}\), \(\cap\), and \(\cup\) denote Euclidean norm, transpose,
intersection, and union. Metric-local
functions—including histograms, classifiers, content-word extraction, and
character counting—are defined immediately below their equations.</p>
<div class="metric-equation">\[
\widehat\phi=\frac{1}{|\mathcal C_\phi|}\sum_{c\in\mathcal C_\phi}\phi_c
\]</div>
<p>Here \(\phi_c\) is metric \(\phi\)'s value in cell \(c\), and
\(\mathcal C_\phi\) is the set of finite-scored cells shared by every compared
system for that metric family. Thus \(\widehat\phi\) is the displayed unweighted
cell mean. The 95% interval resamples cells, not posts, so a large cell does not
receive more weight than a small one.</p>
</aside>
"""
    return notation + f'<div class="metric-groups">{"".join(sections)}</div>'


def hyperparameters(rep_run: Path):
    """Exact settings of the selected configuration, read from the run records.

    Every value here is taken from the `transfer_fn.*.json` written beside the
    outputs, so it is the configuration that actually produced the scored
    generations rather than a description of the intended one.
    """
    def cfg(fn):
        p = rep_run / f"transfer_fn.{fn}.json"
        return json.loads(p.read_text()) if p.exists() else {}

    rows = []
    for fam, colour, cols in FAMILIES[1:]:
        for fn, lbl in cols:
            d = cfg(fn)
            if not d:
                continue
            if fam == "LoRA":
                tc = d.get("training_config", {})
                setting = (f"r={tc.get('r')}, α={tc.get('alpha')}, dropout={tc.get('dropout')}, "
                           f"lr={tc.get('lr')}, modules={len(tc.get('modules', []))} projections")
                sel = f"epoch {d.get('best_epoch')}, dev NLL {d.get('best_dev_nll', float('nan')):.4f}"
            elif fam == "soft prompt":
                tc = (d.get("training") or {}).get("config", {})
                tr = d.get("training") or {}
                setting = (f"{tc.get('n_virtual_tokens')} virtual tokens, lr={tc.get('lr')}, "
                           f"init={tc.get('init')}, weight decay={tc.get('weight_decay')}")
                sel = f"epoch {tr.get('epoch')}, dev NLL {tr.get('dev_loss', float('nan')):.4f}"
            else:
                setting = (f"layer {', '.join(map(str, d.get('layers', [])))}, α={d.get('alpha')}, "
                           f"scope={d.get('scope')}")
                sd = d.get("selected_on_dev") or {}
                sel = (f"centroid cos {sd.get('centroid_cos', float('nan')):.4f} "
                       f"vs {sd.get('baseline_unsteered_centroid_cos', float('nan')):.4f} unsteered")
            gen = (f"{d.get('n_samples')} draws, T={d.get('temperature')}, "
                   f"top-p={d.get('top_p', '—')}, max_new_tokens={d.get('max_new_tokens')}")
            rows.append(
                f"<tr><th scope='row' style='color:{colour}'>{b1.esc(fam)}</th>"
                f"<td>{b1.esc(lbl)}</td><td>{b1.esc(setting)}</td>"
                f"<td>{b1.esc(sel)}</td><td>{b1.esc(gen)}</td></tr>"
            )
    return f"""<figure class="tablefig">
<figcaption><strong>Selected configurations, as run</strong><br>
<span class="sub">Read from the run records written beside the outputs, so these are the settings that
produced the scored generations. Selection was on the validation split in every case.</span></figcaption>
<div class="tscroll"><table class="res hp">
<thead><tr><th>family</th><th>variant</th><th>fitted setting</th><th>selected</th><th>decoding</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>
</figure>"""


def master_table(rep):
    """Every scored row, for readers who want the quantity this page did not discuss.

    The curated tables above select rows to make an argument. This one selects
    nothing: it prints every row the evaluation produced, in the order the metric
    modules registered them, with the cell count each was averaged over. Rows whose
    interval is degenerate or whose cell count is very low are visible here as such
    rather than being silently omitted.
    """
    cols = [(fn, lbl) for _f, _c, cs in FAMILIES for fn, lbl in cs]
    keys = sorted({k for d in rep["table"].values() for k in d})
    head_span = "".join(
        f'<th colspan="{len(cs)}" class="fam" style="color:{c}">{b1.esc(f)}</th>'
        for f, c, cs in FAMILIES
    )
    head = "".join(f"<th>{b1.esc(lbl)}</th>" for _fn, lbl in cols)
    refhead = "".join(f"<th>{b1.esc(REF_STYLE[r][1].split(' · ')[0])}</th>" for r in REFERENCES)
    body = []
    for key in keys:
        # `*.overall.*` rows are pooled scalars and carry no cell count.
        ns = [n for fn, _ in cols if (e := entry(rep, fn, key)) and (n := e.get("n_cells"))]
        if not any(entry(rep, fn, key) for fn, _ in cols):
            continue
        n_txt = "pooled" if not ns else (str(ns[0]) if len(set(ns)) == 1 else f"{min(ns)}–{max(ns)}")
        cells = []
        for fn, _ in cols:
            e = entry(rep, fn, key)
            cells.append("<td>—</td>" if e is None else f"<td>{e['mean']:.4g}</td>")
        for r in REFERENCES:
            e = ref_entry(key, r)
            cells.append("<td class='ref'>—</td>" if e is None else f"<td class='ref'>{e['mean']:.4g}</td>")
        body.append(
            f"<tr><th scope='row'>{b1.esc(mname(key))}</th>"
            + "".join(cells)
            + f"<td class='ref'>{n_txt}</td><td class='key'><code>{b1.esc(key)}</code></td></tr>"
        )
    return f"""<figure class="tablefig">
<figcaption><strong>Master table: every scored quantity</strong><br>
<span class="sub">All {len(body)} rows produced by the evaluation, unfiltered, with the number of
cells each was averaged over in the final column. Rows discussed above appear here unchanged; rows
not discussed are included so that a reader can check a quantity this page did not select.
<code>*.overall.*</code> rows are pooled counts rather than per-cell means and carry no interval.
<code>distributional.mmd2</code> is the unbiased U-statistic and is not constrained to be
non-negative; see the limitations.</span></figcaption>
<div class="tscroll"><table class="res master">
<thead>
<tr><td></td>{head_span}<th colspan="{len(REFERENCES)}" class="fam">reference</th><th class="fam">cells</th><th class="fam">identifier</th></tr>
<tr><td></td>{head}{refhead}<th></th><th></th></tr>
</thead>
<tbody>{"".join(body)}</tbody>
</table></div>
</figure>"""


def resolution(rep):
    """Which stated comparisons survive the interval test."""
    pairs = [
        ("soft_prompt_sp_target_lm_paired", "llm_rewrite_claude", "trm.trm",
         "soft prompt matches the target pool better than Claude zero-shot"),
        ("lora_lora_target_lm_paired", "llm_rewrite_claude", "trm.trm",
         "LoRA matches the target pool better than Claude zero-shot"),
        ("soft_prompt_sp_target_lm_paired", "aspect_prompt_claude", "trm.trm",
         "soft prompt matches the target pool better than aspect-aware prompting"),
        ("lora_lora_target_lm_paired", "soft_prompt_sp_target_lm_paired", "trm.trm",
         "LoRA matches the target pool better than the soft prompt"),
        ("steering_steer_target_lm_paired", "llm_rewrite_claude", "trm.trm",
         "steering matches the target pool better than Claude zero-shot"),
        ("aspect_prompt_claude", "llm_rewrite_claude", "trm.trm",
         "aspect-aware prompting matches the target pool better than zero-shot"),
        ("soft_prompt_sp_target_lm_aspect", "soft_prompt_sp_target_lm", "trm.trm",
         "aspect conditioning improves the soft prompt over topic-only"),
    ]
    out = []
    for a, bb, key, text_ in pairs:
        ea, eb = entry(rep, a, key), entry(rep, bb, key)
        if not ea or not eb:
            continue
        direction = ea.get("direction", -1)
        if direction < 0:
            supported = ea["ci_high"] < eb["ci_low"]
        else:
            supported = ea["ci_low"] > eb["ci_high"]
        verdict = "supported" if supported else "not resolved"
        cls = "ok" if supported else "no"
        out.append(
            f"<tr><td>{b1.esc(text_)}</td>"
            f"<td class='num'>{ea['mean']:.3f} [{ea['ci_low']:.2f}, {ea['ci_high']:.2f}]</td>"
            f"<td class='num'>{eb['mean']:.3f} [{eb['ci_low']:.2f}, {eb['ci_high']:.2f}]</td>"
            f"<td class='{cls}'>{verdict}</td></tr>"
        )
    n_ok = sum(1 for r in out if ">supported<" in r)
    return f"""<figure class="tablefig">
<figcaption><strong>Which comparisons the intervals actually separate</strong><br>
<span class="sub">A difference is a finding only when the two intervals do not overlap. This table is
generated by running that test, not written by hand.</span></figcaption>
<div class="tscroll"><table class="res claims">
<thead><tr><th>claim</th><th>first column</th><th>second column</th><th>verdict</th></tr></thead>
<tbody>{"".join(out)}</tbody></table></div>
<p class="tnote">{n_ok} of {len(out)} stated comparisons are supported.</p>
</figure>""", n_ok, len(out)


def metric_resolution(rep: dict, key: str, title: str) -> str:
    """Apply the interval rule to useful family and conditioning comparisons."""
    pairs = [
        ("soft_prompt_sp_target_lm_paired", "llm_rewrite_claude",
         "retrieved-source soft prompt vs Claude zero-shot"),
        ("lora_lora_target_lm_paired", "llm_rewrite_claude",
         "retrieved-source LoRA vs Claude zero-shot"),
        ("steering_steer_target_lm_paired", "llm_rewrite_claude",
         "retrieved-source steering vs Claude zero-shot"),
        ("aspect_prompt_claude", "llm_rewrite_claude",
         "aspect-aware prompting vs Claude zero-shot"),
        ("soft_prompt_sp_target_lm_aspect", "soft_prompt_sp_target_lm",
         "aspect soft prompt vs topic-only soft prompt"),
        ("lora_lora_target_lm_aspect", "lora_lora_target_lm",
         "aspect LoRA vs topic-only LoRA"),
    ]
    rows = []
    for first, second, label in pairs:
        a, z = entry(rep, first, key), entry(rep, second, key)
        if not a or not z:
            continue
        direction = a.get("direction", METRICS.get(key, {}).get("dirn", -1))
        first_better = (
            a["ci_high"] < z["ci_low"] if direction < 0
            else a["ci_low"] > z["ci_high"]
        )
        second_better = (
            z["ci_high"] < a["ci_low"] if direction < 0
            else z["ci_low"] > a["ci_high"]
        )
        verdict = (
            "first better" if first_better else
            "second better" if second_better else
            "not resolved"
        )
        rows.append(
            f"<tr><td>{b1.esc(label)}</td>"
            f"<td class='num'>{a['mean']:.3f} [{a['ci_low']:.2f}, {a['ci_high']:.2f}]</td>"
            f"<td class='num'>{z['mean']:.3f} [{z['ci_low']:.2f}, {z['ci_high']:.2f}]</td>"
            f"<td class='{'ok' if first_better else 'warn' if second_better else 'no'}'>"
            f"{verdict}</td></tr>"
        )
    if not rows:
        return ""
    return f"""<figure class="tablefig compact-claims">
<figcaption><strong>{b1.esc(title)}</strong><br>
<span class="sub">The verdict is generated from interval separation. Point estimates alone do not
support a finding.</span></figcaption><div class="tscroll"><table class="res claims">
<thead><tr><th>comparison</th><th>first</th><th>second</th><th>verdict</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div></figure>"""


def aspect_results(rep: dict, full: dict) -> str:
    """Render the frozen-rater aspect comparison on a systems-only base."""
    key = "aspect.prevalence_jsd"
    if not rep or not any(entry(rep, fn, key) for fn in TRAINED + PROMPTED):
        return ""
    cells = int((rep.get("common_cells") or {}).get("aspect", 0))
    rows = []
    ordered = [fn for _fam, _colour, cols in FAMILIES for fn, _label in cols]
    labels = {fn: (fam, label) for fam, _colour, cols in FAMILIES for fn, label in cols}
    for fn in ordered:
        e = entry(rep, fn, key)
        if not e:
            continue
        fam, label = labels[fn]
        rows.append(
            f"<tr><th scope='row'>{b1.esc(fam)}</th><td>{b1.esc(label)}</td>"
            f"<td class='num'>{e['mean']:.3f}</td>"
            f"<td class='num'>[{e['ci_low']:.3f}, {e['ci_high']:.3f}]</td>"
            f"<td class='num'>{e.get('n_cells', cells)}</td></tr>"
        )
    for fn, label in (("identity", "Source-copy floor"),
                      ("shuffle_control", "Wrong-topic control")):
        e = entry(rep, fn, key)
        if e:
            rows.append(
                f"<tr class='ref'><th scope='row'>reference</th><td>{label}</td>"
                f"<td class='num'>{e['mean']:.3f}</td>"
                f"<td class='num'>[{e['ci_low']:.3f}, {e['ci_high']:.3f}]</td>"
                f"<td class='num'>{e.get('n_cells', cells)}</td></tr>"
            )
    oracle = entry(full, "target_sample", key)
    if oracle:
        rows.append(
            "<tr class='ref'><th scope='row'>reference</th><td>Authentic-post oracle</td>"
            f"<td class='num'>{oracle['mean']:.3f}</td>"
            f"<td class='num'>[{oracle['ci_low']:.3f}, {oracle['ci_high']:.3f}]</td>"
            f"<td class='num'>{oracle.get('n_cells', 0)}</td></tr>"
        )

    comparisons = [
        ("soft_prompt_sp_target_lm", "llm_rewrite_claude",
         "topic-only soft prompt vs Claude zero-shot"),
        ("soft_prompt_sp_target_lm_paired", "llm_rewrite_claude",
         "retrieved-source soft prompt vs Claude zero-shot"),
        ("lora_lora_target_lm_paired", "llm_rewrite_claude",
         "retrieved-source LoRA vs Claude zero-shot"),
        ("soft_prompt_sp_target_lm_aspect", "soft_prompt_sp_target_lm",
         "aspect soft prompt vs topic-only soft prompt"),
        ("aspect_prompt_claude", "llm_rewrite_claude",
         "aspect-aware prompting vs Claude zero-shot"),
    ]
    claims = []
    for first, second, label in comparisons:
        a, z = entry(rep, first, key), entry(rep, second, key)
        if not a or not z:
            continue
        supported = a["ci_high"] < z["ci_low"]
        claims.append(
            f"<tr><td>{b1.esc(label)}</td><td class='num'>{a['mean']:.3f}</td>"
            f"<td class='num'>{z['mean']:.3f}</td>"
            f"<td class='{'ok' if supported else 'no'}'>"
            f"{'supported' if supported else 'not resolved'}</td></tr>"
        )
    return f"""
<h2 id="aspect-result"><span class="section-no">11</span> Aspect-distribution matching is measurable, but only on {cells} cells</h2>
<p>The frozen Study 3 rater labels the presence of each cell's evaluative lenses in authentic and
generated posts. The headline value is the mean Jensen–Shannon divergence between generated and
authentic target-platform aspect prevalence. Lower is better. It preserves absence as information
and does not normalise away differences in the number of aspects raised per post.</p>
<figure class="tablefig"><figcaption><strong>Aspect-prevalence divergence</strong><br>
<span class="sub">Every system is averaged over the same {cells} cells. The authentic-post oracle
is shown on its own smaller base. Values and intervals are read from the aspect report.</span></figcaption>
<div class="tscroll"><table class="res"><thead><tr><th>family</th><th>configuration</th>
<th>JSD</th><th>interval</th><th>cells</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></figure>
<figure class="tablefig"><figcaption><strong>Predeclared aspect comparisons</strong><br>
<span class="sub">A comparison is supported only when its intervals do not overlap.</span></figcaption>
<div class="tscroll"><table class="res claims"><thead><tr><th>comparison</th><th>first</th>
<th>second</th><th>verdict</th></tr></thead><tbody>{''.join(claims)}</tbody></table></div></figure>
<p>The target-resampling diagnostic is lower than every system, and the wrong-topic control has the
highest point estimate. The metric therefore passes its directional baseline checks on this base.
These results remain exploratory because only {cells} cells satisfy the vocabulary and pool-size
requirements and the automated presence rater has not yet been compared with human labels.</p>
"""


def extended_metric_results(rep: dict) -> str:
    """Figures that decompose distribution match, style and content retention."""
    cols = PROMPTED + TRAINED

    def best(key: str, lower: bool = True) -> tuple[str, float]:
        valid = [(fn, mean(rep, fn, key)) for fn in cols if mean(rep, fn, key) is not None]
        fn, value = (min(valid, key=lambda x: x[1]) if lower else
                     max(valid, key=lambda x: x[1]))
        return fn, value

    labels = {fn: (fam, label) for fam, _colour, rows in FAMILIES for fn, label in rows}
    centroid_fn, centroid_value = best("distributional.centroid_distance")
    topic_fn, topic_value = best("distributional.topic_jsd")
    style_fn, style_value = best("structural.mean_feature_jsd")
    platform_fn, platform_value = best("classifier.stylometric_calibration_gap")
    retain_fn, retain_value = best("semantic.content_word_retention", lower=False)

    def name(fn: str) -> str:
        fam, label = labels[fn]
        return f"{fam}, {label}"

    def relation(first: str, second: str, key: str, lower: bool) -> str:
        a, z = entry(rep, first, key), entry(rep, second, key)
        if not a or not z:
            return "unavailable"
        if lower:
            if a["ci_high"] < z["ci_low"]:
                return "first"
            if z["ci_high"] < a["ci_low"]:
                return "second"
        else:
            if a["ci_low"] > z["ci_high"]:
                return "first"
            if z["ci_low"] > a["ci_high"]:
                return "second"
        return "unresolved"

    centroid_relation = relation(
        "lora_lora_target_lm_paired", "llm_rewrite_claude",
        "distributional.centroid_distance", True,
    )
    topic_relation = relation(
        "lora_lora_target_lm_paired", "llm_rewrite_claude",
        "distributional.topic_jsd", True,
    )
    style_relation = relation(
        "lora_lora_target_lm_paired", "llm_rewrite_claude",
        "structural.mean_feature_jsd", True,
    )
    platform_relation = relation(
        "lora_lora_target_lm_paired", "llm_rewrite_claude",
        "classifier.stylometric_calibration_gap", True,
    )
    retain_relation = relation(
        "lora_lora_target_lm_paired", "llm_rewrite_claude",
        "semantic.content_word_retention", False,
    )

    relation_text = {
        "first": "The retrieved-source LoRA interval is wholly better than Claude zero-shot.",
        "second": "The Claude zero-shot interval is wholly better than retrieved-source LoRA.",
        "unresolved": "Retrieved-source LoRA and Claude zero-shot are not resolved on this measure.",
        "unavailable": "The predeclared comparison is unavailable on this measure.",
    }

    return f"""
<h2 id="centroid-result"><span class="section-no">06</span> Pool location and full distribution shape do not give the same ordering</h2>
{dotplot(rep, "distributional.centroid_distance",
         "Distance between generated and authentic pool centres",
         "This measure compares only the mean embedding of each pool. It can reward the correct centre even when the generated pool has the wrong spread or internal shape.",
         axis_label="Centroid cosine distance", better="lower")}
<p>The lowest centroid-distance point is {b1.esc(name(centroid_fn))} at
{centroid_value:.3f}. This figure should be read beside Triangle-Rank, not as a substitute for it.
A configuration can place its average output near the authentic average while producing candidates
that are too similar to one another or occupy the wrong parts of the target distribution.</p>
<p><strong>Interval result.</strong> {relation_text[centroid_relation]}</p>
{metric_resolution(rep, "distributional.centroid_distance", "Resolved centroid-distance comparisons")}

<h2 id="topic-result"><span class="section-no">07</span> Topic-mixture matching separates target style from target content</h2>
{dotplot(rep, "distributional.topic_jsd",
         "Divergence from the authentic target topic mixture",
         "Authentic posts define one shared clustering of the embedding space. The score compares how generated and authentic pools distribute their mass across those clusters.",
         axis_label="Topic-cluster Jensen–Shannon divergence", better="lower")}
<p>The lowest point estimate is {b1.esc(name(topic_fn))} at {topic_value:.3f}. The wrong-topic
control is especially important here: it already has authentic target-platform form, so its poor
position shows that the measure responds to requested content rather than rewarding platform cues
alone. This is still a coarse cluster-based content measure and does not identify which evaluative
aspect moved.</p>
<p><strong>Interval result.</strong> {relation_text[topic_relation]}</p>
{metric_resolution(rep, "distributional.topic_jsd", "Resolved topic-mixture comparisons")}

<h2 id="writing-result"><span class="section-no">08</span> Low-level writing habits tell a different story from semantic embeddings</h2>
{dotplot(rep, "structural.mean_feature_jsd",
         "Divergence across twenty-one observable writing habits",
         "Each feature is binned on shared edges spanning generated, authentic target and source posts. The figure averages the per-feature divergences.",
         axis_label="Mean writing-habit Jensen–Shannon divergence", better="lower")}
<p>The lowest point estimate is {b1.esc(name(style_fn))} at {style_value:.3f}. The features cover
length, punctuation, pronouns, sentence form and formatting. They are deliberately transparent and
low-level. A low value means that the marginal feature distributions look target-like; it does not
show that an output is persuasive, natural, or faithful to its source.</p>
<p><strong>Interval result.</strong> {relation_text[style_relation]}</p>
{metric_resolution(rep, "structural.mean_feature_jsd", "Resolved writing-habit comparisons")}

<h2 id="platform-result"><span class="section-no">09</span> Platform resemblance without topic vocabulary remains incomplete</h2>
{dotplot(rep, "classifier.stylometric_calibration_gap",
         "Gap from the authentic target rate under a writing-habit classifier",
         "The classifier sees only the transparent surface features, not lexical topic features. Zero means the generated target-classification rate matches authentic posts.",
         axis_label="Stylometric calibration gap", better="lower")}
<p>The smallest point estimate is {b1.esc(name(platform_fn))} at {platform_value:.3f}. This is a
calibration objective, not a request to maximise the target label: authentic posts do not achieve a
perfect classification rate, so saturation would itself be miscalibrated. Agreement with the
semantic figures indicates a broad shift; disagreement identifies a method that moves content
without reproducing observable platform form, or vice versa.</p>
<p><strong>Interval result.</strong> {relation_text[platform_relation]}</p>
{metric_resolution(rep, "classifier.stylometric_calibration_gap", "Resolved platform-calibration comparisons")}

<h2 id="retention-result"><span class="section-no">10</span> Specific source terms expose content loss hidden by fluent rewrites</h2>
{dotplot(rep, "semantic.content_word_retention",
         "Share of the source post's content words retained",
         "For each task, the score measures how many unique non-stopword source terms longer than three characters reappear in the output.",
         axis_label="Content-word retention", better="higher")}
<p>The highest non-reference point estimate is {b1.esc(name(retain_fn))} at {retain_value:.3f}.
Unlike embedding similarity, this measure cannot give credit for a fluent paraphrase that discards
the source's concrete terminology. It is intentionally strict: valid synonyms count as losses, so
the figure is evidence about lexical detail retention rather than complete semantic fidelity.</p>
<p><strong>Interval result.</strong> {relation_text[retain_relation]}</p>
{metric_resolution(rep, "semantic.content_word_retention", "Resolved content-retention comparisons")}
"""


def reverse_results(rep: dict, full: dict) -> str:
    """Build the reverse-direction page from its independent comparison base."""
    if not rep:
        return ""
    reverse_key = {
        "lora_lora_target_lm": "lora_s3r_lora_target_lm",
        "lora_lora_target_lm_paired": "lora_s3r_lora_target_lm_paired",
        "lora_lora_target_lm_aspect": "lora_s3r_lora_target_lm_aspect",
        "soft_prompt_sp_target_lm": "soft_prompt_s3r_sp_target_lm",
        "soft_prompt_sp_target_lm_paired": "soft_prompt_s3r_sp_target_lm_paired",
        "soft_prompt_sp_target_lm_aspect": "soft_prompt_s3r_sp_target_lm_aspect",
        "steering_steer_target_lm": "steering_s3r_steer_target_lm",
        "steering_steer_target_lm_paired": "steering_s3r_steer_target_lm_paired",
        "steering_steer_target_lm_aspect": "steering_s3r_steer_target_lm_aspect",
    }

    def rev_entry(report: dict, fn: str, metric: str) -> dict | None:
        return entry(report, reverse_key.get(fn, fn), metric)

    canonical = dict(rep)
    canonical["table"] = {
        next((old for old, new in reverse_key.items() if new == fn), fn): values
        for fn, values in rep.get("table", {}).items()
    }
    canonical_full = dict(full)
    canonical_full["table"] = {
        next((old for old, new in reverse_key.items() if new == fn), fn): values
        for fn, values in full.get("table", {}).items()
    }
    global REF_REPORT_FULL, REF_REPORT_SYSTEMS
    saved_full, saved_systems = REF_REPORT_FULL, REF_REPORT_SYSTEMS
    REF_REPORT_FULL, REF_REPORT_SYSTEMS = canonical_full, canonical
    try:
        reverse_figures = f"""
<h2>Distribution match on the reverse task</h2>
{dotplot(canonical, "trm.trm",
         "Distance from the pool of authentic LinkedIn posts",
         "The reverse task uses its own target pool and reference anchors. Zero is the matched-distribution ideal.",
         axis_label="TRM score", better="lower")}
{metric_resolution(canonical, "trm.trm", "Resolved reverse Triangle-Rank comparisons")}

<h2>Source preservation on the reverse task</h2>
{dotplot(canonical, "semantic.source_similarity",
         "Similarity to the supplied Reddit source post",
         "Higher values retain more of the input in the primary embedding space; the unchanged-source reference is one by construction.",
         axis_label="Cosine similarity to source", better="higher")}
{metric_resolution(canonical, "semantic.source_similarity", "Resolved reverse source-similarity comparisons")}

<h2>Writing-habit match on the reverse task</h2>
{dotplot(canonical, "structural.mean_feature_jsd",
         "Divergence from authentic LinkedIn writing habits",
         "The same twenty-one transparent surface features are evaluated with LinkedIn as the target.",
         axis_label="Mean writing-habit Jensen–Shannon divergence", better="lower")}
{metric_resolution(canonical, "structural.mean_feature_jsd", "Resolved reverse writing-habit comparisons")}

<h2>Topic-mixture match on the reverse task</h2>
{dotplot(canonical, "distributional.topic_jsd",
         "Divergence from the authentic LinkedIn topic mixture",
         "The wrong-topic control tests whether authentic LinkedIn form can succeed without the requested content.",
         axis_label="Topic-cluster Jensen–Shannon divergence", better="lower")}
{metric_resolution(canonical, "distributional.topic_jsd", "Resolved reverse topic-mixture comparisons")}
"""
    finally:
        REF_REPORT_FULL, REF_REPORT_SYSTEMS = saved_full, saved_systems
    metrics = [
        ("trm.trm", "Triangle-Rank Metric", "lower"),
        ("semantic.source_similarity", "Similarity to source", "higher"),
        ("classifier.stylometric_calibration_gap", "Writing-habit calibration gap", "lower"),
        ("degeneracy.length_ratio_vs_real", "Length ratio", "near one"),
    ]
    labels = {fn: (fam, label) for fam, _colour, cols in FAMILIES for fn, label in cols}
    ordered = [fn for _fam, _colour, cols in FAMILIES for fn, _label in cols]
    body = []
    for fn in ordered:
        vals = []
        for key, _name, _better in metrics:
            e = rev_entry(rep, fn, key)
            vals.append(
                "<td>—</td>" if not e else
                f"<td class='num'>{e['mean']:.3f}<sup>{e.get('n_cells', '')}</sup></td>"
            )
        fam, label = labels[fn]
        body.append(
            f"<tr><th scope='row'>{b1.esc(fam)}</th><td>{b1.esc(label)}</td>"
            + "".join(vals) + "</tr>"
        )
    for fn, label in (("identity", "Source-copy floor"),
                      ("shuffle_control", "Wrong-topic control")):
        vals = []
        for key, _name, _better in metrics:
            e = entry(rep, fn, key)
            vals.append("<td>—</td>" if not e else
                        f"<td class='num'>{e['mean']:.3f}<sup>{e.get('n_cells', '')}</sup></td>")
        body.append(f"<tr class='ref'><th scope='row'>reference</th><td>{label}</td>"
                    + "".join(vals) + "</tr>")
    vals = []
    for key, _name, _better in metrics:
        e = entry(full, "target_sample", key)
        vals.append("<td>—</td>" if not e else
                    f"<td class='num'>{e['mean']:.3f}<sup>{e.get('n_cells', '')}</sup></td>")
    body.append("<tr class='ref'><th scope='row'>reference</th><td>Authentic-post oracle</td>"
                + "".join(vals) + "</tr>")

    pairs = [
        ("soft_prompt_sp_target_lm_paired", "llm_rewrite_claude",
         "soft prompt vs Claude zero-shot"),
        ("lora_lora_target_lm_paired", "llm_rewrite_claude",
         "LoRA vs Claude zero-shot"),
        ("steering_steer_target_lm_paired", "llm_rewrite_claude",
         "steering vs Claude zero-shot"),
        ("aspect_prompt_claude", "llm_rewrite_claude",
         "aspect-aware vs Claude zero-shot"),
        ("soft_prompt_sp_target_lm_aspect", "soft_prompt_sp_target_lm",
         "aspect soft prompt vs topic-only"),
        ("lora_lora_target_lm_paired", "soft_prompt_sp_target_lm_paired",
         "LoRA vs soft prompt"),
    ]
    claims = []
    for first, second, label in pairs:
        a, z = rev_entry(rep, first, "trm.trm"), rev_entry(rep, second, "trm.trm")
        if not a or not z:
            continue
        supported = a["ci_high"] < z["ci_low"]
        claims.append(
            f"<tr><td>{b1.esc(label)}</td>"
            f"<td class='num'>{a['mean']:.3f} [{a['ci_low']:.2f}, {a['ci_high']:.2f}]</td>"
            f"<td class='num'>{z['mean']:.3f} [{z['ci_low']:.2f}, {z['ci_high']:.2f}]</td>"
            f"<td class='{'ok' if supported else 'no'}'>"
            f"{'supported' if supported else 'not resolved'}</td></tr>"
        )
    trm_cells = int((rep.get("common_cells") or {}).get("trm", 0))
    space = rep.get("embedding_space") or {}
    space_label = f"{space.get('model', '')}, dim {space.get('dim', '')}".strip(", ")
    heads = "".join(
        f"<th>{b1.esc(name)}<br><small>{b1.esc(better)}</small></th>"
        for _key, name, better in metrics
    )
    return f"""
<div class="page-intro"><p class="section-label">Direction check</p>
<h1>Reddit to LinkedIn</h1>
<p class="lede">The full method grid is rerun with Reddit as the source and LinkedIn as the target.
It is scored independently because the target distribution, task pools and reference anchors differ
from the forward direction. Embedding-derived values use {b1.esc(space_label)}.</p></div>

<h2>All configurations on the reverse task</h2>
<figure class="tablefig"><figcaption><strong>Reverse-direction scorecard</strong><br>
<span class="sub">Superscripts give the cell base for each value. Every system column in a metric
uses the same cells; the authentic oracle retains its own smaller base.</span></figcaption>
<div class="tscroll"><table class="res"><thead><tr><th>family</th><th>configuration</th>{heads}</tr>
</thead><tbody>{''.join(body)}</tbody></table></div></figure>

{reverse_figures}

<h2>Which reverse comparisons are resolved</h2>
<figure class="tablefig"><figcaption><strong>Triangle-Rank comparisons</strong><br>
<span class="sub">Lower is better. A difference is supported only when the intervals do not overlap.
The shared comparison base is {trm_cells} cells.</span></figcaption>
<div class="tscroll"><table class="res claims"><thead><tr><th>comparison</th><th>first</th>
<th>second</th><th>verdict</th></tr></thead><tbody>{''.join(claims)}</tbody></table></div></figure>

<h2>How to read the direction comparison</h2>
<p>The forward and reverse values must not be treated as columns in one ranking. Each direction has
its own authentic target pool and reference anchors. Agreement in the winning family is evidence of
directional stability; a different ordering is evidence that the intervention depends on which
platform is being modelled. Neither permits subtracting the two Triangle-Rank values as an effect
size.</p>
"""


def embedding_stability(run: Path) -> str:
    """Compare the predeclared TRM claims across every completed space."""
    reports = []
    for path in sorted(run.glob(f"report.{SPLIT}*.json")):
        data = json.loads(path.read_text())
        if not (data.get("common_cells") or {}).get("trm"):
            continue
        slug = (data.get("embedding_space") or {}).get("slug")
        if slug and all((r.get("embedding_space") or {}).get("slug") != slug for r in reports):
            reports.append(data)
    if len(reports) < 2:
        return ""
    pairs = [
        ("soft_prompt_sp_target_lm_paired", "llm_rewrite_claude", "soft prompt vs Claude"),
        ("lora_lora_target_lm_paired", "llm_rewrite_claude", "LoRA vs Claude"),
        ("soft_prompt_sp_target_lm_paired", "aspect_prompt_claude", "soft prompt vs aspect-aware"),
        ("lora_lora_target_lm_paired", "soft_prompt_sp_target_lm_paired", "LoRA vs soft prompt"),
        ("soft_prompt_sp_target_lm_aspect", "soft_prompt_sp_target_lm", "aspect vs topic soft prompt"),
    ]
    rows = []
    supported_counts = [0] * len(pairs)
    for rep in reports:
        space = rep.get("embedding_space") or {}
        cells = int((rep.get("common_cells") or {}).get("trm", 0))
        vals = []
        for i, (first, second, _label) in enumerate(pairs):
            a, z = entry(rep, first, "trm.trm"), entry(rep, second, "trm.trm")
            supported = bool(a and z and a["ci_high"] < z["ci_low"])
            supported_counts[i] += int(supported)
            vals.append(f"<td class='{'ok' if supported else 'no'}'>"
                        f"{'supported' if supported else 'not resolved'}</td>")
        label = f"{space.get('model', space.get('backend', ''))}, dim {space.get('dim', '')}"
        rows.append(f"<tr><th scope='row'>{b1.esc(label)}</th><td>{cells}</td>{''.join(vals)}</tr>")
    head = "".join(f"<th>{b1.esc(label)}</th>" for _a, _z, label in pairs)
    summary = "; ".join(
        f"{b1.esc(label)}: {count} of {len(reports)}"
        for count, (_a, _z, label) in zip(supported_counts, pairs, strict=True)
    )
    cell_counts = sorted({int((r.get("common_cells") or {}).get("trm", 0)) for r in reports})
    cell_text = str(cell_counts[0]) if len(cell_counts) == 1 else f"{cell_counts[0]}–{cell_counts[-1]}"
    return f"""
<h2 id="embedding-stability"><span class="section-no">12</span> Some primary comparisons depend on the embedding space</h2>
<p>Every space scores the same outputs and uses a {cell_text}-cell comparison base. Only the
representation and its cosine distance change. The table reruns the interval-separation rule rather
than comparing point estimates.</p>
<figure class="tablefig"><figcaption><strong>Triangle-Rank findings across embedding spaces</strong><br>
<span class="sub">Each cell says whether the first configuration's interval lies wholly below the
second configuration's interval. Values from different rows are not numerically comparable.</span></figcaption>
<div class="tscroll"><table class="res claims"><thead><tr><th>space</th><th>cells</th>{head}</tr>
</thead><tbody>{''.join(rows)}</tbody></table></div>
<p class="tnote">Supported-space counts: {summary}.</p></figure>
<p>The fitted-method advantage is therefore not a universal property of the Triangle-Rank Metric.
The retrieved-source soft-prompt comparison is the most stable of the tested claims. The report
retains Gemma-512 as its primary, preregistered space and treats this table as a sensitivity check.</p>
"""


# --------------------------------------------------------------------------
# Qualitative
# --------------------------------------------------------------------------

def _jsonl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def qualitative(n_examples=3):
    """Source, authentic target, and one output per family, for several tasks.

    Tasks are chosen as the first eligible task from distinct cells, with both
    cells and task identifiers sorted. This is deterministic, cannot be tuned to
    a method, and avoids presenting three examples of one topic as if they showed
    the breadth of the test set.
    """
    cols = [("llm_rewrite_claude", "prompting", "Claude zero-shot"),
            ("steering_steer_target_lm_paired", "steering", "retrieved source"),
            ("soft_prompt_sp_target_lm_paired", "soft prompt", "retrieved source"),
            ("lora_lora_target_lm_paired", "LoRA", "retrieved source")]
    colour = {fn: c for _f, c, cs in FAMILIES for fn, _l in cs}
    outs = {}
    for fn, _f, _l in cols:
        path = RUN / f"outputs.{fn}.{SPLIT}.jsonl"
        if not path.exists():
            return ""
        outs[fn] = {}
        for r in _jsonl(path):
            outs[fn].setdefault(r["task_id"], r)
    tasks = {t["task_id"]: t for t in _jsonl(ROOT / "data" / f"tasks.{SPLIT}.jsonl")}
    posts = {p["post_id"]: p for p in _jsonl(ROOT / "data" / f"posts.{SPLIT}.jsonl")}
    common = sorted(set.intersection(*[set(v) for v in outs.values()]) & set(tasks))
    eligible = [t for t in common
                if all((outs[fn][t].get("output_text") or "").strip() for fn, _f, _l in cols)]
    picks = []
    seen_cells = set()
    for task_id in sorted(eligible, key=lambda t: (tasks[t]["cell_id"], t)):
        cell_id = tasks[task_id]["cell_id"]
        if cell_id in seen_cells:
            continue
        picks.append(task_id)
        seen_cells.add(cell_id)
        if len(picks) == n_examples:
            break
    if not picks:
        return ""

    def clip(t, n=420):
        t = " ".join((t or "").split())
        return t if len(t) <= n else t[: n - 1].rstrip() + "…"

    blocks = []
    for tid in picks:
        task = tasks[tid]
        inner = [f'<div class="qsrc"><div class="qlab">source · the LinkedIn post supplied to every '
                 f'method</div><blockquote>{b1.esc(clip(task["source_text"], 480))}</blockquote></div>']
        reference_ids = task.get("target_reference_ids") or []
        rt = next((posts[post_id]["text"] for post_id in reference_ids if post_id in posts), "")
        if rt.strip():
            inner.append(f'<div class="qreal"><div class="qlab">authentic test reference · Reddit '
                         f'post from the same cell</div><blockquote>{b1.esc(clip(rt))}</blockquote></div>')
        for fn, fam, var in cols:
            t = outs[fn][tid].get("output_text") or ""
            inner.append(f'<div class="qcell" style="border-left-color:{colour.get(fn, "#8b8b8b")}">'
                         f'<div class="qlab">{b1.esc(fam)} · {b1.esc(var)}</div>'
                         f"<blockquote>{b1.esc(clip(t))}</blockquote></div>")
        blocks.append(f'<div class="qgroup"><div class="qhead">Cell <code>{b1.esc(task["cell_id"])}</code></div>'
                      + "".join(inner) + "</div>")
    return (f'<figure class="qual"><figcaption><strong>Qualitative comparison: {len(picks)} tasks, '
            f'every family</strong><br><span class="sub">The first eligible task from each of '
            f'{len(picks)} distinct cells after sorting by cell and task identifier. The selection is '
            f'fixed before reading the text. Passages are truncated for length; nothing else is altered.'
            '</span></figcaption>'
            + "".join(blocks) + "</figure>")




def method_notation():
    """Define every symbol used by the Methods-page equations."""
    return r"""
<aside class="method-notation">
<h3>Notation used in the diagrams and objectives</h3>
<dl class="symbol-list">
<div><dt>\(a,d_{\mathrm{dom}},\tau,s\)</dt><dd>Audience, domain, topic and supplied source post.
\(I(a,d_{\mathrm{dom}},\tau,s)\) is the rendered rewrite instruction.</dd></div>
<div><dt>\(y=(y_1,\ldots,y_T),\ g\)</dt><dd>An authentic target completion and a generated
post. \(T\) is its token count; \(q\in\{1,\ldots,T\}\) is a token position and \(y_{<q}\)
denotes the preceding completion tokens.</dd></div>
<div><dt>\(c,\ x=[c;y]\)</dt><dd>The rendered conditioning context and the resulting token sequence
when it is followed by completion \(y\). \(c_{\mathrm{topic}}\) is the topic-only instance of
that context. More generally, \(x\) denotes the sequence passed to the model; steering passes an
authentic post directly rather than concatenating a separate context and completion.</dd></div>
<div><dt>\(p_\theta,\ \theta\)</dt><dd>The base model's next-token distribution and its frozen
parameters. \(p_{\theta,P}\) additionally conditions on learned soft prompt \(P\);
\(\mathcal L(P)\) is its completion-only negative log-likelihood, with \(\log\) denoting the
natural logarithm. The dot in \(p_\theta(\cdot\mid c)\) is the continuation sampled conditional
on context \(c\), and \(\sim\) means “is sampled from.”</dd></div>
<div><dt>\(E,\ d_{\mathrm{model}},\ P_i\)</dt><dd>The token-embedding map, its width, and learned
virtual-token vector \(i\). \(P=[P_1;\ldots;P_{32}]\); semicolons denote sequence
concatenation and \(\leftarrow\) denotes replacement by the expression on the right.</dd></div>
<div><dt>\(h_\ell,\ \bar h_\ell,\ v_\ell,\ \bar n_\ell\)</dt><dd>For input \(x\),
\(h_\ell(x,q)\) is the residual state at hidden-state index \(\ell\) and token position \(q\), and
\(\bar h_\ell(x)\) is its arithmetic mean over completion positions. \(v_\ell\) is the
target-minus-source mean direction and \(\bar n_\ell\) is the mean residual norm at that index.</dd></div>
<div><dt>\(\alpha_{\mathrm{st}},\ \Delta_\ell\)</dt><dd>The dimensionless steering strength and the
resulting offset added at hidden-state index \(\ell\).</dd></div>
<div><dt>\(R_{\mathrm{train}},S_{\mathrm{train}}\)</dt><dd>The authentic Reddit and LinkedIn
training-post sets used to fit the steering direction.</dd></div>
<div><dt>\(W,W',A_{\mathrm L},B_{\mathrm L},r,\alpha_{\mathrm L}\)</dt><dd>A frozen projection, its
LoRA-modified value, the two trainable low-rank factors, adapter rank, and LoRA scale. The factor
shapes are \(A_{\mathrm L}\in\mathbb R^{r\times d_{\mathrm{in}}}\) and
\(B_{\mathrm L}\in\mathbb R^{d_{\mathrm{out}}\times r}\), where \(\mathbb R\) is the real
numbers and \(d_{\mathrm{in}},d_{\mathrm{out}}\) are the projection's input and output widths.</dd></div>
<div><dt>\(\kappa,S_\kappa^{\mathrm{train}},x^*\)</dt><dd>A bilateral cell, its train-split
LinkedIn posts, and the source post retrieved for target completion \(y\).
\(\arg\max_{x\in A}f(x)\) returns the member of \(A\) with the largest \(f(x)\).</dd></div>
<div><dt>\(v_{\mathrm{tfidf}},\cos\)</dt><dd>The fitted TF–IDF vector map and cosine similarity,
\(\cos(u,v)=u^{\mathsf T}v/(\lVert u\rVert_2\lVert v\rVert_2)\), with
\(\mathsf T\) denoting transpose.</dd></div>
<div><dt>\(\mathcal A(y),a_j,w_j,w_{j,p},\delta_j\)</dt><dd>The positive aspect-score set for
completion \(y\), aspect index \(j\), its per-post score, its corpus emphasis weight on platform \(p\),
and its Reddit-minus-LinkedIn emphasis skew.</dd></div>
<div><dt>\(|A|,\ \sum,\ \lVert z\rVert_2\)</dt><dd>Set cardinality, finite summation, and
Euclidean norm.</dd></div>
</dl>
</aside>
"""


def method_map():
    """A code-grounded visual map of the four intervention mechanisms."""
    cards = [
        (
            "prompting", "#2a78d6", "01", "rendered instruction", "input",
            r"g\sim p_\theta(\,\cdot\mid I(a,d_{\mathrm{dom}},\tau,s)\,)",
            "No fitting step. Zero-shot, few-shot and aspect-aware prompts change only the "
            "instruction and supplied context; the GPT or Claude model remains fixed.",
        ),
        (
            "steering", "#eb6834", "02", "residual stream", "hidden",
            r"h_\ell\leftarrow h_\ell+\Delta_\ell,\quad "
            r"\Delta_\ell=\alpha_{\mathrm{st}}\frac{v_\ell}{\lVert v_\ell\rVert_2}\bar n_\ell",
            "Mean target-minus-source activations are fitted without gradients. A forward hook "
            "adds the selected offset at every prompt and generated position.",
        ),
        (
            "soft prompt", "#1baf7a", "03", "input embeddings", "embedding",
            r"E(x)\leftarrow[P_1,\ldots,P_{32};E(x)]",
            "The base model is frozen. One global 32-vector prompt is optimised on completion-only "
            "negative log-likelihood and selected by validation loss.",
        ),
        (
            "LoRA", "#4a3aa7", "04", "projection weights", "weights",
            r"W' = W+\frac{\alpha_{\mathrm L}}{r}B_{\mathrm L}A_{\mathrm L}",
            "Only the low-rank matrices are trained. The selected adapter places rank-16 updates "
            "in all attention and feed-forward projections.",
        ),
    ]
    body = []
    stages = ("input", "embedding", "hidden", "weights", "output")
    for fam, colour, number, locus, active, equation, detail in cards:
        route = []
        for stage in stages:
            cls = " active" if stage == active else ""
            route.append(f'<span class="route-stage{cls}">{stage}</span>')
            if stage != stages[-1]:
                route.append('<b aria-hidden="true">→</b>')
        body.append(
            f'<div class="method-node" style="--family:{colour}">'
            f'<div class="method-num">{number}</div><h3>{b1.esc(fam)}</h3>'
            f'<div class="method-flow">{"".join(route)}</div>'
            f'<p class="method-locus">Intervention at {b1.esc(locus)}</p>'
            f'<div class="method-eq">\\({equation}\\)</div>'
            f'<p class="method-detail">{b1.esc(detail)}</p></div>'
        )
    return (
        '<figure class="concept-figure method-map"><figcaption><strong>Where each method changes '
        'the generation process</strong><br><span class="sub">The highlighted stage is the only '
        'locus changed by that family. Equations match the implemented intervention; the selected '
        'run settings appear below.</span></figcaption>'
        f'<div class="method-grid">{"".join(body)}</div></figure>'
    )


def conditioning_map(tmanifest):
    """Diagram the three training renderings emitted by build_training.py."""
    tp = tmanifest["counts"]["examples"]
    aspect = tmanifest["aspect"]["counts"]
    cards = [
        (
            "T", "Topic only", tp["train"],
            r"c_{\mathrm{topic}}=(a,d_{\mathrm{dom}},\tau,\mathrm{Reddit})",
            "No source text is present during fitting. At inference, the LinkedIn post is added "
            "to the shared content block.",
        ),
        (
            "R", "Retrieved source", tp["train"],
            r"x^*=\arg\max_{x\in S_\kappa^{\mathrm{train}}}"
            r"\cos(v_{\mathrm{tfidf}}(x),v_{\mathrm{tfidf}}(y))",
            "The nearest train-split LinkedIn post from the same cell is included as context. It "
            "is a topical neighbour, not a paired reference.",
        ),
        (
            "A", "Aspects", aspect["train"]["n_with_aspects"],
            r"\mathcal A(y)=\{(a_j,w_j):w_j>0\}",
            "The positive aspect coordinates of the authentic completion are included. No source "
            "post is supplied, so this variant models evaluative emphasis rather than transfer.",
        ),
    ]
    body = []
    for tag, title, n_train, equation, detail in cards:
        body.append(
            f'<article class="condition-card"><span class="condition-tag">{tag}</span>'
            f'<h3>{b1.esc(title)}</h3><div class="condition-eq">\\({equation}\\)</div>'
            f'<p>{b1.esc(detail)}</p><div class="condition-count"><strong>{n_train}</strong> '
            'train completions</div></article>'
        )
    return (
        '<figure class="concept-figure conditioning-map"><figcaption><strong>One authentic '
        'target completion, three conditioning records</strong><br><span class="sub">For the '
        'target-side records shown here, the completion is the real train-split Reddit post '
        '\\(y\\). The branch changes only the context rendered before it. Soft prompting and LoRA '
        'mask that context from the loss. Steering mean-pools completion-token activations and also '
        'processes authentic LinkedIn records to form the difference of means.</span></figcaption>'
        '<div class="condition-source"><span>shared response</span><strong>authentic Reddit post '
        '\\(y\\)</strong><small>never a synthetic rewrite</small></div>'
        f'<div class="condition-grid">{"".join(body)}</div></figure>'
    )


def data_pipeline(manifest, tmanifest, coverage, n_methods):
    """Counts from corpus construction to the scored primary comparison."""
    examples = tmanifest["counts"]["examples"]
    coverage_text = " · ".join(f"{name} {coverage[name]}" for name in sorted(coverage))
    cards = [
        ("Corpus", f"{manifest['n_posts']:,}", "posts",
         f"{manifest['n_cells']} bilateral cells · {manifest['n_dense_cells']} dense"),
        ("Training signal", f"{examples['train']:,}", "authentic Reddit posts",
         f"{tmanifest['counts']['n_cells']['train']} cells · {examples['dev']} validation posts"),
        ("Evaluated methods", str(n_methods), "configurations",
         f"plus {len(REFERENCES)} named references"),
        ("Scored comparison", str(coverage['trm']), "cells on the primary measure",
         coverage_text),
    ]
    body = []
    for i, (label, value, unit, note) in enumerate(cards):
        if i:
            body.append('<div class="flow-arrow" aria-hidden="true">→</div>')
        body.append(
            f'<div class="flow-card"><div class="flow-label">{b1.esc(label)}</div>'
            f'<div class="flow-value">{b1.esc(value)}</div><div class="flow-unit">{b1.esc(unit)}</div>'
            f'<p>{b1.esc(note)}</p></div>'
        )
    return (
        '<figure class="concept-figure data-flow"><figcaption><strong>From corpus to scored '
        'comparison</strong><br><span class="sub">Minimum pool sizes reduce the cell base for '
        'distributional measures. The coverage line reports the actual base for every metric '
        'family.</span></figcaption><div class="flow-grid">'
        + "".join(body) + '</div></figure>'
    )


EXTRA_CSS = """
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


def build():
    full = _report(RUN)
    systems = _report(RUN_SYS)
    aspect_systems = (
        json.loads(ASPECT_REPORT.read_text())
        if STUDY_LABEL == "Study 3" and ASPECT_REPORT.exists() else {}
    )
    aspect_full = (
        json.loads(ASPECT_REPORT_FULL.read_text())
        if STUDY_LABEL == "Study 3" and ASPECT_REPORT_FULL.exists() else {}
    )
    reverse_full = {}
    reverse_systems = {}
    if STUDY_LABEL == "Study 3" and (REVERSE_RUN / _reverse_name).exists():
        reverse_full = _report(REVERSE_RUN)
    if STUDY_LABEL == "Study 3" and (REVERSE_RUN_SYS / _reverse_name).exists():
        reverse_systems = _report(REVERSE_RUN_SYS)
    # Figures read like-for-like controls from the systems-only report and the
    # oracle from the full report. Their cell counts are printed with the labels.
    REF_REPORT_FULL.clear()
    REF_REPORT_FULL.update(full)
    REF_REPORT_SYSTEMS.clear()
    REF_REPORT_SYSTEMS.update(systems)
    manifest = json.loads((DATA / "manifest.json").read_text())
    tmanifest = json.loads((DATA / "training" / "manifest.json").read_text())
    corpus_posts = [
        json.loads(line)
        for split_name in ("train", "val", "test")
        for line in (DATA / f"posts.{split_name}.jsonl").read_text().splitlines()
        if line.strip()
    ]
    linkedin_total = sum(p.get("platform") == "linkedin" for p in corpus_posts)
    reddit_total = sum(p.get("platform") == "reddit" for p in corpus_posts)
    linkedin_job_titles = sum(
        p.get("platform") == "linkedin" and bool(p.get("job_title")) for p in corpus_posts
    )
    reddit_subreddits = sum(
        p.get("platform") == "reddit" and bool(p.get("subreddit")) for p in corpus_posts
    )
    internal_selection = tmanifest.get("selection", {}).get("source") == "train"
    selection_fraction = float(tmanifest.get("selection", {}).get("fraction", 0.15))
    selection_threshold = f"{selection_fraction:.2f}"
    wrapping_text = (
        "LoRA, soft prompting and steering place the shared rendered context in "
        "the same base-model chat template at fitting and inference. Steering "
        "also receives the complete source post, so wrapping and source truncation "
        "are held constant across the three mechanisms."
        if internal_selection else
        "LoRA and soft prompting place the rendered context in the model's chat "
        "template. Steering uses raw rendered text, which is a known confound."
    )
    selection_limit = (
        "<h2>Selection is isolated from the scored reference pool</h2>"
        "<p>Early stopping and hyperparameter selection use a deterministic train slice "
        f"defined by a {selection_threshold} blake2b(post_id, seed) hash threshold. Validation and test are "
        "untouched references, so this study can score their merged heldout pool without "
        "selection leakage.</p>"
        if internal_selection else
        "<h2>The trained methods select on part of the pool they are scored against, unless scored on test</h2>"
        "<p>Early stopping and hyperparameter selection use validation, so this study "
        "scores test alone.</p>"
    )
    split_description = (
        "This study is scored on the merged validation-and-test heldout split. Model "
        "selection uses only a deterministic slice carved from train, so neither part "
        "of the scored reference pool informed selection."
        if internal_selection else
        "This study is scored on test alone because validation selected the trained methods."
    )
    cleaning_limit = (
        "<h2>Scraped text is normalised before scoring</h2>"
        "<p>The v2 rebuild decodes HTML entities before splitting or scoring. This "
        "removes the one-sided token artefact present in the Study 2 inputs.</p>"
        if internal_selection else
        "<h2>These numbers were computed before the scraped text was cleaned</h2>"
        "<p>Study 2 predates HTML-entity decoding, so its classifier rows carry a "
        "one-sided scrape artefact.</p>"
    )
    cleaning_dataset = (
        "<h2>Text normalisation</h2><p>HTML entities are decoded once during the v2 "
        "dataset build, before any split, fit or score is computed.</p>"
        if internal_selection else
        "<h2>Known defect in the text as scored</h2><p>Study 2 was computed before "
        "HTML-entity decoding was applied.</p>"
    )
    steering_selection_pool = (
        "the train-internal selection slice" if internal_selection else "the validation split"
    )
    subpopulation_limit = (
        "<h2>The selected corpus does not support a bilateral job-title comparison</h2>"
        f"<p>Subreddit labels are present for {reddit_subreddits} of {reddit_total} Reddit posts, "
        f"but job-title labels are present for {linkedin_job_titles} of {linkedin_total} LinkedIn "
        "posts. The job-title and subreddit infusion rows lack the shared audience-room mapping "
        "needed to compare matched populations across platforms. Study 3 therefore evaluates the "
        "existing audience-by-topic cells and does not report a job-title or subreddit "
        "subpopulation result.</p>"
        if internal_selection else ""
    )

    coverage = {name: int(count) for name, count in (systems.get("common_cells") or {}).items()}
    primary_cells = coverage.get("trm", 0)
    n_methods = sum(len(cols) for _family, _colour, cols in FAMILIES)
    space = systems.get("embedding_space") or {}
    space_name = f"{space.get('model', 'TF-IDF')}, dim {space.get('dim', '')}".strip(", ")

    # --- computed claims, never typed -------------------------------------
    def beats(a, bb, key="trm.trm"):
        ea, eb = entry(systems, a, key), entry(systems, bb, key)
        return bool(ea and eb and ea["ci_high"] < eb["ci_low"])

    trained_beating = sorted(
        t for t in TRAINED if any(beats(t, p) for p in PROMPTED)
    )
    n_trained_beating = len(trained_beating)
    best_prompt = min(PROMPTED, key=lambda f: mean(systems, f, "trm.trm") or 9e9)
    best_trained = min(TRAINED, key=lambda f: mean(systems, f, "trm.trm") or 9e9)
    best_trained_trm = mean(systems, best_trained, "trm.trm")

    # Which families actually clear every prompting column, computed rather than
    # asserted. An earlier draft of this page claimed all three did; steering does
    # not, and the sentence would have been false in the published report.
    def family_cols(name):
        return [c for f, _c, cs in FAMILIES for c, _l in cs if f == name]

    best_prompt_trm = min(mean(systems, p, "trm.trm") or 9e9 for p in PROMPTED)
    clears_all = sorted(
        fam for fam, _c, _cs in FAMILIES[1:]
        if all((mean(systems, c, "trm.trm") or 9e9) < best_prompt_trm for c in family_cols(fam))
    )
    clears_txt = " and ".join(clears_all) if clears_all else "no family"
    len_ratio = {f: mean(systems, f, "degeneracy.length_ratio_vs_real") for f in PROMPTED + TRAINED}
    worst_len = max(len_ratio, key=lambda f: len_ratio[f] or 0)
    lbl = {fn: l for _f, _c, cs in FAMILIES for fn, l in cs}
    fam_of = {fn: f for f, _c, cs in FAMILIES for fn, _l in cs}
    observed_reduction = 1 - best_trained_trm / best_prompt_trm
    best_soft = min(family_cols("soft prompt"),
                    key=lambda f: mean(systems, f, "trm.trm") or 9e9)
    best_lora = min(family_cols("LoRA"),
                    key=lambda f: mean(systems, f, "trm.trm") or 9e9)
    adapted_focus = family_cols("soft prompt") + family_cols("LoRA")
    min_prompt_source = min(mean(systems, fn, "semantic.source_similarity") for fn in PROMPTED)
    max_adapted_source = max(mean(systems, fn, "semantic.source_similarity") for fn in adapted_focus)
    oracle_primary_cells = (ref_entry("trm.trm", "target_sample") or {}).get("n_cells", 0)
    oracle_primary_unit = "cell" if oracle_primary_cells == 1 else "cells"

    res_html, n_ok, n_claims = resolution(systems)
    aspect_html = aspect_results(aspect_systems, aspect_full)
    extended_html = extended_metric_results(systems)
    reverse_html = reverse_results(reverse_systems, reverse_full)
    embedding_html = embedding_stability(EMBEDDING_RUN) if STUDY_LABEL == "Study 3" else ""
    direction_limit = (
        "<h2>Two directions, one base model and one primary embedding space</h2>"
        "<p>Study 3 evaluates LinkedIn to Reddit and Reddit to LinkedIn on independent comparison "
        "bases. The values are not directly subtractable because each direction has a different "
        "target distribution and reference pool. All trained families still share one base model, "
        "and the primary findings in each direction use one embedding space.</p>"
        if reverse_html else
        "<h2>One direction, one base model, one embedding space</h2>"
        "<p>Everything here is LinkedIn to Reddit. The reverse direction is supported and "
        "unexamined. All three trained families share one base model, and every embedding-derived "
        "row is computed in one space.</p>"
    )

    # ---------------- index ------------------------------------------------
    idx = f"""
<div class="page-intro">
<p class="section-label">Research question</p>
<h1>Adaptation can match the target distribution better than direct prompting, but it preserves
less of the source.</h1>
<p class="lede">This study compares prompting, activation steering, soft prompts and LoRA on the
same LinkedIn-to-Reddit transfer task. It asks whether fitting a small number of parameters on real
target-platform posts improves the distribution of the generated pool. Its primary measure is the
Triangle-Rank Metric (TRM).</p>
</div>

<div class="summary-grid" aria-label="Study summary">
  <div class="summary-card primary"><span class="summary-kicker">Supported reach</span>
    <strong>{n_trained_beating} / {len(TRAINED)}</strong>
    <p>trained configurations outperform at least one prompting configuration on the primary
    measure with separated intervals.</p></div>
  <div class="summary-card"><span class="summary-kicker">Best observed score</span>
    <strong>{best_trained_trm:.3f}</strong>
    <p>{b1.esc(fam_of[best_trained])}, {b1.esc(lbl[best_trained])};
    {observed_reduction:.0%} below the best prompting point estimate of {best_prompt_trm:.3f}.</p></div>
  <div class="summary-card"><span class="summary-kicker">Primary comparison base</span>
    <strong>{primary_cells} cells</strong>
    <p>The per-post metric families cover up to {max(coverage.values())} cells. Reference rules state
    their own cell bases.</p></div>
</div>

<aside class="finding-banner">
  <div><span>Distribution match</span><strong>{b1.esc(clears_txt)}</strong>
  <p>Every soft-prompt and LoRA configuration has a lower observed Triangle-Rank score than the best
  prompting configuration.</p></div>
  <div><span>Content retention</span><strong>Method ranges overlap</strong>
  <p>The weakest prompting point on source similarity is {min_prompt_source:.3f}; the strongest soft
  prompt or LoRA point is {max_adapted_source:.3f}. The highest point remains a prompt, but adaptation
  does not uniformly preserve less than prompting.</p></div>
  <div><span>Unresolved objective</span><strong>No joint winner</strong>
  <p>The evaluated methods do not occupy the desirable combination of prompting-level content
  preservation and adaptation-level distribution match.</p></div>
</aside>

{family_overview(systems)}

<div class="story-grid">
  <section><span class="story-num">01</span><h2>What improved</h2>
  <p>Soft prompts and LoRA move the generated pool closer to the distribution of authentic Reddit
  posts. The full results show the interval for every configuration and test each stated
  comparison.</p></section>
  <section><span class="story-num">02</span><h2>What was traded away</h2>
  <p>Prompting keeps more of the supplied LinkedIn post. Adaptation learns the target pool without
  paired source-and-target examples, so a lower distributional score does not establish successful
  content transfer.</p></section>
  <section><span class="story-num">03</span><h2>What explains the change</h2>
  <p>Output length explains part of the soft-prompt result, but not the LoRA result. The results page
  places length, triangle geometry and source preservation beside the primary measure.</p></section>
</div>

<h2>Study design in brief</h2>
<p>Four families are evaluated under one task definition. Prompting changes the input only.
Steering adds a fixed platform-difference vector to the model's hidden state. A soft prompt learns a
short sequence of input vectors. LoRA learns low-rank adapters within the model. The three adaptable
families share the base model, prompt and quantisation.</p>
<p>Each adaptable family is fitted under three training variants: topic only, a retrieved same-cell
source post, or the evaluative aspects foregrounded by the target post. The corpus contains no paired
LinkedIn and Reddit posts, so only the aspect variant directly couples its conditioning information
to the completion. The <a href="methods.html">Methods page</a> gives the full training design.</p>

<div class="next-grid">
  <a href="results.html"><span>Read next</span><strong>Results and figures</strong>
  <small>Primary finding, trade-off, mechanism and full tables</small></a>
  <a href="methods.html"><span>Study design</span><strong>Methods</strong>
  <small>Families, variants, controls and selected configurations</small></a>
  <a href="limitations.html"><span>Boundary conditions</span><strong>Limitations</strong>
  <small>Five-cell primary base, reference coverage and known confounds</small></a>
</div>
"""

    # ---------------- methods ----------------------------------------------
    tp = tmanifest["counts"]["examples"]
    methods = rf"""
<h1>Methods</h1>
<p class="lede">Four intervention families address one transfer task. The three adaptation
families share a rendered context and a Llama base model; they differ in what is fitted and where
the fitted signal enters generation.</p>

{method_notation()}

{method_map()}

<h2>Traits, aspects and style are different units of analysis</h2>
<p><strong>Traits</strong> are audience-global, cross-topic latent dispositions learned because they
help predict opinions; they need not be stated in one post. <strong>Aspects</strong> are topic-local
evaluative lenses actually invoked in a post, such as cost, latency or ethics. <strong>Style
features</strong> are observable linguistic forms such as function words, syntax, punctuation,
length, pronouns and formatting. The eight LLM-clustered axes discussed by Vectorial are referred to
as <em>discovered platform-writing dimensions</em>, not as established stylistic features.</p>
<p>Study 3 evaluates neither Sapiens traits nor the reported platform trait lists. It also does not
test the prior claims that LinkedIn has greater aspect diversity, that twelve aspects are shared, or
that eight discovered writing dimensions separate the platforms. Those are inputs from prior
analysis, not findings of this experiment.</p>
<p>The report's writing-habit analysis uses reproducible feature families from linguistic style
research: function words and lexical form, syntax, punctuation, and discourse formatting.
Kestemont (2014) motivates function words as comparatively content-independent signals;
Sundararajan and Woodard (2018) show why lexical features can leak subject matter and why syntax is
useful across domains; and Gero et al. (2019) operationalise controllable style through pronouns,
prepositions and subordinate clauses. The harness's word and sentence lengths, punctuation,
pronouns and formatting counts fit these families. Promotional and critique term lists are labelled
as content-sensitive probes rather than universal style dimensions.</p>
<p class="cite">Sources: <a href="https://aclanthology.org/W14-0908/">Kestemont, 2014</a>;
<a href="https://aclanthology.org/C18-1238/">Sundararajan and Woodard, 2018</a>;
<a href="https://aclanthology.org/W19-8628/">Gero et al., 2019</a>.</p>

<h2>Pass 2 human validation is specified separately from aspect discovery</h2>
<p>The validation task freezes the existing cell-specific aspect vocabulary and asks annotators only
whether one named aspect is present in one post. Platform, generation method and automated labels are
hidden. Each item supplies the aspect name, its operational definition, inclusion and exclusion
rules, and examples from a separate pilot set. Annotators choose present, absent or cannot determine
and quote an evidence span when present. Two annotators label every evaluation item; disagreements
are adjudicated without revealing the automated score.</p>
<p>The sample is stratified by audience, topic, platform and automated score, with boundary cases
oversampled and sampling weights retained for prevalence estimates. The report will include
human–human agreement, automated-rater precision, recall and F1 against adjudicated labels, and
errors by platform, topic, audience and text length. Confidence intervals resample posts or cells,
not individual post–aspect pairs. This validates application of a fixed vocabulary. It does not
validate the completeness or neutrality of the discovery pass.</p>

<h2>The problem the training data has to solve</h2>
<p>No LinkedIn post in this corpus has a corresponding Reddit post. That is why the harness compares
pools rather than pairs, and it is also why a trained method cannot simply be fitted to
source-and-target examples. Inventing those pairs with a language model would fit the study to
another model's opinion of the answer, which the evaluation would then measure agreement with.</p>

<p>Instead, soft prompting and LoRA are trained as conditional language models over <strong>real
Reddit posts from the train split</strong>. Steering uses the same rendered records but estimates a
target-minus-source activation difference from authentic Reddit and LinkedIn posts rather than an
optimisation loss. At inference the task's LinkedIn post is supplied as content to carry across. All
fitting signals remain authentic and inside the train split. Three variants differ only in the
context rendered around a post:</p>

{conditioning_map(tmanifest)}

<div class="table-shell" role="region" aria-label="Training variant counts" tabindex="0">
<table class="res simple">
<thead><tr><th>variant</th><th>what the model is conditioned on</th><th>train</th><th>dev</th><th>test</th></tr></thead>
<tbody>
<tr><th scope="row">topic only</th><td>the audience, the domain and the topic</td>
    <td>{tp['train']}</td><td>{tp['dev']}</td><td>{tp['test']}</td></tr>
<tr><th scope="row">retrieved source</th><td>the above, plus the most similar source post in the same cell</td>
    <td>{tp['train']}</td><td>{tp['dev']}</td><td>{tp['test']}</td></tr>
<tr><th scope="row">topic + aspects</th><td>the above topic fields, plus the evaluative aspects the real post foregrounds</td>
    <td>{tmanifest['aspect']['counts']['train']['n_with_aspects']}</td>
    <td>{tmanifest['aspect']['counts']['dev']['n_with_aspects']}</td>
    <td>{tmanifest['aspect']['counts']['test']['n_with_aspects']}</td></tr>
</tbody></table></div>

<p>The retrieved-source variant deserves a caveat that the numbers cannot supply. Its pairing is
built by retrieval within a cell, and retrieval can only find what is there: a retrieved pair
averages {0.137:.3f} cosine similarity against {0.079:.3f} for a random pair from the same cell, and
under one per cent of pairs reach 0.3. It removes the structural mismatch between training and
inference, where the model would otherwise never have seen the content block it is asked to
condition on, but it does not teach content preservation, because there is no content
correspondence in this corpus to teach.</p>

<p>The aspect variant drops content transfer as the objective altogether. Rather than asking the
model to carry a specific post across, it supplies the evaluative dimensions the real post
emphasises and asks for a post that foregrounds them. Its context and its completion are genuinely
coupled, which is not true of the other two. It costs coverage: aspect vocabularies exist for
{tmanifest['aspect']['counts']['train']['n_cells']} of the cells present in training.</p>

<h2>Baselines and controls</h2>
<p>Three reference columns bound the attainable range, and every figure in this report draws all
three so that no system value is read without them. They are named explicitly below because a
reference read only by intuition is a reference that will eventually be misread.</p>

<div class="ctrl floor">
<h4><code>identity</code> — the source-copy floor</h4>
<p>Emits the source LinkedIn post unchanged. It is simultaneously the floor on transfer, since no
transformation has occurred, and the ceiling on content preservation, since it attains a cosine
similarity to the source of exactly one and a copy rate of one by construction. A method that fails
to improve on this column on the platform-match measures has performed no transfer.</p>
</div>

<div class="ctrl">
<h4><code>shuffle_control</code> — the wrong-topic negative control</h4>
<p>Emits an authentic Reddit post drawn from a <em>different</em> cell. It therefore has perfect
target-platform register and no relationship to the requested subject. Its purpose is diagnostic: a
measure that scores this column as highly as the oracle is indexed on register and blind to content,
which is the study's principal identified risk. Its position is checked on every embedding-derived
measure, and a measure that fails to rank it worst is not reported as evidence.</p>
</div>

<div class="ctrl oracle">
<h4><code>target_sample</code> — the authentic-post oracle</h4>
<p>Emits an authentic Reddit post drawn from the same cell, from the train split only, so that it
never emits the specific post it is scored against. It is the practical ceiling: it shows what the
measures return when the generated pool is in fact drawn from the target distribution. It is the
reference for every diagnostic whose ideal is a middle value rather than an extreme, and it produces
no output on the cells reserved for zero-shot generalisation, which is why its cell count is printed
whenever it differs from the systems'.</p>
</div>

<p>The tension between the first and the third is the most diagnostic reading in the report. A method
that improves on <code>identity</code> against the platform classifier while approaching its
similarity-to-source score is performing substantive work; one that improves on the classifier by
discarding the input will show a corresponding fall in similarity to source and in content-word
retention.</p>

<h2>The four families</h2>

<h3 style="color:#2a78d6">Prompting</h3>
<p>The zero-shot implementation in <code>transfer/llm_rewrite.py</code> sends the audience, domain,
topic, source post and a target-platform writing hint to a fixed GPT or Claude model. Formally it
draws \(g\sim p_\theta(\cdot\mid I(a,d_{{\mathrm{{dom}}}},\tau,s))\), with no parameter update. Few-shot prompting adds
the longest available authentic train-split posts from the same cell as style exemplars; an entirely
held-out cell therefore falls back to zero-shot rather than borrowing from the evaluation split.</p>
<p>The aspect-aware prompt changes one additional input. For aspect \(j\), it computes an emphasis
weight separately within each platform and uses the skew
\(\delta_j=w_{{j,\mathrm{{Reddit}}}}-w_{{j,\mathrm{{LinkedIn}}}}\). At most four aspects with
\(\delta_j\geq0.05\) are brought forward and at most three with \(\delta_j\leq-0.05\) are played
down. The prompt instructs the model to reweight only information already present in the source.</p>

<h3 style="color:#eb6834">Steering</h3>
<p><code>methods/steering/fit_steering.py</code> performs one forward pass over each authentic
train-split post. At hidden-state index \(\ell\), it mean-pools only the completion-token residual
states and forms the global platform direction</p>
<div class="method-formula">\[
v_\ell=\frac{{1}}{{|R_{{\mathrm{{train}}}}|}}\sum_{{x\in R_{{\mathrm{{train}}}}}}\bar h_\ell(x)
-\frac{{1}}{{|S_{{\mathrm{{train}}}}|}}\sum_{{x\in S_{{\mathrm{{train}}}}}}\bar h_\ell(x).
\]</div>
<p>No gradient is taken. <code>runtime.steering_deltas</code> normalises the raw direction and restores
the natural layer scale with the mean activation norm \(\bar n_\ell\):
\(\Delta_\ell=\alpha_{{\mathrm{{st}}}}(v_\ell/\lVert v_\ell\rVert_2)\bar n_\ell\). A forward hook adds
\(\Delta_\ell\) to every sequence position on every decoding step. Layer, strength and global versus
per-cell scope are selected on validation centroid similarity subject to a degeneracy constraint.
The reported configurations select global scope.</p>

<h3 style="color:#1baf7a">Soft prompt</h3>
<p><code>methods/soft_prompt/train.py</code> prepends one shared matrix
\(P\in\mathbb R^{{32\times d_{{\mathrm{{model}}}}}}\) to every embedded input and freezes the complete base model:
\(E(x)=[P;E(c);E(y)]\). Its label tensor is \(-100\) over the rendered context \(c\), so the only
optimised term is the authentic completion:</p>
<div class="method-formula">\[
\mathcal{{L}}(P)=-\sum_{{q=1}}^{{T}}\log p_{{\theta,P}}(y_q\mid c,y_{{<q}}),
\qquad \theta\ \text{{frozen}}.
\]</div>
<p>The selected prompt is global rather than cell-specific because most cells contain too few
training posts to fit a separate continuous prefix and held-out cells would have no prefix. The
32 vectors are initialised from a natural-language instruction, optimised with AdamW, evaluated on
completion-only dev loss after each epoch, and restored from the best epoch.</p>

<h3 style="color:#4a3aa7">LoRA</h3>
<p><code>methods/lora/train.py</code> uses the same completion-only causal-language-model objective,
but modifies selected projection matrices as
\(W'=W+(\alpha_{{\mathrm{{L}}}}/r)B_{{\mathrm{{L}}}}A_{{\mathrm{{L}}}}\), with
\(B_{{\mathrm{{L}}}}\in\mathbb R^{{d_{{\mathrm{{out}}}}\times r}}\) and
\(A_{{\mathrm{{L}}}}\in\mathbb R^{{r\times d_{{\mathrm{{in}}}}}}\). The base matrix \(W\) is
frozen. The selected <code>all-r16</code> preset uses \(r=16\),
\(\alpha_{{\mathrm{{L}}}}=32\), dropout 0.05, and updates the
<code>q</code>, <code>k</code>, <code>v</code>, <code>o</code>, <code>gate</code>, <code>up</code>
and <code>down</code> projections. Dev negative log-likelihood is measured before the first update
and after every epoch; the best adapter is retained and patience-based early stopping prevents a
later overfit epoch from replacing it.</p>

{hyperparameters(RUN)}

<h2>What was held constant</h2>
<p>Steering, soft prompting and LoRA use
<code>meta-llama/Llama-3.1-8B-Instruct</code> in four-bit NF4 and import the rendered context from
<code>data/build_training.py</code>. The fitting sequence limit is 768 tokens, and the completion is
always an authentic post from the designated split. LoRA and soft prompting place the rendered
context in the model's chat template at both fitting and inference.</p>
<p>{wrapping_text}</p>
<p>Prompt tokens are masked from the loss. The selection source is recorded in the training
manifest and run records; scored posts are never opened by the training scripts.</p>
"""

    # ---------------- results ----------------------------------------------
    results = f"""
<div class="page-intro">
<p class="section-label">Complete analysis</p>
<h1>Results</h1>
<p class="lede">The comparison contains {n_methods} method configurations and three named
references. The primary Triangle-Rank Metric (TRM) covers {primary_cells} shared {SPLIT} cells;
distributional measures cover {coverage.get('distributional', 0)}, and per-post semantic,
classifier and degeneracy measures cover {coverage.get('semantic', 0)}. All embedding-derived
values use {b1.esc(space_name)}.</p>
</div>

<nav class="result-nav" aria-label="Results on this page">
  <a href="#primary-result">Primary result</a>
  <a href="#mechanism">Output behaviour</a>
  <a href="#trade-off">Transfer trade-off</a>
  <a href="#stability">Cell stability</a>
  <a href="#centroid-result">Metric decomposition</a>
  <a href="#aspect-result">Aspect match</a>
  <a href="#embedding-stability">Space stability</a>
  <a href="#full-results">Full tables</a>
  <a href="#metric-definitions">Metric definitions</a>
</nav>

<aside class="reading-key"><strong>Rows within each adaptable family</strong>
<span><b>T</b> topic only</span><span><b>R</b> retrieved source</span><span><b>A</b> aspects</span>
<a href="methods.html">Definitions and training details</a></aside>

<h2 id="primary-result"><span class="section-no">01</span> Soft prompts and LoRA produce the lowest observed distributional scores</h2>

{dotplot(systems, "trm.trm",
         "Distance from the pool of real Reddit posts",
         "Zero indicates that a pool of generated posts cannot be distinguished from a pool of authentic ones.",
         axis_label="TRM score",
         better="lower")}

<p>Every soft-prompt and LoRA configuration has a lower observed score than the best prompting
configuration. The best points are {b1.esc(fam_of[best_soft])} {b1.esc(lbl[best_soft])} at
{mean(systems, best_soft, 'trm.trm'):.3f} and {b1.esc(fam_of[best_lora])}
{b1.esc(lbl[best_lora])} at {mean(systems, best_lora, 'trm.trm'):.3f}, against
{b1.esc(lbl[best_prompt])} at {best_prompt_trm:.3f}. Steering lies on the prompting side of the
comparison.</p>

{res_html}

<h2 id="mechanism"><span class="section-no">02</span> Adaptation moves the triangle-rank profile toward the null</h2>

{dotplot(systems, "trm.rank_i2",
         "How often the authentic-authentic edge is longest",
         "The rank indicator I₂ is one third for a matched pool. Values above it indicate generated points concentrated inside the authentic cloud; values below require the in-edge-shortest rate to distinguish displacement from excess spread.",
         axis_label="In-edge-longest rate (I₂)",
         ideal=1/3, ideal_label="one third · matched rank profile", better="neither extreme")}

<p>Prompted outputs have an in-edge-longest rate well below one third and, in the full table, an
in-edge-shortest rate well above one third. Together those two indicators mean that the generated
points more often lie outside the authentic reference cloud; the pattern is not evidence of simple
mode collapse. Soft prompts and LoRA move both ranks toward one third. The wrong-topic control is
the extreme case—very low on the longest rate and very high on the shortest rate—which confirms
that the profile responds to semantic displacement even when platform register is authentic.</p>

{length_tradeoff(systems)}

<p>Length is part of the mechanism, but it is not a complete explanation. The best soft-prompt
configuration produces posts at {mean(systems, best_soft, 'degeneracy.length_ratio_vs_real'):.2f}
times authentic length. The best LoRA configuration reaches a similar Triangle-Rank score while
remaining {mean(systems, best_lora, 'degeneracy.length_ratio_vs_real'):.2f} times longer. Steering
is the longest family, with its most extreme configuration reaching {len_ratio[worst_len]:.1f}
times authentic length.</p>

<h2 id="trade-off"><span class="section-no">03</span> Distribution match is purchased with source loss</h2>

{tradeoff(systems)}

<p>The highest source-similarity point is a prompt, but the prompting and adaptation ranges overlap.
The adaptable methods learn from unpaired target-platform posts, and their lowest Triangle-Rank
scores occur below the prompting range on source similarity. The lower-right region remains empty:
none of the evaluated configurations combines the highest observed content preservation with the
lowest observed distribution score.</p>

<h2 id="stability"><span class="section-no">04</span> The primary result rests on a small, variable cell base</h2>

{strip_by_cell(systems, "trm.trm",
    "The per-cell values behind the aggregate",
    "Each dot is one cell. The aggregate is a mean over these, and with a base this small it is "
    "worth seeing whether it rests on agreement between cells or on one of them.",
    f"TRM score · {SPACE_SHORT}", better="lower")}

<p>Cell-level values vary substantially. The report therefore supports selected differences between
adaptation and prompting, but it does not establish an ordering between LoRA and the soft prompt.</p>

<h2><span class="section-no">05</span> The best training variant depends on the family</h2>

{variant_slopes(systems, "trm.trm",
    "Does the training variant matter, and does it matter the same way for each family?",
    "Three variants of each trained family on the primary measure.",
    "TRM score", better="lower")}

<p>The lines are not parallel. The retrieved-source variant is the best choice for LoRA and for the
soft prompt but not for steering, so the variant cannot be selected once and applied to every family.
The figure motivates a larger ablation; the current comparison does not settle the within-family
ordering.</p>

{extended_html}

{aspect_html}

{embedding_html}

<h2 id="full-results"><span class="section-no">13</span> Full results</h2>

{table(systems, TABLE_GROUPS[0][2], TABLE_GROUPS[0][0], TABLE_GROUPS[0][1])}
{table(systems, TABLE_GROUPS[1][2], TABLE_GROUPS[1][0], TABLE_GROUPS[1][1])}
{table(systems, TABLE_GROUPS[2][2], TABLE_GROUPS[2][0], TABLE_GROUPS[2][1])}

<h2><span class="section-no">14</span> Qualitative comparison</h2>
<p>The examples make the numerical trade-off concrete. Prompting produces coherent expository
rewrites that keep the source topic. Soft-prompt outputs often resemble the short title-like form of
the target pool while dropping source specifics. LoRA is also shorter than prompting, but it retains
more source content than the soft prompt in some tasks. Steering frequently runs to a long templated
answer.</p>

{qualitative()}

<section class="metric-definitions" id="metric-definitions">
<h2><span class="section-no">15</span> Metric definitions and equations</h2>
<p>Each equation below states the finite-sample quantity implemented by the evaluation code. This
distinguishes, for example, the tie-aware Triangle-Rank indicators from a generic rank statistic and
the classifier calibration gap from a target-rate maximisation objective.</p>
{metric_glossary(TABLE_GROUPS)}
</section>

<details class="appendix"><summary>Master table: every scored row</summary>
{master_table(systems)}
</details>
"""

    # ---------------- limitations ------------------------------------------
    limits = f"""
<h1>Limitations</h1>
<p class="lede">The conditions under which the results above hold, and the claims they do not
support.</p>

<h2>Prompt structure across trained methods</h2>
<p>{wrapping_text}</p>

<h2>The primary comparison base is {primary_cells} cells</h2>
<p>Each metric family restricts every method column to one shared cell set, but minimum pool sizes
make those sets different. The Triangle-Rank Metric covers {coverage.get('trm', 0)} cells,
distributional and structural measures cover {coverage.get('distributional', 0)}, and the per-post
semantic, classifier and degeneracy measures cover {coverage.get('semantic', 0)}. Resampling is over
cells rather than posts, since posts within one cell are not exchangeable with posts in another.
The primary intervals are correspondingly wide. <strong>LoRA against the soft prompt is not resolved
on the Triangle-Rank Metric</strong>, and the report does not order them.</p>

<h2>The authentic-post oracle covers {oracle_primary_cells} {oracle_primary_unit} on the primary measure</h2>
<p>The systems-only report excludes the oracle from the common-cell intersection so that its limited
coverage does not reduce the method comparison. Identity and the wrong-topic control are therefore
drawn from the systems-only report on the same {primary_cells}-cell base as the methods. The oracle is
overlaid from the full report and covers {oracle_primary_cells} {oracle_primary_unit} on the
Triangle-Rank Metric. Its
rule is a calibration marker rather than a like-for-like aggregate, and every figure label and table
cell states that base.</p>

<h2>The unbiased MMD estimator is reported but not used here</h2>
<p>The report carries an unbiased squared MMD (distributional.mmd2) computed as the U-statistic of
Gretton et al., in which the diagonal terms are removed to eliminate the O(1/n) bias. That statistic
is unbiased for a non-negative population quantity and is consequently not itself non-negative: on
pools whose true MMD&sup2; is near zero it takes negative values with probability approaching one
half, which is what the oracle column (target_sample) exhibits. The V-statistic form retains the
diagonal, equals the squared norm of the difference of empirical kernel mean embeddings, and is
non-negative by construction, at the cost of a positive bias that is severe at these pool sizes. On
this corpus the guard <code>MIN_MMD_POOL = 5</code> leaves the row scored on too few cells to
support a comparison, so the tables above report centroid distance
(distributional.centroid_distance) instead and the MMD row should not be read as a result.</p>

{selection_limit}

{subpopulation_limit}

<h2>Steering is reported at a selected strength, and the selection is consequential</h2>
<p>At the unselected default strength the offset produced degenerate output: repetitive fragments
continuing to the generation limit, with vocabulary variety (degeneracy.distinct_2) far below every
other column. All three steering columns would then have been read as a failure of the method rather
than of a default. The layer and strength reported here were selected on {steering_selection_pool} by
maximising the cosine between the generated and authentic pool centroids subject to a degeneracy
constraint, with the selected strengths between one tenth and one half of that default. Steering
nonetheless does not reach the level the other two trained families reach on the triangle score
(trm.trm), and that result should be read as applying to activation steering at its best available
setting on this corpus, under this generation configuration, rather than to the method in
general.</p>

<h2>Length accounts for much of the effect, and is partly a generation setting</h2>
<p>The largest single difference between the families is output length
(degeneracy.length_ratio_vs_real). For the soft prompt and for LoRA this is learned from the target
distribution. For steering it is not: the base model does not emit an end-of-sequence token under
this prompt and continues to <code>max_new_tokens</code>, so steering's length ratio reflects the
decoding configuration rather than the intervention. An additive offset in the residual stream alters
the conditional distribution over tokens without altering the model's stopping behaviour, and a
comparison that isolated the intervention would require equalising that behaviour across
families.</p>

<h2>Aspect conditioning rests on vocabularies built with the platform visible</h2>
<p>The aspect vocabularies were extracted with platform labels visible to the extracting model, so
the vocabulary may have been constructed to separate the platforms, which is the same signal the
aspect-conditioned columns are then scored on. A blind rerun is pending. Every aspect-conditioned
result is provisional until it lands. Coverage is a second constraint: vocabularies join to
{tmanifest['aspect']['counts']['train']['n_cells']} of the cells present in training, and the
aspect-aware prompting column ran unconditioned on roughly half its outputs.</p>

<h2>TRM is not a topic or aspect score</h2>
<p>The Triangle-Rank Metric compares complete generated and authentic distributions in
{b1.esc(space_name)} cosine space. Removing two measured platform directions leaves the primary
ordering nearly unchanged, which shows only that those two directions do not explain the result.
It does not establish that TRM is dominated by topic, and it cannot determine whether the intended
evaluative aspects were transferred.</p>

{cleaning_limit}

{direction_limit}
"""

    dataset = f"""
<h1>Dataset</h1>
<p class="lede">The corpus, the cell structure, the split rule, and the three conditionings of the
target text used to fit the adaptation methods. This page is self-contained; nothing on it requires
the methods page.</p>

{data_pipeline(manifest, tmanifest, coverage, n_methods)}

<h2>Corpus and cell structure</h2>
<p>The corpus pairs an occupational audience with a topic to form a <em>bilateral cell</em>, a cell
being retained only where both platforms carry at least
{manifest['config']['min_posts_per_platform']} posts. The harness holds {manifest['n_posts']} posts
in {manifest['n_cells']} such cells, of which {manifest['n_dense_cells']} are dense, meaning at least
{manifest['config']['dense_min_posts_per_platform']} posts on each platform. Coverage, rather than
corpus size, is the binding constraint: of roughly fifteen hundred topic clusters in the source data,
only these {manifest['n_cells']} carry sufficient mass on both platforms.</p>
<p>Source platform is {manifest['direction']['source']} and target platform is
{manifest['direction']['target']}. Of the {manifest['n_posts']} posts,
{manifest['platform_totals'][manifest['direction']['source']]} are drawn from the source platform and
{manifest['platform_totals'][manifest['direction']['target']]} from the target.</p>

<h2>Splits</h2>
<p>Splits are assigned at the level of the post, stratified within a cell and platform, in the
proportions {manifest['config']['split_ratios'][0]:.0%}, {manifest['config']['split_ratios'][1]:.0%}
and {manifest['config']['split_ratios'][2]:.0%}. Assignment is
<code>blake2b(post_id, seed)</code> rather than a shuffle, so the mapping is a pure function of the
identifier and appending newly scraped posts never reshuffles an existing assignment. A cell-level
split was rejected because cells hold between five and roughly one hundred and fifty posts, and every
metric in the harness compares pools; a cell-level split would leave single-digit pools.</p>
<p>Leakage is controlled by role rather than by partition. A task's exemplars are drawn only from
train, while the authentic posts it is scored against come from the split being scored. Separately,
{manifest['n_heldout_cells']} whole cells are withheld from train and validation entirely, which
provides a zero-shot set for topics from which no exemplar has ever been seen.</p>
<p>{split_description}</p>

<h2>The trainable projection</h2>
<p>There is no paired data. No source-platform post in this corpus has a corresponding
target-platform post, which is why the harness compares pools rather than pairs and why no generated
post has a gold reference. A trained method therefore cannot be fitted to source-and-target examples
without first inventing them, and inventing them with a language model would fit the study to another
model's opinion of the answer, which the evaluation would then measure agreement with.</p>
<p>Each method is instead fitted as a conditional language model over authentic train-split
target-platform posts, with the source post supplied at inference as content to be carried across.
Restricting the projection to target-platform posts leaves <strong>{tp['train']} training
examples</strong> over {len(tmanifest['counts']['by_cell']['train'])} cells, with {tp['dev']} for
selection and {tp['test']} held out. That figure is the entire training signal behind every trained
column in this study and belongs in any reading of the results. It is also small enough that LoRA
overfitting was the expected outcome; it did not occur, and every run improved on the base model
before early stopping engaged.</p>

<h2>The three conditionings</h2>
<p>The completion is always the authentic post, verbatim. The variants differ only in what conditions
it, and all three share one prompt template, so a method may be fitted to each without anything
changing but the file it reads.</p>
<div class="table-shell" role="region" aria-label="Three conditioning variants" tabindex="0">
<table class="res simple">
<thead><tr><th>variant</th><th>conditioned on</th><th>train</th><th>dev</th><th>test</th><th>cells</th></tr></thead>
<tbody>
<tr><th scope="row">topic only</th><td>audience, domain and topic</td>
    <td>{tp['train']}</td><td>{tp['dev']}</td><td>{tp['test']}</td>
    <td>{len(tmanifest['counts']['by_cell']['train'])}</td></tr>
<tr><th scope="row">retrieved source</th><td>the above, plus the nearest same-cell source post under TF-IDF cosine</td>
    <td>{tp['train']}</td><td>{tp['dev']}</td><td>{tp['test']}</td>
    <td>{len(tmanifest['counts']['by_cell']['train'])}</td></tr>
<tr><th scope="row">aspects</th><td>the topic fields, plus the evaluative dimensions the post foregrounds</td>
    <td>{tmanifest['aspect']['counts']['train']['n_with_aspects']}</td>
    <td>{tmanifest['aspect']['counts']['dev']['n_with_aspects']}</td>
    <td>{tmanifest['aspect']['counts']['test']['n_with_aspects']}</td>
    <td>{tmanifest['aspect']['counts']['train']['n_cells']}</td></tr>
</tbody></table></div>
<p><strong>Topic only</strong> presents no source post during fitting, so the content block supplied
at inference is structurally unseen. <strong>Retrieved source</strong> removes that mismatch by
pairing each target with the nearest same-cell source post, but the pairing is weak by construction:
a retrieved pair averages 0.137 cosine against 0.079 for a random same-cell pair, and fewer than one
per cent reach 0.3. Retrieval recovers topical neighbours because the corpus contains no content
correspondences to recover, so the variant supplies the shape of the inference context rather than a
supervision signal for content preservation. <strong>Aspects</strong> abandons content transfer as the
objective and conditions on the evaluative dimensions the target post emphasises; its context and
completion are genuinely coupled, unlike the other two, at the cost of coverage, since aspect
vocabularies join to {tmanifest['aspect']['counts']['train']['n_cells']} of the cells present in
training.</p>

{cleaning_dataset}
"""

    OUT.mkdir(parents=True, exist_ok=True)

    # This study is scored in one embedding space, so any stale replication page
    # from a previous build is removed rather than left in the output directory.
    stale = OUT / "embedding-space.html"
    if stale.exists():
        stale.unlink()

    css = (ROOT / "experimental-notes" / "style.css").read_text() + EXTRA_CSS
    (OUT / "style.css").write_text(css, encoding="utf-8")
    pages = [
        ("index.html", "Overview", idx),
        ("methods.html", "Methods", methods),
        ("results.html", "Results", results),
        ("dataset.html", "Dataset", dataset),
        ("limitations.html", "Limitations", limits),
    ]
    if reverse_html:
        pages.insert(2, ("reverse.html", "Reverse direction", reverse_html))
    for slug, title, body in pages:
        html = study_page(
            slug, title, body, corpus_cells=manifest["n_cells"],
            primary_cells=primary_cells, n_methods=n_methods, space_name=space_name,
        )
        (OUT / slug).write_text(html, encoding="utf-8")
    print(f"Wrote {len(pages)} pages and a stylesheet to {OUT}")


if __name__ == "__main__":
    build()
