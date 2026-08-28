# 0. Experimentation guide

This document is the reasoning behind the harness rather than a description of it.
The other files in `docs/` explain *what* each component does; this one explains
*why* it is built the way it is, records the mistakes that were made and caught
during the first study, and states the rules a future experimenter should follow
to avoid repeating them. Read it before extending the harness or interpreting a
new run.

The other documents:

| Document | Purpose |
|---|---|
| [01-dataset.md](01-dataset.md) | Bilateral cells, split design, JSONL schemas |
| [02-transfer-functions.md](02-transfer-functions.md) | The interface and how to add a function |
| [03-metrics.md](03-metrics.md) | Each metric, its base distance, and its failure modes |
| [04-interpreting-results.md](04-interpreting-results.md) | A worked reading of a result table |
| [05-limitations.md](05-limitations.md) | What the harness cannot currently answer |
| [06-reporting.md](06-reporting.md) | Figure conventions, the report generator, and publishing |

Agents should also read [../AGENTS.md](../AGENTS.md), which states the same
constraints as enforceable invariants and lists the checks that must pass.

---

## 1. The problem, stated precisely

The scientific object is a **transfer function**: a map from a source-platform
post to a prediction of how the same audience would express that content on the
target platform. The measurement problem has one property that determines the
entire design.

**There is no one-to-one correspondence between posts across platforms.** A given
LinkedIn post has no single Reddit counterpart. This means no generated post has a
gold reference, which rules out every reference-based score in the BLEU and ROUGE
family. Evaluation is necessarily **distributional**: a *pool* of generated posts
is compared against a *pool* of authentic target posts.

Everything else follows from that sentence. If a future variant of the task does
have paired references — a survey-response prediction benchmark, for instance —
then most of the machinery here is unnecessary and a simpler pairwise metric is
correct. Check this assumption first, because it is load-bearing.

---

## 2. The three principles

Three commitments run through every design decision. When adding a metric or a
transfer function, check the new code against all three.

### 2.1 A single number is never trusted

Every headline number has at least one companion that can detect the way it is
gamed. The platform classifier measures whether text reads as the target
platform; on its own it rewards emitting the single most target-like string
available, which destroys the content. So it is always read alongside a semantic
preservation measure and a distributional measure. The first study's central
finding is exactly a case where the classifier and the distributional metric
disagree, and the disagreement is the result.

**Rule:** a proposed metric that can be maximised by a degenerate output is not
finished until its companion guard exists.

### 2.2 Baselines bracket the range before any system is scored

Three reference columns are computed on every run:

| Reference | Emits | Establishes |
|---|---|---|
| `identity` | the source post unchanged | floor on transfer, ceiling on preservation |
| `target_sample` | a real target post from the cell's train split | the practical ceiling for distribution matching |
| `shuffle_control` | a real target post from a *different* cell | the noise floor: correct register, wrong topic |

These are instrumentation, not candidate systems. Their purpose is to calibrate
the metrics. If a metric cannot separate the three — most importantly, if it
scores `shuffle_control` as highly as `target_sample` — that metric is
style-indexed and blind to semantics, and it must not be reported.

`target_sample` is subtle and was the source of the first serious confusion (see
§5.1). It emits a *different* real post, so it scores near zero on source
preservation. That is correct behaviour, not failure. It is the ceiling for
*distribution matching only*; `identity` is the ceiling for *preservation*.
Neither reference is the target, and no working system can occupy both corners at
once.

### 2.3 Style and semantics are measured against each other

The project's stated central risk is that embeddings organise posts by writing
style rather than subject. Every embedding-based metric is therefore reported
twice: raw, and after the platform-discriminating linear directions are projected
out ("style-removed"). A function that looks strong raw but no better than
`shuffle_control` once style is removed has learned only the register. The
style-removal is a lower bound (two linear directions cannot capture all of
register), so a small raw-vs-removed gap is reassuring but a large one is a
warning rather than a verdict.

---

## 3. Why the distribution-aware metric is the right one

The harness adopts the Triangle-Rank Metric (TRM) of Chan et al.,
*Distribution Aware Metrics for Conditional Natural Language Generation*,
LREC-COLING 2024. Three facts make it the correct choice, and each is a
constraint a replacement metric would also have to satisfy:

1. **The regime is tens of references, not thousands.** A bilateral cell holds
   5–150 posts. MAUVE, the obvious alternative, needs thousands of samples for its
   k-means density estimate and degrades badly below that. TRM was designed for
   exactly the small-sample regime this dataset lives in.

2. **The failure mode to detect is mode-seeking.** Metrics based on distance to
   the nearest reference reward generating the *centre* of the reference
   distribution — the bland, average output. Prompt-based rewriting is strongly
   disposed toward exactly this. TRM measures location and spread jointly through
   its triangle-rank counts, so it penalises a system that hits the average tone
   while collapsing the variety. A centroid-distance metric cannot see this.

3. **It has a meaningful null.** Under the hypothesis that candidates and
   references come from one distribution, each of the three triangle ranks occurs
   one third of the time. This gives both a permutation-test p-value per cell and
   an interpretable diagnostic (excess in the "candidate is inside" rank is the
   signature of collapse).

**TRM is a meta-metric.** It is built on top of a pairwise text distance — cosine
in the harness embedding space here — and the choice of that distance determines
what it can detect. The base distance is recorded in every report
(`notes.<fn>.trm.pairwise_distance`) and must always be stated alongside a value.
Values computed over different base distances are not comparable. If a future run
switches to a neural embedding, every TRM number changes and the comparison must
be redone from scratch.

---

## 4. Scaling without collecting data

The binding constraint on this dataset is **coverage**, not corpus size: only 30
of 1,525 topic clusters have enough posts on both platforms to form a cell, and
the group-level metrics need a minimum number of posts per cell. Before
requesting more data, exhaust the two axes that enlarge the evaluation for free.
Both were applied in the first study and together roughly doubled coverage.

### 4.1 Merge the held-out splits for scoring

Validation and test are both held out of every transfer function's exemplar pool,
so merging them into a single `heldout` split introduces no leakage and roughly
doubles the reference posts available per cell. This is a pure win and should be
the default evaluation split.

The mechanism matters and is worth preserving exactly: `exemplar_ids` are always
drawn from **train**, and only the *scoring* references widen. Any future split
change must keep that invariant, because it is the entire leakage-control
argument.

### 4.2 Draw several candidates per source post

The distributional metrics compare a pool of candidates against a pool of
references. Drawing `k` candidates per source post at temperature 1.0 enlarges the
candidate pool by a factor of `k` with no new corpus data, and it is what the TRM
paper prescribes. It is also the only way to observe how much a model varies when
asked the same question twice — which turned out to be a headline result.

**This exposed and required a real fix.** The LLM cache originally keyed only on
the prompt, so the second through `k`-th draws would have been served the
identical cached completion, and the "candidate pool" would have been `k` copies
of one post. The cache key now includes a variant index. **Any future caching
layer must incorporate the sample index**, or multi-candidate sampling silently
degenerates.

The effect was not cosmetic. With one draw per post the vocabulary-variety metric
could not distinguish the rewriters from the source, because the repetition
happens *between* draws on different inputs. With four draws it separated them
clearly and became a supported finding. A single-sample run understates the very
failure the harness exists to detect.

### 4.3 What scaling did to the conclusions

Moving from 66 single-sample tasks to 126 tasks × 4 candidates took TRM coverage
from 5 cells to 10 (counted over the *system* columns; the oracle covers 6 of
those 10, per §5.1) and the classifier metrics from 17 to 22, and it resolved
three comparisons the smaller run could not separate. Notably, the claim
"rewriting improves platform register" went from *unresolved* (and therefore not
claimed) to *supported*. Scaling is not just precision; it changes which
statements the report is entitled to make.

---

## 5. Errors made during the first study, and the rules they produced

These are recorded because they are the failures most likely to recur. Each is a
case where a plausible-looking number was wrong, and none would have raised an
exception.

### 5.1 Averaging over inconsistent cell subsets

The first result table averaged each column over whatever cells it happened to
cover. Because `target_sample` produces nothing on held-out cells, it was scored
on an easier subset than `identity`, and the oracle appeared *worse* than the
identity baseline on MMD — an impossible result that a reader would either
disbelieve or, worse, believe.

**Rule:** columns in a comparison table must be averaged over a common set of
cells. The table restricts to the cells every *system* column covers, and any
reference scored on fewer prints its own count as a superscript. Never let a
column's coverage silently define the comparison base.

### 5.2 Reading point estimates without their spread

The first draft stated "few-shot is the only configuration that improves both
axes" and "the calibration gap falls from 0.355 to 0.166" as findings. Checking
the bootstrap intervals showed that **every calibration-gap interval overlapped**
and the few-shot-vs-zero-shot differences were within noise. Two stated findings
were unsupported.

**Rule:** a difference is a finding only when the bootstrap intervals over cells
do not overlap. The report carries an explicit resolution table listing every
comparison with a supported/not-resolved verdict, generated by the overlap test
rather than written by hand. Point estimates go in tables and figures; the prose
states only what the intervals support. Numbers in the running text were later
removed for this reason — they invited exactly the over-reading that produced the
error.

### 5.3 A statistic that is noisy at small n

The MMD estimator returns negative values and large outliers on pools of 2–4,
which then dominated any average over cells. This is what produced the impossible
oracle result in §5.1 alongside the subset problem.

**Rule:** every distributional statistic has a minimum pool size below which it
returns NaN and is excluded, rather than contributing noise. The test suite pins
this: `tests/test_metrics.py` asserts MMD is NaN below the threshold and checks
that TRM recovers its theoretical null and detects the collapse signature. A new
metric needs the same guard and the same test.

### 5.4 Hardcoded figure bounds

A per-cell figure had its y-axis fixed at 0.95 while the scaled run produced
values up to 1.33, so the largest points rendered *above* the plot, invisible or
clipped. Similarly, several figure captions were hardcoded from the pre-scaling
run and became false after scaling ("all five scored cells" when there were ten).

**Rule:** figure axis limits are derived from the data they plot, never fixed.
Counts and ranges quoted in captions are computed from the report, never typed.
A layout audit (`_CHAR_W`-based width checking) runs against the built SVG to
catch text that overflows the viewBox, because monospace figure text that wraps
past the edge is easy to miss in a quick look.

### 5.5 Claims that drift from the data during editing

Late in the process a note read "Claude is above the identity baseline in every
one of the ten scored cells." Verifying rather than trusting the sentence showed
it was **nine** of ten; one cell reverses. The claim was corrected and the
exception named.

**Rule:** before writing any count or "in every/all" claim into prose, compute it
against the current report. This is cheap and catches the class of error where a
figure or number was right for an earlier run and silently became wrong.

### 5.6 A metric whose value is not what it appears

Topic-JSD scores the LLM columns as badly as the shuffle control, which looks like
complete topical failure — but source similarity stays high and the topic is
preserved on inspection. The metric is computed over k-means clusters fit on
*authentic* posts, so it appears to detect machine origin as well as topic drift.

**Rule:** when two metrics disagree, inspect the outputs before believing either.
Topic-JSD is documented as unreliable for generated text rather than reported as a
finding, and the report says so explicitly. A metric that cannot separate the
baselines cleanly (§2.2) is a candidate for this treatment.

---

## 6. Interpreting a result table: the fixed procedure

Follow this order. It is the sequence that makes the errors in §5 visible.

1. **Check the baselines first.** `identity` should be worst on transfer and
   perfect on preservation; `target_sample` should be best on the distributional
   metrics; `shuffle_control` should have good register but bad topic agreement.
   If any of these is violated, the metric is broken and no system conclusion is
   admissible.

2. **Read transfer and preservation together, never separately.** A gain on the
   classifier that comes with a drop in source similarity is content being
   discarded, not transfer.

3. **Consult the distributional metric, and expect it to disagree.** The classifier
   is a per-post measure; TRM is a per-group measure. They answer different
   questions and their disagreement is informative.

4. **Check raw against style-removed.** A result that survives style removal is
   about content; one that collapses to the shuffle-control level was about
   register.

5. **Consult the resolution table.** Only differences with non-overlapping
   intervals are findings. Everything else is a hypothesis for a larger run.

6. **When two metrics disagree, read the actual generated text.** Several of the
   most important observations — the repeated closing-question structure, the
   topic-JSD artifact — came from reading outputs, not from the numbers.

---

## 7. Practical operation

**Environment.** `uv venv && uv pip install -e .` for the offline path;
`.[embeddings]` adds the neural backend. Nothing is installed into the system
Python. Credentials live in `.env` (`OPENROUTER_API_KEY`).

**The embedding space is chosen per evaluation.** `--embedding-backend`,
`--embedding-model` and `--embedding-dim` on `evaluate` select the space, and it
applies to every embedding-based metric at once: the semantic, distributional and
TRM metrics, the style-residual projection, the topic clusters, and the
self-similarity guard in `degeneracy`. The classifier and structural metrics do
not use it, so their values are a useful check that two runs differ only in the
space. Since scores from two spaces are not comparable, the report is filed under
the space that produced it: the default keeps `report.<split>.json`, and anything
else appends the space slug. A single space is fit once per evaluation and shared
across every transfer function, so encodings are memoised per text; both backends
encode a document independently of its batch, which is what makes that safe.

**One model client for every provider.** All models route through OpenRouter's
OpenAI-compatible endpoint, so a Claude-vs-GPT comparison differs only in a model
string and is not confounded by two SDKs' differing retry and sampling defaults.
Model slugs drift; refresh with `vectorial-eval models --refresh` before assuming
a name is current, because the first attempt in this study used stale slugs.

**Caching is by full request.** Generation is the expensive step and metrics are
the step you iterate on, so completions are cached on disk keyed on the full
request including the sample index (§4.2). This lets a metric bug be fixed and the
run re-scored for free. It also means a stale cache silently serves old outputs —
clear the cache directory if a prompt or model changes in a way the key does not
capture.

**Failures are values, not exceptions.** A refusal or persistent API error returns
`ok=False` and is recorded, because a function that silently dropped 5% of a cell
would bias every distributional metric toward whatever it kept. The degeneracy
metric surfaces the failure rate, so a quietly-failing run is visible rather than
hidden.

**Parallelism.** The generation step is concurrent (bounded by
`LLMConfig.max_concurrency`). Independent evaluation passes can run at once by
pointing a second run directory at the same outputs via symlink; the two write
different report files. This halves wall-clock when producing both the full and
the system-only tables.

**Determinism.** Splits are `blake2b(post_id, seed)`, not a shuffle, so adding data
never reshuffles existing assignments. The report site build is deterministic:
rebuilding without changing the reports produces byte-identical pages. Verify this
after touching the generator — a non-deterministic build usually means an
unsorted dict or a `hash()` leaked in somewhere.

**Publishing.** `scripts/publish_report.sh` mirrors the built site to
`s3://vectorial-reports/<timestamp>/` and refreshes `latest/`, using the `wasabi`
AWS profile. Two non-obvious details are baked in: `aws s3 cp --recursive`
segfaults against the Wasabi endpoint, so uploads are per-object via
`s3api put-object`; and content types must be set explicitly, because the S3
default of `application/octet-stream` makes browsers download the HTML rather than
render it.

---

## 8. The largest open gap

The harness measures the *combination* of platform register and topic. It does not
measure **aspect** — the evaluative dimension through which a topic is discussed,
which the project meetings identify as a strong transfer signal. A function that
reproduces register and topic while inverting the aspect distribution would incur
no penalty under any current metric. An aspect-distribution metric is the
single most valuable extension and fits the existing metric registry without
structural change. It should be the first addition once the dataset's aspect layer
exists.

Two further gaps are worth restating because they bound what any result here can
claim: the harness cannot separate *population shift* (different people appear on
each platform) from *behavioural shift* (the same people write differently), which
would require cross-platform identity linkage the dataset lacks; and every metric
is a distribution comparison, so none establishes that a function has recovered
the *mechanism* of platform difference rather than merely matching its surface
statistics. The interpretability requirement from the client side is not satisfied
by these numbers alone — the structural feature breakdown and the judge's reported
cues are the closest available approximations, and both are descriptive.
