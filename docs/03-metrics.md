# 3. Metrics

Every metric is **distributional**: it compares a pool of generated posts against a
pool of authentic target posts within a single bilateral cell. This follows from the
absence of one-to-one post correspondence, which leaves pairwise reference-based
scores undefined.

Metrics return cell-level scores. Aggregation to a headline figure occurs once, in
the reporting layer, so that per-cell variance remains visible. With approximately
30 viable cells, a single dense cell would otherwise dominate an unweighted mean.

All metrics are registered and selectable:

```bash
.venv/bin/vectorial-eval evaluate --split test --metrics classifier trm semantic
```

---

## `classifier` — platform-classifier confusion

A LinkedIn-versus-Reddit classifier is trained on authentic posts from the train
split alone, then applied to each transfer function's generated posts.

The naive interpretation — that a higher target-platform rate is preferable — is
incorrect in isolation, and this metric is constructed to expose that. A function
emitting the single most Reddit-like string available would approach a rate of 1.0
while destroying the content entirely. Three quantities are therefore reported
jointly:

| Score | Interpretation |
|---|---|
| `target_rate` | Proportion of generated posts classified as target platform |
| `calibration_gap` | Absolute difference between that rate and the rate for *authentic* target posts. **This is the score to read.** |
| `mean_target_prob` | Mean predicted probability; more sensitive than the hard rate when outputs cross the boundary only marginally |

The authentic target pool does not itself classify at 1.0. The objective is to
*match* the authentic distribution, not to saturate the classifier.

Two classifiers are fit deliberately. The `lexical` classifier (TF-IDF with logistic
regression) is strong but keys substantially on topic vocabulary and platform
artefacts. The `stylometric` classifier uses only the interpretable surface features
and cannot exploit topic vocabulary. Agreement between them constitutes evidence
that a transfer is stylistic rather than an artefact of word choice. Held-out
validation accuracy for both is recorded in the report notes, so that a weak
classifier is not relied upon implicitly.

---

## `structural` — interpretable feature agreement

Where the classifier metric yields a single figure, this metric provides the
decomposition that addresses the interpretability requirement raised on the client
side: which specific properties distinguish these audiences, and which of them did
the transfer function reproduce.

Twenty-one surface features are extracted per post (`features/stylometry.py`),
selected to reflect the platform differences observed by the team: promotional
vocabulary, question rate, hedging, first- and second-person usage, emoji, hashtags,
code markers, list structure, and length statistics.

For each feature and cell:

| Score | Definition |
|---|---|
| `jsd` | Jensen-Shannon divergence between binned generated and authentic distributions |
| `effect_gap` | \|d(generated, source) − d(authentic, source)\|, where d denotes Cohen's d |
| `feature_coverage` | Proportion of features whose authentic source-to-target shift was reproduced to within half its magnitude |

`effect_gap` asks whether the function displaced the feature by the *correct
magnitude* relative to the authentic platform difference, which is a stricter
criterion than displacement in the correct direction.

The report notes contain `worst_features` and `best_features`, which together
constitute the table appropriate for a client asking which properties the model
failed to capture.

---

## `trm` — Triangle-Rank Metrics

Implements the metric family of Chan et al., *Distribution Aware Metrics for
Conditional Natural Language Generation*, LREC-COLING 2024.

### Motivation

The paper's setting corresponds closely to the present one. It addresses conditional
generation in which reference diversity carries information rather than noise, and
in which only tens of references are available per condition. Both conditions hold
here: a bilateral cell contains between 5 and 150 authentic target posts, and the
*dispersion* of an audience's Reddit expression is the signal to be reproduced
rather than noise to be averaged away.

The paper further identifies the failure mode this harness most requires detecting.
Metrics based on distance to the nearest reference reward generation at the mode of
the reference distribution — the bland, central output. Prompt-based rewriting is
strongly disposed toward exactly this behaviour. Such a system scores well on
centroid distance while conveying little information about the audience.

MAUVE addresses the same concern but requires thousands of samples for its k-means
density estimate and degrades in the low-reference regime, which excludes it here.

### Base distance

TRM is a **meta-metric** defined over an underlying pairwise text distance, not a
distance in its own right. The choice of base distance determines what the statistic can
detect, so it must always be stated alongside a TRM value.

This implementation uses **cosine distance in the harness embedding space**:

| Configuration | Base distance |
|---|---|
| Default (`--embedding-backend tfidf-svd`) | Cosine over TF-IDF + truncated SVD, 256 dimensions |
| `--embedding-model google/embeddinggemma-300m` | Cosine over EmbeddingGemma, Matryoshka width per `--embedding-dim` |
| `trm_style_residual` | The same cosine distance, after platform-discriminating directions are projected out |

The base distance is recorded in every report under
`notes.<fn>.trm.pairwise_distance` and `notes.<fn>.trm.distance_space`, and the space
itself under the top-level `embedding_space`, so a value can always be traced to the
space it was computed in. TRM values computed over different base distances are not
comparable.

Because they are not comparable, a report is filed under the space that produced it.
The default space keeps `report.<split>.json`; any other space appends its slug, so
the two runs sit side by side rather than one replacing the other:

```bash
.venv/bin/vectorial-eval --run-dir runs/scaled evaluate --split heldout
# -> runs/scaled/report.heldout.json

.venv/bin/vectorial-eval --run-dir runs/scaled evaluate --split heldout \
    --embedding-model google/embeddinggemma-300m --embedding-dim 256
# -> runs/scaled/report.heldout.embeddinggemma-300m-d256.json
```

`--report-tag` overrides the suffix. `vectorial-eval report --report-tag <slug>`
re-prints a report filed this way.

The statistic requires only that `d(x,x) = 0`, so any text distance may be substituted,
including asymmetric learned distances such as BERTScore.

### The statistic

For candidates *C* and references *R* under a pairwise distance *S*, enumerate every
triangle with one vertex in *C* and two in *R*. Designate the R–R edge the *in* edge
and the two C–R edges the *cross* edges, then record the rank assumed by the in-edge:

| Indicator | Condition | Interpretation |
|---|---|---|
| I₀ | in-edge shortest | Candidates lie outside or between the references |
| I₁ | in-edge intermediate | — |
| I₂ | in-edge longest | Candidates are less dispersed than the references |

Under the null hypothesis that *C* and *R* are drawn from the same distribution each
rank is equiprobable, so each indicator has expectation ⅓. The directed statistic is
the squared deviation from that uniform profile,

```
Q(C,R) = Σₖ ( mean(Iₖ) − ⅓ )²
```

and the reported statistic symmetrises it: `TRM(C,R) = Q(C,R) + Q(R,C)`.

TRM equals zero when the two sets are distributionally indistinguishable and
increases as they diverge in either location or dispersion. **Lower is better.**

### Significance

A raw TRM value has no natural scale, so each cell additionally receives a
permutation-test p-value: *C* and *R* are pooled, reshuffled into groups of the
original sizes, and the proportion of permuted statistics at least as large as the
observed statistic is recorded. Per-cell p-values are combined using the harmonic
mean p-value (Wilson, 2019), following the paper. The harmonic mean remains valid
under arbitrary dependence between tests, which is the relevant situation here since
cells share audience rooms, an embedding space, and a generating model.

**A large p-value is the favourable outcome**: it indicates that the generated pool
is statistically indistinguishable from authentic target posts.

| Score | Direction | Interpretation |
|---|---|---|
| `trm` | lower better | 0 denotes indistinguishability |
| `trm_pvalue` | higher better | >0.05 denotes indistinguishability at conventional levels |
| `trm_style_residual` | lower better | The same statistic after style directions are projected out |
| `rank_i2` | diagnostic | Elevated values indicate candidate under-dispersion, i.e. mode collapse |
| `frechet` | lower better | Frechet distance between fitted Gaussians; the kernel-based companion metric |

`rank_i2` is the specific diagnostic for the homogenisation to which LLM rewriting
is prone.

TRM requires at least four candidates and four references per cell. On small splits
this restricts coverage considerably; `--split train` provides greater mass.

---

## `distributional` — embedding-space distribution matching

| Score | Definition |
|---|---|
| `mmd2` | Unbiased squared maximum mean discrepancy under an RBF kernel |
| `centroid_distance` | Cosine distance between pool centroids |
| `topic_jsd` | JSD between pool distributions over a shared k-means partition of the embedding space |

`topic_jsd` is the closest analogue of the "KLD/JSD over post pools" formulation
discussed in the meetings, computed over discretised semantic clusters.

Each statistic is additionally reported in the **style-residual** space. This pairing
is the central diagnostic. In the raw space, a function that merely adopts Reddit
register will appear to have matched the distribution; in the residual space only
topical agreement survives. A function whose raw score is favourable but whose
residual score is no better than `shuffle_control` has acquired register alone.

MMD is suppressed to NaN when either pool contains fewer than five items. The
unbiased estimator is severely noisy at that scale, returning negative values and
occasional large outliers; averaging these across cells previously produced a table
in which the oracle appeared inferior to the identity baseline.

---

## `semantic` — semantic preservation

| Score | Definition |
|---|---|
| `source_similarity` | Cosine similarity between each generated post and its source |
| `content_word_retention` | Proportion of the source's distinctive content words surviving into the output |
| `target_pool_similarity` | Mean similarity to the authentic target pool |

`identity` scores 1.0 on the first two by construction; `shuffle_control` scores near
the corpus floor. A useful transfer function must remain high here *while* moving on
the classifier metric.

`content_word_retention` is reported alongside embedding similarity because the
latter can remain high while specific details are discarded. This matters given the
stated requirement that transfer preserve topic and stance rather than register
alone.

---

## `degeneracy` — failure-mode guards

These render the transfer scores non-gameable.

| Score | Detects |
|---|---|
| `empty_rate`, `failure_rate` | Silent discarding of difficult cases |
| `copy_rate` | Near-verbatim reproduction of the source, i.e. absence of transfer |
| `distinct_2` | Vocabulary collapse |
| `pool_self_similarity` | Mode collapse: outputs more similar to one another than authentic posts are |
| `length_ratio_vs_real` | Systematic length mismatch |

`pool_self_similarity` should be compared against the value attained by
`target_sample`, which represents the authentic self-similarity of the target pool.
Values materially above it indicate homogenisation.

---

## `judge` — LLM-as-a-judge

The naive design — presenting a generated post and requesting a quality rating —
measures the judge's prior regarding the platform rather than whether the transfer
matched this audience on this topic, and is moreover susceptible to length and
fluency bias.

The judge is therefore framed as a **discrimination task**, the natural analogue of
a distributional metric: *N* authentic target posts from the cell are presented
alongside one candidate, and the judge is asked to identify the machine-written
item. If the transfer function is effective, the judge cannot exceed chance,
1/(N+1). `detection_rate` is consequently the score, **lower is better**, and
`chance_rate` is reported alongside it so that the reader is never comparing against
an implied zero.

This design has three properties the rubric formulation lacks: it is grounded in the
cell's authentic posts, so it tests audience and topic agreement rather than generic
platform register; position bias is controlled by randomising the candidate's slot;
and it possesses meaningful floor and ceiling values.

The judge also names the cue that identified the candidate. These aggregate into a
frequency table (`top_tells`) constituting the qualitative counterpart to the
structural metric's worst-feature list.

By default the judge is drawn from a different model family than the rewriter, since
LLM judges exhibit measurable self-preference bias. A cross-family panel with
majority voting is available:

```bash
.venv/bin/vectorial-eval evaluate --split test --metrics judge \
    --judge-panel gemini-flash gpt-5.6-small claude-haiku
```

---

## Aggregation and statistical discipline

### Common cell base

Every column in a comparison must be averaged over the **same** cells. The table
restricts to cells that every *system* column covers; a reference column scored on
fewer is averaged over the portion it covers and prints its own count as a
superscript.

This is not a nicety. Allowing each column's own coverage to define its average
once made the oracle appear *worse* than doing nothing, because it produces no
output on cells held out for generalisation and was therefore scored on a
different, harder subset. The subset bias was measured directly and is 0.03 or
smaller on every metric, an order of magnitude below the differences the table
reports, which is why averaging over the portion covered is acceptable while
letting coverage vary silently is not.

### Unweighted mean over cells

Cell-level scores are averaged **unweighted**. Size-weighting was rejected because
the largest cell holds roughly fifteen times the posts of the median, so a
weighted mean would principally report performance on a single topic.

### Bootstrap intervals and the resolution rule

Resampling is over **cells**, not posts, because cells are the independent unit.
For each of `n_bootstrap` iterations (default 500), cells are resampled with
replacement and the mean recomputed; the reported interval is the empirical 2.5
and 97.5 percentiles.

**A difference is a finding only when the two intervals do not overlap.** The
report generates a resolution table by running this overlap test over every stated
comparison, rather than relying on a human reading point estimates off a table.
Two claims in the first draft of the first study were unsupported by this test and
had to be withdrawn.

Prose states only what the table supports. Point estimates and intervals belong in
tables and figures, not in the running text, because numbers in prose invite
exactly the over-reading that produced those errors.

`dense_mean` and `heldout_mean` are recorded per score in the JSON report, so
performance on dense cells and zero-shot generalisation can be read separately.

### Direction and diagnostic rows

Each metric declares a `directions` map: `+1` where higher is better, `-1` where
lower is better, and `0` for a **diagnostic** quantity where neither extreme is
the goal.

Diagnostic rows still have a target, and the report marks the value closest to it
rather than leaving the row unmarked:

| Row | Target |
|---|---|
| `classifier.target_rate` | the rate authentic target posts achieve |
| `trm.rank_i2` | exactly ⅓, the value under a perfect distributional match |

A new metric that reports a quantity whose ideal is a middle value should declare
direction `0` and supply its reference in `TABLE_ROWS`.

### Minimum pool sizes

Every distributional statistic returns `NaN` below a minimum pool size and its
cell is excluded, rather than contributing noise. The unbiased MMD estimator on
pools of 2–4 returns negative values and occasional large outliers that dominate
any average over cells; `MIN_MMD_POOL = 5` and TRM's `MIN_POOL = 4` exist for this
reason, and `tests/test_metrics.py` pins both.

A new distributional metric needs the same guard and the same test.
