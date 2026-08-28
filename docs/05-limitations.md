# 5. Limitations

Questions this harness cannot currently answer, and the reasons why.

## Coverage is the binding constraint

Of 1,525 topic clusters, only 30 carry at least five posts on both platforms. The
merged held-out split contains 126 source posts. The group-similarity metric needs
at least four generated and four real posts in a cell, which leaves 10 scoreable
cells; the classifier and content metrics reach 22.

Two changes raised that coverage without collecting anything new. Merging the
validation and test splits for scoring roughly doubled the real posts available
per cell, and drawing four candidates per source post instead of one enlarged the
generated set by the same factor. Together they took the group-similarity metric
from 5 cells to 10 and the classifier metrics from 17 to 22, and resolved three
comparisons the earlier run could not separate. Both are documented in
[01-dataset.md](01-dataset.md) and
[02-transfer-functions.md](02-transfer-functions.md).

Further gains now require more posts in the single-platform rooms; the levers
available within the current corpus are exhausted.

The comparisons this study still cannot separate are those between the zero-shot
and few-shot configurations. Their point estimates favour few-shot on every
metric, and the gap is smaller than the variation between cells.

## The dataset is explicitly provisional

The team has stated that the current CSV is suitable for structure and prototyping
but should not be treated as the final optimised dataset. Cluster assignments carry
an estimated 3.7% problematic reassignment rate, and the stability curve
(ARI ≈ 0.80 at 90% subsample) has not yet plateaued.

Every cell in this harness is defined by `final_topic`. Cluster instability
therefore propagates directly into cell membership, and consequently into every
metric. Results should be regarded as conditional on the current clustering rather
than as properties of the audiences.

## Room canonicalisation is an assumption

`--room-mode canonical` merges `CTOs` with `CTO Reddit`, and similarly for five
other pairs, thereby nearly doubling usable cells. This asserts that the two scrapes
sample the same underlying audience. That assertion is plausible but unverified, and
if it fails for a given room, that room's cells conflate two populations.

Any finding that does not replicate under `--room-mode strict` should be treated as
provisional.

## Removing style is only partly possible

`style_residual_space` removes two linear directions: the decision boundary of a
linear platform classifier, and the between-platform difference of means. Register
is not confined to a two-dimensional linear subspace, so residual style signal
certainly remains.

A residualised score should therefore be read as "at least this much of the effect
survives style removal", never as "this effect is purely semantic".

## Embedding choice materially affects conclusions

The default TF-IDF backend keys substantially on lexical overlap. As documented in
[04-interpreting-results.md](04-interpreting-results.md), `topic_jsd` appears to
detect synthetic provenance in addition to topic drift under this backend, which
renders that particular figure unreliable for LLM-generated text.

The neural backend (`google/embeddinggemma-300m`) is expected to behave better but
introduces its own confound: it may itself encode register. This is the project's
identified central risk, and the harness mitigates it only partially, through the
raw-versus-residual comparison.

No conclusion should be reported without replication across backends. That
replication has now been run on the scaled study across six spaces — TF-IDF and
five neural spaces spanning two model families and, for EmbeddingGemma, four
Matryoshka widths — and is presented in
`experimental-notes/embedding-space.html`. Its outcome bounds what the study can
claim, and the numbers below are the reason this section exists rather than a
reassurance:

- Seven of the ten stated comparisons reach the same verdict in every space. Two
  are supported only under TF-IDF and unresolved in every neural space — "Claude
  rewriting degrades distribution match relative to the source" and "Claude
  rewriting raises self-similarity above the authentic Reddit level" — and are
  therefore not results.
- Both rest on the triangle score, and the TF-IDF triangle score fails the harness's
  own baseline-separation test: it ranks the wrong-topic `shuffle_control` second
  best of five columns rather than worst, so it is largely reading register rather
  than subject. Every neural space ranks it worst. This is why the neural triangle
  score is reported at 512 dimensions, where the separation is clean; 256 is
  marginal.
- Of the fifteen rows in the results table, eight are computed in the embedding
  space and seven are not. The seven that are not are identical across all six runs,
  which establishes that the runs differ in the space and in nothing else.
- The ordering of the system columns changes on six of those eight rows, the
  triangle score among them.

The classifier, structural and vocabulary measures are unaffected by the space, so
any conclusion resting on them alone is not exposed to this limitation.

## The classifier is imperfect

Held-out validation accuracy is approximately 0.79 (lexical) and 0.85
(stylometric), recorded in the report notes. It is not a reliable oracle. The
`calibration_gap` formulation is partially robust to this — a systematically
miscalibrated classifier affects generated and authentic pools comparably — but the
absolute `target_rate` figures should not be over-interpreted.

## The LLM judge has not been validated against human annotation

The discrimination framing controls position bias, is grounded in authentic cell
posts, and possesses meaningful floor and ceiling values. It has nevertheless not
been calibrated against human judgement on this corpus. Whether judge detection rate
correlates with human ability to distinguish these posts remains unestablished.

The `RAPIDATA_*` credentials present in `.env` suggest an available human-annotation
pathway; connecting it would permit the judge to be validated rather than assumed.

## Aspect-level evaluation is absent

The meetings describe a second representational layer above topic clusters: the
distinction between *stimuli* (the object under discussion) and *aspects* (the
evaluative dimension through which it is discussed), with differing aspect skew
across platforms within a shared topic identified as a strong transfer signal.

The harness does not model aspects. A transfer function that reproduces
platform register and topic while inverting the aspect distribution would not be
penalised by any current metric. Adding an aspect-distribution metric is the single
most valuable extension, and would slot into the existing metric registry unchanged.

## Results to date were produced before HTML entities were decoded

The scrape left HTML entities in the post text, principally `&amp;`, with `&gt;`
and `&lt;` also present. They occur in 48 of 939 posts, and **every one of those
posts is a Reddit post**, so the encoded entity is a token that appears on one
platform and never on the other.

Two consequences. The lexical platform classifier is free to key on it, which
means part of what the classifier metrics reward is the reproduction of a
scraping artefact rather than a property of how the audience writes. And `&gt;`
is Reddit's markdown quote marker, so leaving it encoded removes the quoting
structure that the writing-habit features attempt to measure.

`build_dataset.load_frame` now decodes entities once, and
`tests/test_build_training.py` pins the behaviour. The fix applies from the next
`vectorial-eval build` onward. Every number reported before that rebuild was
computed on text that still carried the entities, so the classifier and
writing-habit figures in those runs carry this artefact. The effect is bounded by
the 5.1 per cent of posts involved, and it is one-directional: it inflates the
apparent separability of the two platforms rather than hiding it.

## Only one transfer direction is evaluated

All results are LinkedIn → Reddit. The reverse direction is supported
(`--source reddit --target linkedin`) but has not been examined. Given the sharply
asymmetric platform characteristics, results in the reverse direction should not be
assumed to mirror these.

## Population shift versus behavioural shift is not separated

The meetings distinguish *population* shift (different people appear on different
platforms) from *behavioural* shift (the same people express themselves differently).
This harness measures only their combination. A transfer function could score well by
modelling either, and nothing here distinguishes the two.

Doing so would require identity linkage across platforms, which the dataset does not
provide.

## Metrics are correlational, not causal

Every metric compares distributions. None establishes that a transfer function has
recovered the *mechanism* of platform difference. A function could match every
distributional statistic while implementing an entirely unrelated internal process —
which is the reason the interpretability requirement raised on the client side is not
satisfied by these numbers alone. The `structural` metric's per-feature breakdown and
the judge's `top_tells` are the closest available approximations, and both are
descriptive rather than causal.
