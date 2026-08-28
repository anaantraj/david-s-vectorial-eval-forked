# 8. Transfer-function roadmap

The remaining transfer functions, specified precisely enough to be built, with the
harness changes each requires, the cost of each against the actual size of this
dataset, and the specific failure that would sink it.

This is a design document. Nothing in it has been run.

## Where the method set stands

The project committed to a range that brackets the problem: an in-domain predictor
as a ceiling, a no-adaptation baseline as a floor, and existing cross-domain
methods in between rather than only internal variants. The named list was prompt
tuning, in-context learning, LoRA fine-tuning, steering, and plug-and-play language
modelling, with trait mediation as the intended novel contribution.

| Function | Role | State |
|---|---|---|
| `identity` | Floor | Implemented |
| `target_sample` | Oracle reference | Implemented |
| `shuffle_control` | Negative control | Implemented |
| `llm_rewrite` | Prompting | Implemented |
| `llm_fewshot` | In-context learning | Implemented |
| Aspect-aware prompting | Prompting, aspect-conditioned | In progress |
| Soft-prompt tuning | Prompt tuning | In progress |
| LoRA | Parameter-efficient fine-tuning | In progress |
| Steering | Activation steering | In progress |
| **Plug-and-play LM** | Classifier-guided decoding | **Section 1** |
| **In-domain predictor** | Trained ceiling | **Section 2** |
| **Trait-mediated transfer** | Novel contribution | **Section 3** |
| **Reverse direction** | Second transfer direction | **Section 4** |

## The scale everything below is designed against

These numbers constrain every method in this document and should be read before any
of them.

| Quantity | Value |
|---|---|
| Bilateral cells | 30, of which 6 dense and 4 reserved for zero-shot |
| Posts in viable cells | 939 (444 LinkedIn, 495 Reddit) |
| Train-split posts | 630 (293 LinkedIn, 337 Reddit) |
| Train-split transfer tasks | 293 |
| Merged held-out tasks (val plus test) | 126 |
| Median posts per cell per platform | 7.5 |
| Cells scoreable by the group-similarity metric | 10 |
| Cells scoreable by the classifier and content metrics | 22 |

The consequences are direct and should not be argued around.

**Anything that must be estimated per cell is out of reach.** A per-cell classifier,
a per-cell trait model, or a per-cell adapter would be fitted on a median of seven
or eight posts. Every method below therefore estimates its parameters globally or
per room, and carries the cell identity in the conditioning text rather than in the
parameters.

**Anything that requires thousands of training examples is out of reach.** With 337
train-split Reddit posts, a full fine-tune, a learned latent-variable model with a
free bottleneck, or a discriminator with a large capacity will memorise rather than
generalise. Low-rank adaptation, soft prompts of a few dozen tokens, and small
discriminators are the viable capacity range.

**Only large differences will resolve.** Under invariant 3 a difference is a finding
only when the bootstrap intervals over cells do not overlap. With 10 to 22 scoreable
cells, the intervals already observed in `runs/scaled` are wide. The study is
powered to detect large effects and should not be planned as though it will separate
close ones. The forward comparison between zero-shot and few-shot prompting is
already unresolved for exactly this reason.

---

## 1. Plug-and-play language modelling

### Formulation

Guide decoding from the base model with an attribute discriminator instead of
changing the model's weights. Two forms are available.

The original PPLM form runs gradient ascent on the cached key and value activations
at each step to raise the discriminator's probability of the target attribute. It
requires several backward passes per generated token, and its behaviour depends on
step size, number of steps, and a fluency term that must be tuned jointly with them.

The FUDGE form leaves the model untouched and reweights the next-token
distribution. For the top *k* continuations proposed by the base model, a
discriminator scores each partial sequence for the probability that the completed
post belongs to the target platform, and the two scores are combined:

```
log p(x_t | x_<t) + lambda * log p_disc(target_platform | x_<=t)
```

**Use the FUDGE form.** It costs one small forward pass per candidate token rather
than several backward passes through an 8B model, its single knob is easier to
report honestly, and it is the modern equivalent that the method list was pointing
at. Implement it as a `LogitsProcessor` over `meta-llama/Llama-3.1-8B-Instruct`,
with the same prompt `llm_rewrite` uses, so the comparison against prompting
isolates the guidance term.

The discriminator is a small encoder fine-tuned on train-split posts truncated at
random prefix lengths and labelled by platform. It is **global**, not per cell:
337 target-platform posts cannot support 30 discriminators. Cell identity stays in
the prompt, where it already lives for every other function.

### The circularity problem, and how to avoid it

The harness already trains a platform classifier in `metrics/classifier.py`. It is
the obvious discriminator and it must not be used. Guiding decoding against the same
function that scores the output turns `classifier.target_rate` and
`classifier.calibration_gap` into a measure of how hard the guidance was pushed. The
metric was constructed to expose exactly the failure mode that guidance invites,
which is raising the platform score by discarding content, and it cannot expose it
while it is the objective.

Three mitigations, to be applied together:

1. **Disjoint data.** Partition the train split by `blake2b(post_id, seed_guidance)`
   into a guidance half and a metric half, and fit the guidance discriminator on the
   guidance half only. This costs the metric classifier roughly half its training
   data, which at a held-out accuracy of 0.79 lexical and 0.85 stylometric is a real
   cost and must be re-measured, not assumed.
2. **Disjoint feature space.** The guidance discriminator is a subword-level neural
   model over prefixes. The metric classifiers are a bag of word n-grams and 21
   hand-built surface features. These are different families, so agreement between
   them is evidence rather than tautology.
3. **Declare the classifier row diagnostic for this function.** Report the method on
   the measures that involve no platform classifier: the triangle score, the
   distribution metrics, semantic preservation, the structural feature breakdown,
   and the judge. Fix this before the run, not after seeing the numbers.

One further exposure is worth stating. The style-residual space used by the
distribution metrics is itself derived from a linear platform classifier fitted on
the train split. Residualised scores for a guided decoder are therefore partly
exposed to the same circularity, more weakly than the classifier metric but not by
zero. The raw-space scores are clean.

### What the harness needs that does not exist

- A local generation wrapper around the cached Llama weights, with a
  `LogitsProcessor` hook. The soft-prompt, LoRA and steering work in progress needs
  the same thing; it should be built once and shared rather than three times.
- Per-draw seeding. `--n-samples k` must produce *k* distinct candidates. The LLM
  path enforces this through the draw index in the cache key (invariant 5); the
  local path needs the equivalent, which is a generator seed derived from
  `(task_id, sample_index)`. Without it the candidate pool is *k* copies of one
  post, silently.
- A training script and a checkpoint location for the guidance discriminator. It is
  a few tens of megabytes and belongs with the adapters, not in the model cache.
- A split of the train pool into guidance and metric halves, plumbed through
  `_fit_classifiers` so the metric classifier can be refitted on its half.

### Compute and data

The cheapest method here in compute and the most expensive in engineering. The
discriminator trains on 315 posts in minutes on one 4090. Generation of 126 tasks at
four draws, with top-50 rescoring at roughly 200 tokens per post, is on the order of
five million short forward passes through a small encoder, which batches into well
under an hour. No cluster time beyond a single GPU.

### The risk that sinks it

**Lambda has no honest tuning signal.** It trades platform score against content
preservation continuously, and there is no paired held-out example to select it on.
Tuning it against a harness metric on the validation split is tuning on the
evaluation, since validation posts are half of the merged scoring split. Set lambda
on train-split source posts by matching the authentic train-split target rate, then
freeze it and never touch it again. If a sweep is run at all, it must be declared in
the report.

The secondary risk is that a discriminator trained on 315 posts keys on surface
artefacts, principally hashtags and promotional openers. Guidance then improves the
platform score by deleting LinkedIn markers rather than by adopting Reddit
structure. The structural metric's per-feature breakdown is the diagnostic: a
function that moves only the emoji, hashtag and promotional-vocabulary features while
leaving question rate and list structure unchanged has done the shallow thing.

### Effort

Four to six engineer-days, conditional on the shared local generation wrapper
existing. Ten or more if it has to be built here.

---

## 2. The in-domain predictor

### Why this is not `target_sample`

`target_sample` emits a real Reddit post. It is an oracle. It cannot be beaten by any
generator, it is not a model, and it says nothing about what a system trained on
target-domain data can achieve. The ceiling the project promised is a **model trained
on the target domain**, evaluated under the same generation regime as every system
column. That function does not exist and its absence is a gap in the promised
baseline set.

### Formulation

`in_domain_lm`. Take the conditional target-language model that
`data/build_training.py` already prepares, which is trained on real train-split
target posts conditioned on `(room, topic, platform)`, and adapt
`meta-llama/Llama-3.1-8B-Instruct` with LoRA on it. At inference, supply only the
task's `(room, topic)` and sample a post. **The source post is not placed in the
context.**

This is the same training run as the LoRA transfer function. The two differ only in
inference-time conditioning: the LoRA transfer function additionally supplies
`task.source_text` as the content to be carried across, and `in_domain_lm` does not.
The ceiling therefore costs one extra inference pass over an adapter that is already
being trained, which makes it the cheapest item in this document and the reason it
should be built first.

Use identical rank, schedule, decoding temperature and `--n-samples` to the LoRA
function, so that the difference between the two columns is the source conditioning
and nothing else.

### What it is a ceiling on, and what it is not

It upper-bounds the pool-matching measures that do not involve the source post:
the classifier metric, the triangle score, the distribution metrics, the structural
feature agreement and the judge. It has the target distribution and carries none of
the burden of preserving a particular source post's content.

It is emphatically **not** a ceiling on `semantic.source_similarity` or
`content_word_retention`. On those it will score near the `shuffle_control` level
within its own cell, because it is producing a topically appropriate post that has
no relation to the specific source. That is the correct and useful reading: the
target for a real transfer function is to score like `in_domain_lm` on distribution
match while scoring like `identity` on content preservation. Stating both columns
side by side is what makes the trade-off legible.

On the four zero-shot cells it has never seen the `(room, topic)` pair and falls
back to whatever the base model infers from the two strings. This degradation is the
same one `llm_fewshot` exhibits and should be reported the same way, through
`heldout_mean`.

### What the harness needs that does not exist

- A transfer function that ignores `task.source_text`. Trivial in code, but the
  degeneracy and semantic metrics will read oddly for it and the report needs to say
  why rather than leaving a reader to conclude the model is broken.
- **A decision about column role, and a change to the figure convention.**
  `in_domain_lm` is a reference column, not a system column, because it is a ceiling.
  Under invariant 2 the common cell base is defined by the system columns and a
  reference column prints its own coverage. The reporting rule in
  [06-reporting.md](06-reporting.md) currently requires every comparison figure to
  show exactly three references. Adding a fourth is a deliberate change to that
  convention and must be made in `build.py`, not worked around per figure.
- **A train-internal development slice.** `data/build_training.py` currently emits
  its dev split from validation posts, and validation is half of the merged `heldout`
  split that everything is scored on. Selecting a checkpoint or a hyperparameter by
  perplexity on validation target posts is selection on part of the scoring pool. It
  is not exemplar leakage and it does not violate invariant 1, but it is a real
  optimistic bias on the trained columns and it applies to the soft-prompt, LoRA and
  steering functions as much as to this one. Carve a dev slice out of *train* by
  `blake2b(post_id, seed_dev)`, select on that, and leave validation untouched. This
  is worth fixing now, while three trainable functions are still being built, rather
  than after their numbers are in a table.

### Compute and data

337 train-split target posts across 30 cells. LoRA at rank 8 to 16 for a small
number of epochs on one 4090, checkpointing every few hundred steps and every epoch
so a killed job resumes. Under an hour of GPU time. No new data.

### The risk that sinks it

**Memorisation.** At 337 examples a low-rank adapter can reproduce training posts,
at which point the ceiling collapses into `target_sample` and stops being
informative about what a trained model can do. Measure it directly before reporting
the column: compute the maximum similarity of each generated post against the cell's
train pool and compare it against the same statistic for `target_sample` against a
disjoint pool. If generations are near-duplicates of training posts, reduce rank or
epochs until they are not, and report the check.

The second risk is that on the four zero-shot cells it produces generic
platform-flavoured text, which drags a headline mean computed over all cells. The
`dense_mean` and `heldout_mean` fields already exist for this and should carry the
reading.

### Effort

One to two engineer-days on top of the LoRA work already in flight. The
train-internal dev slice is a further day and benefits three other functions.

---

## 3. Trait-mediated transfer

The intended novel contribution. The stated bar is that it must outperform prompt
tuning to be claimed as a method rather than reframed as a broader task. This
section is the longest because it is the one where a vague formulation would waste
the most time.

### What a trait is

A trait is a **latent, named, population-level attribute that governs how an
audience discusses a subject, as distinct from what it discusses**. A trait vector
is a point in a fixed low-dimensional space with named axes. A post's trait vector
is the expression of those attributes in that post; a population's trait vector is a
statistic over its posts.

Three candidate spaces, in decreasing order of readiness:

**(a) The eight Vectorial stylistic dimensions.** They arrive with descriptions and
0/1 anchors, they are already computed for 490 of the 939 harness posts, and they are
interpretable by construction, which is the requirement the client keeps raising.
Blocked on obtaining the rater itself, per
[07-vectorial-aspect-style-integration.md](07-vectorial-aspect-style-integration.md).

**(b) A learned bottleneck.** An encoder maps a post to eight to sixteen dimensions,
trained to support generation of the post while a residualising or adversarial term
removes topic information. Not interpretable without a post-hoc naming pass.

**(c) Aspect distributions** over the per-cluster aspect vocabularies. Blocked on
the blind rerun, since the vocabularies were discovered with platform labels visible
and may have been constructed to separate the platforms.

**Adopt (a), with (b) as the fallback if the rater does not arrive.** The argument is
scale. At 630 train-split posts a free bottleneck has no chance of being identified
from data; a supplied rubric contributes the inductive bias we cannot afford to
learn. Choosing (b) first would be choosing to spend the entire data budget
discovering axes that (a) hands over for the price of an email.

### How traits are learned without paired data

They are not learned per pair. They are estimated per `(cell, platform)` as a
population statistic over train-split posts:

```
t(c, p)     = mean trait vector over train-split posts of cell c on platform p
sigma(c, p) = per-dimension standard deviation over the same posts
```

No pairing is required because the object being compared is a distribution of trait
vectors, not a post against a post. This is the same reason the rest of the harness
is distributional.

### Estimating the population-level trait shift

Define the shift and the dispersion ratio for cell *c*:

```
Delta(c) = t(c, target) - t(c, source)
rho(c)   = sigma(c, target) / sigma(c, source)
```

The whole method rests on `Delta` being **systematic** rather than a per-cell
accident. Test that before building anything on top of it, with an additive
fixed-effects decomposition over the 30 cells:

```
Delta(c) ~ Delta_platform + Delta_room(room(c)) + epsilon(c)
```

Fit it leave-one-cell-out and report predictive error on the held-out cell, not
in-sample fit. **This is the go/no-go test and it costs two days once trait vectors
exist.** If `Delta` is not predictable for a cell the model has not seen, the method
has no mechanism on unseen cells and must be scoped to seen cells, which is a much
weaker claim and should be decided deliberately rather than discovered at
submission time.

This decomposition is the collaborators' decomposed-transfer idea made concrete.
The **topic-level style shift** is the within-cell term, readable directly from a
cell's own exemplars and available only on the 26 non-zero-shot cells. The
**population-level trait shift** is `Delta_platform + Delta_room`, which generalises
to an unseen topic inside a known room. The decomposition is testable at this scale
precisely because rooms repeat across cells even though cells do not repeat.

### How traits condition generation

Three variants, in increasing depth and cost.

**(i) Trait-conditioned prompting.** Rate the source post, apply `Delta(c)` clipped
to the valid range, and place the resulting target trait vector in the prompt as
named dimensions with target levels and short instructions derived from the rubric
anchors. Requires the rater at inference time, which is one extra call per post.
Directly comparable to `llm_rewrite` and to the aspect-aware prompting being built
in parallel. This is the variant that can be built now.

**(ii) Trait-conditioned soft prompt.** A small network maps a trait vector to *n*
soft-prompt tokens prepended to the Llama context. Train it on train-split target
posts to raise the likelihood of each post given its own trait vector and its
`(room, topic)`. At inference, supply the shifted trait vector rather than the
observed one. **This is the variant that is a method rather than a prompt, and it is
the one that has to beat soft-prompt tuning**, because it is the same machinery with
the trait vector as the only addition. That is the cleanest possible statement of
the contribution: identical base model, identical adaptation mechanism, one extra
conditioning signal.

**(iii) Trait-directed steering.** Regress an activation-space direction onto each
trait dimension using train-split posts, then steer by the amount `Delta` requires.
Strictly more expensive and strictly less identifiable at 337 posts. Record it as
future work and do not schedule it.

### What "outperform prompt tuning" has to mean

State the bar before running, because it will be tempting to soften it afterwards.
Variant (ii) beats soft-prompt tuning when the bootstrap intervals over cells do not
overlap on at least one headline score, the verdict holds in more than one embedding
space per [05-limitations.md](05-limitations.md), and there is no offsetting loss on
semantic preservation. An overlapping interval is not a result under invariant 3, and
two claims in the first study had to be withdrawn for exactly this.

Given 10 to 22 scoreable cells, only a large difference will clear that bar. The
honest outcome if it does not clear it is the reframing that has already been named
as the alternative, in which trait mediation is presented as an analysis of the
transfer problem rather than as a winning method. Planning for that outcome now costs
nothing; discovering it in September costs the submission.

### What the harness needs that does not exist

- **The trait rater, callable on generated text.** This is the blocker named in
  section 3 of [07-vectorial-aspect-style-integration.md](07-vectorial-aspect-style-integration.md).
  Vectorial shipped ratings of real posts, not the rater. Without it there is no way
  to place a generated post in the trait space, which blocks both the conditioning
  and the metric. A reimplementation from the published rubric is possible but
  introduces a rater-identity confound sitting exactly on the axis being measured,
  so it requires re-rating a held-out slice of real posts and reporting the
  agreement rate.
- **A `trait` metric.** Per-cell distance between the generated and authentic target
  trait distributions, in the dispersion form rather than the cosine form for the
  reasons given in 07, plus the per-dimension breakdown that answers the
  interpretability request. It needs the standard minimum-pool guard, a test pinning
  it, and the baseline separation check: if it does not rank `shuffle_control` worst,
  it is reading register and must not be reported.
- **Trait vectors materialised per post**, as a side table keyed on `post_id`, so
  population statistics are computed once rather than re-rated every run.
- **The eleven unrated cells rated.** The method is otherwise defined on 19 of 30
  cells, which shrinks the common comparison base under invariant 2 and drops the
  largest cell in the corpus.

### Compute and data

Negligible GPU for variants (i). Variant (ii) is one soft-prompt training run on
337 target posts, comparable to the soft-prompt function already in flight, under an
hour on one 4090. The real cost is rater calls: rating 939 real posts plus every
generated post, which at four draws over 126 tasks per system is roughly 500 calls
per system column.

### The risk that sinks it

**Circularity between the rater and the generator.** If an LLM rates the traits, an
LLM generates the text, and the trait space was itself constructed by an LLM with
platform labels visible, then the method can score well by producing text the rater
scores as shifted, without any real shift having occurred. Two guards: the trait
metric must pass the baseline separation test, and the headline claim must also hold
on the classifier, structural and vocabulary measures, which do not involve the
rater at all. If the claim holds only on the trait metric, it is not a claim about
the world.

**Too few degrees of freedom in the decomposition.** With roughly twelve rooms across
30 cells, a room term is estimated from a median of two or three cells. Report
leave-one-out predictive error, never in-sample fit, and be prepared for the room
term to be indistinguishable from the platform term.

### Effort

| Piece | Days |
|---|---|
| Go/no-go decomposition test, once trait vectors exist | 2 |
| Variant (i), trait-conditioned prompting | 3 to 4 |
| `trait` metric with guards and tests | 2 to 3 |
| Variant (ii), trait-conditioned soft prompt, including training | 6 to 10 |

Two to three and a half engineer-weeks in total, gated on an external dependency
that is not under our control.

---

## 4. The reverse direction

### Formulation

Nothing conceptual is new. Rebuild with `--source reddit --target linkedin` and
rerun the method set. The value is that platform asymmetry is a claim the study
currently makes and does not test, and it is named in
[05-limitations.md](05-limitations.md) as an open question.

The cells are unchanged, since a cell is a room crossed with a topic and holds both
platforms. What changes is which posts become tasks and which become exemplars.
Tasks are drawn one per source post, so the forward direction draws from 444 LinkedIn
posts and the reverse from 495 Reddit posts, and the exemplar pool becomes the 293
train-split LinkedIn posts. **The reverse direction is not the smaller experiment.**
The exact task counts must be measured by running the build rather than assumed: the
forward direction yields 66 test tasks from 89 LinkedIn test posts because
zero-shot-cell and empty-reference-pool exclusions remove the rest, and the same
reductions will apply on the reverse side.

### What the harness needs that does not exist

- **A direction-safe output path.** `manifest.json` records the direction, but the
  build writes to the same `data/` directory regardless, so a reverse build silently
  overwrites the forward one and any run directory still holding forward outputs
  would then be scored against reverse references. This is the single most likely way
  to produce wrong numbers without an error in this whole document. Make the
  direction part of the output path, and make `evaluate` refuse to score outputs
  against a manifest whose direction does not match the one recorded in the run
  directory.
- A check that no prompt or figure hardcodes the platform names. The rewriter is
  already direction-ready: `llm_rewrite.py` carries platform descriptions for both
  LinkedIn and Reddit and selects on the task. The report generator has not been
  audited for this.

The metrics themselves are direction-agnostic in code, since `EvalContext` carries
the source and target platforms, and the structural metric's `feature_coverage`
inverts cleanly because it is defined against the authentic source-to-target shift
in whichever direction that is.

### Compute and data

No new data and no GPU. One rebuild, one pass of the LLM systems, and one evaluation
per embedding space. At four draws over roughly 130 to 160 tasks it is a little more
API spend than the forward run, not less.

### The risk that sinks it

**The two directions are not comparable as numbers.** LinkedIn is measurably more
register-coherent within a topic than Reddit: on the Vectorial style space, LinkedIn
is tighter in 32 of 38 clusters, with mean dispersion 0.559 against Reddit's 0.703.
Generating toward the tighter distribution is an easier target, so reverse-direction
scores are likely to look better on every pool-matching measure for reasons that have
nothing to do with the transfer function.

Do not place forward and reverse scores in one table as though they were systems in a
comparison. They are two different tasks. The object that is comparable across
directions is each system's position relative to its own `identity`,
`shuffle_control` and `target_sample` anchors. Report the reverse direction as its
own table with its own anchors, and state the asymmetry claim in those terms.

### Effort

Two to three engineer-days, including the direction-safe build path. The build path
is a correctness fix worth making regardless of whether the reverse experiment is
run.

---

## Priority sequence

Today is 5 August. ICLR closes 25 September, which is 51 days away. ARR October
closes 12 October, 68 days away.

**Set an internal generation freeze of 11 September for the ICLR submission.** The
replication across embedding spaces required by [05-limitations.md](05-limitations.md)
is not optional, it runs after generation, and the six-space replication on the
scaled study is what withdrew two claims from the first draft. Two weeks between
freeze and deadline covers that replication, figure regeneration and writing. It does
not cover discovering a problem.

The sequence below assumes one engineer and is ordered by dependency and by cost of
being wrong, not by interest.

| Window | Work | Why here |
|---|---|---|
| 5 to 12 Aug | In-domain ceiling, and the train-internal dev slice | Cheapest item, closes a promised baseline, and the dev slice fixes an optimistic bias on three functions still being built |
| by 8 Aug | Send the rater request and the three data issues to Vectorial | Everything trait-mediated is blocked on this. If the request goes out later than this week, trait mediation does not fit ICLR at all |
| 12 to 22 Aug | Reverse direction, with the direction-safe build path | No dependencies, closes a named limitation, and the build path is a correctness fix |
| 18 to 29 Aug | Plug-and-play guided decoding | Completes the promised list of existing cross-domain methods. Needs the shared local generation wrapper to be stable first |
| 25 Aug onward | Trait mediation: go/no-go decomposition test, then variant (i) | The go/no-go runs before any generation code is written |
| 1 to 11 Sep | Trait-conditioned prompting run, full evaluation, replication across spaces | Freeze |

### What fits before ICLR

The full baseline set with a trained ceiling, the five promised cross-domain methods
including plug-and-play, both transfer directions, and trait mediation in its
prompting form. That is a complete methods table against what was promised, with the
novel contribution present and evaluated.

### What does not fit before ICLR

Trait-conditioned soft prompts and trait-directed steering. The `trait` metric with a
validated rater. The `style` and `aspect` metrics from
[07-vectorial-aspect-style-integration.md](07-vectorial-aspect-style-integration.md).
Human validation of the judge.

### What ARR adds

Seventeen days over ICLR. That is enough for trait-conditioned soft prompts **only
if** the rater arrived by early September and the decomposition go/no-go passed. It
is not enough to add both the soft-prompt variant and a validated trait metric, and
attempting both would deliver a method whose own metric has not passed the baseline
separation check.

### The decision that has to be made, and when

**By 1 September**, decide what the submission claims. If the rater has arrived and
the decomposition test passed, trait mediation is presented as a method, with variant
(i) as the ICLR evidence and variant (ii) as the ARR extension. If the rater has not
arrived, or the decomposition does not predict held-out cells, the submission is the
evaluation harness and the method comparison, with trait mediation described as the
prompting variant and its limitation stated plainly. The second framing is a good
paper. It is only a bad outcome if the decision is made in the last week, when there
is no time left to write it properly.
