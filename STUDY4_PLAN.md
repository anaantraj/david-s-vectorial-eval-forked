# Study 4 plan

## Unsupervised platform adaptation and the content-transfer frontier

### Status

This revision incorporates the August 2026 data audit, the latest rerun, and the Study 4 planning
meeting. It supersedes the earlier audience-conditioning-only design.

The executed two-platform study is explicitly a **four-audience case study**, not the planned
multi-audience confirmatory study. The supplied exports provide four reviewed bilateral mappings,
so acceptance gates 1 through 4 are not met. Instructional-design Reddit contributes 8,936 records
after clone resolution rather than 10,000. The report must retain these shortfalls and must not make
an audience-generalisation claim. Twitter and Quora were not present in the supplied bucket and
remain future replications rather than silently missing experimental columns.

For this case-study execution, `platform_prior_lm` on the full available corpus, capped at 10,000
records per audience during collection, is the frozen primary Stage A base. It contains 34,901
Reddit training records and 16,103 LinkedIn training records. The raw-text arm and the nested
1,000-record-per-audience arm are ablations; held-out frontier
results do not select the primary base. This preserves the merged held-out split for final scoring
without using it for model selection. The choice follows the study's substantive intervention:
measuring whether the supplied audience and profile details add value beyond raw target text.

The executed structured condition uses prompt schema `study4-rich-priors-v3`. It reserves at most
128 tokens and gives each available metadata family a bounded share. It includes platform,
audience, record type, community, sanitized thread title and context, engagement, post dimensions, train-author
categorical or keyword labels, and train-derived audience keywords, dimensions, and communities.
Profile summaries and highlights are excluded because they are generated prose that can reproduce
author identity. Supplied room traits remain audit-only because their source covers whole rooms,
including held-out authors. Treating them as train-only priors would violate the split.
Some Reddit comments retain their parent thread's title or body. Normalized matches to any
validation or test post, including a scored post contained inside a longer metadata string or a
substantial metadata excerpt contained inside the scored post, are blanked from model-visible
metadata before structured-prior training.
The authentic comment completion, record count, raw-text control, and audience aggregates remain
unchanged. Sanitization manifests record source and output hashes and the number removed by field.

The coarse-to-fine arm uses a deterministic disjoint partition within each audience. The platform
stage receives the first stable-ranked half and the audience-prior stage receives the remainder.
Their union is the frozen primary training corpus, so the two stages together expose exactly the
same target completions as one full primary pass. Reusing the complete corpus at both stages would
double the token budget and is not the reported comparison.

The rerun did not materially change Study 3. LoRA and soft prompting remain the strongest fitted
families. Lower-strength steering performs worse on the new data. Direct prompting and steering do
not reproduce target-platform writing as well as LoRA and soft prompting. The strongest methods move
toward authentic target distributions but also move away from the supplied source. Study 4 therefore
asks whether unsupervised platform and audience adaptation can move the frontier toward the desired
combination of high source preservation and low distributional error.

A scale arm for steering was added on 2026-08-27 (section 4.5). Every steering result to date is an
8B result because that is what one card holds, so `steering_ndif` runs the same construction on
NDIF-hosted 70B and 405B, where the residual stream is exposed for read and write. One artifact is
fitted so far — `target_lm` on `Llama-3.1-70B-Instruct`, in
`runs/steering/steering.target_lm.ndif-Llama-3.1-70B-Instruct.pt`. **No results exist from this arm
yet**: the layer-and-alpha sweep has not run, and remote columns are never pooled with the 4-bit
local columns.

The new S3 corpus supports this direction, but not without reconstruction. It contains raw profile
post and comment histories grouped by audience room. It also contains test rooms, clones, repeated
exports, direct identifiers, heterogeneous schemas, and summaries that may have been generated from
posts later used for evaluation. Raw records must be deduplicated, author-grouped, time-filtered, and
split before any derived audience artifact is computed.

---

## 1. Research questions

### Primary question

**Does unsupervised target-platform or target-audience language-model adaptation improve the
content-preservation versus distribution-match frontier for cross-platform transfer?**

The primary comparison is not the single lowest Triangle-Rank Metric (TRM) value. It is the frontier
between:

- source cosine similarity and source content-word retention, where higher is better; and
- TRM against authentic target posts, where lower is better.

A method improves the frontier only if it lowers TRM without reducing source preservation, or
improves source preservation without increasing TRM. A point that merely moves down and left is a
different trade-off, not an unqualified improvement.

### Secondary questions

1. Does audience-level unsupervised adaptation improve on platform-level adaptation?
2. At what corpus size do additional unpaired posts stop producing useful frontier gains?
3. Do LoRA and soft prompting retain their advantage after every method begins from the same
   unsupervised-adapted base?
4. Can explicit aspect extraction followed by transfer preserve more source content than direct
   generation?
5. Can platform, audience, and trait steering vectors be combined without redundant directions
   dominating the intervention?
6. Does a Llama instruction-tuned prompting baseline reproduce the Claude result, or is Claude's
   movement away from LinkedIn specific to its instruction and safety tuning?
7. Do learned platform transformations compose across three or more platforms?
8. Does the platform contrast direction survive at scale, or is it an artifact of the 8B model the
   local ceiling forces every steering result onto?

---

## 2. What Study 3 establishes

Study 3 used 1,112 posts across 33 bilateral audience-by-topic cells. Its primary TRM comparison
covered 11 shared held-out cells, and its per-post measures covered 25. The authentic-post reference
covered six cells on TRM.

The results relevant to Study 4 are:

- LoRA and soft prompts produced the lowest observed distributional errors.
- The retrieved-source soft prompt was the most stable fitted-versus-prompted comparison across the
  six tested embedding spaces.
- LoRA and soft prompting captured low-level target writing habits better than direct prompting.
- Steering remained on the prompting side of the primary comparison. Reducing its strength did not
  reverse that result in the new-data rerun.
- Methods with low TRM often lost source-specific content. No evaluated configuration occupied the
  desired high-preservation, low-TRM corner.
- Claude preserved less LinkedIn resemblance than the other direct prompting systems, but the study
  did not isolate whether this came from platform adaptation or model-specific instruction tuning.
- Aspect labeling was too sparse for a confirmatory result. Existing automated aspect judgments
  were exploratory and lacked human validation.
- The corpus lacked a shared job-title and subreddit audience map. Audience labels were not present
  in the current flat dataset.

Study 4 must preserve the harness rules that made those statements defensible: common cell bases,
minimum pool sizes, cell-level intervals, train-only exemplars, scored-split references, recorded
failures, sample-indexed cache keys, and separate reporting for every embedding space.

---

## 3. Data construction

### 3.1 Required record schema

Every unsupervised record must have:

- stable post or comment identifier;
- stable platform-specific author identifier;
- platform label;
- canonical audience label;
- canonical source room identifier and export identifier;
- text and, where available, separate title and body;
- creation timestamp and collection timestamp;
- record type, such as post or comment;
- duplicate and clone lineage;
- optional topic, aspect, and trait annotations with their provenance.

The audience label is currently missing from the flat experiment dataset. Vectorial must add it or
provide a frozen room-to-audience crosswalk before Study 4 training can begin. Job title and
subreddit are not substitutes because they lack a shared bilateral mapping.

Topic labels are optional for unsupervised language-model training. They are required for the
evaluation cells and for any topic-controlled audience comparison. If topic assignment is costly,
label the evaluation pool and a training diagnostic sample rather than the complete unsupervised
corpus.

### 3.2 S3 audit findings

The first read-only audit found 117 LinkedIn room directories and 115 Reddit room directories.
LinkedIn contained 24,723 objects and approximately 10.8 GB. The room inventory mixes production,
test, mock, cloned, merged, and repeated exports. Only `Backend Engineer` and
`Instructional Designer` match exactly by name across platforms, although several further pairs are
plausible after normalisation.

The four Study 3 audiences have substantial profile coverage:

| Canonical audience | LinkedIn profile lower bound | Reddit profile lower bound |
|---|---:|---:|
| backend engineer | 340 | 148 |
| full-stack engineer | 127 | 155 |
| instructional designer | 44 | 256 |
| education-technology engineer | 351 | 267 |

Additional candidate pairs include CTO, engineering manager, product manager, QA engineer,
performance or growth marketer, e-commerce apparel, OpenAI community, and founders. These require a
reviewed crosswalk. A repeated or cloned room does not count as another audience.

The platforms expose different profile schemas. LinkedIn includes employment and biography fields.
Reddit includes activity counts and generated summaries but no observed job title or industry.
Confirmatory bilateral features must therefore be computed identically from raw post and comment
text. Platform-specific profile fields are descriptive only.

### 3.3 Leakage and deduplication

Profile histories overlap Study 3. In a deterministic sample from the four matched audiences, 61 of
316 LinkedIn post URLs and 43 of 345 Reddit post URLs matched the existing flat corpus. Shipped room
summaries, profile summaries, traits, keywords, highlights, exposure analytics, and inferred
dimensions may therefore encode scored posts.

Do not treat those shipped derived fields as confirmatory ground truth. They are useful weak labels
and must be retained. Use them only for profiles assigned wholly to train, aggregate them into
platform-by-audience priors, and compare them with features recomputed from raw text. Recompute all
confirmatory features after the following operations:

1. Resolve canonical exports using declared clone and merge lineage, timestamps, hashed profile
   membership, and object ETags.
2. Deduplicate by platform post ID, normalised URL, and content hash.
3. Group every post and comment belonging to one platform author.
4. Restrict both platforms to a common observation window.
5. Assign authors, not posts, to deterministic splits using `blake2b(author_id, seed)`.
6. Remove every held-out author and post before computing a training summary, vocabulary, cluster,
   aspect, trait, or steering vector.

### 3.3.1 Prior construction from the full exports

The language-model corpus includes posts and comments, with record type explicitly conditioned so a
comment-heavy audience does not silently teach the post generator to emit replies. Preserve titles,
subreddits or LinkedIn author headlines, thread context, context summaries, flair, labels,
dimensions, engagement, and timestamps as structured fields.

Construct three train-only prior layers:

1. A platform prior from raw post and comment text, record-type rates, length, communities, and
   structural features.
2. A platform-by-audience prior from canonical audience membership, communities, frequent
   train-author weak labels, dimensions, and keywords. Whole-room highlights and traits require a
   train-only reconstruction before they can be used.
3. A source-content prior from post labels, thread context, entities, claims, numbers, and aspects.

Profile summaries, highlights, keywords, room descriptions, room traits, post dimensions, and
context summaries are weak labels. Only labels tied to profiles assigned wholly to train may
condition training; whole-room summaries and traits in the supplied export are audit-only. They
never define evaluation truth. Priors supplied for a held-out example are aggregates estimated from
train authors, never that author's own profile. Report an ablation between raw-text-only adaptation
and prior-conditioned adaptation so the contribution of these details is measured rather than
assumed.

Existing Study 3 post IDs retain their prior split roles where possible. When an author-grouped split
conflicts with a post-level split, the author-level assignment wins and the reconstructed benchmark
is versioned as Study 4 rather than compared row-for-row with Study 3.

### 3.4 Scale targets

The collection target is at least 10,000 usable posts or comments per accepted audience and platform.
The preferred range is 50,000 to 100,000, based on the working hypothesis that quality gains flatten
in that region. One million records per audience is not a default requirement.

These are training-volume targets, not evaluation requirements. Evaluation still needs enough
held-out authentic posts in each audience-by-topic cell. Adding many posts to one audience does not
narrow an interval whose independent unit is the audience.

Freeze a data scale ladder after the audit:

- 1,000 records per audience and platform;
- 10,000 records;
- 50,000 records;
- 100,000 records, where available.

Every rung is a deterministic nested sample of the next rung. Never resample a different 10,000 for
each run. This makes the scale curve a data-size comparison rather than a sample-composition
comparison.

### 3.5 Acceptance gates

Begin the confirmatory run only when:

1. At least eight non-cloned bilateral audiences have a collaborator-confirmed crosswalk.
2. At least six audiences contain 10,000 usable training records on both platforms, or the study is
   explicitly relabeled as a four-audience case study.
3. At least 20 held-out audience-by-topic cells across at least six audiences clear the TRM minimum
   pools on a common systems base.
4. No audience contributes more than one third of the scoreable cells.
5. Collection timestamps and a common observation window are known.
6. The audience labels and deduplication manifest are frozen before training.

---

## 4. Training design

Study 4 has two stages. This prevents a large method-by-scale-by-platform grid from obscuring the
primary question.

### 4.1 Stage A: unsupervised adaptation and scale selection

Continue causal language-model training on authentic, unpaired text. No synthetic rewrite and no
source-target pairing enters this stage.

Train the following bases with identical token budgets and decoding validation:

| Base | Unsupervised corpus | Purpose |
|---|---|---|
| `base_none` | none | Original instruction-tuned checkpoint |
| `base_platform` | target-platform text across training audiences | Platform adaptation |
| `base_audience` | target-platform text from the requested audience | Audience adaptation |
| `base_platform_then_audience` | platform corpus followed by audience corpus | Tests coarse-to-fine adaptation |

Use `meta-llama/Llama-3.1-8B-Instruct`, 4-bit NF4, batch size 1, and maximum sequence length 768, as
required by the measured cluster ceiling. All bases receive the same effective token budget at a
given scale. Report examples, tokens, steps, epochs, learning rate, and validation loss. A run with
more tokens is a compute-and-data intervention and must not be described as a pure data-size effect.

Run the full scale ladder for `base_platform` on one preregistered direction and a balanced subset of
audiences. Select the smallest rung whose frontier improvement is within a frozen tolerance of the
next rung. Confirm that choice once on `base_audience`. Use the selected scale for Stage B. The
100,000-record rung is skipped when unavailable; missing rungs are reported rather than filled by
oversampling.

### 4.2 Stage B: transfer adaptation

Starting from the selected unsupervised base, fit the transfer methods. The existing harness
requires all three training variants for every fitted family:

- `target_lm`: room and topic only;
- `target_lm_paired`: room and topic plus a retrieved same-cell source post;
- `target_lm_aspect`: room and topic plus foregrounded aspects.

Run LoRA and soft prompting under all three variants. The aspect variant remains exploratory until
coverage and rater validation pass the gates in Section 7. Do not silently compare it on a smaller
or easier cell set.

The confirmatory grid is:

| Method | Starting base | Role |
|---|---|---|
| direct Llama prompt | `base_none` | Model-controlled zero-shot baseline |
| direct Llama prompt | selected unsupervised base | Tests whether pretraining alone is sufficient |
| soft prompt | selected unsupervised base | Study 3's most stable fitted method |
| LoRA | selected unsupervised base | Expected strongest practical method |
| steering | `base_none` | Primary mechanistic baseline; no unsupervised fit required |
| steering | selected unsupervised base | Recomputed-vector interaction experiment |
| extract-then-transfer prompt | selected unsupervised base | Explicit content-plan baseline |

Claude remains a reference prompting system but is not the only zero-shot baseline. Comparing Claude
with a Llama prompt under the same task helps identify model-specific instruction tuning.

### 4.3 Extract-then-transfer prompting

This is a two-step method, not hidden chain-of-thought collection:

1. Produce a short structured content plan containing topic, claims, entities, numbers, named tools,
   requested action, and optional labeled aspects.
2. Render a target-platform post from that plan and the target audience context.

The structured plan is stored and scored. It must not contain private reasoning. Add plan recall for
entities, numbers, and content words as a diagnostic. Use the same generation budget as direct
prompting or report the extra calls and tokens explicitly.

### 4.4 Steering decomposition

Compute steering vectors from train-only mean activation differences. At minimum estimate:

- target-platform minus source-platform;
- target-audience minus comparison-audience within platform;
- trait-present minus trait-absent, only for validated traits.

Every contrast used together must be computed by teacher-forcing text through the **same frozen
checkpoint**. Activations from `base_none`, `base_platform`, and `base_audience` occupy related but
not identical parameterisations and their vectors must not be mixed. The primary steering baseline
uses `base_none`, where no unsupervised training is required. A second steering run may recompute all
contrasts from the selected unsupervised base to test the interaction, but it is a complete
re-estimation rather than reusing vectors fitted on the original model.

Raw mean-difference vectors are generally correlated. Form a matrix of centred, unit-norm candidate
vectors at each layer and use an SVD or equivalent orthogonal basis. Record singular values and the
mapping from original contrasts to basis directions. Apply coefficients after norm calibration so
that adding two correlated concepts does not double the intervention magnitude.

Sweep platform and audience coefficients on the train-internal selection set. The sweep objective is
Pareto dominance under a degeneracy constraint, not centroid distance alone. Report the entire
selected frontier rather than choosing the lowest-TRM point after seeing held-out results.

Run one small LoRA-plus-steering interaction experiment. It is secondary and stops unless at least
one coefficient setting improves both primary axes on selection data. Do not expand it into the full
grid when it only exchanges content for style.

### 4.5 Scale generalisation of steering on NDIF

Every steering number in Studies 2 through 4 is an 8B number. The ceiling is one 23.5 GB card, and
per `docs/09-cluster-training-config.md` it is the 128k-vocab logits tensor rather than the weights
that sets it. "The platform contrast is a translation in the residual stream" is therefore a claim
about one small model, not about the mechanism. This arm answers secondary question 8 by running the
identical difference-in-means construction on NDIF-hosted `Llama-3.1-70B-Instruct` and
`Llama-3.1-405B-Instruct`, whose residual streams NDIF exposes for read **and** write. No generation
API can serve this arm at any price: the intervention is inside the model.

The arm is `steering_ndif`, and it subclasses the local `steering` transfer function, overriding only
model loading and the batch generation call. Vector resolution, provenance, scope grouping, the draw
loop and the per-item failure accounting are the same code, so the two arms differ in where the
forward pass happens and in nothing else. See `docs/11-ndif-steering.md`.

**A steering vector belongs to the residual stream it was read from.** The local artifacts have 4,096
columns; 70B's stream is 8,192 wide and 405B's is 16,384. The existing artifacts cannot be applied at
scale, so this arm refits with `methods.steering.fit_steering_ndif`. The `base_model` guard refuses
any artifact whose model does not match the model being run.

**Remote columns are not comparable with local columns.** Section 10 fixes 4-bit NF4 across every
method so quantisation is a constant of the study rather than a difference between its columns. NDIF
serves its own precision, so that constant does not cover this arm; `load_in_4bit` is recorded as
`None` rather than `False`, because `False` already names the local bf16 ablation. The remote column
is a scale contrast against the **local steering column** and against nothing else. Two further
properties are recorded per output rather than assumed: sampling is seeded on the NDIF worker, so
draws are reproducible against a given deployment but not across a redeploy; and the checkpoint is
NDIF's, so the deployment entry and the nnsight version are written into `fit_config`.

Access constrains what can be run. A standard key may only use models NDIF has **pinned**. As of
2026-08-27 that is `gpt-j-6b`, `gemma-2-9b-it`, `Llama-3.1-8B`, `Llama-3.1-70B`,
`Llama-3.1-70B-Instruct` and `Llama-3.1-405B-Instruct`. Two consequences matter for the design:

- `Llama-3.1-8B-Instruct`, the local study's base, is hot but **not** pinned, so the local 8B results
  cannot be replicated remotely as a correctness check on this arm.
- The pinned *base* models have no chat template. Fitting or generating on one would compare an
  unwrapped remote arm against the wrapped local columns, which is the confound STUDY3 item 1 was
  opened for. The backend refuses by default.

#### Execution status and where the results are

| | state |
|---|---|
| `target_lm` on `Llama-3.1-70B-Instruct` | **fitted** 2026-08-27, 630/630 posts, 0 failures, 834s |
| `target_lm_paired`, `target_lm_aspect` on 70B | not run |
| any variant on `Llama-3.1-405B-Instruct` | not run |
| dev sweep over layer and alpha | **not run** |
| scored transfer outputs | **none** |

```
runs/steering/steering.target_lm.ndif-Llama-3.1-70B-Instruct.pt     fitted artifact, 81 MB
runs/steering/steering.target_lm.ndif-Llama-3.1-70B-Instruct.json   counts, fit_config, diagnostics
runs/steering/logs/fit.target_lm.ndif-Llama-3.1-70B-Instruct.log    fit log
runs/checkpoints/steering_target_lm.ndif-Llama-3.1-70B-Instruct/    resume checkpoints
```

The artifact holds an `(81, 8192)` global vector over 337 target and 293 source train posts, with own
vectors for 23 of 26 cells and 4 of 5 audiences. Cells too thin: `backend_engineer::agentic_coding`,
`backend_engineer::react_state_management`, `edtech_engineering::education_technology_jobs`. Audience
too thin: `product_leadership`. Per-layer `act_norm` rises monotonically from 0.62 to 90.7 and the
relative shift `‖v_i‖/act_norm_i` from 0.07 to 0.27, peaking at the last hidden state.

**There are no results from this arm yet, and the artifact must not be scored until the sweep runs.**
Layer and alpha are unselected. A spot check at layer 40 and alpha 0.5 left the output structurally
Reddit-like but code-switching into Spanish and German, which is the intervention pushing off-manifold
with language identity breaking first, and is what the degeneracy guard in section 6 exists to catch.
The next step is `methods.steering.sweep_steering` against the artifact above, under the same
Pareto-dominance-under-a-degeneracy-constraint objective as section 4.4, after which this arm reports
alongside the local steering column as a scale contrast.

---

## 5. Evaluation design

### 5.1 Directions and splits

The primary directions remain LinkedIn to Reddit and Reddit to LinkedIn. Exemplars come only from
train. Authentic references come only from the scored split. Model selection uses a deterministic
slice of train. The primary scored split is merged held-out.

Add an unseen-topic evaluation by withholding complete topic clusters within each audience. No
completion, exemplar, audience summary, unsupervised validation example, or steering contrast from a
held-out topic may enter task-specific fitting or selection. Unsupervised platform pretraining may
include other topics but not the held-out evaluation text or authors.

All methods draw at least four candidates per task. The sample variant remains in the cache key.
Failures emit an empty `TransferOutput` with `ok=False` and remain in every denominator.

### 5.2 Reference columns

Every comparison includes:

- `identity`, the unchanged source and preservation ceiling;
- `shuffle_control`, an authentic target post from another cell;
- `target_sample`, an authentic target post from the correct cell;
- `audience_shuffle`, the same method conditioned on another target-platform audience.

The audience shuffle is a system control. It uses a frozen derangement matched on training volume and
broad topic coverage. It participates in the common systems cell base. `target_sample` and
`shuffle_control` draw distinct authentic posts up to the available pool size.

### 5.3 Primary frontier analysis

The main figure plots source cosine similarity on the horizontal axis and TRM on the vertical axis.
It includes every system setting and the three named references. Lower and farther right is better.
Axis limits derive from the data.

Compute the non-dominated frontier from cell-aggregated values. For each proposed method, compare it
with the corresponding no-unsupervised-pretraining condition using paired cells. A frontier gain is
supported only when the interval for one axis separates and the other axis is non-inferior under a
frozen margin, or both axes separate in the favourable direction.

The margins frozen before generation are 0.01 cosine-similarity units for source preservation and
0.01 TRM units for distributional error. For candidate-minus-baseline differences, improvement
requires either an upper TRM interval below zero with a lower source-similarity interval of at least
-0.01, or a lower source-similarity interval above zero with an upper TRM interval of at most 0.01.
Point-estimate frontier membership is descriptive and is never promoted to a finding by itself.

TRM always states its pairwise cosine distance and embedding space. Repeat the frontier verdict in
the same six embedding spaces used by Study 3. Values from different spaces are never pooled.

### 5.4 Independent units

Audience-by-topic cells nested within one audience share training data and conditioning. The primary
interval uses a two-stage bootstrap: resample audiences, then resample eligible topic cells within
each selected audience. Report the Study 3 cell-only bootstrap as a sensitivity analysis, not as the
basis for an audience-generalisation claim.

Per-post preservation outcomes use an author-clustered bootstrap nested within audience. Multiple
posts from one author are never treated as independent observations.

---

## 6. Metrics

### Primary axes

- TRM in the preregistered embedding space, lower better;
- cosine similarity to the supplied source, higher better.

### Required guards

- source content-word retention;
- entity and number retention;
- topic-mixture divergence;
- centroid distance;
- writing-habit divergence over the existing transparent features;
- length ratio to authentic target text;
- distinct-bigram ratio and within-pool similarity;
- near-copy and failure rates;
- TRM triangle-rank components.

The classifier metrics remain calibration measures. Maximising the target-platform label is not the
objective because authentic target posts do not achieve a perfect rate.

### Scale-curve outcome

For each data rung, report frontier hypervolume relative to fixed reference bounds, validation loss,
training tokens, and wall-clock compute. Hypervolume bounds are frozen from reference columns before
method outputs are inspected. Also report each axis directly so the aggregate cannot hide a content
loss.

---

## 7. Aspects and traits

Unsupervised training does not require explicit aspect labels. It may encode recurring traits and
aspects in its latent state, but that is a hypothesis rather than an observed fact. Explicit labels
serve three narrower purposes:

1. explain what differs between authentic and generated pools;
2. condition the `target_lm_aspect` variant;
3. define a validated trait steering contrast.

The confirmatory aspect gate requires:

- a vocabulary extracted without platform labels;
- coverage of at least 20 common-base cells across at least six audiences;
- the complete rater prompt, model, version, and parser;
- human annotation on a frozen held-out sample;
- a preregistered agreement threshold met separately on authentic and generated text.

Until all gates pass, aspect-conditioned results, trait steering, and claims that a method learned an
aspect distribution are exploratory. Sparse labels are never imputed from the generated output being
scored.

Add one exploratory aspect-planning model when coverage permits. It predicts a set of target aspects
from audience and topic, then generates the post conditioned on the predicted set. This differs from
`target_lm_aspect`, which is given the authentic post's foregrounded aspects and learns how the target
platform expresses them. Evaluate aspect selection and text generation separately so an apparently
good post score cannot hide incorrect aspect prediction.

---

## 8. Cross-platform extension

Twitter and Quora should be collected with the same platform, audience, author, timestamp, and text
schema. The target is 10,000 or more records per accepted audience and platform. They enter Study 4
only after the two-platform confirmatory artifacts are frozen.

Run direct transfer and the following composition baseline on audiences present on all three
platforms:

1. train Reddit-to-LinkedIn and Reddit-to-Twitter transformations;
2. infer LinkedIn-to-Twitter by composing the learned transformations;
3. compare with a directly trained LinkedIn-to-Twitter method on the same cells.

For steering, composition means adding orthogonalised contrast coefficients. For LoRA and soft
prompts there is no assumed algebra; any adapter composition rule is a separately defined method.
Failure is informative, but the experiment is exploratory because modern language-model
representations need not support linear domain arithmetic.

Also train one unified multi-direction LoRA with explicit ordered platform tokens, for example
`<source:reddit><target:linkedin>`. Each minibatch draws directions from a balanced sampler so the
largest platform does not define the model. Compare this model with separately trained directional
LoRAs. Test a withheld direction such as LinkedIn-to-Twitter only when its direct model and held-out
references exist. Success means the withheld-direction frontier approaches the direct model;
multilingual analogies do not justify assuming that it will.

Quora is a fourth-platform replication after the three-platform analysis, not another factor in the
initial grid.

---

## 9. Confirmatory hypotheses

Test in this order:

1. LoRA from the selected unsupervised base improves the LinkedIn-to-Reddit primary frontier over
   LoRA from `base_none`.
2. The improvement repeats for Reddit-to-LinkedIn.
3. Audience-level adaptation improves the frontier over platform-level adaptation.
4. Soft prompting from the selected unsupervised base improves over its `base_none` counterpart.
5. Extract-then-transfer prompting improves entity, number, and content-word retention without a
   worse TRM interval than direct Llama prompting.
6. The selected LoRA frontier dominates the preregistered `base_none` steering frontier. Steering
   recomputed on the unsupervised base is a separately reported interaction experiment.

A finding requires interval separation under the two-stage bootstrap and must repeat in the
preregistered embedding space. The six-space table states how often it replicates. Later hypotheses
are exploratory after the first failed gate within their family. Point estimates remain in tables
and figures and are not promoted to prose findings.

The scale curve is confirmatory only for the preregistered platform, direction, audiences, and method.
All other scale curves are replications.

---

## 10. Compute plan on Cthulhu

Before writing or launching training code, follow `docs/09-cluster-training-config.md`.

Use `cthulhu2` for the initial data audit, token counting, and Stage A scale sweep if its GPUs satisfy
the recorded free-memory check. Do not assume a whole node is available. Select cards at launch time
and preserve the cluster's checkpoint and resume behaviour.

The fixed Llama configuration is 4-bit NF4, batch size 1, maximum length 768. Run every method under
the same quantisation. Log peak memory, tokens per second, training tokens, elapsed time, selected
checkpoint, and data-manifest hash.

The section 4.5 scale arm is the one exception, and it does not run on Cthulhu at all: the forward
pass is on NDIF, in NDIF's own precision, so it neither consumes cluster GPUs nor shares the
quantisation constant. It needs `NDIF_API_KEY` in `.env` and the `[ndif]` extra installed. Budget
about 14 minutes per variant at 70B and roughly ten times that at 405B; NDIF is a shared NSF resource
and the fit queues against itself, so run one variant at a time.

Recommended sequence:

1. Sync code once to the shared NFS path and verify the cluster environment.
2. Build the de-identified manifest and author-grouped splits without copying raw identifiers into
   the repository.
3. Count usable records and tokens per accepted audience and platform.
4. Run one 1,000-record smoke fit and generation check.
5. Run the platform scale ladder on separate available cards.
6. Select the data rung using only the frozen selection criterion.
7. Train audience and platform-then-audience bases at the selected rung.
8. Run the confirmatory LoRA, soft-prompt, steering, and prompting grid.
9. Score all outputs locally or on CPU workers, then run the six embedding spaces.
10. Freeze the two-platform report before adding Twitter or Quora.

Credentials used to read S3 are process-scoped cluster secrets. They are not placed in scripts,
shell history, manifests, logs, checkpoints, or the repository. The Study 4 manifest records bucket,
region, object ETags, and hashes, but never credentials or direct profile identifiers.

---

## 11. Reporting

The generated report contains:

1. data audit, crosswalk, deduplication, and split coverage;
2. unsupervised scale curve;
3. primary content-versus-TRM frontier;
4. training-method comparison;
5. forward and reverse directions;
6. unseen-topic evaluation;
7. aspect and trait exploration;
8. steering scale contrast at 70B and 405B, reported against the local steering column only and
   never pooled with the 4-bit columns;
9. embedding-space sensitivity;
10. optional Twitter and Quora extension;
11. limitations.

Every comparison column uses the same cells. Reference columns print their own counts when coverage
differs. Every figure reads from run reports, axis limits derive from data, and all counts and
universal claims are computed. Report prose states only interval-supported findings.

Before completion:

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check src/
SOURCE_DATE_EPOCH=1000000000 .venv/bin/python experimental-notes/build.py
```

Build the report twice with the pinned epoch and confirm byte identity, then build once without it.

---

## 12. Limitations

There is no cross-platform person linkage. Study 4 cannot separate the same person's behavioural
change from differences in who participates on each platform.

Unsupervised target training can improve platform fluency while weakening source conditioning. That
is why the frontier, not target resemblance alone, is the primary object.

Audience labels are constructed mappings. A LinkedIn professional room and a Reddit community may
not represent the same population even when their names are similar. The crosswalk is an explicit
measurement assumption.

The task still has no gold target rewrite. Authentic target posts calibrate distributions but do not
say how one particular source should be rewritten.

Modern LLM activations need not decompose linearly into platform, audience, and trait directions.
Orthogonalisation prevents redundant magnitude but does not make a direction causally pure.

The proposed 50,000-to-75,000-record inflection point comes from another generation domain and is a
planning prior, not evidence about this task. Study 4's nested scale curve must determine whether it
holds here.

Twitter and Quora change both platform form and population. Their results are separate replications,
not extra samples for the original Reddit and LinkedIn estimand.
