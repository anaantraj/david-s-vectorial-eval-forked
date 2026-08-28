# Study 3 plan

What the second study's limitations permit us to fix with the corpus we already hold, and what
they do not. Every limitation recorded in `reports/study2/limitations.html` appears below exactly
once, under one of the two headings, with the cost of addressing it and the gain expected.

The organising question is deliberately narrow. A limitation is **solvable now** if it can be
removed by compute, code, or a decision, using `final_dataset_v1.csv` as it stands. It is **not
solvable now** if removing it requires posts we have not collected, labels we do not hold, or an
artifact a collaborator owns. Several limitations that read as methodological turn out to be the
first kind, and one that reads as a small implementation detail turns out to be the second.

Dates below assume the ICLR deadline of 25 September and the ARR October deadline of 12 October.

---

## Part I — Solvable with the data we already hold

### 1. The steering columns are confounded with prompt wrapping

**Status: solvable, and it invalidates a published column until it is.**

The steering artifacts are fitted and generate on the raw rendered context, while the LoRA and
soft-prompt functions apply the base model's chat template at both fitting and inference. Every
steering-against-LoRA and steering-against-soft-prompt comparison in Study 2 therefore compounds the
intervention with the wrapping. A second, smaller asymmetry sits beside it: the steering function
truncates the source post to 3000 characters where the other two pass it whole.

**Do:** refit the three steering artifacts under `methods.steering` with `wrap_prompt` applied at
fit and at inference, remove the truncation, and regenerate. Nothing else changes.

**Cost:** three fits and three generation passes. Under an hour of wall-clock on idle cards.

**Gain:** the steering columns become a measurement of activation steering. Until then they are not,
and the report says so.

**This is the first thing to do.** It is cheap, and it is the only item on this list that makes an
existing published claim honest rather than merely stronger.

### 2. Selection on validation forced the study onto the smaller split

**Status: solvable, and it is the highest-leverage item here.**

The trained methods early-stop and select hyperparameters on the validation split. The harness's
usual scored split, `heldout`, is the union of validation and test, so scoring there would let a
trained method select against roughly half of its own reference pool. Study 2 therefore scored on
`test` alone, which is clean but smaller.

The cost of that choice is measurable and large:

| Scored split | Triangle-Rank Metric cells | Classifier cells |
|---|---|---|
| `test` (Study 2) | 5 | 17 |
| `heldout` (Study 1) | 10 | 22 |

**Do:** carve the selection set out of **train** rather than using validation. `build_training`
already assigns splits by `blake2b(post_id, seed)`, so a deterministic sub-split of train costs one
argument and preserves append-safety. Retrain all nine trained columns against it, then score on
`heldout`.

**Cost:** 50 of 337 training examples move from fitting to selection, leaving 287. All nine
configurations retrain, roughly half a day of cluster time including the sweeps.

**Gain:** the Triangle-Rank Metric base doubles from 5 cells to 10 and the classifier base rises
from 17 to 22, with no leakage. Study 1 recorded that this same change resolved three previously
unresolved comparisons. **LoRA against the soft prompt is currently unresolved and is the study's
most conspicuous open question; doubling the base is the only lever we hold that might resolve it.**

Whether 287 examples fit as well as 337 is itself an empirical question worth reporting, since it
bears directly on how much of the trained methods' advantage is sample-limited.

### 3. The oracle is drawn once per task, which collapses its coverage

**Status: solvable, and the cause is a routing decision rather than a data limit.**

Every system column draws four candidates per source post, following the multi-candidate procedure
the Triangle-Rank Metric's originating work prescribes. `target_sample` draws one. Its pool is
therefore a quarter the size of every column it calibrates, and since the metric requires at least
four candidates and four references in a cell, the oracle clears that threshold in **4 of 13 cells**
and survives the common-cell intersection in **1**. Every oracle rule in every figure of Study 2
rests on that single cell.

The cause is a decision in `cli.cmd_transfer`: `--n-samples` is routed to any function whose
constructor accepts it, and the three reference columns were excluded as "single-draw by
construction". That is right for `identity`, which is deterministic, and wrong for `target_sample`,
which samples an authentic post from the cell's train split and can perfectly well sample four
distinct ones. `shuffle_control` is the same case.

**Do:** route `--n-samples` to `target_sample` and `shuffle_control`, drawing distinct authentic
posts per draw, and confirm the draws differ. Leave `identity` single-draw and say why.

**Cost:** an hour, no GPU. Regeneration of two reference columns and a rescore.

**Gain:** the oracle returns to a base comparable with the systems it calibrates, which matters more
than it sounds: the oracle is the reference for every diagnostic whose ideal is a middle value, and a
reference computed on one cell is not a reference. Combined with item 2 this is the difference
between a calibration marker and a like-for-like column.

### 4. Length accounts for much of the effect and is partly a decoding setting

**Status: solvable.**

Output length is the largest single difference between the families. For LoRA and the soft prompt it
is learned. For steering it is not: the base model does not emit an end-of-sequence token under this
prompt and runs to `max_new_tokens`, so steering's length ratio of roughly 12.5 reflects decoding
rather than the intervention.

**Do:** two things, and report both. First, equalise stopping behaviour across families, either by
giving every family the same stop criterion or by fitting a stop token. Second, add a
**length-controlled** evaluation: resample or truncate pools to a common length distribution and
recompute the distributional measures. If the trained families' advantage survives length control,
that is a much stronger claim than the one Study 2 can make; if it does not, the honest finding is
that this task reduces largely to length calibration, which is publishable and more interesting than
it sounds.

**Cost:** one generation pass per family plus a metric variant. A day.

**Gain:** separates "the trained methods learned the target distribution" from "the trained methods
learned how long a post is". Study 2 cannot distinguish these, and a reviewer will ask.

### 5. Results were computed before the scraped text was cleaned

**Status: solvable, mechanically.**

HTML entities survived the scrape, and every post carrying one is a Reddit post, so the encoded
entity is a token appearing on one platform and never the other. The lexical platform classifier is
free to key on it. `build_dataset.load_frame` now decodes entities once, pinned by a test, but every
number in Study 2 predates the fix.

**Do:** rebuild the dataset, retrain, rescore. Splits are a pure function of the post identifier, so
cell membership does not move and the rebuild is safe.

**Cost:** subsumed by item 2, which already requires a full retrain. Free if sequenced after it.

**Gain:** removes a one-sided artifact from the classifier rows. The effect is bounded by the five
per cent of posts involved and is known to inflate apparent separability, so the corrected numbers
should be slightly *less* favourable. Reporting that movement honestly is worth more than the
movement itself.

### 6. One embedding space

**Status: solvable, and it is a prerequisite for any claim we intend to publish.**

Every embedding-derived row in Study 2 is computed in EmbeddingGemma at 512 dimensions. Study 1
scored six spaces and found that seven of ten comparisons held in all of them while two held only
under TF-IDF and were consequently withdrawn.

**Do:** rescore the Study 2 outputs in the same six spaces and regenerate the replication page. The
outputs already exist; this is evaluation only.

**Cost:** six evaluation passes over existing outputs. A few hours, no generation.

**Gain:** every claim acquires a replication verdict. Given that Study 1 withdrew two claims on
exactly this test, running it before submission rather than after review is not optional.

### 7. One transfer direction

**Status: solvable.**

Everything is LinkedIn to Reddit. The harness supports `--source reddit --target linkedin` and it
has never been run. The corpus is close to balanced, with 444 LinkedIn and 495 Reddit posts.

**Do:** build the reverse dataset, train the same three families, score.

**Cost:** a full replication of the pipeline in the other direction. Two days of cluster time.

**Gain:** a second direction turns a single result into a pair and tests whether the trained
families' advantage is a property of the task or of Reddit's particular characteristics. Given the
sharply asymmetric platforms, the reverse direction should not be assumed to mirror these results,
and if it does not, that asymmetry is itself the finding.

### 8. One base model

**Status: solvable, cheaply and partially.**

All three trained families share `meta-llama/Llama-3.1-8B-Instruct`, so nothing separates the
mechanism from that model's behaviour.

**Do:** repeat the soft-prompt and LoRA arms on a second cached base, `Qwen3-1.7B`, which is already
in the HuggingFace cache and needs no download. Steering should be included only after item 1.

**Cost:** six fits on a small model. Under a day.

**Gain:** demonstrates that the trained-beats-prompted result is not an artifact of one checkpoint.
A second base at a different scale also gives a weak scaling signal, which is more than we have now.

### 9. The missing baselines from the promised set

**Status: solvable; these are implementations, not experiments.**

The project committed to an in-domain predictor as the ceiling, and to plug-and-play controlled
decoding among the cross-domain methods. Neither exists. `target_sample` is an oracle that emits
real posts, not a model trained on target-domain data, so the promised ceiling is genuinely absent.

**Do:** implement both, per `docs/08-transfer-function-roadmap.md`. For plug-and-play, note the
circularity risk flagged there: guiding decoding with the same classifier that scores the output
would be self-fulfilling, so the guidance classifier must be trained on a disjoint split or in a
different feature space.

**Cost:** the in-domain ceiling is a day. Plug-and-play is two to three.

**Gain:** completes the baseline set the project promised and gives the table a real ceiling.

### 10. The unbiased MMD row is uninterpretable here

**Status: solvable only in the sense that it can be reported correctly.**

The U-statistic estimator removes the diagonal to eliminate the O(1/n) bias, and is consequently not
a squared norm and not constrained to be non-negative. On pools whose true discrepancy is near zero
it is negative roughly half the time, which is what the oracle exhibits. With `MIN_MMD_POOL = 5` it
scores too few cells here to support any comparison.

**Do:** report the biased V-statistic alongside it, which is non-negative by construction at the cost
of a positive bias, and state the bias rather than choosing between them silently. Do not attempt to
rescue the unbiased row by lowering the pool guard; the guard exists because the estimator's variance
at pools of two to four dominates any average.

**Cost:** an afternoon.

**Gain:** an interpretable dispersion measure. This does not raise cell coverage, which is a data
problem and appears in Part II.

### 11. The aspect join is incomplete for reasons that are partly ours

**Status: partially solvable now.**

Nineteen of thirty harness cells carry an aspect vocabulary, while nineteen of the collaborators'
thirty-six shipped clusters match no cell at all. Those two facts are related: clusters were named
differently from `final_topic`, so vocabularies that exist are not being found. Five probable
renames are listed in `docs/10-aspect-vocabulary-coverage.md`.

**Do:** replace the string join with a match on aspect-description similarity, which
`docs/10` shows separates the probable pairs cleanly, and treat the five candidates as confirmed only
after the collaborators verify them.

**Cost:** a day, plus one message.

**Gain:** four cells move from fallback to conditioned, including the two largest uncovered ones.
The aspect-aware prompting column currently runs unconditioned on roughly half its outputs, which
caps how much signal it can show.

---

## Part II — Not solvable with the data we hold

### A. Cell coverage, and therefore the width of every interval

**This is the binding constraint on the entire project and no amount of compute touches it.**

The Triangle-Rank Metric requires at least four generated and four authentic posts in a cell. Of the
thirty bilateral cells, only **six carry ten or more posts on both platforms**. Item 2 above raises
the scored base from five cells to ten by removing the selection dependence, and that exhausts the
levers available within the corpus; `AGENTS.md` says as much.

Everything downstream follows from this. LoRA against the soft prompt may remain unresolved even at
ten cells. The bootstrap intervals will stay wide because they are resampled over cells and there
are few cells. No estimator choice fixes it.

**Needs:** more posts in cells that already exist, not more cells. The most efficient collection is
targeted: raise the twenty-four cells that currently sit between five and nine posts on their
thinner platform up to ten or more. That is a scraping request with a specific target list, and it
is the single highest-value thing the collaborators could do for the evaluation.

### B. Population shift against behavioural shift

The project distinguishes *population* shift, meaning different people appear on different
platforms, from *behavioural* shift, meaning the same people write differently. The harness measures
only their combination, and a transfer function could score well by modelling either.

**Needs:** identity linkage across platforms, which the dataset does not provide and which is
difficult to obtain ethically at scale from public data. This is not a compute problem and should be
stated as out of scope rather than left as an open item that looks addressable.

### C. Trait-mediated transfer, the intended novel contribution

Blocked on the collaborators' rating harness: the prompt, model, version, and parser used to produce
the style and aspect ratings. Without it, generated posts cannot be placed in the same trait space as
real posts, which blocks both the conditioning and the trait metric.

Reimplementing the rater from the shipped dimension definitions is possible but introduces a
rater-identity confound sitting exactly on the axis being measured: generated posts scored by our
rater against real posts scored by theirs. That is not a shortcut worth taking.

**Needs:** the rating harness from Vectorial, plus a re-rated held-out slice to establish agreement.
**This has been outstanding since the aspect package arrived and now gates the novel contribution
rather than only the metric. It is the most urgent thing to request.**

### D. The blind aspect rerun

The aspect vocabularies were extracted with platform labels visible to the extracting model, so the
vocabulary may have been constructed to separate the platforms, which is the signal the
aspect-conditioned columns are then scored on. Every aspect-conditioned result is provisional.

**Needs:** the collaborators' blind rerun, already agreed. We could rerun it ourselves, but the
vocabulary would then differ from the one their analyses use, and the two would stop being
comparable.

### E. Aspect vocabularies for the genuinely uncovered cells

Beyond the five probable renames in item 10, five topics have no plausible counterpart in the shipped
set: `ruby on rails releases`, `engineering management challenges`, `education technology jobs`,
`online learning courses`, and `api development`. `fullstack_engineer::career_development`, at 147
posts the largest cell in the corpus, has no vocabulary under any label.

**Needs:** extraction for those cells. This is collaborator work, not ours.

### F. Title and body as separate fields

Reddit submissions carry a title and a selftext body natively, and `final_dataset_v1.csv` flattens
both into one `post_text`. This is why link submissions appear as degenerate title-only outputs and
why the length distribution has the shape it does. Modelling and scoring them separately would be a
better task definition and would explain a large part of what the length ratio currently measures.

**Needs:** re-extraction at the scrape. Apify returns both fields, so the information was available
and was discarded during flattening. It cannot be recovered from the CSV.

### G. A third platform, and additional subpopulations

Twitter, and knowledge-worker groups beyond software engineers, are dataset contributions rather
than method contributions. They require collection.

### H. Downstream task validation

Ad-engagement prediction and survey prediction both require ground truth we do not hold. Stack
Overflow survey data is a concrete external source and would need to be joined to the audiences,
which is a project in itself.

### I. The judge has never been validated against human annotation

The `RAPIDATA_*` credentials in `.env` suggest an available annotation pathway, so this is closer to
solvable than the rest of Part II, but it needs human annotation time and money rather than compute,
and it needs a protocol. Listed here because it cannot be done on the cluster tonight.

---

---

## Running the work: cluster harness and evaluation tooling

Everything in Part I is executed through the tooling below. This section is
operational rather than argumentative, and it is written on the assumption that
whoever picks the work up has not run any of it before. The failure modes listed
are the ones that actually occurred during Study 2, each of which cost time and
none of which announced itself.

### The shape of the environment

`/home/davidchan` is NFS-shared across `cthulhu1` to `cthulhu6`, so the
repository is synced **once** and every node sees it. The node named in a command
only decides which machine does the work. Each node holds six RTX 4090s with
23.5 GB usable, and the cards are shared with their owners, who may take one at
any time.

The cluster interpreter is `~/micromamba/envs/vectorial/bin/python`, built for
this project because no existing environment carried `peft`. It holds
torch 2.11 with CUDA 12.8, transformers 5.14, peft 0.20, trl 1.9 and
bitsandbytes 0.50. `transformers` is on the version 5 line, which removed
deprecated interfaces, so code written against version 4 idioms should be checked
rather than assumed. The package itself is **not** pip-installed there; anything
that imports `vectorial_eval` needs `PYTHONPATH` set to include `src/`.

Models are read from `~/.cache/huggingface`, which already holds every checkpoint
this project uses. The shared disk sits above ninety per cent, so downloads are
disabled by default in the launchers and only adapters, soft prompts and steering
vectors are written, never a merged model.

### The five scripts

| Script | Purpose |
|---|---|
| `scripts/cluster/sync.sh [node]` | Push the repository. Idempotent; a second run copies nothing. |
| `scripts/cluster/pick_gpu.py --n K` | Print `host gpu` lines for K cards under about 500 MiB. |
| `scripts/cluster/launch.sh --job NAME -- script.py [args]` | Launch a **training** job detached, with a log, a PID file and a checkpoint directory. |
| `scripts/cluster/generate.sh --fn FN --tag TAG --artifact ENV=VALUE` | Launch a **generation** pass from a trained artifact. |
| `scripts/cluster/status.sh [--job NAME] [--tail N]` | One row per job: alive or not, latest checkpoint and its age, and whether the log ends in an out-of-memory error or a kill. |

`sync.sh` pushes with `--delete` and excludes `runs/` in its entirety. That
exclusion is load-bearing rather than tidy: training artifacts are produced on the
cluster and never exist locally, so a sync that did not exclude them would delete
them. During Study 2 it deleted a completed set of steering vectors between one
job finishing and the next launching. Bring results back with an explicit pull:

```bash
rsync -az cthulhu1.ist.berkeley.edu:'~/Projects/vectorial/runs/methods/' runs/methods/
```

### A full pass, end to end

```bash
# 1. push code and data
scripts/cluster/sync.sh cthulhu1

# 2. train. One job per configuration; they are independent and run concurrently.
scripts/cluster/launch.sh --job lora-target_lm-all-r16 -- \
    src/vectorial_eval/methods/lora/train.py --variant target_lm --config all-r16

# 3. watch. Poll this rather than the logs; it is one line per job.
scripts/cluster/status.sh --quiet

# 4. select the configuration on dev, never on test
.venv/bin/python src/vectorial_eval/methods/soft_prompt/select_best.py --remote

# 5. generate from the selected artifact
scripts/cluster/generate.sh --fn lora --tag lora_target_lm \
    --artifact VECTORIAL_LORA_ADAPTER=runs/checkpoints/lora-target_lm-all-r16/best

# 6. pull the outputs back
rsync -az cthulhu1.ist.berkeley.edu:'~/Projects/vectorial/runs/methods/' runs/methods/

# 7. score, twice: with the oracle and without it
.venv/bin/vectorial-eval --run-dir runs/methods evaluate --split test \
    --metrics structural classifier distributional trm semantic degeneracy \
    --embedding-model google/embeddinggemma-300m --embedding-dim 512

# 8. build the report
cd experimental-notes && SOURCE_DATE_EPOCH=1000000000 ../.venv/bin/python build_study2.py
```

### Artifact selection is by environment variable, and the names differ

Each trained transfer function loads its artifact from an environment variable.
The names are not interchangeable and passing the wrong one is silent:

| Function | Variable | Points at |
|---|---|---|
| `lora` | `VECTORIAL_LORA_ADAPTER` | an adapter directory, usually `.../best` |
| `soft_prompt` | `VECTORIAL_SOFT_PROMPT_DIR` | an adapter directory |
| `steering` | `VECTORIAL_STEERING_ARTIFACT` | a `.pt` vector file |

During Study 2 all three steering variants were launched with
`VECTORIAL_STEERING_PATH`, which nothing reads. Every run silently loaded the
same default artifact and produced byte-identical output, and the three columns
were briefly reported as a variant comparison. **Always check the run record
before trusting a comparison**, since each generation writes the artifact it
actually loaded and its digest:

```bash
python -c "import json;d=json.load(open('runs/methods/transfer_fn.steering_steer_target_lm.json'));\
print(d['artifact'], d['artifact_sha256'][:12], d['layers'], d['alpha'])"
```

If two columns that should differ report the same digest, they are the same run.

### Steering needs a selection pass before it means anything

The steering transfer function reads its layer, strength and scope from
`chosen.<artifact stem>.json` beside the artifact, and warns and falls back to
unselected defaults when that file is absent. `sweep_steering.py` performs the
selection on the validation split, maximising the cosine between the generated
and authentic pool centroids subject to a degeneracy constraint:

```bash
python -m vectorial_eval.methods.steering.sweep_steering \
    --artifact runs/steering/steering.target_lm.pt \
    --layers 12,16,20 --alphas 0.1,0.25,0.5,1.0 \
    --out runs/steering/sweep.target_lm.json
```

List arguments are **comma-separated**, not space-separated. The sweep writes the
`chosen.*.json` the transfer function reads; earlier it wrote only to `--out`,
which left the selection where nothing would load it, and the whole study
generated at the unselected default of 1.0. At that strength the model produces
repetitive fragments running to the token limit; the selected strengths are
between one tenth and one half of it.

### The memory ceiling is set by the vocabulary, not the weights

Measured on a live card before any training was launched, and recorded in
`docs/09-cluster-training-config.md`:

| Configuration | Batch | Length | Peak | Headroom |
|---|---|---|---|---|
| bf16 + gradient checkpointing | 1 | 768 | 22.6 GB | 0.9 GB |
| bf16 + gradient checkpointing | 2 | 768 | OOM | — |
| **4-bit NF4 + double quant** | **1** | **768** | **19.1 GB** | **4.4 GB** |

Batch size is bound by the 128k-vocabulary logits tensor rather than by the model
weights, so batch size one is required in either precision. Use 4-bit for every
family even where sixteen-bit would fit: 22.6 GB of 23.5 GB dies the moment a card
owner's process joins the GPU, and identical quantisation across families makes it
a constant of the study rather than a difference between its columns.

### Jobs are killed, so everything resumes

The cards are shared and jobs are killed without warning. Every training script
uses `scripts/cluster/checkpointing.py`, which writes atomically to a temporary
path and renames, keeps the last two checkpoints, and resumes from the latest on
relaunch. **Relaunching the identical command is the recovery procedure**; it is
the default path rather than an option. `status.sh` reports the latest checkpoint
step and its age, which is how a stalled job is told from a slow one.

### Evaluation and reporting

Scoring is separate from generation so that a metric defect can be corrected and
results rescored without paying for generation again. Two run directories are
needed: the full one, and a systems-only one that excludes the oracle so that its
lower coverage does not shrink the common-cell intersection every column is
averaged over.

```bash
mkdir -p runs/methods_sys && cd runs/methods_sys
for f in ../methods/outputs.*.jsonl ../methods/transfer_fn.*.json; do
  case "$(basename "$f")" in *target_sample*) continue;; esac
  ln -sf "$f" "$(basename "$f")"
done
```

Report values from two embedding spaces are not comparable, and a report is filed
under the space that produced it. Do not force two spaces into one file with
`--report-tag`.

The site is generated by `experimental-notes/build_study2.py`, which reads the
report JSON and never transcribed numbers. It takes its run directories from the
environment, so both studies build from one checkout:

```bash
VECTORIAL_REPORT_RUN=runs/methods VECTORIAL_REPORT_RUN_SYS=runs/methods_sys \
VECTORIAL_REPORT_SPLIT=test VECTORIAL_REPORT_OUT=reports/study3 \
SOURCE_DATE_EPOCH=1000000000 .venv/bin/python experimental-notes/build_study2.py
```

Pin `SOURCE_DATE_EPOCH` when checking that a build is deterministic, since every
page carries its generation time and two builds either side of a minute boundary
otherwise differ in one character per page.

### Checks that must pass before a result is reported

```bash
.venv/bin/python -m pytest tests/ -q      # currently 116 tests
.venv/bin/ruff check src/
```

Beyond those, three checks are specific to this project and each exists because
its absence produced a plausible and wrong number:

1. **Figure overflow.** Text is wrapped by pixel estimate rather than by the
   renderer, so a long label can leave the viewBox without any error. Audit
   rendered `<text>` extents against each viewBox, including rotated nodes and
   accounting for `text-anchor`, before publishing.
2. **Baseline separation.** A metric that does not rank the wrong-topic control
   worst is reading register rather than transfer and must not be reported. The
   TF-IDF triangle score failed exactly this test, which is why two claims were
   withdrawn from the first study.
3. **Interval overlap.** A difference is a finding only when the two intervals do
   not overlap. The resolution table is generated by running that test, never
   written by hand.


## Sequence

The ordering follows from cost against what each item unblocks, not from interest.

**Week of 10 August — correctness before anything else.**
Item 1, the steering refit, because a published column is currently not measuring what it claims.
Item 2, the train-carved selection set, and item 3, the oracle draw count, because it doubles the evaluation base and every subsequent
number should be computed on the larger one. Item 5 rides along with the retrain at no cost.

**Week of 17 August.**
Item 6, the six-space replication, since it must precede any claim we intend to defend. Item 4, the
length-controlled evaluation, because it is the question a reviewer will ask first and it may change
what the headline claim is. Item 11, the aspect join.

**Week of 24 August — the grad student starts around 20 August and item 8 is well-scoped for them.**
Item 9, the in-domain ceiling and plug-and-play. Item 8, the second base model.

**Week of 31 August.**
Item 7, the reverse direction. Item 10, the MMD estimator pair. Trait-mediated transfer if and only
if the rating harness has arrived; if it has not, it is out of scope for ICLR and the paper is framed
around the benchmark and the adaptation comparison instead.

**From 8 September.**
Writing, with results frozen. Nothing new enters the table after this date.

## What to ask the collaborators for this week

These gate Part II items and none of them is expensive on their side.

1. **The rating harness** — prompt, model, version, parser, for both raters. Gates the novel
   contribution (C).
2. **Targeted collection** against a named list of the twenty-four cells that sit between five and
   nine posts on their thinner platform. This is the only lever on the binding constraint (A).
3. **Confirmation of the five probable cluster renames** in `docs/10`, and the cluster-naming
   mapping, so the join can be made on a stable key (item 11, E).
4. **A date for the blind aspect rerun** (D).
5. **Title and body as separate fields** in the next extraction, and confirmation of whether the
   existing scrape can be re-exported with them rather than re-collected (F).

## What Study 3 can claim if Part I is completed and Part II is not

A benchmark and a method comparison, on one direction, one corpus, and ten scored cells, with
replication across six embedding spaces, a length-controlled ablation, a complete baseline set
including a genuine in-domain ceiling, and every comparison reported with its interval and its
resolution verdict. That is a defensible ICLR blog-track or workshop contribution and an honest
short paper.

It is not the paper the project set out to write. That paper needs the trait-mediated method, which
needs the rating harness, and a wider dataset, which needs collection. Both are requests rather than
work, which is why they head the list above.
