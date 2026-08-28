# Vectorial × BAIR — Transfer Function Experimentation Harness

An evaluation harness for **cross-platform audience transfer functions**: given the
manner in which an occupational audience expresses itself on LinkedIn, predict how
the same audience expresses itself on Reddit.

The harness provides infrastructure rather than a modelling contribution. Its
purpose is to ensure that when a substantive transfer function becomes available —
steering vectors, LoRA adapters, or the trait-mediated formulation toward which the
project is moving — it can be evaluated against fixed splits, fixed baselines, and a
fixed metric suite without per-experiment modification.

```
CSV ──► build ──► JSONL splits ──► transfer ──► outputs ──► evaluate ──► report
                  (bilateral cells)  (fn registry)          (metric registry)
```

**Continuing this project?** Start with [HANDOFF.md](HANDOFF.md) — current state of
every arm, the immediate next action, and where the data lives.

## Installation

The project is managed with [uv](https://docs.astral.sh/uv/) and installs into an
isolated virtual environment.

```bash
uv venv --python 3.12
uv pip install -e .

# Optional neural embedding backend (adds torch and sentence-transformers)
uv pip install -e ".[embeddings]"

# Optional remote steering on NDIF (adds nnsight)
uv pip install -e ".[ndif]"
```

Credentials for the LLM-backed components are read from `.env`:

```bash
echo "OPENROUTER_API_KEY=sk-or-..." >> .env
echo "NDIF_API_KEY=..." >> .env          # only for the steering_ndif arm
```

## Data

`data/`, `runs/` and the bulk source CSVs are not in git. They are distributed as three
archives on Wasabi object storage:

| Archive | Size | Contents |
|---|---|---|
| `vectorial-study34-data-2026-08-28.tar.zst` | 84 MB | `data/` in full — Study 3 and Study 4 splits, priors, aspects, training sets and manifests — plus `final_dataset_v1.csv` |
| `vectorial-study34-runs-2026-08-28.tar.zst` | 122 MB | `runs/` minus checkpoints — every scored run and report JSON behind the published numbers |
| `vectorial-study34-checkpoints-2026-08-28.tar.zst` | 433 MB | `runs/checkpoints/`, almost entirely the two NDIF steering resume payloads |

Unpack from the repository root:

```bash
BASE=https://s3.us-west-1.wasabisys.com/vectorial-reports/data
for a in data runs checkpoints; do
  curl -fL -O "$BASE/vectorial-study34-$a-2026-08-28.tar.zst"
  zstd -dc "vectorial-study34-$a-2026-08-28.tar.zst" | tar -xf -
done
```

> **These archives are on a publicly readable bucket.** `vectorial-reports` carries a
> `PublicReadReports` policy granting `s3:GetObject` to `*`, so anyone with the URL can
> download them without credentials — including the raw prior variants described below.
> This was a deliberate choice; to reverse it, delete the `data/` prefix and re-upload
> to a private bucket.

Checksums and the difference between the raw and `rich_safe` prior variants are in
[HANDOFF.md](HANDOFF.md#5-data-runs-and-checkpoints). The short version: structured-prior
training reads `rich_safe`; the raw files exist so the sanitizer can be re-run and audited.

## Usage

Dataset construction, the non-LLM baselines, and the full non-LLM metric suite
execute entirely offline:

```bash
.venv/bin/vectorial-eval build
.venv/bin/vectorial-eval transfer --fn identity target_sample shuffle_control
.venv/bin/vectorial-eval evaluate --split test
```

The LLM-backed components require a key. Evaluate on the merged `heldout` split
and draw several candidates per source post; both widen coverage at no data cost
(see [docs/01-dataset.md](docs/01-dataset.md) and
[docs/02-transfer-functions.md](docs/02-transfer-functions.md)):

```bash
.venv/bin/vectorial-eval --run-dir runs/scaled transfer \
    --fn llm_rewrite llm_fewshot --split heldout --n-samples 4 \
    --model claude-opus --tag claude
.venv/bin/vectorial-eval --run-dir runs/scaled transfer \
    --fn llm_rewrite --split heldout --n-samples 4 --model gpt-5.6 --tag gpt
.venv/bin/vectorial-eval --run-dir runs/scaled evaluate --split heldout --metrics \
    structural classifier distributional trm semantic degeneracy judge
```

The embedding space that the semantic, distributional and TRM metrics are
computed in is selected on `evaluate`, and reported results should be replicated
under a second one. Values from two spaces are not comparable, so the report is
filed under the space that produced it rather than replacing the previous one:

```bash
.venv/bin/vectorial-eval --run-dir runs/scaled evaluate --split heldout \
    --embedding-model google/embeddinggemma-300m --embedding-dim 256
# -> runs/scaled/report.heldout.embeddinggemma-300m-d256.json
```

The two spaces are compared in
[experimental-notes/embedding-space.html](experimental-notes/embedding-space.html).

## Tests

```bash
uv pip install -e ".[dev]"
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check src/
```

The suite covers the statistical core: that TRM recovers its theoretical null, that
it detects the under-dispersion signature for which it was adopted, and that the
shared numerical helpers behave correctly at their boundaries. Defects in these
would not raise an exception but would instead yield a plausible yet incorrect
result table.

## Presentation site

A self-contained static site summarising the results for circulation lives in
`experimental-notes/`. Open `experimental-notes/index.html` directly, or zip the
folder — it has no JavaScript and no external assets. Regenerate it from the
current reports with:

```bash
.venv/bin/python experimental-notes/build.py
```

Figures and qualitative examples are both extracted from the report JSON and the run
outputs rather than transcribed, so the site cannot silently drift from the data it
describes. The build is deterministic: rebuilding without changing the reports produces
byte-identical pages.

## Documentation

| Document | Contents |
|---|---|
| [docs/00-experimentation-guide.md](docs/00-experimentation-guide.md) | **Start here.** The reasoning behind the harness, the errors made and caught, and the rules for future experiments |
| [docs/01-dataset.md](docs/01-dataset.md) | Bilateral cells, split design, JSONL schemas |
| [docs/02-transfer-functions.md](docs/02-transfer-functions.md) | The interface, the baselines, and how to add a new function |
| [docs/03-metrics.md](docs/03-metrics.md) | Each metric, its interpretation, and its failure modes |
| [docs/04-interpreting-results.md](docs/04-interpreting-results.md) | A worked reading of an observed result table |
| [docs/05-limitations.md](docs/05-limitations.md) | Questions this harness cannot currently answer |
| [docs/06-reporting.md](docs/06-reporting.md) | Figure conventions, the report generator, and publishing |
| [docs/07-vectorial-aspect-style-integration.md](docs/07-vectorial-aspect-style-integration.md) | The shipped aspect package, what it contains, and how it joins to the harness cells |
| [docs/08-transfer-function-roadmap.md](docs/08-transfer-function-roadmap.md) | The four transfer functions not yet built, their cost against the real scale, and the sequence against the submission dates |
| [docs/09-cluster-training-config.md](docs/09-cluster-training-config.md) | Measured memory ceiling, the required batch and sequence settings, and the three training variants |
| [docs/10-aspect-vocabulary-coverage.md](docs/10-aspect-vocabulary-coverage.md) | Which cells the aspect vocabulary covers and which are missing |

Agents working in this repository should read [AGENTS.md](AGENTS.md) first; it
states the invariants that must not be broken and the checks that must pass.

## Design commitments

**Evaluation is distributional rather than pairwise.**
No one-to-one correspondence exists between posts across platforms, so no generated
post has a gold reference. Every metric compares a *pool* of generated posts against
a *pool* of authentic target posts within a single bilateral cell. This excludes
BLEU- and ROUGE-style scoring and determines the remainder of the design.

**Baselines establish the attainable range before any system is evaluated.**
`identity` (reproducing the source) is the floor on transfer and simultaneously the
ceiling on semantic preservation. `target_sample` (an authentic target post) is the
practical ceiling. `shuffle_control` (an authentic target post drawn from a
different topic) is the negative control that detects style-only matching. A metric
that fails to separate these three is not measuring transfer and should not be
relied upon for substantive systems.

**Style and semantics are measured separately and against one another.**
The project has identified as its principal risk that embeddings may organise posts
by writing style rather than by subject. Every embedding-based metric is therefore
reported twice: once in the raw space, and once with the platform-discriminating
directions projected out. A function that performs well in the raw space but no
better than `shuffle_control` after style removal has learned only the tone.

**Distribution matching is assessed with a small-sample-appropriate statistic.**
Bilateral cells contain tens of posts, not thousands. The harness therefore adopts
the Triangle-Rank Metric of Chan et al. (LREC-COLING 2024), which was designed for
precisely this regime and which penalises generation at the mode of the reference
distribution, the dominant failure mode of prompt-based rewriting. TRM is defined
over a pairwise distance (cosine in the harness embedding space); that base
distance is recorded in every report and values computed over different distances
are not comparable.

**A difference is a finding only when the intervals separate.** Cell-level scores
are bootstrapped over cells, and the report generates a resolution table by
running the overlap test over every stated comparison. Two claims in the first
draft of the first study failed this test and were withdrawn.

## Repository layout

```
src/vectorial_eval/
  config.py              sweepable settings; model and embedding registries
  llm.py                 cached OpenRouter client (Claude, GPT, Gemini, Grok)
  evaluate.py            evaluation driver, aggregation, and reporting
  cli.py                 build / transfer / evaluate / report / models
  data/
    schema.py            PostRecord, CellRecord, TransferTask, TransferOutput
    build_dataset.py     CSV to bilateral cells to train/val/test JSONL
  features/
    stylometry.py        21 interpretable register features
    embeddings.py        embedding backends and style-residual projection
  transfer/
    base.py              TransferFunction interface and registry
    baselines.py         identity, target_sample, shuffle_control
    llm_rewrite.py       zero-shot and few-shot LLM rewriters
  metrics/
    base.py              Metric interface, EvalContext, shared statistics
    classifier.py        platform-classifier confusion and calibration
    structural.py        per-feature distributional agreement
    distributional.py    MMD, centroid distance, topic JSD
    trm.py               Triangle-Rank Metrics and Frechet distance
    semantic.py          semantic preservation and degeneracy guards
    judge.py             LLM-as-a-judge, framed as a discrimination task

experimental-notes/      generated static report site (build.py + HTML + CSS)
scripts/publish_report.sh  stage to reports/<timestamp>/ and upload to Wasabi
reports/                 timestamped copies of published reports
tests/test_metrics.py    statistical core: TRM null, collapse signature, guards
AGENTS.md                invariants and checks for agents working in this repo
HANDOFF.md               state of every arm, the next action, and where the data lives
STUDY3_PLAN.md           Study 3 design
STUDY4_PLAN.md           Study 4 design, execution status, and acceptance gates
```

## Dataset state

Constructed from `final_dataset_v1.csv` under default settings:

- **30 bilateral cells** (6 dense, 4 reserved for zero-shot generalisation)
- **939 posts** and **419 transfer tasks**
- Source platform LinkedIn; target platform Reddit
- Evaluation uses the merged `heldout` split: **126 source posts**, four candidates
  drawn per post, giving 504 generated posts per language-model configuration

Metric coverage on that run: 22 cells for the classifier and content metrics, 15
for the writing-habit and distribution metrics, 10 for the group-similarity
metric.

Coverage, rather than harness capability, is the binding constraint: only 30 of
1,525 `final_topic` clusters carry at least five posts on both platforms. See
[docs/05-limitations.md](docs/05-limitations.md).

## Reference

Chan, D. M., Ni, Y., Ross, D. A., Vijayanarasimhan, S., Myers, A., and Canny, J. F.
*Distribution Aware Metrics for Conditional Natural Language Generation.*
LREC-COLING 2024, pp. 5064–5095.
