# Handoff

Written for whoever picks this project up after Study 4. It says what exists, what
state each arm is in, what the immediate next action is, and where the things that
are not in git live.

Read in this order:

1. This file — current state and the next action.
2. [docs/00-experimentation-guide.md](docs/00-experimentation-guide.md) — why the harness is
   built the way it is, and the errors it exists to prevent.
3. [AGENTS.md](AGENTS.md) — the invariants and the checks that must pass before anyone
   claims a result.
4. [STUDY4_PLAN.md](STUDY4_PLAN.md) — the design you are continuing.

---

## 1. Getting to a working checkout

```bash
git clone git@github.com:DavidMChan/vectorial-eval.git
cd vectorial-eval
uv venv --python 3.12
uv pip install -e ".[dev]"
```

Then fetch the data archive (section 5) and unpack it into the repository root, and
write a `.env`:

```bash
echo "OPENROUTER_API_KEY=sk-or-..." >> .env    # LLM transfer functions and the judge metric
echo "NDIF_API_KEY=..."             >> .env    # only the steering_ndif arm
```

Confirm the checkout is sound before changing anything:

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check src/
```

## 2. What is settled

Study 3 established the method comparison at 8B under 4-bit NF4, in both the forward
and reverse directions, over fixed bilateral cells. Those numbers are in `runs/study3*`
and are reported in the published notes.

Study 4 executed as a **four-audience case study**, not the planned multi-audience
confirmatory study. This distinction is load-bearing and must survive into any future
write-up:

- Acceptance gates 1 through 4 in `STUDY4_PLAN.md` §3.5 are **not met**. Four reviewed
  bilateral mappings were available, not the planned set.
- Instructional-design Reddit contributes 8,936 records after clone resolution, not 10,000.
- Twitter and Quora were never in the supplied exports. They are future replications, not
  silently missing columns.
- **No audience-generalisation claim may be made from the current results.**

The frozen Stage A primary base is `platform_prior_lm` on the full available corpus,
capped at 10,000 records per audience: 34,901 Reddit and 16,103 LinkedIn training
records. The raw-text arm and the nested 1,000-per-audience arm are ablations. Held-out
frontier results did not select it — that split is reserved for final scoring.

## 3. The immediate next action

**Run the NDIF steering sweep.** This is the one arm that is fitted but unscored, and
`STUDY4_PLAN.md` §4.5 is explicit that the artifact must not be scored until the sweep
selects a layer and an alpha.

| | state |
|---|---|
| `target_lm` on `Llama-3.1-70B-Instruct` | fitted 2026-08-27, 630/630 posts, 0 failures |
| `target_lm_paired`, `target_lm_aspect` on 70B | not run |
| any variant on `Llama-3.1-405B-Instruct` | not run |
| dev sweep over layer and alpha | **not run — do this first** |
| scored transfer outputs from this arm | none |

The fitted artifact is `runs/steering/steering.target_lm.ndif-Llama-3.1-70B-Instruct.pt`
(81 MB, an `(81, 8192)` global vector over 337 target and 293 source train posts; own
vectors for 23 of 26 cells and 4 of 5 audiences).

Known-thin, do not report per-cell numbers for these: `backend_engineer::agentic_coding`,
`backend_engineer::react_state_management`, `edtech_engineering::education_technology_jobs`;
and the `product_leadership` audience.

Entry point: `methods.steering.sweep_steering`, under the same
Pareto-dominance-under-a-degeneracy-constraint objective as §4.4.

A spot check at layer 40, alpha 0.5 produced structurally Reddit-like output that
code-switched into Spanish and German. That is the intervention pushing off-manifold with
language identity breaking first. It is exactly what the degeneracy guard in §6 exists to
catch, so treat a degeneracy-guard failure in the sweep as informative, not as a bug.

## 4. Open threads after that

- `target_lm_paired` and `target_lm_aspect` at 70B, then the 405B scale contrast. Report
  against the local steering column only; never pool with the 4-bit columns.
- The scale-curve inflection (`STUDY4_PLAN.md` §12): the 50,000–75,000-record prior comes
  from another generation domain. The nested scale curve has to decide whether it holds here.
- Twitter and Quora as separate replications, if those exports ever arrive.
- Widening beyond four audiences, which is what would actually close acceptance gates 1–4.

## 5. Data, runs and checkpoints

None of this is in git — `data/`, `runs/` and the bulk CSVs are ignored deliberately. They
are distributed as three archives:

| Archive | Size | Contents |
|---|---|---|
| `vectorial-study34-data-2026-08-28.tar.zst` | 84 MB | `data/` in full — Study 3 and Study 4 splits, priors, aspects, training sets, manifests — plus `final_dataset_v1.csv` |
| `vectorial-study34-runs-2026-08-28.tar.zst` | 122 MB | `runs/` minus checkpoints — every scored run, report JSON and output pool behind the published numbers |
| `vectorial-study34-checkpoints-2026-08-28.tar.zst` | 433 MB | `runs/checkpoints/` — almost entirely the two 324 MB NDIF steering resume payloads; the LoRA and soft-prompt adapters are small |

Download and unpack from the repository root:

```bash
BASE=https://s3.us-west-1.wasabisys.com/vectorial-reports/data
for a in data runs checkpoints; do
  curl -fL -O "$BASE/vectorial-study34-$a-2026-08-28.tar.zst"
done
for a in data runs checkpoints; do
  zstd -dc "vectorial-study34-$a-2026-08-28.tar.zst" | tar -xf -
done
```

Verify before trusting anything derived from them:

```
bdc93df035484fb04d76df3da805e6a30394b4754aefc46f4ca999624f3f48e9  vectorial-study34-data-2026-08-28.tar.zst
d5813b1145384a7b0b196a7ceaa6aea9cdc8a46ec52bcad8098fe2c4d7505572  vectorial-study34-runs-2026-08-28.tar.zst
2dbbb12f892a3d87c52be77d671e5102462c144c6d7373739bc0f7d612ca0643  vectorial-study34-checkpoints-2026-08-28.tar.zst
```

```bash
shasum -a 256 -c SHA256SUMS
```

> **These archives are on a publicly readable bucket.** `vectorial-reports` carries a
> `PublicReadReports` policy granting `s3:GetObject` to `*`, so anyone with the URL can
> download them without credentials — including the raw prior variants described below.
> This was a deliberate choice; to reverse it, delete the `data/` prefix and re-upload
> to a private bucket.

You only need the checkpoints archive if you are resuming NDIF steering. Everything in
sections 2 and 3 except the sweep runs from the first two.

### Raw and sanitized prior variants

Study 4 prior files ship in two variants, and the difference matters:

- `*.train.jsonl` / `*.dev.jsonl` — **raw**. These carry `profile_weak_labels.summary` and
  `.highlights`, which are generated prose that names real people, their employers and their
  education history.
- `*.rich_safe.*.jsonl` — sanitized, with a `.sanitization.json` manifest recording source and
  output hashes and what was removed by field.

`STUDY4_PLAN.md` §Status excludes profile summaries and highlights from model-visible priors
precisely because they can reproduce author identity. **Structured-prior training reads the
`rich_safe` variants.** The raw files are retained so the sanitizer
(`scripts/sanitize_study4_prior_context.py`) can be re-run and audited, not for training and
not for redistribution.

Room traits remain audit-only: their source covers whole rooms including held-out authors, so
treating them as train-only priors would violate the split.

## 6. Things that will bite you

- **Never install into system Python.** Everything runs through `.venv/bin/…`.
- **The report build must be deterministic.** Build twice under
  `SOURCE_DATE_EPOCH=1000000000`, confirm byte identity, then build once without it so the
  committed HTML states when it was really generated. A non-deterministic build usually means
  an unsorted dict or a leaked `hash()`.
- **Values from different embedding spaces are not comparable.** Reports are filed under the
  space that produced them rather than replacing each other.
- **TRM values computed over different base distances are not comparable.** The base distance
  is recorded in every report.
- **A difference is a finding only when the bootstrap intervals separate.** Two claims in the
  first draft of the first study failed this test and were withdrawn.
- **`aws s3 cp --recursive` segfaults against the Wasabi endpoint.** `scripts/publish_report.sh`
  uploads one object at a time for this reason.
- 4-bit NF4, batch size 1, max length 768 on `Llama-3.1-8B-Instruct`. The 128k-vocab logits
  tensor sets the memory ceiling, not the weights. See
  [docs/09-cluster-training-config.md](docs/09-cluster-training-config.md).
