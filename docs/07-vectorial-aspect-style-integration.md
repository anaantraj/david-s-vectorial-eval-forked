# 7. Integrating the Vectorial style and aspect package

Working notes on `aspect_stylistics.zip` (received 2026-07-29) and what it takes to
turn it into transfer-function metrics. Everything below was verified against the
shipped files and the current `runs/scaled` reports.

## What arrived

| File | Contents |
|---|---|
| `stylistic/stylistic_dimensions.json` | 8 style dimensions, each with a description and 0/1 anchors |
| `stylistic/stylistic_embeddings.csv` | 2,130 real posts × 8 dimensions, 0–1 |
| `stylistic/cluster_stylistic_similarity_clean.csv` | 38 clusters, LL/RR/LR cosine + `style_gap` |
| `aspects/aspects_*.json` | 36 per-cluster aspect vocabularies (extracted with `gpt-4.1-mini`) |
| `aspects/aspect_embeddings.csv` | 812 real posts × per-cluster aspect scores |
| `aspects/phase_d_similarity.csv` | 36 clusters, LL/RR/LR on aspect vectors |

Both per-post files are *ratings of real posts*. Neither contains a rater we can run.

## It joins to the harness, on 19 of 30 cells

Every `post_id` in both files resolves against `final_dataset_v1.csv`. Against the
built splits:

- stylistic: 490 of 939 harness posts rated
- aspect: 488 of 939 harness posts rated

Coverage is essentially all-or-nothing per cell — a cell is either fully rated or
untouched — which makes the joinable subset clean to define. **19 of 30 cells have
≥4 rated posts on both platforms**, for both files.

The 11 uncovered cells are not a random sample. They include the largest cell in the
corpus (`fullstack_engineer::career_development`, 147 LinkedIn posts) and the entire
`edtech_engineering` room (`ai_in_education` at 65 Reddit posts,
`online_learning_courses` at 56). Rating those is cheap and roughly doubles the
Reddit mass available to an aspect metric.

## The blocker: we cannot score generated posts

This is the one thing that has to be resolved before any of this becomes a metric.

Every harness metric compares a pool of *generated* posts against a pool of authentic
target posts. Vectorial shipped vectors for real posts only. To place a generated post
in the same space we need the rater itself — the prompt, the model, the version, and
the output parsing — not the outputs it produced.

The rubric is reproducible in principle: `stylistic_dimensions.json` carries names,
descriptions and both anchors, and the aspect files name the extraction model. We
could reimplement a rater from that. But a reimplementation is not comparable to their
numbers at the level of precision the resolution rule demands, and we would be
comparing generated posts scored by our rater against real posts scored by theirs —
a rater-identity confound sitting exactly on the axis being measured.

**Ask Vectorial for the rating harness, and re-rate a held-out slice of real posts
with it to establish rater agreement.** Everything downstream is blocked on this.

For aspects there is a second blocker they already flagged: the discovery pass ran
with platform labels visible to the LLM. The blind rerun has to land before the
aspect vocabularies can support a claim about platform difference, because the
vocabulary itself may have been constructed to separate the platforms.

## Data issues to send back

**1. The zero-vector removal was not applied to the shipped file.** The README states
that zero-vector posts "have been removed" from `stylistic_embeddings.csv`. All 132
are still present. `zero_vector_audit.txt` documents exactly these 132, and the
cluster summary was computed after removing them, so the summary and the per-post
file disagree — 17 of the 38 clusters have counts in
`cluster_stylistic_similarity_clean.csv` that are lower than the per-post file, always
in the direction of the removed posts.

This matters more than a bookkeeping error. An all-zero vector has undefined cosine
similarity; anyone using the file as shipped gets silent zeros or NaNs in 6.2% of
rows, concentrated in Software Engineering (8.9%) and in LinkedIn Data & Analytics
(16.2%). Filtering by vector norm before use is mandatory.

**2. `stylistic_embeddings.csv` has a duplicated header row.** Row 2 repeats the
column names. A naive `read_csv` yields one junk record.

**3. Clusters are keyed by `(domain, final_topic)`, not `final_topic`.** Six topics
span two domains — `ruby on rails ecosystem` and `rust programming language` each
appear under both Open Source and Software Engineering, and so on. The stylistic
files respect this; `phase_d_similarity.csv` does not carry a domain column at all.
Joining aspects on `final_topic` alone silently merges those clusters.

`phase_d_similarity.csv` is otherwise internally consistent — all 36 rows match the
per-post file exactly.

## The style finding replicates, but the metric scale is wrong

Their headline reproduces exactly: LL > LR in 34 of 38 clusters, mean `style_gap`
0.0338. It also survives a change of statistic, which is the more important result —
measuring within-platform dispersion as mean L2 distance to the cluster centroid,
LinkedIn is tighter in **32 of 38** clusters, mean dispersion 0.559 versus Reddit's
0.703.

So the substantive claim — LinkedIn is more register-coherent within a topic, Reddit
is more variable — holds. But cosine is the wrong instrument for it. The 8 dimensions
are all non-negative, so the space is compressed into a narrow band: two *randomly
paired* posts from anywhere in the corpus average a cosine of 0.773 (p5 0.431, p95
0.972). A `style_gap` of 0.034 is being read off a floor of 0.77. The dispersion
framing puts the same finding at a 26% relative difference on a scale that starts at
zero.

**Adopt the dispersion form, not the cosine form.** `style_gap` should not go into the
harness as defined.

The gap does not appear to be a small-sample artefact — correlation with minimum pool
size is 0.150 — but 15 of the 38 clusters have fewer than 5 posts on one platform, so
cluster-level values there should not be read individually.

## The framing to correct at the meeting

The package is built around the claim that Phase C explains our self-similarity
finding: *"this is probably what's behind your self-similarity finding: the transfer
function is hitting the LinkedIn mean rather than Reddit's variance."*

That finding was withdrawn. Per [05-limitations.md](05-limitations.md), "Claude
rewriting raises self-similarity above the authentic Reddit level" is supported only
under TF-IDF and unresolved in every neural space, so it is not a result. The numbers
in `runs/scaled`:

| Space | `llm_rewrite_claude` | `target_sample` |
|---|---|---|
| TF-IDF + SVD 256 | 0.654 [0.595, 0.709] | 0.390 [0.241, 0.538] |
| EmbeddingGemma d128 | 0.853 [0.826, 0.883] | 0.863 [0.816, 0.907] |
| EmbeddingGemma d256 | 0.842 [0.815, 0.873] | 0.844 [0.790, 0.892] |
| EmbeddingGemma d512 | 0.825 [0.796, 0.859] | 0.828 [0.771, 0.880] |
| EmbeddingGemma d768 | 0.815 [0.784, 0.852] | 0.811 [0.748, 0.866] |
| all-MiniLM-L6-v2 d384 | 0.566 [0.502, 0.634] | 0.549 [0.443, 0.661] |

Under every neural space the intervals overlap and the point estimates are within
0.01. Generated pools are not measurably more homogeneous than authentic Reddit pools.

This does not make Phase C wrong. LinkedIn really does look more register-coherent
than Reddit in their style space. It means Phase C is not explaining that, and the
package's stated bridge to our work does not hold. Worth raising early and without
drama — the finding is theirs to keep, the causal story is the part to drop.

There is a live and more interesting question underneath it. Their claim is about
*real* LinkedIn versus *real* Reddit dispersion. Ours was about *generated* versus
*real* dispersion. Whether a transfer function reproduces Reddit's dispersion is a
question we can now actually ask in the style space, once we can rate generated posts
— and `trm.rank_i2` already says the answer is not the one assumed. In the d512 space
the LLM systems sit at 0.068–0.129 against `target_sample`'s 0.261 and a null of ⅓,
with `rank_i0` at 0.68–0.73. That signature is candidates sitting *outside* the
reference cloud, not collapsed inside it. The problem is location, not variance.

## Proposed next steps

1. **Request the rating harness from Vectorial** — prompt, model, version, parser, for
   both the style rater and the aspect rater. Blocking for everything below.
2. **Send back the three data issues** with the zero-vector one flagged as
   correctness-affecting.
3. **Ask for the 11 uncovered cells to be rated**, prioritising `career_development`
   and the `edtech_engineering` room.
4. **Confirm the blind aspect rerun is in flight** and get an expected date.
5. **Implement `style` as a harness metric** once the rater lands: per-cell dispersion
   match and centroid distance in the 8-dimensional space, plus the per-dimension
   breakdown, which is the interpretable table the client keeps asking for. It slots
   into the metric registry unchanged and needs the standard minimum-pool guard and a
   test.
6. **Implement `aspect` as a harness metric** after the blind rerun: per-cell JSD
   between the generated and authentic aspect distributions. This is what
   [05-limitations.md](05-limitations.md) names as the single most valuable extension —
   a function that reproduces register and topic while inverting aspect emphasis is
   currently unpenalised by anything we measure.
7. **Re-run the baseline separation test on both new metrics before trusting them.**
   A metric that does not rank `shuffle_control` worst is reading register, not
   transfer. The TF-IDF triangle score failed exactly this test and it is the reason
   two claims were withdrawn.

## What to have for the next meeting

- The three data issues, sent before the meeting rather than raised in it.
- The dispersion-versus-cosine reframing, with the 32/38 and the random-pair floor.
- The self-similarity correction, with the six-space table.
- The 19-of-30 coverage number and the specific ask for the missing 11.
- A one-slide sketch of the `style` and `aspect` metric definitions, so the rater
  request is concrete about what the metric needs to consume.
