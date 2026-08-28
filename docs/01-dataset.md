# 1. Dataset construction

`build_dataset.py` converts `final_dataset_v1.csv` into JSONL splits organised
around **bilateral cells**.

```bash
.venv/bin/vectorial-eval build \
    --room-mode canonical \
    --min-posts 5 \
    --source linkedin --target reddit
```

## The bilateral cell

A cell comprises **one audience room crossed with one topic cluster, partitioned by
platform**. It is the core unit of analysis identified in the project meetings, and
it is also the unit at which every metric is computed, since it is the smallest
grouping for which "the manner in which this audience discusses this subject on
Reddit" constitutes a well-defined distribution.

```
cell_id = "backend_engineer::ruby_on_rails_ecosystem"
          ├── linkedin: 21 posts
          └── reddit:   96 posts
```

## Room canonicalisation

The raw CSV contains 24 distinct `audience_room_name` values, but these are not
consistent across platforms. The LinkedIn component of the CTO room is labelled
`CTOs`, whereas its Reddit counterpart is labelled `CTO Reddit`. Grouped naively,
such pairs never form a bilateral cell.

| Raw name | Platform | Canonical identifier |
|---|---|---|
| `CTOs` | linkedin | `cto` |
| `CTO Reddit` | reddit | `cto` |
| `Engineering Manager` | linkedin | `engineering_manager` |
| `Engineering Manager reddit` | reddit | `engineering_manager` |
| `Fullstack Engineers` | linkedin | `fullstack_engineer` |
| `Full Stack Developer Reddit` | reddit | `fullstack_engineer` |

Only two raw rooms — `Backend Engineer (linkedin+reddit)` and
`Instructional Designer` — already contain both platforms. The consequence is
substantial:

| `--room-mode` | Viable cells (≥5 posts per platform) |
|---|---|
| `strict` (raw names) | 19 |
| `canonical` (default) | **30** |

`canonical` is the default because it nearly doubles usable coverage. It is
nevertheless an **assumption rather than an established fact**: it asserts that
`CTOs` and `CTO Reddit` sample the same underlying audience. Where this assumption
is untenable for a particular room, the corresponding entry should be removed from
`ROOM_CANONICAL_MAP` in `config.py`. Both modes should be run and compared whenever
a result appears surprising, since a finding that obtains only under `canonical` may
be an artefact of an unwarranted merge.

## Split design

Splits are assigned at the **post** level, stratified within `(cell, platform)`.

*Why not partition by cell?* With 30 cells containing between 5 and 150 posts each,
a cell-level partition would yield single-digit cell counts per split. Since every
metric is a distribution comparison requiring a pool, such a design is not viable.

*How is leakage then controlled?* By role rather than by partition:

- `exemplar_ids` — target posts on which a transfer function may condition.
  **Always drawn from the train split.**
- `target_reference_ids` — target posts against which a task is scored.
  **Always drawn from the task's own split.**

A function may therefore condition on train-split Reddit posts while being evaluated
against held-out test-split Reddit posts from the same cell. The two sets are
disjoint by construction.

**Zero-shot generalisation** is handled along a second, separate axis. Fifteen
percent of cells are flagged `heldout_cell`; their posts appear only in the test
split and they carry no exemplars. A few-shot function degrades silently to
zero-shot behaviour on these cells, which renders the generalisation penalty
directly measurable rather than latent.

## The merged `heldout` split

Evaluation uses a fourth split, `heldout`, which is the union of validation and
test. Both are already excluded from every transfer function's exemplar pool, so
merging them introduces no leakage, and it roughly doubles the number of real
posts available to score against in each cell.

This matters because the group-level metrics require a minimum number of posts
per cell and were otherwise limited by how the split was cut rather than by the
size of the corpus. Merging raised coverage of the group-similarity metric from 5
cells to 10.

Only the **scoring references** widen. `exemplar_ids` still draw exclusively from
train, so the leakage-control argument is unchanged. `_rebuild_tasks_for_heldout`
in `build_dataset.py` re-points val and test tasks at the union of both reference
pools and relabels their split; it does not touch exemplars.

```bash
.venv/bin/vectorial-eval transfer --fn llm_rewrite --split heldout
.venv/bin/vectorial-eval evaluate --split heldout
```

`heldout` should be the default evaluation split. Use `test` alone only when
deliberately reproducing an earlier single-split run.

**Determinism.** Split assignment is computed as `blake2b(post_id, seed)` rather
than by `random.shuffle` or Python's per-process-salted `hash()`. The addition of
new material therefore never reshuffles existing assignments — a necessary property
given that the corpus continues to grow.

## Artifacts

```
data/
  posts.{train,val,test,heldout}.jsonl    one record per post
  cells.jsonl                             one record per bilateral cell
  tasks.{train,val,test,heldout}.jsonl    one record per transfer request
  manifest.json                           configuration, counts, per-cell inventory
```

The `heldout` files are the union of `val` and `test` (see above). They duplicate
those records rather than replacing them, so the three-way split remains available.

Three files are emitted rather than one because the underlying units genuinely
differ. A task's reference pool is a *set* of posts; flattening this into one row
per generation would replicate the entire pool on every row.

### `PostRecord`

```json
{
  "post_id": "t3_1jrcl48",
  "cell_id": "fullstack_engineer::react_state_management",
  "room": "fullstack_engineer",
  "room_raw": "Frontend Engineer Reddit",
  "topic": "react state management",
  "platform": "reddit",
  "domain": "Software Engineering",
  "text": "Several React and js frameworks stand out for...",
  "post_type": "opinion",
  "split": "train",
  "legacy_labels": {"sentiment": "neutral", "stance": "informational",
                    "original_topic": "frontend frameworks discussion",
                    "topic_narrow": "js frameworks comparison"}
}
```

The fields `sentiment`, `stance`, `original_topic`, and `topic_narrow` are
namespaced under `legacy_labels` because the team identified them as artefacts of
earlier extraction passes. They are preserved so that a subsequent pass may revisit
them, but are held apart from the primary record surface to prevent inadvertent use.

Dataset v2 also supplies `subreddit` for Reddit posts and `job_title` for
LinkedIn posts. The builder preserves both fields on `PostRecord` for
subpopulation analysis. They are platform-specific labels, not a shared audience
axis. Rows that lack `audience_room_name` are excluded from bilateral-cell
construction rather than being combined into a synthetic `nan` room. Including
those rows in a cross-platform experiment requires an explicit, reviewed mapping
between job titles and subreddits. The mapping must be recorded as experiment
configuration because it changes the population being measured.

### `TransferTask`

```json
{
  "task_id": "backend_engineer::agentic_coding::urn:li:activity:7...",
  "cell_id": "backend_engineer::agentic_coding",
  "source_platform": "linkedin",
  "target_platform": "reddit",
  "source_text": "Y'all really shouldn't sleep on Temporal for...",
  "target_reference_ids": ["t3_abc", "t3_def", "..."],
  "exemplar_ids": ["t3_xyz", "..."],
  "heldout_cell": false
}
```

No gold target field is present. This absence encodes the governing constraint at
the level of the schema: the reference is a pool, not a string.

## Normalisation

Applied within `load_frame`, in order:

1. Platform restricted to `linkedin` and `reddit`.
2. Text length constrained to the interval [30, 8000] characters.
3. Deduplication on `post_id`, followed by deduplication on exact `post_text`.
4. Rows carrying an empty `final_topic` removed.
5. Room canonicalisation applied.

This yields 20,529 raw rows, reduced to 14,672 after normalisation, of which 939
fall within viable bilateral cells. The final reduction is by far the largest and
reflects **coverage rather than filtering**: the majority of posts belong to a
`(room, topic)` pair carrying no mass on the opposing platform.

## Tuning coverage

### Current build

Under default settings: **30 bilateral cells** (6 dense, 4 reserved for zero-shot),
939 posts, 419 transfer tasks. Per split — posts: train 630, val 131, test 178,
heldout 309. Tasks: train 293, val 60, test 66, **heldout 126**.

| Setting | Effect |
|---|---|
| `--min-posts 3` | Approximately 65 cells, but pools too small for TRM or MMD |
| `--min-posts 5` (default) | 30 cells |
| `--min-posts 8` | 8 cells, all dense |
| `--room-mode strict` | 19 cells, no merge assumptions |
| `topic_field="topic_narrow"` | Substantially more, and substantially smaller, clusters |

The `dense_min_posts_per_platform` threshold (default 10) tiers cells rather than
discarding them: sparse cells are retained in the dataset and reported separately,
consistent with the guidance to prioritise dense cells and defer weaker ones.
