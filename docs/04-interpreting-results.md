# 4. Interpreting results

A worked reading of an actual result table, produced by:

```bash
.venv/bin/vectorial-eval build
.venv/bin/vectorial-eval --run-dir runs/scaled transfer \
    --fn identity target_sample shuffle_control --split heldout
.venv/bin/vectorial-eval --run-dir runs/scaled transfer \
    --fn llm_rewrite llm_fewshot --split heldout --n-samples 4 \
    --model claude-opus --tag claude
.venv/bin/vectorial-eval --run-dir runs/scaled transfer \
    --fn llm_rewrite --split heldout --n-samples 4 --model gpt-5.6 --tag gpt
.venv/bin/vectorial-eval --run-dir runs/scaled evaluate --split heldout
```

LinkedIn → Reddit, merged held-out split, 126 source posts, four candidates drawn
per post for each language model, TF-IDF/SVD embeddings at 256 dimensions.

```
metric                              identity  shuffle  gpt-5.6  claude-0  claude-4  oracle
classifier.calibration_gap ↓           0.375    0.214    0.210     0.111     0.078   0.134
semantic.source_similarity ↑           1.000    0.039    0.816     0.583     0.648   0.099
trm.trm ↓                              0.264    0.347    0.504     0.783     0.769   0.132
trm.rank_i2 ·                          0.267    0.107    0.383     0.386     0.415   0.277
degeneracy.pool_self_similarity ↓      0.211    0.043    0.560     0.654     0.659   0.390
degeneracy.distinct_2 ↑                0.937    0.931    0.512     0.693     0.671   0.737
```

Oracle values are averaged over the cells it covers, which is fewer than the
system columns cover; the report prints that count as a superscript.

## Step 1: verify the references behave as designed

Before reading any system column, confirm the three reference columns sit where
they must. If they do not, the metric is at fault and no system conclusion is
admissible.

- `identity` scores 1.000 on source similarity and the worst calibration gap.
  Correct: an unmodified LinkedIn post keeps all its content and is the least
  Reddit-like column.
- `target_sample` attains the best TRM by a wide margin. Correct: real Reddit
  posts should be distributionally indistinguishable from real Reddit posts.
- `shuffle_control` attains a good calibration gap but a poor topic JSD. Correct,
  and this is the separation the control exists to provide: right tone, wrong
  subject.

The references are consistent, so the system columns may be read.

## Step 2: read transfer and preservation together, never separately

The two model families occupy opposite positions on the central trade-off.

**Claude transfers aggressively.** Its calibration gap improves from 0.375 to
0.111, but source similarity falls to 0.583 and specific details are lost.

**GPT-5.6 transfers conservatively.** Source similarity stays at 0.816, but the
calibration gap only reaches 0.210.

Read in isolation, either column looks better than the other. Neither dominates,
so the choice between them is a position on the curve rather than a quality
ranking.

## Step 3: consult the group-level metric, and expect disagreement

This is the most consequential reading in the table.

The classifier reports that Claude's rewrites read more like Reddit than the
source posts do. TRM reports that the same rewrites, taken as a group, are
markedly *further* from real Reddit than untouched LinkedIn posts are:

```
trm.trm ↓        identity 0.264    claude zero-shot 0.783    oracle 0.132
```

Both readings are correct and both are supported by the intervals. The rewrites
hit the average tone of the platform and lose its variety. The explanation appears
in the degeneracy rows:

```
pool_self_similarity ↓   identity 0.211   claude 0.654   oracle 0.390
distinct_2 ↑             identity 0.937   claude 0.693   oracle 0.737
```

Generated posts resemble each other far more than real posts do, and draw on a
narrower vocabulary than either the source posts or the oracle. The corroborating
diagnostic is `rank_i2` at 0.386 against an expected ⅓ under a perfect match; the
oracle sits at 0.277, on the other side of ⅓.

This is exactly the failure a classifier-only evaluation conceals. Reported alone,
the calibration gap would have supported the conclusion that prompt-based
rewriting substantially solves the transfer problem.

## Step 4: check whether the result survives style removal

Compare each TRM figure against its style-removed counterpart. In this run the
values move little (0.783 raw against 0.783 style-removed for Claude zero-shot),
so the result is not an
artefact of the platform-style subspace. Had a column improved sharply after style
removal, the correct conclusion would have been that its apparent success derived
from tone agreement rather than subject agreement.

## Step 5: consult the resolution table before stating anything

Only differences whose bootstrap intervals do not overlap are findings. For this
run:

| Comparison | Verdict |
|---|---|
| Rewriting degrades group similarity relative to the source | supported |
| Rewriting raises self-similarity above the source | supported |
| Rewriting raises self-similarity above the authentic level | supported |
| Rewriting reduces vocabulary variety relative to the source | supported |
| GPT-5.6 preserves more source content than Claude | supported |
| Rewriting reduces the calibration gap relative to the source | supported |
| GPT-5.6 matches the target distribution better than Claude | not resolved |
| Few-shot improves transfer / preservation / distribution over zero-shot | not resolved |

The few-shot comparisons favour few-shot on every point estimate, and the gap is
smaller than the variation between cells. The report states the direction and
declines to claim the magnitude.

Note that scaling changed this table. Under the earlier single-sample run on the
test split alone, "rewriting reduces the calibration gap" was *not* resolved and
therefore was not claimed. Scaling is not only precision; it changes which
statements the report is entitled to make.

## Step 6: when metrics disagree, read the generated text

`topic_jsd` scores the Claude columns (0.907) nearly as badly as the shuffle
control (0.760), which would ordinarily indicate complete topical failure. Source
similarity is 0.583 and manual inspection confirms the topic is preserved. These
findings are inconsistent.

The probable explanation is that `topic_jsd` is computed over k-means clusters fit
on *authentic* posts, so machine-generated text may occupy distinct regions of the
embedding space for reasons of origin rather than subject. The metric is
documented as unreliable for generated text rather than reported as a finding.

Re-scoring the same outputs under `google/embeddinggemma-300m` supports that
explanation. In that space the Claude columns separate clearly from the shuffle
control, which is the ordering inspection of the outputs indicates. The artefact
is a property of the TF-IDF space and not of the generated posts, so `topic_jsd`
should be read only in a space where the baselines separate.

Several of the most useful observations in this study came from reading outputs
rather than from the numbers: the repeated closing-question structure, and the
`topic_jsd` artefact above.

## Summary of findings

1. Rewriting genuinely does make individual posts read more like Reddit.
2. Rewriting makes the *set* of posts less like real Reddit. These are not in
   conflict; one is a property of a post and the other of a collection.
3. The cause is that outputs are too alike: higher self-similarity, narrower
   vocabulary, a closing question about three times as often as real posts, and
   the automated judge naming structural sameness as its cue.
4. GPT-5.6 keeps more of the original content than Claude; Claude changes the tone
   more and keeps less.
5. Showing the model real examples points in the right direction on every measure,
   though the size of the gain is not established.
6. Considerable headroom remains: the best system is far from the oracle on every
   group-level measure.

## Reproducing with stronger embeddings

These results use the offline TF-IDF backend. Replicate under a neural backend
before reporting, particularly for `topic_jsd` and the semantic metrics:

```bash
uv pip install -e ".[embeddings]"
.venv/bin/vectorial-eval --run-dir runs/scaled evaluate --split heldout \
    --embedding-model google/embeddinggemma-300m --embedding-dim 256
```

The second report is written alongside the first as
`report.heldout.embeddinggemma-300m-d256.json` rather than replacing it, because
values from two spaces are not comparable and must not be mixed in one file.

Any conclusion that does not survive both backends is an artefact of the
representation rather than a property of the transfer function. Note that TRM
values are not comparable across base distances, so the switch invalidates every
prior TRM number and the comparison must be redone in full.

This replication has been run on the scaled study across six spaces: TF-IDF and
five neural spaces (EmbeddingGemma at 128/256/512/768 and MiniLM-L6 at 384).
Of the ten comparisons the report states, seven reach the same verdict in every
space. Two are supported only under TF-IDF and unresolved in every neural space:
"Claude rewriting degrades distribution match relative to the source" and "Claude
rewriting raises self-similarity above the authentic Reddit level". Neither is a
result.

Both rest on the triangle score, and the reason they do not replicate is
diagnosable: the TF-IDF triangle score fails the harness's own baseline-separation
test (§2.2). It ranks the wrong-topic `shuffle_control` second best of five
columns, above all three rewriters, which means it is largely reading register
rather than subject. Every neural space ranks `shuffle_control` worst, where a
content-sensitive score must, with EmbeddingGemma at 256 dimensions the one
marginal case (rank 3). For this reason the neural triangle score is reported at
512 dimensions. The identity-versus-Claude gap that the withdrawn claim rests on
is +0.52 and separates under TF-IDF; in every neural space it shrinks to +0.11 to
+0.17 and its intervals overlap, and it does not recover as dimensions increase,
so the effect was specific to the space that fails the test rather than merely
under-measured. The ordering of the system columns changes on six of the eight
space-dependent rows. The full comparison is presented in
`experimental-notes/embedding-space.html`.
