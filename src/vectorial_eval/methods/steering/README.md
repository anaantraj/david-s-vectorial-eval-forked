# Activation steering

The no-gradient arm of the adaptation comparison. Nothing here is trained. The
artifact is a set of per-layer difference-in-means vectors read off the base
model's residual stream, and the whole fit is one forward pass per train-split
post.

## Formulation

For a post `x`, let `h_i(x)` be the residual stream at hidden-state index `i`,
averaged over the tokens of the post body. The tokens of the prompt are excluded
from the average, so what is measured is how the model represents the post text
rather than how it represents the instruction. The steering vector at layer `i`
is the difference of the two platform means over authentic train-split posts:

```
v_i = mean_{x in Reddit_train} h_i(x)  -  mean_{x in LinkedIn_train} h_i(x)
```

At inference the source LinkedIn post is placed in the shared prompt and

```
delta_i = alpha * (v_i / ||v_i||) * act_norm_i
```

is added to the output of the decoder block that produces hidden state `i`, for
every token position, throughout generation. `act_norm_i` is the mean residual
norm at layer `i` measured during the fit, which makes `alpha` a fraction of the
typical activation size at that layer and therefore comparable across layers. A
raw `v_i` would not be: its norm varies by an order of magnitude with depth, and
one alpha would mean something different at every layer.

Hidden-state indexing follows Hugging Face. Index 0 is the embedding output and
is not a valid steering site; index `i` is the output of decoder layer `i - 1`.

## Why this method is expected to help here

The harness's TRM diagnostics on the existing LLM rewriters report `rank_i2`
between 0.068 and 0.129 against 0.261 for `target_sample`, with `rank_i0`
between 0.68 and 0.73. Candidates that were collapsed inside the reference cloud
would show a different pattern. This one says the candidates sit outside the
cloud, which is an error of location rather than of spread. Adding a constant to
the residual stream is a translation, and translation is the intervention that
acts on location directly. The dev sweep is therefore scored on location, and
the artifact records the size of the shift it induces relative to the activation
norm at each layer, so the claim is checkable rather than asserted.

## Scopes

Two vectors are fitted in the same pass.

- **Global**: every train-split Reddit post (337) against every train-split
  LinkedIn post (293), over 26 bilateral cells.
- **Per cell**: the same contrast restricted to one cell, written only for cells
  with at least `--min-cell-posts` (default 3) posts on both platforms. Cells
  below the threshold are listed in the artifact under `counts.cells_too_thin`
  and fall back to the global vector at inference, and `describe()` reports both
  the per-cell post counts and the number of cells that have a vector at all.
  Per-cell mass in this corpus is thin, so the per-cell scope should be read as
  an ablation, not as the headline configuration.

## Hyperparameters and how they were chosen

| Parameter | Value | Why |
|---|---|---|
| Base model | `meta-llama/Llama-3.1-8B-Instruct` | Same base as the soft prompt and the LoRA. A base-model difference would confound the comparison. |
| Quantisation | 4-bit NF4, double quant, bf16 compute | `docs/09-cluster-training-config.md`. Steering would fit in bf16, but 4-bit is used for the headline number so quantisation is a constant of the study. `--bf16` runs the ablation. |
| Max length (fit) | 768 | Covers the 99th percentile of train Reddit post length. |
| Pooling | mean over completion tokens | The prompt is shared across platforms except for its platform line; pooling over the body keeps the contrast on the text. |
| Layer | swept, `8,12,16,20,24` | Chosen on the validation split. Early layers carry token identity and late layers carry next-token decisions; the middle is where a platform-level property is usually linearly available, and the sweep spans it. |
| Alpha | swept, `0.5,1.0,2.0` | A fraction of the residual norm at the steered layer. Above roughly 2 the model is expected to break, which the degeneracy column of the sweep shows directly. |
| Selection rule | maximise `centroid_cos` subject to `degeneracy <= 0.15` | Location is the quantity the method is supposed to move, and the constraint is what stops the sweep selecting a broken model that happens to land near the reference centre. |

The sweep also records `source_cos`, the similarity between a generation and the
source post it was given, so that a gain in location bought by discarding the
content is visible rather than hidden, and it estimates a ceiling for
`centroid_cos` by splitting the authentic dev pool in half and measuring the two
halves against each other.

The sweep numbers select hyperparameters. They are computed in their own
embedding space and are not the harness's metrics; they must not be quoted
alongside them.

## What the fit and the sweep found

The fit reads 630 train-split posts in about 60 seconds on one 4090. The whole
method costs roughly a minute of GPU time per variant, which is why it was the
first of the three to produce numbers.

| Variant | Reddit posts | LinkedIn posts | Cells with own vector |
|---|---|---|---|
| `target_lm` | 337 | 293 | 23 of 26 |
| `target_lm_paired` | 334 | 293 | 22 of 26 |
| `target_lm_aspect` | 186 | 116 | 14 of 17 |

Three posts are skipped in the paired variant because the retrieved source post
fills the 768-token window and leaves no completion token to pool over. They are
counted in `counts.skipped` rather than dropped silently.

The platform difference is a small direction. Its norm is 0.13 to 0.18 of the
mean residual-stream norm at the same layer, rising slightly with depth. That
number is what makes the alpha scale interpretable: an alpha of 0.5 moves the
residual stream about three times as far as the distance between the two
platform means.

The first sweep, over layers 8 to 24 and alphas 0.25 to 2.0, selected layer 24
and alpha 0.5 for all three variants:

| Configuration | `centroid_cos` | `energy_dist` | `source_cos` | degeneracy |
|---|---|---|---|---|
| Unsteered | 0.817 | 0.086 | 0.778 | 0.00 |
| `target_lm`, layer 24, alpha 0.5 | 0.835 | 0.082 | 0.767 | 0.00 |
| `target_lm_paired`, layer 24, alpha 0.5 | 0.839 | 0.080 | 0.781 | 0.00 |
| `target_lm_aspect`, layer 24, alpha 0.5 | 0.839 | 0.084 | 0.758 | 0.13 |
| Authentic posts against each other | 0.942 | −0.006 | — | — |

Two readings follow. The unsteered model sits at 0.817 against a ceiling of
0.942, so the generated pool is genuinely displaced from the authentic pool
rather than sitting inside it, which is what the TRM diagnostics said. Steering
closes part of that gap and does so without paying for it in content: at the
selected setting the similarity to the source post is unchanged within the noise
of sixteen dev tasks. Larger alphas move the centre further at first and then
away from it, and they cost content heavily. At layer 12 and alpha 1.0 the
similarity to the source falls from 0.778 to 0.432 while `centroid_cos` drops to
0.740, and at alpha 2.0 the outputs are degenerate. The useful range of this
intervention is therefore narrow.

Because layer 24 sat at the edge of that grid, a second sweep over layers 24 to
30, alphas 0.35 to 0.75, and both scopes is running under the job names
`steer_sweep2_*`. It writes `sweep2.steering.<variant>.json` and refreshes
`chosen.steering.<variant>.json`, and its grid contains the first sweep's
selection, so the refreshed choice is a comparison made within one run rather
than across two. Its partial results already show layer 26 at alpha 0.5 to 0.75
reaching a higher `centroid_cos` than layer 24 at a larger cost in
`source_cos`, which is the same trade the first sweep showed.

## Variants

`docs/09` requires every method to be run against all three conditionings. All
three are supported by `--variant`:

- `target_lm`: room, domain, topic, platform.
- `target_lm_paired`: the above plus the most similar same-cell post from the
  other platform. The retrieval is run in both directions here, because the
  contrast needs a context for the LinkedIn side as well, and it is restricted
  to train-split posts on both sides.
- `target_lm_aspect`: the above plus the aspects the post foregrounds. Coverage
  is 302 posts over 17 cells rather than 630 over 26.

## Leakage

`fit_steering.py` reads `data/posts.train.jsonl` only, asserts that every record
carries `split == "train"`, and refuses to run otherwise. `sweep_steering.py`
reads the validation split only and asserts the same. Neither opens the test or
held-out files.

## Reproducing

Fit, on the cluster:

```bash
scripts/cluster/sync.sh
scripts/cluster/launch.sh -j steer_fit_target_lm -- \
    src/vectorial_eval/methods/steering/fit_steering.py --variant target_lm
```

Sweep the layer and alpha on the validation split:

```bash
scripts/cluster/launch.sh -j steer_sweep_target_lm -- \
    src/vectorial_eval/methods/steering/sweep_steering.py \
    --artifact runs/steering/steering.target_lm.pt --limit 16
```

Both write to `runs/steering/`. The sweep writes
`chosen.steering.<variant>.json`, which the transfer function reads for its
default layer, alpha, and scope; passing `--layer`/`--alpha` overrides it.

Score it with the harness, on a machine with the artifact and a GPU. The
`transfer` command passes no method arguments, so the variant is selected
through the environment, as it is for the soft prompt:

```bash
.venv/bin/vectorial-eval transfer --fn steering --split heldout --n-samples 4

VECTORIAL_STEERING_VARIANT=target_lm_paired \
  .venv/bin/vectorial-eval transfer --fn steering --split heldout \
  --n-samples 4 --tag paired
# or point at a file directly: VECTORIAL_STEERING_ARTIFACT=runs/steering/steering.target_lm_aspect.pt
```

The variant recorded in the artifact overrides the label, so an artifact cannot
be reported under a conditioning it was not fitted under. If no
`chosen.steering.<variant>.json` sits beside the artifact, the function warns and
falls back to a layer and an alpha that were never selected on dev; do not report
a number produced in that state.

Both scripts checkpoint through `scripts/cluster/checkpointing.py`: the fit every
25 posts, the sweep after every grid point. Relaunching the identical command
resumes.


## The scale arm

Every number this directory produces is an 8B number, because that is what one
23.5 GB card holds. `ndif_backend.py` runs the identical construction on
NDIF-hosted `Llama-3.1-70B-Instruct` and `Llama-3.1-405B-Instruct`, whose residual
streams are exposed for read and write. A vector belongs to the stream it was read
from, so the scale arm fits its own with `fit_steering_ndif.py` and generates
through the `steering_ndif` transfer function.

Remote columns are **not** comparable with local ones — NDIF serves its own
precision rather than the study's 4-bit NF4 constant — only with each other and
with the local steering column as a scale contrast. See
[docs/11-ndif-steering.md](../../../../docs/11-ndif-steering.md).
