# LoRA fine-tuning of the conditional target language model

This directory fits the LoRA arm of the three-method comparison. The fitting
script is `train.py` and the transfer function that generates from what it
produces is `vectorial_eval/transfer/lora.py`, registered as `lora`.

## The formulation

The corpus contains no paired posts. No LinkedIn post has a corresponding Reddit
post, which is why the harness compares pools rather than pairs, and it is also
why this method is not trained on source and target pairs. Constructing pairs
with a language model would make the study measure agreement with that model's
opinion of the transfer rather than agreement with the real target platform.

The objective is instead a language-model loss over authentic train-split Reddit
posts, conditioned on the shared context that
`vectorial_eval.data.build_training` emits and that all three trainable methods
use. The loss is taken on the completion tokens only; the context tokens are
masked with `-100`, because a loss over the context would train the adapter to
reproduce the template. The adapter therefore carries the manner in which a
given occupational audience writes about a given subject on the target platform.
At inference the same context is rendered with the source LinkedIn post in its
content block, and the adapted model writes the post: the weights supply the
manner, the prompt supplies the content to be carried across.

The context is rendered by `render_prompt`, imported rather than reimplemented,
and then wrapped as a user turn by `prompting.wrap_prompt`. Both training and
inference call `wrap_prompt`, so the two cannot drift apart. The base model is
an instruction-tuned checkpoint, so placing the context in a user turn and the
authentic post in the assistant turn works with its instruction tuning rather
than against it.

The inference context is identical for all three training variants. Only the
training context differs. That is what makes the variant a clean axis: any
difference between the three columns is a difference in what the adapter was
fitted on, not in what it was asked.

## The three variants

`vectorial-eval build-training` emits three conditionings of the same authentic
target text, and every method is run against all three (docs/09).

| `--variant` | Training context | Train / dev |
|---|---|---|
| `target_lm` | room, domain, topic, platform | 337 / 69 |
| `target_lm_paired` | the above plus a retrieved same-cell source post | 337 / 69 |
| `target_lm_aspect` | the above plus the aspects the post foregrounds | 186 / 39 |

A number reported without its variant is not interpretable, so the variant is
read back from `training_config.json` inside the adapter directory and recorded
in `describe()` and on every generated output.

## Hyperparameters and why

**Quantisation, batch, and length are fixed by docs/09 and are not swept.**
4-bit NF4 with double quantisation, batch size 1, sequence length 768, gradient
accumulation 8, gradient checkpointing on. The measured peak for this
configuration is 19.1 GB of 23.5, against 22.6 GB for bf16, and these cards are
shared: a run that peaks at 22.6 GB dies the moment an owner's process takes half
a gigabyte. The observed peak in the sweep was 8.9 GB, well inside the measured
ceiling. All three methods use the same quantisation, so it is a constant of the
study rather than a difference between its columns.

**The sweep is three points on the capacity axis.**

| `--config` | rank | alpha | dropout | modules | trainable |
|---|---|---|---|---|---|
| `attn-r4` | 4 | 8 | 0.10 | q, k, v, o | 3,407,872 |
| `attn-r8` | 8 | 16 | 0.05 | q, k, v, o | 6,815,744 |
| `all-r16` | 16 | 32 | 0.05 | q, k, v, o, gate, up, down | 41,943,040 |

With 337 training examples the question is not which of a dozen settings is best
but whether added capacity helps at all, and a wide grid over a set this small
would select on dev noise rather than on signal. The three points span an order
of magnitude of trainable parameters and cross the attention-only to
attention-plus-MLP boundary, which is the choice that actually changes what the
adapter can represent. Alpha is held at twice the rank throughout so that the
effective scale does not move with the rank, and dropout is raised at the
smallest rank only because that configuration trains longest before it overfits.
The learning rate is 1e-4 with 10 steps of warmup and a cosine decay, which is
the standard setting for LoRA at this scale; it was not tuned, because tuning it
against 69 dev examples would be fitting the dev set.

**Selection is by dev loss and nothing else.** The dev loss of the untrained base
model is measured before the first optimiser step and recorded as epoch 0.
Without it there is no way to distinguish an adapter that helps from one that
merely stopped getting worse. The dev loss is then measured after every epoch,
the adapter from the best epoch is kept in `best/`, and training stops after
three epochs without improvement or twelve epochs, whichever comes first. Dev
loss is the token-weighted mean negative log likelihood over completion tokens,
so long and short posts contribute in proportion to their length.

Dev loss selects a configuration within a variant. It does not rank the variants
against one another. `target_lm` and `target_lm_paired` score the same 69 dev
posts under different context, so their losses are on the same targets and may be
read together, but `target_lm_aspect` covers only the 39 dev posts for which
aspect vectors exist and its loss is over a different set of tokens. The variant
comparison that matters is the one the harness makes on generated text, not this
one.

**If no epoch beats the base model, no adapter is written and the log says so in
capitals.** At 337 examples that is a plausible outcome, and it is a finding
about the data regime rather than a failure of the implementation. The transfer
function refuses to substitute the base model for a missing adapter, because a
number produced that way would be reported as a LoRA result while measuring
something else.

**Cells are pooled, not weighted.** Sixteen of the 26 training cells hold fewer
than ten examples and the smallest holds two. Weighting toward the thin cells
would multiply their noise, and holding out per-cell dev sets is impossible at
this size, so every example counts once and the conditioning is left to do the
work. The four zero-shot cells contribute no training signal by construction and
their loss is reported separately wherever they appear.

## Interruption

The cards are shared and jobs are killed without warning. Checkpoints are
written through `scripts/cluster/checkpointing.py` every 100 optimiser steps and
at every epoch boundary, and each one holds the adapter, the optimiser and
scheduler state, and the dev-loss history. Relaunching the identical command
resumes from the last complete checkpoint: the epoch is recovered from the step
count, the shuffle is reproduced from `seed + epoch`, and the completed steps of
the interrupted epoch are skipped. Only the last two checkpoints are kept.

Adapters only are saved, never a merged model. The shared disk is at 91 percent
and a merged 8B copy is 16 GB. The transfer function loads the base model and
applies the adapter at run time for the same reason.

## Reproducing

```bash
# Once, if the training files are not present.
.venv/bin/vectorial-eval build-training

scripts/cluster/sync.sh

# One job per (variant, config). Nine in total; each is roughly half an hour.
for v in target_lm target_lm_paired target_lm_aspect; do
  for c in attn-r4 attn-r8 all-r16; do
    scripts/cluster/launch.sh -j "lora-$v-$c" -- \
      src/vectorial_eval/methods/lora/train.py \
      --variant "$v" --config "$c" --epochs 12 --patience 3 --save-every 100
  done
done

scripts/cluster/status.sh --job lora
```

Relaunching any of those commands after a kill resumes it. The dev-loss curve of
a finished or running job is in
`runs/checkpoints/lora-<variant>-<config>/metrics.json` and the selected adapter
is in `.../best`.

Generation runs on the cluster as well, because the base model is 8B:

```bash
ssh cthulhu1.ist.berkeley.edu
cd ~/Projects/vectorial
export VECTORIAL_LORA_ADAPTER=$PWD/runs/checkpoints/lora-target_lm_paired-attn-r8/best
HF_HOME=$HOME/.cache/huggingface PYTHONPATH=$PWD/src \
  ~/micromamba/envs/vectorial/bin/python -m vectorial_eval.cli \
  --run-dir runs/lora transfer --fn lora --split heldout --tag paired-attn-r8
```

The adapter directory is supplied through `VECTORIAL_LORA_ADAPTER` because the
harness passes no method-specific arguments to a non-LLM transfer function.
`VECTORIAL_LORA_N_SAMPLES` sets the number of candidates drawn per task;
`VECTORIAL_LORA_BATCH`, `VECTORIAL_LORA_MAX_NEW_TOKENS`,
`VECTORIAL_LORA_TEMPERATURE`, and `VECTORIAL_LORA_TOP_P` are the remaining
overrides. Each draw is seeded from the task identifier and the draw index, so
`k` candidates are `k` genuinely different posts rather than `k` copies of one.

Local tests that do not need a GPU: `.venv/bin/python -m pytest tests/test_lora.py -q`.
