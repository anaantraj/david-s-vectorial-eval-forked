# Soft prompt tuning

The first of the three trainable transfer functions. It fits a small set of continuous prompt
embeddings over a completely frozen `meta-llama/Llama-3.1-8B-Instruct`, and nothing else in the
model changes.

## Formulation

There is no paired data in this project, so the objective is not a translation loss. Following
the design recorded in `docs/09-cluster-training-config.md`, the model is trained as a
conditional language model over **real train-split Reddit posts**. Each training example is the
shared context from `build_training.TARGET_LM_PROMPT`, which names the audience room, the domain,
the topic and the target platform, followed by an authentic post as the completion. The loss is
computed on the completion tokens only; the context is masked out with `-100`, because it is
nearly identical across the examples of a cell and scoring it would measure how well the model
predicts a fixed header.

At inference the same template is rendered with its content block filled by the source LinkedIn
post. The context therefore supplies the content to be carried across and the trained prompt
supplies the manner of the target platform.

The context is wrapped as a user turn of the base model's chat template, with the post as the
assistant reply, which is the same wrapping the LoRA method uses. `tests/test_soft_prompt.py`
asserts the two wrappings are byte-identical, since a difference there would appear in the
results as a property of the adaptation method rather than of the prompt.

## One global prompt, not one per cell

A single prompt is fitted over all cells. There are 337 train examples spread across 26 cells,
a median of 7 per cell and a minimum of 2. A per-cell prompt would fit tens of thousands of free
parameters to a handful of posts, and it would leave the four zero-shot held-out cells with no
prompt at all, which is exactly the case the held-out cells exist to measure. Pooling loses
nothing that matters, because the room and the topic are already stated in text in the context
every example shares; what the continuous prompt has to encode is the manner of the target
platform, and that is the part common to all cells.

A per-room or per-cell prompt is the obvious ablation. `train.py --group-by` is reserved for it
and currently refuses to run rather than pretending to implement it.

## Hyperparameters

Fixed by `docs/09-cluster-training-config.md` and identical across all three methods, so they are
constants of the study rather than differences between its columns:

| Setting | Value |
|---|---|
| Base model | `meta-llama/Llama-3.1-8B-Instruct` |
| Quantisation | 4-bit NF4, double quant, bf16 compute |
| Batch size | 1, with gradient accumulation 8 |
| Max length | 768 tokens |
| Gradient checkpointing | on |

Swept, because these are the three that matter for prompt tuning:

| Hyperparameter | Values | Why |
|---|---|---|
| Virtual tokens | 16, 64 | Capacity. 16 tokens is 65k parameters, 64 is 262k, against 337 training examples. Below 16 there is too little room to encode a manner; above 64 the parameter count approaches one free parameter per training character. |
| Learning rate | 3e-3, 3e-2 | Prompt tuning needs a learning rate orders of magnitude above ordinary fine-tuning, because the only gradient signal reaches the input embeddings. These two bracket the range that is reported to work. |
| Initialisation | `text`, `sampled_vocab` | Prompt tuning is sensitive to where it starts. `text` initialises from the tokenised instruction in `INIT_TEXT`, `sampled_vocab` from randomly drawn vocabulary embeddings. |

`peft`'s own `RANDOM` initialisation is available as `--init random` but is not part of the sweep.
It draws from the default `nn.Embedding` normal, whose scale is roughly two orders of magnitude
larger than a Llama input embedding, so it starts far outside the region the frozen model was
trained on. That is a property of the initialiser rather than of soft prompting, which is why
`sampled_vocab` is the random arm instead.

Fixed by argument rather than swept: AdamW, weight decay 0, gradient clipping at 1.0, cosine
schedule with 10 per cent warmup, at most 20 epochs, and early stopping after 4 epochs without an
improvement in dev loss. With 337 examples and accumulation 8 an epoch is 43 optimiser steps, so
the whole budget is at most 860 steps.

## Selection

Dev loss is the mean negative log-likelihood per completion token on `*.dev.jsonl`, measured at
every epoch boundary. The example-weighted mean is recorded next to it because post lengths vary
by two orders of magnitude and the two do not agree. The best epoch is copied to
`<ckpt_dir>/best/`, which is what the transfer function loads, and the full curve is written to
`<ckpt_dir>/summary.json`.

The test and heldout files are never opened by the training script.

## Crash safety

The cluster GPUs are shared and jobs are killed without warning. A checkpoint is written every 50
optimiser steps and at every epoch boundary through `scripts/cluster/checkpointing.py`, and
relaunching the identical command resumes. The example order for an epoch is a pure function of
the seed and the epoch number, so a resumed run consumes exactly the examples the killed run had
not reached. Only the prompt embeddings, the optimiser state and the scheduler state are saved,
which is a few megabytes; no copy of the base model is ever written.

## Reproducing

```bash
# from the Mac
scripts/cluster/sync.sh
scripts/cluster/launch.sh -j sp_target_lm_n32_lr1e-2_text -- \
    src/vectorial_eval/methods/soft_prompt/train.py \
    --variant target_lm --n-virtual-tokens 32 --lr 1e-2 --init text
scripts/cluster/status.sh --job sp_target_lm_n32_lr1e-2_text

# the sweep and the three variants in one command
scripts/cluster/soft_prompt_sweep.sh            # --dry-run to see what it would launch
.venv/bin/python src/vectorial_eval/methods/soft_prompt/select_best.py --remote
```

`select_best.py` reads every run's `summary.json`, prints the dev-loss table, and names the best
configuration. Generation then runs on the cluster, because the base model is 8B:

```bash
ssh cthulhu1.ist.berkeley.edu
cd ~/Projects/vectorial
CUDA_VISIBLE_DEVICES=<free gpu> HF_HOME=~/.cache/huggingface \
  VECTORIAL_SOFT_PROMPT_DIR=runs/checkpoints/<best job>/best \
  ~/micromamba/envs/vectorial/bin/python -m vectorial_eval.cli transfer \
  --fn soft_prompt --split heldout --n-samples 4
```

## Known limitations

The `target_lm` variant has a structural mismatch that belongs to the data rather than to this
implementation: the content block is empty during training and populated at inference, so the
model is asked for a conditioning it never saw. `semantic.source_similarity` measures the
consequence directly. The `target_lm_paired` and `target_lm_aspect` variants supply a content
block during training as well, which is why all three are run and why the variant is recorded
with every number.
