# Remote steering on NDIF

Every steering number in Studies 2–4 is an 8B number. The ceiling is one 23.5 GB
card, and per [docs/09](09-cluster-training-config.md) it is the 128k-vocab logits
tensor rather than the weights that sets it. So "the platform contrast is a
translation in the residual stream" is currently a claim about one small model,
not about the mechanism.

[NDIF](https://ndif.us) hosts large open-weight models with the residual stream
exposed for **read and write**. That is the one thing a generation API cannot
provide, at any price: the intervention happens inside the model. Running the same
difference-in-means construction on `Llama-3.1-70B-Instruct` and
`Llama-3.1-405B-Instruct` turns the 8B claim into a scale-generalisation result.

This arm does **not** replace OpenRouter. NDIF serves open-weight models only, so
it cannot host the frontier prompting baselines (`llm_rewrite`, `llm_fewshot`,
`aspect_prompt`) or the judge; those stay where they are.

## What it is

| | local `steering` | `steering_ndif` |
|---|---|---|
| forward pass | one local GPU | NDIF worker |
| base model | `Llama-3.1-8B-Instruct` | `Llama-3.1-70B-Instruct`, `Llama-3.1-405B-Instruct` |
| quantisation | 4-bit NF4 (study constant) | NDIF's own precision — **not** the constant |
| intervention | forward hook | write inside an nnsight trace |
| fit | `fit_steering` | `fit_steering_ndif` |

`NDIFSteeredGenerator` subclasses `SteeredGenerator` and overrides only `load` and
`_generate_batch`. Vector resolution, provenance, scope grouping, the draw loop
and the per-item failure handling of invariant 6 are the *same code*. The two arms
differ in where the forward pass happens and in nothing else; if they ever differ
in more, the columns stop being comparable. `test_inherits_the_local_run_loop`
pins that.

## Setup

```bash
uv pip install -e ".[ndif]"
echo "NDIF_API_KEY=..." >> .env     # keys from https://login.ndif.us
```

Then check the wiring before committing to a long run:

```bash
.venv/bin/python -m vectorial_eval.methods.steering.smoke_ndif \
    --model meta-llama/Llama-3.1-70B-Instruct --layer 24
```

It uses a *random* vector, so a degenerate continuation under steering is the
pass condition — it shows the write landed. Meaning comes from a fitted vector.

## Running it

A steering vector belongs to the residual stream it was read from. The local
artifacts have 4096 columns; 70B's stream is 8192 wide and 405B's is 16384. The
existing artifacts cannot be applied at scale, so the scale arm fits its own:

```bash
# 1. fit against the remote model (~15 min for the full 674 posts at concurrency 8)
NDIF_API_KEY=... .venv/bin/python -m vectorial_eval.methods.steering.fit_steering_ndif \
    --model meta-llama/Llama-3.1-70B-Instruct --variant target_lm --concurrency 8

# 2. select layer and alpha on dev, exactly as the local arm does
.venv/bin/python -m vectorial_eval.methods.steering.sweep_steering \
    --artifact runs/steering/steering.target_lm.ndif-Llama-3.1-70B-Instruct.pt ...

# 3. generate
VECTORIAL_STEERING_ARTIFACT=runs/steering/steering.target_lm.ndif-Llama-3.1-70B-Instruct.pt \
VECTORIAL_NDIF_MODEL=meta-llama/Llama-3.1-70B-Instruct \
NDIF_API_KEY=... .venv/bin/vectorial-eval transfer --fn steering_ndif --split heldout
```

`SteeringTransfer`'s existing guard refuses an artifact whose `base_model` does
not match the model being run, so a 4096-column vector cannot reach a 16384-wide
stream.

## Reading the results

**A remote column is not comparable with a local column.** The study fixes 4-bit
NF4 across every method so quantisation is a constant rather than a difference
between columns. NDIF serves its own precision, so that constant does not cover
this arm. `load_in_4bit` is recorded as `None`, not `False`, to keep the two from
being pooled by accident. The remote column is comparable with the *local
steering* column as a scale contrast, and that is its purpose.

Two further caveats travel with every output:

- **Sampling is seeded on the worker.** `torch.manual_seed` is called inside the
  trace, so draws are reproducible against the same deployment — a weaker
  guarantee than the local in-process seed, and not guaranteed across a redeploy.
- **The checkpoint is NDIF's.** The deployment entry and the nnsight version are
  written into `fit_config` and into `describe()`, because without them a remote
  result cannot be attributed, let alone reproduced.

## Access, and four things that will waste an afternoon

1. **Only *pinned* models run.** A standard key cannot hotswap; anything else
   fails with *"Model is not pinned and hotswapping is not supported for this API
   key"*. As of 2026-08-26 the pinned set is `gpt-j-6b`, `gemma-2-9b-it`,
   `Llama-3.1-8B`, `Llama-3.1-70B`, `Llama-3.1-70B-Instruct`,
   `Llama-3.1-405B-Instruct`. `check_model` fails fast and names them.

2. **`Llama-3.1-8B-Instruct` — the local study's base — is hot but *not* pinned.**
   Being "hot" on the status page is not sufficient. This means the local 8B
   results cannot be replicated remotely as a correctness check; the nearest
   available comparison is base-vs-base on `Llama-3.1-8B`, and that model has no
   chat template (see 3).

3. **The pinned *base* models have no chat template.** `Llama-3.1-8B` and
   `Llama-3.1-70B` cannot be wrapped by `wrap_prompt`. Running one anyway would
   compare an unwrapped remote arm against wrapped local columns — precisely the
   confound [STUDY3_PLAN](../STUDY3_PLAN.md) item 1 was opened for. The backend
   refuses by default; `require_chat_template=False` accepts it and records
   `chat_wrapped=False`, which is not comparable with the wrapped columns.

4. **The attestation status on the NDIF profile page does not gate this.** A key
   showing *Pending* authenticates and runs pinned models, 405B included. No
   separate 405B application was needed.

## Implementation notes

Three things about nnsight 0.7 that are not obvious and cost real time:

- **The trace body must close over plain locals.** nnsight serialises what the
  body references. A reference to `self` drags the `LanguageModel`, the tokenizer
  and the artifact tensors along, and the remote failure is the unhelpful
  `'NoneType' object has no attribute '__dict__'`. Everything is hoisted before
  the `with`.

- **A decoder block's output is a bare `[B, T, H]` tensor here, not a tuple.**
  This depends on the transformers version on the *worker*, not ours. Indexing it
  as `output[0]` — the tuple idiom — silently yields batch row 0, so only the
  first prompt of each batch gets steered and the rest are recorded as steered
  outputs that were never touched. `probe_block_layout` settles the layout with
  one cheap trace at load rather than guessing;
  `test_bare_tensor_write_steers_every_batch_row` pins it.

- **Reduce inside the trace.** Downloading the raw residual stream for a fit
  would be `[L+1, T, H]` — about 1.6 GB per post on 405B, against 8 MB pooled.
  The reduction also has to happen *per layer before stacking*, because the big
  models are sharded and layers 0 and 79 sit on different CUDA devices.

- nnsight reads the source of the `with` block, so this code must live in an
  importable file. A heredoc or REPL fails with `could not get source code`.
