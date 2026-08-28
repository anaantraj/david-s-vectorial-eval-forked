# 9. Validated cluster training configuration

Measured on `cthulhu1` against a live RTX 4090 before any training was launched. These are not
estimates. Any method that departs from them must say so and give its own measured peak memory.

## Environment

| Item | Value |
|---|---|
| Nodes | `cthulhu1..6.ist.berkeley.edu`, 6 × RTX 4090 (23.5 GB usable) each |
| Filesystem | `/home/davidchan` is NFS-shared across all six nodes; sync once, run anywhere |
| Python | `~/micromamba/envs/vectorial/bin/python` |
| Versions | torch 2.11.0+cu128, transformers 5.14.1, peft 0.20.0, trl 1.9.2, bitsandbytes 0.50.0 |
| HF cache | `~/.cache/huggingface`, 350 GB, already holds every model named below |
| Disk | 91% used, 199 GB free. Save adapters and vectors only, never a merged model |

`transformers` is on the version 5 line. Both `peft` LoRA and `peft` prompt tuning were confirmed
to run forward and backward under it, so the major version bump is not a blocker, but any code
written against version 4 idioms should be checked rather than assumed.

## Base model

`meta-llama/Llama-3.1-8B-Instruct`, fully cached (16 GB, 5 shards, no partial downloads).

All three trainable methods use this same base. A difference in base model between methods would
confound the comparison the table exists to make.

## Measured memory

Batch size is bound by the logits tensor, not by the weights. The vocabulary is 128k, so a
forward pass materialises `batch × sequence × 128256` in bf16 and again in float32 for the loss.
This dominates everything else and is why batch size 1 is required regardless of quantisation.

| Configuration | Batch | Max length | Peak | Headroom |
|---|---|---|---|---|
| bf16 + gradient checkpointing | 1 | 768 | 22.6 GB | 0.9 GB |
| bf16 + gradient checkpointing | 1 | 1024 | 22.9 GB | 0.6 GB |
| bf16 + gradient checkpointing | 2 | 768 | OOM | — |
| **4-bit NF4 + double quant** | **1** | **768** | **19.1 GB** | **4.4 GB** |
| 4-bit NF4 + double quant | 4 | 768 | OOM | — |

## The configuration to use

```
load_in_4bit          = True        # NF4, double quant, bf16 compute
per_device_batch_size = 1
max_length            = 768
gradient_accumulation = 8           # effective batch 8
gradient_checkpointing= True
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

**Use 4-bit for all three methods**, including the ones that would fit in bf16.

The reason is not memory alone. These GPUs are shared and jobs may be killed or joined by their
owners at any time. A configuration that peaks at 22.6 GB of 23.5 GB dies the moment another
process takes half a gigabyte, whereas 19.1 GB survives a co-tenant. Since quantisation is then
identical across all three methods it is a constant of the study rather than a difference between
its columns.

Steering is inference-only and would fit in bf16 comfortably. Run it in 4-bit for the headline
number so it matches the other two, and run the bf16 version as a cheap ablation to demonstrate
that quantisation is not driving the result.

## Sequence length

Chosen from the data rather than by convention. Train-split Reddit posts:

| Percentile | Characters | Approximate tokens |
|---|---|---|
| 50 | 236 | 59 |
| 75 | 955 | 238 |
| 90 | 1,368 | 342 |
| 95 | 1,886 | 471 |
| 99 | 2,718 | 679 |
| 100 | 7,181 | 1,795 |

A maximum length of 768 covers the 99th percentile of completions with room for the prompt. The
single longest post is truncated, which is acceptable; raising the limit to hold it would cost
batch size across every step for one example.

## The three training variants

`vectorial-eval build-training` emits three conditionings of the same authentic target text. The
completion is always a real train-split Reddit post; only the context differs. All three share
`TARGET_LM_PROMPT`, so a method can be trained on each without changing anything but the file.

| Variant | Context supplied | Train / dev / test | Cells |
|---|---|---|---|
| `target_lm` | room, topic, platform | 337 / 69 / 89 | 26 |
| `target_lm_paired` | the above plus a retrieved same-cell source post | 337 / 69 / 89 | 26 |
| `target_lm_aspect` | the above plus the aspects the post foregrounds | 186 / 39 / 51 | 17 |

**`target_lm` has a structural mismatch.** Nothing fills the content block during training, but
the transfer function fills it at inference. A model trained this way has never seen the block it
is asked to condition on, and the likely consequence is that it ignores the source post, which
`semantic.source_similarity` measures directly.

**`target_lm_paired` removes the mismatch but cannot supply real content correspondence.** Each
target is paired with the most similar same-cell source post by TF-IDF retrieval. Measured over
all 337 training pairs, the retrieved pair averages 0.137 cosine against 0.079 for a random
same-cell pair, the best pair in the corpus reaches 0.343, and 0.6 per cent clear 0.3. Retrieval
is therefore recovering topical neighbours, not content matches, because the corpus contains no
content matches to recover. Training on it teaches the model the shape of the inference context
rather than how to preserve content.

**`target_lm_aspect` drops content transfer as the objective.** Instead of asking the model to
carry specific content across, it supplies the evaluative aspects the target post foregrounds and
asks for a post that emphasises them. The context and the completion are genuinely coupled here,
which is not true of the other two, and it aligns with the finding that within a shared topic the
two platforms emphasise different aspects. The cost is coverage: aspect vectors exist for 56 per
cent of train posts and 17 of 26 cells, and the vocabularies are provisional until the blind
rerun lands.

**Run all three for every method.** Nine runs total. Each is roughly a thousand optimiser steps
on a few hundred examples, so the whole grid is minutes of GPU time spread over 36 idle cards,
and the variant axis is the more interesting scientific result: it measures how much of the task
is content transfer and how much is target-distribution generation. Report the variant alongside
the method in every table; a number without its variant is not interpretable.

## Scale

**There are 337 train-split Reddit posts.** That is the entire trainable set for the target-side
conditional language model.

This number should be stated in the paper and in any discussion of the trainable baselines. It is
small enough that LoRA overfitting is the expected outcome rather than a surprise, and a null
result for LoRA at this scale is a finding about the data regime rather than a failure of the
implementation. Dev loss should be watched from the first epoch and early stopping used.
