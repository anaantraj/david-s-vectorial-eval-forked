# 2. Transfer functions

A transfer function maps a source-platform post to a prediction of how the same
audience would express that content on the target platform.

## Interface

```python
class TransferFunction(ABC):
    name: str

    @abstractmethod
    def run(self, tasks: list[TransferTask], corpus: Corpus) -> list[TransferOutput]:
        ...

    def describe(self) -> dict:
        """Metadata recorded alongside results for reproducibility."""
```

Three properties of this interface warrant explanation.

**The entire batch is supplied at once.** Batched implementations — an LLM invoked
with concurrency, or a GPU-resident adapter — require the full batch, and a
per-task interface would compel them to simulate one.

**A `Corpus` object is supplied rather than the raw frame.** Transfer functions
must resolve `exemplar_ids` into text. Mediating this through a read-only lookup
prevents inadvertent access outside the split boundaries established during
dataset construction.

**One output must be returned per input task, including on failure.** Discarding
failures would bias every distributional metric toward whichever subset the
function happened to handle successfully. Failures are recorded in
`TransferOutput.meta` and surfaced by the `degeneracy` metric.

## Implemented functions

### Baselines

These establish the attainable range. Their purpose is diagnostic: a metric that
fails to separate them is not measuring platform transfer.

| Name | Behaviour | Role |
|---|---|---|
| `identity` | Reproduces the source post unchanged | Floor on transfer; ceiling on semantic preservation |
| `target_sample` | Emits an authentic target post from the cell's train split | Practical ceiling |
| `shuffle_control` | Emits an authentic target post from a *different* cell | Negative control for style-only matching |

The tension between the first two is the harness's most diagnostic reading. A
transfer function that improves on `identity` with respect to the platform
classifier while approaching its semantic-preservation score is performing
substantive work; one that improves on the classifier solely by discarding content
will exhibit a corresponding decline in `semantic.source_similarity`.

`target_sample` draws exclusively from the train split, and therefore never emits
the specific post against which it is evaluated. Were it permitted to draw from the
evaluation split, the ceiling would be spuriously perfect.

`shuffle_control` isolates topic from register. Any metric that scores it comparably
to `target_sample` is style-indexed and blind to semantics — precisely the failure
mode identified as the project's principal risk.

### LLM rewriters

| Name | Conditioning |
|---|---|
| `llm_rewrite` | Zero-shot: audience, topic, and target platform only |
| `llm_fewshot` | Additionally conditioned on `k` authentic target posts from the cell's train split |

The difference between these two isolates the contribution of observing the actual
target distribution, as distinct from the model's generic prior regarding platform
norms.

On held-out cells `llm_fewshot` receives no exemplars and therefore degrades to
zero-shot behaviour. This is intentional and renders the zero-shot generalisation
penalty measurable.

Both are surface rewriters. The project has stated explicitly that reducing transfer
to surface rewriting would understate the research framing, and that the intended
direction is trait-mediated. These functions exist to constitute the baseline that a
trait-mediated function must exceed, which is why the prompt makes no attempt at
trait extraction.

## Adding a transfer function

```python
from vectorial_eval.transfer.base import Corpus, TransferFunction, register
from vectorial_eval.data.schema import TransferOutput, TransferTask


@register("steering_vector")
class SteeringVectorTransfer(TransferFunction):
    name = "steering_vector"

    def __init__(self, alpha: float = 1.0, name: str | None = None, **_):
        self.alpha = alpha
        if name:
            self.name = name

    def run(self, tasks: list[TransferTask], corpus: Corpus) -> list[TransferOutput]:
        outputs = []
        for task in tasks:
            # Exemplars are guaranteed to be train-split target posts.
            exemplars = corpus.texts(task.exemplar_ids)
            generated = my_model.generate(task.source_text, exemplars, self.alpha)
            outputs.append(
                TransferOutput(
                    task_id=task.task_id,
                    cell_id=task.cell_id,
                    transfer_fn=self.name,
                    output_text=generated,
                    meta={"alpha": self.alpha},
                )
            )
        return outputs
```

Importing the module registers the function; add the import to
`transfer/__init__.py`. It is then available as `--fn steering_vector` and is scored
by the identical metric suite on the identical splits.

## Multi-candidate sampling

Each LLM transfer function can draw several candidates per source post via
`--n-samples k` (default 1). This is the primary way to enlarge the evaluation
without collecting more data.

The group-level metrics compare a *pool* of generated posts against a *pool* of
real posts, so `k` draws per source post multiply the candidate pool by `k`. It is
what the metric's originating work prescribes, and it is the only way to observe
how much a model varies when asked the same question twice.

```bash
.venv/bin/vectorial-eval transfer --fn llm_rewrite --split heldout --n-samples 4
```

Sampling is at `temperature = 1.0` (`LLMConfig.temperature`). Each draw is emitted
as its own `TransferOutput` carrying `meta.sample_index`, so 126 tasks at `k = 4`
yield 504 outputs.

**The cache key must include the draw index.** `LLMClient._cache_key` takes a
`variant` argument for exactly this reason. Without it every draw after the first
is served the identical cached completion and the candidate pool becomes `k`
copies of one post — silently, with no error. Preserve this in any change to the
caching layer.

This choice is not cosmetic. With a single draw per post the vocabulary-variety
metric could not distinguish the rewriters from the source posts, because the
repetition occurs *between* draws on different inputs rather than within any one
output. At `k = 4` it separates them clearly and becomes a supported finding. A
single-sample run understates the failure mode the harness exists to detect.

## Running

```bash
# Baselines
.venv/bin/vectorial-eval transfer --fn identity target_sample shuffle_control

# LLM rewriters, with a tag distinguishing the model
.venv/bin/vectorial-eval transfer --fn llm_rewrite llm_fewshot \
    --split heldout --n-samples 4 --model claude-opus --tag claude --n-shot 4

# Head-to-head against a second model family
.venv/bin/vectorial-eval transfer --fn llm_rewrite --split heldout \
    --n-samples 4 --model gpt-5.6 --tag gpt

# Inexpensive smoke run over a deterministic subsample
.venv/bin/vectorial-eval transfer --fn llm_rewrite --limit 12
```

`--tag` appends a suffix to the output label so that the same function may be
evaluated under several models without output files colliding.

Model aliases resolve through `MODELS` in `config.py`:

```bash
.venv/bin/vectorial-eval models            # list aliases
.venv/bin/vectorial-eval models --refresh  # query OpenRouter for current releases
```

## Cost control

All LLM calls are cached on disk under `<run-dir>/llm_cache`, keyed on the complete
request: provider, model, temperature, max tokens, both prompts, and the draw
index. Generation is the expensive stage whereas metrics are the stage subject to
iteration, so caching permits a metric defect to be corrected and results
re-scored at no cost. Disable with `--no-cache`.

A stale cache serves old outputs silently. Clear the directory if a prompt or
model changes in a way the key does not capture.

Generation is concurrent, bounded by `LLMConfig.max_concurrency`. Independent
evaluation passes can also run simultaneously by pointing a second run directory
at the same outputs through symlinks, since the two write different report files.
