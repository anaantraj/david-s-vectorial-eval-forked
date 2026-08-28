# Vectorial AI × BAIR Cross-Domain Eval
_Project state, direction, and dataset context for downstream analysis_

## Overview

This project studies **cross-platform transfer of audience behavior**, primarily between **LinkedIn and Reddit**, with the goal of learning a **transfer function** that maps how the same underlying audience or subpopulation expresses itself differently across platforms.

The practical framing is:

- observe an audience on one platform,
- understand how topic, tone, evaluative lens, and behavioral style shift across platforms,
- and learn a transformation that can generate or predict target-platform behavior.

This is **not** just a style-transfer problem. The team repeatedly distinguished:

- **population shift**: different kinds of people show up on different platforms, and
- **behavioral / stylistic shift**: the same or similar people express themselves differently depending on platform norms.

The current project focus is on **occupational audiences** such as software engineers, AI engineers, academic researchers, and edtech-related roles, rather than broad consumer populations.

---

## Current state of the project

### 1. The dataset has moved from early topic extraction to a cluster-based structure

The project began with a more free-form topic extraction setup, but the team found that this led to severe fragmentation and poor usable coverage.

Earlier state:

- ~22,000 posts produced ~17,000 unique labels in one pass, making most of the corpus unusable.
- A later normalization pass reduced this to 4,865 unique labels, with singletons dropping dramatically.

Current state:

- the team has largely shifted from free-form topic extraction to **hierarchical clustering with HDBSCAN** to create more usable topic groupings with meaningful coverage on both platforms.
- this produced **198 topics** in one discussed version of the dataset, but only **~6 topics / bilateral cells** had enough mass on both LinkedIn and Reddit to begin transfer-function work.
- by the July 16 meeting, the team referred to **196 usable clusters currently**, with cluster counts in the low 230s depending on settings.

So the project is no longer in “raw data gathering only” mode. It is in a **structured-but-still-being-validated dataset phase**.

---

### 2. The dataset is large enough to prototype, but not yet final

The corpus discussed across meetings includes:

- **22,954 total posts**,
- **14,316 LinkedIn posts**,
- **8,368 Reddit posts**.

There are:

- **24 audience rooms total**,
- **12 bilateral rooms** with both platforms,
- **12 single-platform rooms**, with targeted scraping underway to fill gaps.

The healthiest cited audience room was **backend engineer**, with roughly **3,000 LinkedIn posts** and **1,500 Reddit posts**.

Important caveat: the team explicitly said the current CSV is usable for **structure and prototyping**, but should not yet be treated as the final optimized dataset for strongest experimental results.

---

### 3. The core unit of analysis is the bilateral cell

A recurring concept is the **bilateral cell**: same audience, same topic, split across LinkedIn and Reddit.

The working schema includes fields such as:

- audience
- platform
- domain
- post text
- URL
- post type
- stimulus type
- final topic

The team described reconstructing cells by grouping on:

- `audience_room + final_topic`,
- then splitting by platform.

Other topic fields like `narrow_topic` and `original_topic` are more granular than `final_topic`, which is the current clustering output.

For any system analyzing the dataset, this bilateral-cell framing is the most important organizing principle.

---

## What the project has established so far

### Platform differences are real and strong

The team repeatedly observed that LinkedIn and Reddit are behaviorally distinct even for similar domains and audiences.

Examples captured in meetings:

- LinkedIn is more strategic, positive, business-case oriented, and self-promotional.
- Reddit is more execution-focused, critical, question-heavy, and detail-oriented.
- LinkedIn generates promotional content at **19×** the rate of Reddit, while Reddit produces far more question-based content.

This matters because it supports the basic premise that there is a meaningful transfer problem to solve, rather than the two platforms just being interchangeable samples of the same discourse.

---

### The transfer function is the main scientific object

The team has discussed multiple implementation options for the transfer function:

- prompt engineering,
- soft prompts,
- LoRA / adapters,
- task vectors,
- steering vectors,
- and other parameter-efficient or test-time steering methods.

The broader idea is to transform source-platform observations into target-platform behavior while preserving relevant audience identity.

By late June / July, the conversation had become more specific: Vectorial’s internal framing converts raw user history into **latent traits**, then feeds those traits into an LLM, rather than simply dumping raw context. They reported this outperformed raw-context baselines on an Amazon review prediction benchmark.

This suggests the project is trending toward a **trait-mediated transfer function**, not merely direct rewriting.

---

### Evaluation remains a major open area

Evaluation has been discussed in several forms:

- post similarity,
- interaction patterns,
- survey response prediction,
- semantic feature comparison,
- stylistic feature comparison,
- lexical feature comparison,
- distribution-level metrics such as KLD / JSD over post pools,
- and external benchmarks such as Stack Overflow survey prediction.

A major constraint is that there is **no one-to-one post matching** in the core setup, so evaluation has to be **distributional**, not pairwise.

There is also an interpretability requirement from the client / product side: users will want to know what distinguishes LinkedIn from Reddit audiences. The team discussed:
- **causal interpretability** through explicit feature bottlenecks, versus
- **post-hoc interpretability** through LLM-generated explanations.

This is unresolved and is a real downstream design constraint for any modeling system.

---

## Current methodological direction

### 1. Clustering first, transfer second

The team has increasingly prioritized validating the cluster structure before treating topics as fixed.

Recent choices:

- moved from Jaccard to **ARI (Adjusted Rand Index)** for cluster stability,
- because Jaccard inflated stability by conditioning on matches,
- while ARI penalizes random assignments and wrong merges.

Reported results:

- **ARI ~0.80 at 90% subsample**, treated as reasonably stable,
- lower stability at 70% and 50% subsamples,
- the stability curve is improving but has **not yet plateaued**,
- more data infusion is expected to help.

Other cluster observations:

- UMAP `n_components=5` was chosen as a practical stable point, though 5–15 looked directionally similar.
- seed sensitivity looked mostly flat.
- 89% of cluster jumps happened at >90% similarity, while problematic reassignments were estimated at **3.7%**.

The project is therefore in a phase where **clustering is considered usable, but still under active calibration**.

---

### 2. Disentangling semantics from style is now central

A major concern is that embeddings may cluster LinkedIn and Reddit posts by **style**, not by **topic**.

This is one of the most important current risks in the project:

- if clusters are style-indexed, transfer learning becomes much less meaningful,
- if clusters are too narrow or trivial, transfer may also become uninteresting.

By July 16, the team’s proposed solution was to explicitly build a **style embedding** and factor stylistic similarity out from total similarity, treating the remainder as semantic signal.

The target “healthy cluster” is one with:

- a good LinkedIn / Reddit mix,
- stylistic differences,
- and aspect differences.

This is a key instruction for any downstream analysis system: **do not assume cluster purity means semantic purity**. Style-semantic disentanglement is an explicit project concern.

---

### 3. Aspect extraction is becoming a second layer over topic clusters

The team increasingly distinguishes between:

- **stimuli**: the object/event/topic being discussed,
- **aspects**: the evaluative lens or dimension through which it is discussed.

Examples:
- a model release might be a stimulus,
- latency, cost, pedagogy, salary growth, or tool effectiveness might be aspects.

The newer aspect extraction method:

- feeds all posts in a cluster into one context window,
- extracts evaluative lenses,
- and constrains them to be atomic, cross-cutting, latent, and not capped to a fixed count.

The ideal outcome is **different aspect skew within the same topic cluster across platforms**, since this provides a stronger signal for transfer-function learning.

This means the likely medium-term representation is:

1. audience
2. topic / cluster
3. platform
4. aspect distribution within cluster

---

## Practical context for a system analyzing the dataset

### Treat the dataset as experimental, not canonical

The dataset is not yet a fixed benchmark. It is still being refined through:

- better clustering,
- more data enrichment,
- embedding comparisons,
- stability analysis,
- and aspect extraction improvements.

Any analysis system should therefore:
- preserve uncertainty,
- avoid overcommitting to cluster labels as ground truth,
- and keep intermediate diagnostics.

---

### Use `final_topic` as the current primary topic field

The discussed structure suggests:

- `final_topic` is the cluster-derived field to use for current experiments,
- `narrow_topic` and `original_topic` are more granular / legacy or auxiliary views.

If reconstructing bilateral cells, group by audience + `final_topic` first.

---

### Ignore some legacy labels in early passes

The team explicitly said some sentiment / opinion labels in the dataset are artifacts of earlier extraction passes and should be ignored in the first iteration.

Likewise, some quality-control labels were carried over from a prior B2B schema and were not yet integrated into the current research workflow.

---

### Expect skewed coverage and sparse bilateral cells

Not every audience-topic combination has healthy mass on both platforms.

This means:
- some cells are strong candidates for modeling,
- some are sparse and useful only for qualitative inspection,
- and some are candidates for future generalization tests rather than direct training.

The team explicitly discussed using dense, stable clusters first and deferring weaker ones.

---

### Platform differences are not just superficial style

The project hypothesis is that LinkedIn vs Reddit differences include:

- tone,
- promotion vs critique,
- question frequency,
- evaluative focus,
- and trait activation under different social settings.

A downstream system should not reduce the transfer problem to surface rewriting alone. That would undershoot the research framing.

---

## Open questions and active directions

### Transfer function design
Still open:
- prompt-only,
- adapter-based,
- steering-based,
- or latent-trait-based transfer.

### Embedding selection
Still open:
- which embedding family best preserves topic while minimizing stylistic confounding,
- whether IR-oriented embeddings outperform more general-purpose ones.

### Stimulus vs aspect formalization
Still open:
- exact operational boundaries between stimulus and aspect,
- how aspect extraction should be constrained,
- and how aspect skew should be used in downstream training.

### Evaluation
Still open:
- best distributional metrics,
- survey-style external benchmarks,
- interpretability mode,
- and how to assess transfer quality in sparse bilateral settings.

### Data growth
Still active:
- expanding single-platform gaps,
- improving topic coverage,
- increasing cluster stability,
- and enriching dense cells for stronger experiments.

---

## Bottom line

The project is currently in a **post-ingestion, pre-final-model** phase.

What is stable enough to rely on:
- the high-level goal,
- the bilateral-cell framing,
- the existence of strong platform differences,
- the need for distribution-level rather than pairwise evaluation,
- and the importance of disentangling semantics, style, and aspect emphasis.

What is still moving:
- cluster calibration,
- embedding choice,
- aspect extraction,
- transfer-function implementation,
- and the final evaluation stack.

A good downstream system should therefore treat the dataset as:
- structured,
- promising,
- partially validated,
- and explicitly designed for iterative research rather than fixed supervised training.
