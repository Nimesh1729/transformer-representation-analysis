# Research Journal — Transformer Representation Analysis

## Project Motivation

The goal of this repository is to understand how pretrained transformer models organize astronomy-related concepts in representation space.

Before studying multimodal encoders, vision-language models, SDSS data, this repository focuses on a controlled text-only question:

```text
How do transformer representations evolve across layers, and how does training objective affect embedding geometry?
```

The project builds directly on earlier representation-learning work involving:

* MNIST classifiers
* hidden-layer embeddings
* PCA
* t-SNE
* UMAP
* CKA
* CNN representation analysis
* autoencoders
* VAEs
* ViTs
* Tiny CLIP-style contrastive learning

The central idea is to apply the same representation-analysis mindset to pretrained transformer models.

---

## Project Standards

This repository follows research-grade development standards:

* Google Python Style Guide
* type hints where practical
* Ruff linting (wherever suitable)
* mypy static type checking (wherever suitable)
* pytest tests where reasonable
* config-driven experiments
* reproducible outputs
* centralized path utilities
* experiment-specific output directories
* timestamped pipeline logs
* continuous README and journal updates

The goal is to maintain the repository at a level suitable for serious research development, not just quick experimentation.

---

## Dataset

The current dataset is a small handcrafted astronomy sentence dataset.

Classes:

* galaxy
* quasar
* star

The dataset is intentionally small and interpretable. Its purpose is not to train a model, but to create a controlled setting for representation analysis.

### Dataset Rationale

The dataset allows us to ask whether pretrained language models naturally separate astronomy-related concepts in embedding space.

Example concepts include:

* galaxies
* quasars
* stars
* spectra
* hydrogen emission
* redshift
* stellar properties

### Limitations

The dataset is currently limited by:

* small sample size
* manually written examples
* limited vocabulary diversity
* no real astronomy corpus
* no SDSS metadata
* no paper abstracts or scientific descriptions yet

Future versions should include richer astronomy text from real sources.

---

## Models Studied

Two models were compared:

### DistilBERT

Model:

```text
distilbert-base-uncased
```

Training objective:

```text
masked language modeling / BERT-style pretraining
```

DistilBERT is a general-purpose language model. It is not specifically trained to produce high-quality sentence embeddings.

### Sentence-BERT

Model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Training objective:

```text
sentence-level semantic similarity / embedding alignment
```

Sentence-BERT is trained to produce useful sentence embeddings for similarity, retrieval, and semantic comparison.

---

## Methods

The repository analyzes representations using:

* CLS embeddings
* mean-pooled embeddings
* PCA
* t-SNE
* UMAP
* cosine similarity
* class similarity matrices
* separation scores
* layer-wise embeddings
* layer-wise cosine similarity
* layer-wise CKA
* layer-wise PCA / UMAP
* attention heatmaps
* attention-head summaries
* target-token attention analysis

The main separation metric is:

```text
separation score = within-class similarity - between-class similarity
```

Higher separation score means the model places examples from the same astronomy class closer together while pushing examples from different classes farther apart.

---

## Milestone 1 — Repository Infrastructure

### Goal

I try to establish a professional, reproducible foundation before model analysis, which is build upon my earlier projects.

### Completed

* repository structure
* virtual environment
* `.gitignore`
* `pyproject.toml`
* editable package installation
* configuration loader
* reproducibility utilities
* logger utility
* pytest infrastructure
* Ruff and mypy setup

### Observation

Starting with infrastructure I learnt from earlier projects prevented early technical debt and made later refactoring easier.

---

## Milestone 2 — DistilBERT Loading and Hidden-State Extraction

### Goal

Load a pretrained transformer model from Hugging Face and inspect hidden states.

### Experiment

Input sentence:

```text
The galaxy contains strong hydrogen emission lines.
```

DistilBERT produced hidden states with shape:

```text
(1, 10, 768)
```

Meaning:

```text
1 sentence
10 tokens
768 features per token
```

Tokenization produced:

```text
['[CLS]', 'the', 'galaxy', 'contains', 'strong', 'hydrogen', 'emission', 'lines', '.', '[SEP]']
```

### Result

DistilBERT returned 7 hidden-state tensors:

```text
Layer 0: embedding output
Layers 1–6: transformer layers
```

### Interpretation

DistilBERT preserves the same hidden dimension across layers while changing the content of the representation.

This established the basic representation-extraction pipeline.

---

## Milestone 3 — CLS Embeddings and Mean-Pooled Embeddings

### Initial Approach

The first sentence embedding was extracted from the final-layer CLS token:

```text
hidden_states[-1][:, 0, :]
```

This produced embeddings of shape:

```text
(15, 768)
```

for the astronomy sentence dataset.

### Observation

CLS embeddings produced very high cosine similarities across both same-class and different-class pairs.

This suggested that DistilBERT's CLS token is not ideal as a sentence-level semantic embedding.

### Mean Pooling

Mean pooling was added as an alternative.

The mean-pooling function averages token embeddings while ignoring padding tokens using the attention mask.

### Result

Mean pooling reduced similarity compression and produced more useful sentence-level geometry.

### Interpretation

For this repository, mean-pooled embeddings are preferred for the main model comparison.

---

## Milestone 4 — PCA, t-SNE, and UMAP

### Goal

Visualize astronomy sentence embeddings in two dimensions.

### Methods

Applied:

* PCA
* t-SNE
* UMAP

to final-layer mean-pooled embeddings.

### DistilBERT Observation

DistilBERT showed partial semantic structure, but the classes overlapped substantially.

Galaxy and quasar examples showed some separation, while star examples were more scattered.

### Sentence-BERT Observation

Sentence-BERT showed substantially clearer structure.

Quasar sentences occupied a more distinct region of embedding space, and the overall geometry better reflected semantic class differences.

### Interpretation

Visual analysis supports the quantitative similarity results.

Sentence-BERT produces more useful sentence-level embeddings for astronomy concepts than DistilBERT.

---

## Milestone 5 — Similarity Analysis

### Goal

Measure whether same-class sentences are closer than different-class sentences.

### DistilBERT Mean Similarity Results

```text
galaxy-galaxy: 0.885
quasar-quasar: 0.918
star-star: 0.866
galaxy-quasar: 0.868
galaxy-star: 0.858
quasar-star: 0.872
```

DistilBERT produced high absolute similarities for nearly all class pairs.

### DistilBERT Separation Score

```text
within-class similarity ≈ 0.890
between-class similarity ≈ 0.866
separation score ≈ 0.024
```

### Sentence-BERT Mean Similarity Results

```text
galaxy-galaxy: 0.515
quasar-quasar: 0.668
star-star: 0.485
galaxy-quasar: 0.408
galaxy-star: 0.406
quasar-star: 0.333
```

Sentence-BERT produced lower absolute similarities but much larger differences between related and unrelated classes.

### Sentence-BERT Separation Score

```text
within-class similarity ≈ 0.556
between-class similarity ≈ 0.383
separation score ≈ 0.174
```

### Key Result

Sentence-BERT achieved approximately:

```text
0.174 / 0.024 ≈ 7.4×
```

stronger semantic separation than DistilBERT.

### Interpretation

DistilBERT compresses astronomy sentences into a dense semantic region.

Sentence-BERT creates a more discriminative embedding space, consistent with its sentence-similarity training objective.

---

## Milestone 6 — Layer-Wise Embedding Extraction

### Goal

Study how representations evolve across transformer layers.

Embeddings were extracted from every layer:

```text
layer_0.npy
layer_1.npy
layer_2.npy
layer_3.npy
layer_4.npy
layer_5.npy
layer_6.npy
```

Each layer embedding file had shape:

```text
(15, 768)
```

### Rationale

Final-layer analysis alone is incomplete. To understand a transformer, we need to inspect representation evolution across depth.

---

## Milestone 7 — Layer-Wise Cosine Similarity

### Goal

Measure how much sentence embeddings move across layers.

### DistilBERT Observation

DistilBERT showed substantial movement across depth.

Example:

```text
Layer 0 ↔ Layer 6 cosine similarity ≈ 0.236
Layer 5 ↔ Layer 6 cosine similarity ≈ 0.678
```

### Interpretation

Individual sentence vectors change significantly as they pass through the network.

---

## Milestone 8 — Layer-Wise CKA

### Goal

Measure whether the overall representation geometry is preserved across layers.

CKA asks a different question from cosine similarity.

Cosine similarity asks:

```text
Does the same sentence vector move?
```

CKA asks:

```text
Does the whole dataset keep a similar relational geometry?
```

### DistilBERT CKA

Layer 0 to Layer 6:

```text
CKA ≈ 0.821
```

### Sentence-BERT CKA

Layer 0 to Layer 6:

```text
CKA ≈ 0.946
```

### Unexpected Result

Sentence-BERT had higher cross-layer CKA than DistilBERT.

This was unexpected because Sentence-BERT also had much stronger final-layer semantic separation.

### Interpretation

Sentence-BERT does not appear to improve semantic separation by completely reorganizing the representation space.

Instead, it appears to preserve the overall geometry while sharpening class-relevant semantic structure.

This is one of the most important findings of the repository.

---

## Milestone 9 — Layer-Wise Class Separation

### Goal

Determine which transformer layer contains the strongest semantic separation.

### DistilBERT Layer Separation

```text
Layer 0: 0.086
Layer 1: 0.045
Layer 2: 0.031
Layer 3: 0.015
Layer 4: 0.009
Layer 5: 0.014
Layer 6: 0.024
```

### Best DistilBERT Layer

```text
Layer 0
```

### Sentence-BERT Layer Separation

```text
Layer 0: 0.103
Layer 1: 0.048
Layer 2: 0.040
Layer 3: 0.045
Layer 4: 0.048
Layer 5: 0.049
Layer 6: 0.174
```

### Best Sentence-BERT Layer

```text
Layer 6
```

### Interpretation

DistilBERT has its strongest semantic separation at the embedding layer.

Sentence-BERT has its strongest semantic separation at the final layer.

This reflects the different training objectives:

```text
DistilBERT: masked-language modeling
Sentence-BERT: sentence-level semantic similarity
```

---

## Milestone 10 — Cross-Model Layer Trajectory Comparison

### Goal

Compare the layer-wise semantic separation curves of DistilBERT and Sentence-BERT.

### Observation

DistilBERT trajectory:

```text
high at layer 0
decreases through middle layers
remains weak at final layer
```

Sentence-BERT trajectory:

```text
strong at layer 0
drops in middle layers
large jump at final layer
```

### Interpretation

The final Sentence-BERT layer is qualitatively important.

The separation curve suggests that sentence-similarity fine-tuning strongly shapes the final representation.

However, CKA shows that this final-layer improvement does not come from a complete reorganization of the representation space.

Instead, the final layer appears to sharpen existing semantic directions.

---

## Milestone 11 — Layer-Wise PCA and UMAP

### Goal

Visualize representation evolution at selected layers.

Selected layers:

```text
Layer 0
Layer 3
Layer 6
```

### Observation

DistilBERT:

* Layer 0 showed the clearest class separation in full-space metrics.
* Later layers became smoother but less discriminative.
* PCA and UMAP showed partial structure but substantial overlap.

Sentence-BERT:

* Final-layer visualizations showed clearer semantic organization.
* Quasar sentences became especially well-separated.
* Visual results agreed with quantitative separation analysis.

### Interpretation

Low-dimensional plots and full-space metrics must be interpreted together.

PCA/UMAP can reveal visual structure, but separation scores measure discriminability in the full embedding space.

---

## Milestone 12 — Attention Analysis

### Goal

Move beyond hidden states and inspect attention behavior.

### Attention Shape

For a sample sentence, DistilBERT attention tensors had shape:

```text
(1, 12, 12, 12)
```

Meaning:

```text
1 sentence
12 attention heads
12 query tokens
12 key tokens
```

DistilBERT produced 6 attention tensors, one for each transformer layer.

### Initial Attention Heatmaps

Early-layer attention appeared more distributed.

Later-layer attention often focused on punctuation and boundary tokens such as:

```text
.
[SEP]
```

### Interpretation

Some attention heads appear to act as boundary or aggregation heads.

---

## Milestone 13 — Attention Head Discovery

### Goal

Identify attention heads that focus on astronomy-relevant tokens.

Target tokens included:

```text
quasar token pieces
spectrum
hydrogen
emission
lines
redshift token pieces
```

### Observation

Some heads assigned elevated attention to astronomy-relevant tokens such as:

* spectrum
* hydrogen
* emission
* quasar subword pieces

Other heads focused strongly on punctuation.

### Interpretation

Attention heads exhibit different functional roles.

Some heads appear content-sensitive, while others focus on sentence boundaries or punctuation.

This is an initial step toward mechanistic interpretability.

---

## Milestone 14 — Experiment-Aware Pipeline Refactor

### Motivation

The project grew from simple scripts into a multi-stage representation-analysis pipeline.

Running scripts manually became risky because it was easy to forget the correct order or overwrite outputs.

### Work Completed

Added:

* experiment configs
* centralized CLI handling
* centralized path utilities
* experiment-specific outputs
* full pipeline runner
* timestamped logs

### Pipeline Command

Example:

```bash
python scripts/run_full_pipeline.py --config configs/experiments/distilbert.yaml --embedding-type mean
```

### Output Organization

Outputs are now separated by experiment:

```text
outputs/
├── distilbert/
└── sentence_bert/
```

Logs are stored separately:

```text
logs/
├── distilbert/
└── sentence_bert/
```

### Interpretation

This refactor turned the repository from a collection of scripts into a reproducible research pipeline.

---

## Core Findings

### Finding 1 — Sentence-BERT Separates Astronomy Concepts Better

Sentence-BERT achieved approximately 7.4× stronger semantic separation than DistilBERT.

```text
DistilBERT separation:    0.024
Sentence-BERT separation: 0.174
```

### Finding 2 — Best Semantic Layer Depends on Training Objective

```text
DistilBERT best layer:    layer 0
Sentence-BERT best layer: layer 6
```

### Finding 3 — Sentence-BERT Maintains Higher Cross-Layer CKA

```text
DistilBERT layer 0 ↔ layer 6 CKA:    0.821
Sentence-BERT layer 0 ↔ layer 6 CKA: 0.946
```

### Finding 4 — Sentence-BERT Sharpens Existing Structure

The combination of high CKA and improved separation suggests that Sentence-BERT fine-tuning sharpens existing semantic structure rather than completely reorganizing the representation space.

### Finding 5 — DistilBERT Drifts Without Improving Separation

DistilBERT representations change across layers, but this does not improve astronomy concept separation.

---

## Main Interpretation

The current working interpretation is:

```text
DistilBERT representations drift across layers while semantic separation worsens.

Sentence-BERT representations remain geometrically stable while semantic separation improves.

Sentence-level semantic fine-tuning appears to sharpen existing structure rather than reorganize the embedding space.
```

This is the central result of the repository so far.

---

## Unexpected Findings

### 1. DistilBERT's Best Layer Was Layer 0

The initial expectation was that deeper layers would be more semantic.

Instead, DistilBERT showed the strongest class separation at the embedding layer.

Possible explanation:

The class labels are strongly tied to lexical identity:

```text
galaxy
quasar
star
```

Layer 0 already captures these lexical differences.

Later transformer layers mix contextual information and reduce explicit class separation.

### 2. Sentence-BERT's CKA Was Higher Than DistilBERT's

The initial expectation was that Sentence-BERT's final layer might reorganize the representation space dramatically.

Instead, CKA remained extremely high across layers.

Possible explanation:

Sentence-BERT fine-tuning sharpens the existing geometry rather than replacing it.

### 3. Visual Separation and Full-Space Separation Are Not Identical

Some PCA plots appeared visually cleaner even when full-space separation scores were lower.

This reinforces the importance of combining visualization with quantitative metrics.

---

## Current Limitations

The current study is limited by:

* small handcrafted dataset
* only three astronomy categories
* no real scientific text corpus
* no SDSS metadata
* only two models compared
* limited attention analysis
* no statistical uncertainty estimates
* no repeated dataset variants

The current findings are useful but preliminary.

---

## Next Steps

### Short-Term

* generate CKA heatmaps
* polish layer trajectory plots
* improve README formatting and add timestamps
* improve journal formatting and add time stamps
* add a short final report
* add more tests for path utilities and pipeline behavior
* convince users to read the journal

### Medium-Term

* expand astronomy sentence dataset
* compare additional models:

  * BERT
  * RoBERTa
  * MiniLM
* analyze real astronomy abstracts or metadata
* compare scientific vs non-scientific sentence embeddings
* improve attention-head analysis

### Long-Term

To use this repository as the bridge into:

```text
LLaVA / Open VLM Study
↓
SDSS Data Analysis
↓
Mini Astronomy model Prototype
↓
Reasoning Audit
```

---

The next roadmap step is to move from:

```text
text embeddings
```

to:

```text
image + text embeddings
```

through an open VLM / LLaVA-style study.

---

## Current Status

The repository currently contains:

* working Hugging Face model loading
* DistilBERT and Sentence-BERT configs
* CLS and mean embedding extraction
* PCA / t-SNE / UMAP visualization
* similarity analysis
* layer-wise analysis
* CKA analysis
* attention analysis
* experiment-aware outputs
* full reproducible pipeline
* timestamped logs
* documented key findings

The project has reached a stable first research milestone, till now.
