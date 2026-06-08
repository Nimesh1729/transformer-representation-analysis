# Transformer Representation Analysis

This is what I think as an amateur is a research-grade representation-analysis project studying how pretrained transformer models encode astronomy-related concepts. This project contains a refactored and professional level of code which I had written previously.

My goal is not to train a classifier, but to understand how transformer embedding spaces behave across models, layers, and training objectives.

## Project Goal

This repository investigates the question:

```text
How do pretrained transformer representations organize astronomy concepts?
```

Specifically, it studies whether sentences about:

* galaxies
* quasars
* stars

occupy meaningful regions of embedding space, and how this structure changes across transformer layers.

The project compares:

* `distilbert-base-uncased`
* `sentence-transformers/all-MiniLM-L6-v2`

using representation-analysis tools such as:

* CLS embeddings
* mean-pooled embeddings
* PCA
* t-SNE
* UMAP
* cosine similarity
* class-separation scores
* layer-wise similarity
* Centered Kernel Alignment (CKA)
* attention-head analysis

## Motivation

This project extends earlier work on MNIST representation analysis, CKA, PCA, t-SNE, UMAP, CNNs, autoencoders, VAEs, ViTs, and Tiny CLIP-style alignment.

```text
text representations
        ↓
vision-language representations
        ↓
astronomy data representations
        ↓
multimodal alignment
        ↓
Mini Astronomy model
```

Before studying multimodal systems, this repository focuses on understanding transformer representations in a controlled text-only setting.

## Key Findings

### 1. Sentence-BERT Produces Better Astronomy Sentence Embeddings

Using mean-pooled embeddings, Sentence-BERT produced substantially stronger semantic separation between astronomy concepts than DistilBERT.

| Model         | Separation Score |
| ------------- | ---------------: |
| DistilBERT    |            0.024 |
| Sentence-BERT |            0.174 |

Sentence-BERT achieved approximately **7.4× stronger semantic separation** than DistilBERT.

The separation score is defined as:

```text
within-class similarity - between-class similarity
```

### 2. Best Semantic Layer Depends on Training Objective

| Model         | Best Layer | Separation Score |
| ------------- | ---------: | ---------------: |
| DistilBERT    |    Layer 0 |            0.086 |
| Sentence-BERT |    Layer 6 |            0.174 |

DistilBERT reaches its highest class separation at the embedding layer.

Sentence-BERT reaches its highest class separation at the final layer.

This suggests that masked-language-model pretraining and sentence-similarity fine-tuning produce different layer-wise representation trajectories.

### 3. Sentence-BERT Preserves Representation Geometry

Layer 0 to Layer 6 CKA:

| Model         | Layer 0 ↔ Layer 6 CKA |
| ------------- | --------------------: |
| DistilBERT    |                 0.821 |
| Sentence-BERT |                 0.946 |

This was an unexpected result.

Sentence-BERT improves semantic separation while maintaining very high cross-layer CKA. This suggests that sentence-similarity fine-tuning sharpens existing semantic structure rather than completely reorganizing the representation space.

### 4. DistilBERT Drifts While Separation Worsens

DistilBERT layer-wise separation decreases with depth:

```text
Layer 0: 0.086
Layer 1: 0.045
Layer 2: 0.031
Layer 3: 0.015
Layer 4: 0.009
Layer 5: 0.014
Layer 6: 0.024
```

Interpretation:

DistilBERT representations change across layers, but the change does not improve astronomy concept separation.

### 5. Sentence-BERT Final Layer Sharpening

Sentence-BERT layer-wise separation:

```text
Layer 0: 0.103
Layer 1: 0.048
Layer 2: 0.040
Layer 3: 0.045
Layer 4: 0.048
Layer 5: 0.049
Layer 6: 0.174
```

Interpretation:

The final layer is strongly shaped by sentence-level similarity fine-tuning.

## Repository Structure
Thanks to chatgpt for generating the structures in this document, and anywhere else, credit where it is due.
```text
transformer_representations/
│
├── configs/
│   ├── base.yaml
│   └── experiments/
│       ├── distilbert.yaml
│       └── sentence_bert.yaml
│
├── data/
│   └── astronomy_sentences.csv
│
├── scripts/
│   ├── run_full_pipeline.py
│   ├── run_visual_analysis.py
│   ├── run_similarity_analysis.py
│   ├── extract_layer_embeddings.py
│   ├── run_layer_similarity.py
│   ├── run_layer_cka.py
│   ├── run_layer_class_similarity.py
│   ├── analyze_best_layer.py
│   ├── run_layer_visualizations.py
│   ├── inspect_attention.py
│   ├── visualize_attention.py
│   ├── summarize_attention_heads.py
│   ├── find_token_heads.py
│   └── compare_layer_trajectories.py
│
├── src/
│   ├── analysis/
│   │   ├── cka.py
│   │   ├── layer_similarity.py
│   │   ├── pca.py
│   │   ├── similarity.py
│   │   ├── tsne.py
│   │   └── umap.py
│   │
│   ├── extraction/
│   │   ├── attention_extractor.py
│   │   ├── dataset_embeddings.py
│   │   ├── embedding_extractor.py
│   │   ├── hidden_states.py
│   │   └── layer_extractor.py
│   │
│   ├── models/
│   │   └── load_model.py
│   │
│   ├── utils/
│   │   ├── cli.py
│   │   ├── config_loader.py
│   │   ├── logger.py
│   │   ├── paths.py
│   │   └── reproducibility.py
│   │
│   └── visualization/
│       ├── attention.py
│       └── scatter.py
│
├── tests/
│   ├── test_config_loader.py
│   ├── test_dataset_embeddings.py
│   ├── test_hidden_states.py
│   ├── test_reproducibility.py
│   └── test_similarity.py
│
├── logs/
├── outputs/
├── README.md
├── research_journal.md
├── requirements.txt
└── pyproject.toml
```

## Setup

Creating and activating a virtual environment is recommended.

On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install the project in editable mode:

```bash
pip install -e .
```

This allows imports such as:

```python
from src.utils.config_loader import load_config
```

to wpork cleanly across scripts and tests. The pathing mechanics would break if this isn't implemented.

## Running the Full Pipeline

The project supports experiment-aware runs.

### DistilBERT

```bash
python scripts/run_full_pipeline.py --config configs/experiments/distilbert.yaml --embedding-type mean
```

### Sentence-BERT

```bash
python scripts/run_full_pipeline.py --config configs/experiments/sentence_bert.yaml --embedding-type mean
```

The `--embedding-type` argument supports:

```text
cls
mean
```

Mean embeddings are the preferred default for the main comparison because they produced better sentence-level geometry than CLS embeddings.

## Pipeline Stages

The full pipeline runs:

1. Embedding extraction
2. PCA, t-SNE, and UMAP visualization
3. Cosine similarity analysis
4. Layer-wise embedding extraction
5. Layer-wise cosine similarity
6. Layer-wise CKA
7. Layer-wise class similarity
8. Best-layer analysis
9. Layer-wise PCA and UMAP visualization

Attention analysis is currently treated as exploratory and is run separately.

## Output Structure

Outputs are experiment-specific. Thanks to chatGPT again.

```text
outputs/
├── distilbert/
│   ├── embeddings/
│   ├── layer_embeddings/
│   ├── analysis/
│   │   ├── similarity/
│   │   └── layers/
│   ├── visualizations/
│   │   ├── embeddings/
│   │   └── layers/
│   └── attention/
│
└── sentence_bert/
    ├── embeddings/
    ├── layer_embeddings/
    ├── analysis/
    ├── visualizations/
    └── attention/
```

Logs are saved separately:

```text
logs/
├── distilbert/
└── sentence_bert/
```

Pipeline logs are timestamped and include the embedding type.

Example:

```text
logs/distilbert/2026-06-08_01-25-30_mean.log
```

## Individual Script Usage

### Extract Embeddings

```bash
python main.py --config configs/experiments/distilbert.yaml
```

### Visualize Embeddings

```bash
python scripts/run_visual_analysis.py --config configs/experiments/distilbert.yaml --embedding-type mean
```

### Similarity Analysis

```bash
python scripts/run_similarity_analysis.py --config configs/experiments/distilbert.yaml --embedding-type mean
```

### Layer Embeddings

```bash
python scripts/extract_layer_embeddings.py --config configs/experiments/distilbert.yaml
```

### Layer-wise CKA

```bash
python scripts/run_layer_cka.py --config configs/experiments/distilbert.yaml
```

### Best Layer Analysis

```bash
python scripts/analyze_best_layer.py --config configs/experiments/distilbert.yaml
```

### Layer Trajectory Comparison

```bash
python scripts/compare_layer_trajectories.py
```

## Attention Analysis

Attention analysis can be run separately.

### Inspect Attention Shapes

```bash
python scripts/inspect_attention.py --config configs/experiments/distilbert.yaml
```

### Visualize Attention Heatmaps

```bash
python scripts/visualize_attention.py --config configs/experiments/distilbert.yaml
```

### Summarize Attention Heads

```bash
python scripts/summarize_attention_heads.py --config configs/experiments/distilbert.yaml
```

### Find Heads Attending to Target Tokens

```bash
python scripts/find_token_heads.py --config configs/experiments/distilbert.yaml
```

Attention analysis revealed that some heads focus heavily on punctuation or boundary tokens, while others assign elevated attention to astronomy-relevant tokens such as:

* spectrum
* hydrogen
* emission
* quasar token pieces

## Dataset

The current dataset is a small handcrafted astronomy sentence dataset with three classes:

* galaxy
* quasar
* star

The dataset is intentionally small and interpretable. Its purpose is to support representation-analysis experiments before moving to larger astronomy corpora.

Current limitations:

* small sample size
* manually written examples
* limited scientific diversity
* no real astronomy metadata yet

Future versions should include richer scientific text from real astronomy sources.

## Development Standards

This repository follows research-grade coding standards:

* Google Python Style Guide
* type hints where practical
* Ruff linting and formatting (not strict)
* mypy static type checking (not strict)
* pytest tests where reasonable
* config-driven experiments
* reproducible outputs
* experiment-specific output directories
* continuous README and research journal maintenance

Run tests:

```bash
pytest
```

Run Ruff:

```bash
ruff check .
```

Run mypy :

```bash
mypy src
```

## Research Interpretation

The central finding so far is:

```text
DistilBERT representations drift across layers while semantic separation worsens.

Sentence-BERT representations remain geometrically stable while semantic separation improves.

Sentence-similarity fine-tuning appears to sharpen existing semantic structure rather than reorganizing it.
```

## Roadmap

## Status

This repository is currently a functional transformer representation-analysis pipeline with experiment-aware outputs, reproducible logs, cross-model comparison, layer-wise analysis, and early attention inspection.
