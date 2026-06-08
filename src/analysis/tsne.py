"""t-SNE utilities for representation analysis."""

import numpy as np
from sklearn.manifold import TSNE


def compute_tsne(
    embeddings: np.ndarray,
    n_components: int = 2,
    perplexity: float = 5.0,
    random_state: int = 42,
) -> np.ndarray:
    """Compute t-SNE projections.

    Args:
        embeddings: Input embeddings with shape `(num_samples, embedding_dim)`.
        n_components: Number of t-SNE components.
        perplexity: t-SNE perplexity. Must be less than number of samples.
        random_state: Random seed.

    Returns:
        t-SNE projections with shape `(num_samples, n_components)`.
    """
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )

    return tsne.fit_transform(embeddings)
