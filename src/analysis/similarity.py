"""Similarity analysis utilities."""

import numpy as np


def cosine_similarity_matrix(
    embeddings: np.ndarray,
) -> np.ndarray:
    """Compute pairwise cosine similarity matrix.

    Args:
        embeddings: Embedding matrix with shape
            (num_samples, embedding_dim).

    Returns:
        Cosine similarity matrix with shape
            (num_samples, num_samples).
    """
    normalized_embeddings = embeddings / np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    return normalized_embeddings @ normalized_embeddings.T
