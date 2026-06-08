"""UMAP utilities for representation analysis."""

import numpy as np
import umap


def compute_umap(
    embeddings: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 5,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """Compute UMAP projections.

    Args:
        embeddings: Input embeddings with shape `(num_samples, embedding_dim)`.
        n_components: Number of UMAP components.
        n_neighbors: Number of local neighbors used by UMAP.
        min_dist: Minimum distance between embedded points.
        random_state: Random seed.

    Returns:
        UMAP projections with shape `(num_samples, n_components)`.
    """
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )

    return reducer.fit_transform(embeddings)
