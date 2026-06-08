"""PCA utilities for representation analysis."""

import numpy as np
from sklearn.decomposition import PCA


def compute_pca(
    embeddings: np.ndarray,
    n_components: int = 2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute PCA projections.

    Args:
        embeddings: Input embeddings with shape `(num_samples, embedding_dim)`.
        n_components: Number of PCA components.
        random_state: Random seed.

    Returns:
        A tuple containing:
            - PCA projections with shape `(num_samples, n_components)`.
            - Explained variance ratio with shape `(n_components,)`.
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    projected_embeddings = pca.fit_transform(embeddings)

    return projected_embeddings, pca.explained_variance_ratio_
