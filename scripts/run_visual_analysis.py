"""Run PCA, t-SNE, and UMAP visualization for saved embeddings."""

import numpy as np

from src.analysis.pca import compute_pca
from src.analysis.similarity import cosine_similarity_matrix
from src.analysis.tsne import compute_tsne
from src.analysis.umap import compute_umap
from src.utils.cli import parse_args
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.paths import get_embeddings_dir, get_visualizations_dir
from src.visualization.scatter import plot_2d_embeddings


def main() -> None:
    """Run visual analysis on saved CLS embeddings."""
    logger = get_logger(__name__)

    args = parse_args()

    config = load_config(args.config)

    embedding_type = args.embedding_type

    experiment_name = config["experiment"]["name"]
    embeddings_dir = get_embeddings_dir(experiment_name)

    visualizations_dir = get_visualizations_dir(experiment_name) / "embeddings"
    visualizations_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = embeddings_dir / f"{embedding_type}_embeddings.npy"

    logger.info("Experiment: %s", experiment_name)
    logger.info("Embedding type: %s", embedding_type)
    logger.info("Loading embeddings from %s", embeddings_path)
    logger.info("Saving visualizations to %s", visualizations_dir)

    labels_path = embeddings_dir / "labels.npy"

    embeddings = np.load(embeddings_path)
    labels = np.load(labels_path, allow_pickle=True)

    output_dir = get_visualizations_dir(experiment_name) / "embeddings"

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    similarity = cosine_similarity_matrix(
        embeddings,
    )

    logger.info(
        "Average similarity: %.4f",
        similarity.mean(),
    )

    plot_title_prefix = f"{experiment_name} {embedding_type.upper()} embeddings"

    pca_embeddings, explained_variance = compute_pca(embeddings)

    plot_2d_embeddings(
        embeddings_2d=pca_embeddings,
        labels=labels,
        title=f"PCA of {plot_title_prefix}",
        output_path=visualizations_dir / f"{embedding_type}_pca.png",
    )

    logger.info(
        "PCA explained variance ratio: %s",
        explained_variance,
    )

    tsne_embeddings = compute_tsne(embeddings)

    plot_2d_embeddings(
        embeddings_2d=tsne_embeddings,
        labels=labels,
        title=f"t-SNE of {plot_title_prefix}",
        output_path=visualizations_dir / f"{embedding_type}_tsne.png",
    )

    umap_embeddings = compute_umap(embeddings)

    plot_2d_embeddings(
        embeddings_2d=umap_embeddings,
        labels=labels,
        title=f"UMAP of {plot_title_prefix}",
        output_path=visualizations_dir / f"{embedding_type}_umap.png",
    )

    logger.info("Saved visualizations to %s", output_dir)


if __name__ == "__main__":
    main()
