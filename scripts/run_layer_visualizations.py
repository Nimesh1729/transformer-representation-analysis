"""Generate PCA and UMAP visualizations for selected layers."""

import numpy as np

from src.analysis.pca import compute_pca
from src.analysis.umap import compute_umap
from src.utils.cli import parse_args
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.paths import (
    get_layer_embeddings_dir,
    get_visualizations_dir,
)
from src.visualization.scatter import plot_2d_embeddings


def main() -> None:
    """Generate PCA and UMAP plots for selected layers."""
    logger = get_logger(__name__)
    args = parse_args()

    config = load_config(args.config)
    experiment_name = config["experiment"]["name"]

    input_dir = get_layer_embeddings_dir(experiment_name)

    output_dir = get_visualizations_dir(experiment_name) / "layers"
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = np.load(
        input_dir / "labels.npy",
        allow_pickle=True,
    )

    selected_layers = [0, 3, 6]

    for layer_index in selected_layers:
        embeddings = np.load(input_dir / f"layer_{layer_index}.npy")

        pca_embeddings, explained_variance = compute_pca(embeddings)

        plot_2d_embeddings(
            embeddings_2d=pca_embeddings,
            labels=labels,
            title=f"{experiment_name}: Layer {layer_index} PCA",
            output_path=output_dir / f"layer_{layer_index}_pca.png",
        )

        logger.info(
            "Experiment %s | Layer %d PCA variance: %s",
            experiment_name,
            layer_index,
            explained_variance,
        )

        umap_embeddings = compute_umap(embeddings)

        plot_2d_embeddings(
            embeddings_2d=umap_embeddings,
            labels=labels,
            title=f"{experiment_name}: Layer {layer_index} UMAP",
            output_path=output_dir / f"layer_{layer_index}_umap.png",
        )

        logger.info(
            "Experiment %s | Finished layer %d",
            experiment_name,
            layer_index,
        )


if __name__ == "__main__":
    main()
