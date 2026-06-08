"""Run class-similarity analysis for every transformer layer."""

import numpy as np
import pandas as pd

from src.analysis.similarity import cosine_similarity_matrix
from src.utils.cli import parse_args
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.paths import get_analysis_dir, get_layer_embeddings_dir


def compute_class_similarity_summary(
    similarity_matrix: np.ndarray,
    labels: np.ndarray,
) -> pd.DataFrame:
    """Compute average similarity between class pairs.

    Args:
        similarity_matrix: Pairwise similarity matrix.
        labels: Class labels.

    Returns:
        DataFrame containing class-pair similarities.
    """
    unique_labels = sorted(set(labels.tolist()))

    rows = []

    for label_a in unique_labels:
        for label_b in unique_labels:
            mask_a = labels == label_a
            mask_b = labels == label_b

            similarities = similarity_matrix[np.ix_(mask_a, mask_b)]

            if label_a == label_b:
                similarities = similarities[
                    ~np.eye(
                        similarities.shape[0],
                        dtype=bool,
                    )
                ]

            rows.append(
                {
                    "label_a": label_a,
                    "label_b": label_b,
                    "mean_similarity": float(similarities.mean()),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    """Run class similarity analysis for all layers."""
    logger = get_logger(__name__)

    args = parse_args()
    config = load_config(args.config)
    experiment_name = config["experiment"]["name"]

    input_dir = get_layer_embeddings_dir(experiment_name)

    output_dir = get_analysis_dir(experiment_name) / "layers"
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = np.load(
        input_dir / "labels.npy",
        allow_pickle=True,
    )

    layer_paths = sorted(input_dir.glob("layer_*.npy"))

    all_results = []

    for layer_path in layer_paths:
        layer_name = layer_path.stem

        embeddings = np.load(layer_path)

        similarity_matrix = cosine_similarity_matrix(embeddings)

        similarity_df = compute_class_similarity_summary(
            similarity_matrix=similarity_matrix,
            labels=labels,
        )

        similarity_df["layer"] = layer_name

        all_results.append(similarity_df)

        logger.info(
            "Processed %s",
            layer_name,
        )

    results_df = pd.concat(
        all_results,
        ignore_index=True,
    )

    output_path = output_dir / "layer_class_similarity.csv"

    results_df.to_csv(
        output_path,
        index=False,
    )

    logger.info(
        "Saved results to %s",
        output_path,
    )

    logger.info(
        "\n%s",
        results_df,
    )


if __name__ == "__main__":
    main()
