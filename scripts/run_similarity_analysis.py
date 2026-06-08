"""Run cosine similarity analysis on saved embeddings."""

import numpy as np
import pandas as pd

from src.analysis.similarity import cosine_similarity_matrix
from src.utils.cli import parse_args
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.paths import get_analysis_dir, get_embeddings_dir


def compute_class_similarity_summary(
    similarity_matrix: np.ndarray,
    labels: np.ndarray,
) -> pd.DataFrame:
    """Compute average within-class and between-class similarities.

    Args:
        similarity_matrix: Pairwise similarity matrix with shape
            (num_samples, num_samples).
        labels: Class labels with shape (num_samples,).

    Returns:
        DataFrame containing average similarity for each label pair.
    """
    unique_labels = sorted(set(labels.tolist()))
    rows = []

    for label_a in unique_labels:
        for label_b in unique_labels:
            mask_a = labels == label_a
            mask_b = labels == label_b

            similarities = similarity_matrix[np.ix_(mask_a, mask_b)]

            if label_a == label_b:
                similarities = similarities[~np.eye(similarities.shape[0], dtype=bool)]

            rows.append(
                {
                    "label_a": label_a,
                    "label_b": label_b,
                    "mean_similarity": float(similarities.mean()),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    """Run similarity analysis on saved embeddings."""
    logger = get_logger(__name__)
    args = parse_args()

    config = load_config(args.config)
    experiment_name = config["experiment"]["name"]
    embedding_type = args.embedding_type

    embeddings_dir = get_embeddings_dir(experiment_name)
    output_dir = get_analysis_dir(experiment_name) / "similarity"
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = embeddings_dir / f"{embedding_type}_embeddings.npy"
    labels_path = embeddings_dir / "labels.npy"

    embeddings = np.load(embeddings_path)
    labels = np.load(labels_path, allow_pickle=True)

    similarity_matrix = cosine_similarity_matrix(embeddings)
    similarity_summary = compute_class_similarity_summary(
        similarity_matrix=similarity_matrix,
        labels=labels,
    )

    matrix_path = output_dir / f"{embedding_type}_cosine_similarity_matrix.npy"
    summary_path = output_dir / f"{embedding_type}_class_similarity_summary.csv"

    np.save(matrix_path, similarity_matrix)
    similarity_summary.to_csv(summary_path, index=False)

    logger.info("Experiment: %s", experiment_name)
    logger.info("Embedding type: %s", embedding_type)
    logger.info("Similarity matrix shape: %s", similarity_matrix.shape)
    logger.info("Saved similarity matrix to %s", matrix_path)
    logger.info("Saved similarity summary to %s", summary_path)
    logger.info("\n%s", similarity_summary)


if __name__ == "__main__":
    main()
