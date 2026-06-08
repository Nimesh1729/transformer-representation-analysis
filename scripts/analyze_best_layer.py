"""Analyze which transformer layer best separates classes."""

import pandas as pd

from src.utils.cli import parse_args
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.paths import get_analysis_dir


def compute_layer_separation_scores(
    class_similarity: pd.DataFrame,
) -> pd.DataFrame:
    """Compute within-class, between-class, and separation scores.

    Args:
        class_similarity: DataFrame with columns:
            layer, label_a, label_b, mean_similarity.

    Returns:
        DataFrame with one row per layer.
    """
    rows = []

    for layer, layer_df in class_similarity.groupby("layer"):
        within_df = layer_df[layer_df["label_a"] == layer_df["label_b"]]
        between_df = layer_df[layer_df["label_a"] != layer_df["label_b"]]

        within_similarity = within_df["mean_similarity"].mean()
        between_similarity = between_df["mean_similarity"].mean()

        rows.append(
            {
                "layer": layer,
                "within_similarity": within_similarity,
                "between_similarity": between_similarity,
                "separation_score": within_similarity - between_similarity,
            }
        )

    return pd.DataFrame(rows).sort_values(
        by="separation_score",
        ascending=False,
    )


def main() -> None:
    """Analyze best semantic-separation layer."""
    logger = get_logger(__name__)
    args = parse_args()

    config = load_config(args.config)
    experiment_name = config["experiment"]["name"]

    layer_analysis_dir = get_analysis_dir(experiment_name) / "layers"

    input_path = layer_analysis_dir / "layer_class_similarity.csv"
    output_path = layer_analysis_dir / "best_layers.csv"

    class_similarity = pd.read_csv(input_path)

    separation_scores = compute_layer_separation_scores(class_similarity)

    separation_scores.to_csv(output_path, index=False)

    logger.info("Experiment: %s", experiment_name)
    logger.info("Saved separation scores to %s", output_path)
    logger.info("\n%s", separation_scores)


if __name__ == "__main__":
    main()
