"""Compare layer-wise semantic separation across models."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.logger import get_logger


def load_separation_scores(
    experiment_name: str,
) -> pd.DataFrame:
    """Load best-layer separation scores for one experiment.

    Args:
        experiment_name: Experiment name.

    Returns:
        DataFrame with layer-wise separation scores.
    """
    input_path = (
        Path("outputs") / experiment_name / "analysis" / "layers" / "best_layers.csv"
    )

    scores = pd.read_csv(input_path)
    scores["experiment"] = experiment_name

    return scores


def main() -> None:
    """Compare DistilBERT and Sentence-BERT layer trajectories."""
    logger = get_logger(__name__)

    experiments = [
        "distilbert",
        "sentence_bert",
    ]

    all_scores = [load_separation_scores(experiment) for experiment in experiments]

    comparison = pd.concat(
        all_scores,
        ignore_index=True,
    )

    comparison["layer_index"] = (
        comparison["layer"].str.replace("layer_", "", regex=False).astype(int)
    )

    comparison = comparison.sort_values(
        by=[
            "experiment",
            "layer_index",
        ]
    )

    output_dir = Path("outputs/comparisons")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = output_dir / "layer_separation_comparison.csv"
    figure_path = output_dir / "layer_separation_comparison.png"

    comparison.to_csv(
        csv_path,
        index=False,
    )

    plt.figure(figsize=(7, 5))

    for experiment, experiment_df in comparison.groupby("experiment"):
        plt.plot(
            experiment_df["layer_index"],
            experiment_df["separation_score"],
            marker="o",
            label=experiment,
        )

    plt.xlabel("Layer")
    plt.ylabel("Separation score")
    plt.title("Layer-wise Semantic Separation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        figure_path,
        dpi=300,
    )
    plt.close()

    logger.info("Saved comparison CSV to %s", csv_path)
    logger.info("Saved comparison figure to %s", figure_path)
    logger.info("\n%s", comparison)


if __name__ == "__main__":
    main()
