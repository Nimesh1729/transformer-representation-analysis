"""Run layer-wise cosine similarity analysis."""

import numpy as np
import pandas as pd

from src.analysis.layer_similarity import compute_layer_similarity_matrix
from src.utils.cli import parse_args
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.paths import get_analysis_dir, get_layer_embeddings_dir


def main() -> None:
    """Compute cosine similarity between transformer layers."""
    logger = get_logger(__name__)

    args = parse_args()
    config = load_config(args.config)
    experiment_name = config["experiment"]["name"]

    input_dir = get_layer_embeddings_dir(experiment_name)
    output_dir = get_analysis_dir(experiment_name) / "layers"
    output_dir.mkdir(parents=True, exist_ok=True)

    layer_paths = sorted(input_dir.glob("layer_*.npy"))
    layer_embeddings = [np.load(path) for path in layer_paths]

    similarity_matrix = compute_layer_similarity_matrix(layer_embeddings)

    layer_names = [path.stem for path in layer_paths]

    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=layer_names,
        columns=layer_names,
    )

    output_path = output_dir / "layer_cosine_similarity.csv"
    similarity_df.to_csv(output_path)

    logger.info("Saved layer cosine similarity matrix to %s", output_path)
    logger.info("\n%s", similarity_df)


if __name__ == "__main__":
    main()
