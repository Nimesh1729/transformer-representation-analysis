"""Compute CKA similarities between transformer layers."""

import numpy as np
import pandas as pd

from src.analysis.cka import linear_cka
from src.utils.cli import parse_args
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.paths import (
    get_analysis_dir,
    get_layer_embeddings_dir,
)


def main() -> None:
    """Run layer-wise CKA analysis."""
    logger = get_logger(__name__)

    args = parse_args()

    config = load_config(args.config)

    experiment_name = config["experiment"]["name"]

    input_dir = get_layer_embeddings_dir(
        experiment_name,
    )
    output_dir = get_analysis_dir(experiment_name) / "layers"
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    layer_paths = sorted(input_dir.glob("layer_*.npy"))

    layer_embeddings = [np.load(path) for path in layer_paths]

    num_layers = len(layer_embeddings)

    cka_matrix = np.zeros((num_layers, num_layers))

    for i in range(num_layers):
        for j in range(num_layers):
            cka_matrix[i, j] = linear_cka(
                layer_embeddings[i],
                layer_embeddings[j],
            )

    layer_names = [path.stem for path in layer_paths]

    cka_df = pd.DataFrame(
        cka_matrix,
        index=layer_names,
        columns=layer_names,
    )

    output_path = output_dir / "layer_cka.csv"

    cka_df.to_csv(output_path)

    logger.info(
        "Saved layer CKA matrix to %s",
        output_path,
    )

    logger.info("\n%s", cka_df)


if __name__ == "__main__":
    main()
