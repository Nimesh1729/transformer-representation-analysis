"""Extract and save mean-pooled embeddings from every DistilBERT layer."""

import numpy as np

from src.extraction.dataset_embeddings import load_sentence_dataset
from src.extraction.layer_extractor import extract_layer_mean_embeddings
from src.models.load_model import load_model, load_tokenizer
from src.utils.cli import parse_args
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.paths import get_layer_embeddings_dir
from src.utils.reproducibility import set_seed


def main() -> None:
    """Extract layer-wise mean-pooled embeddings."""
    logger = get_logger(__name__)

    args = parse_args()
    config = load_config(args.config)

    experiment_name = config["experiment"]["name"]

    model_name = config["model"]["name"]
    seed = config["system"]["seed"]

    set_seed(seed)

    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name)

    dataset = load_sentence_dataset("data/astronomy_sentences.csv")
    texts = dataset["text"].tolist()
    labels = dataset["label"].to_numpy()

    output_dir = get_layer_embeddings_dir(experiment_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_path = output_dir / "labels.npy"
    np.save(labels_path, labels)

    num_layers = model.config.num_hidden_layers + 1
    logger.info(
        "Found %d hidden-state layers.",
        num_layers,
    )

    for layer_index in range(num_layers):
        embeddings = extract_layer_mean_embeddings(
            texts=texts,
            tokenizer=tokenizer,
            model=model,
            layer_index=layer_index,
        )

        output_path = output_dir / f"layer_{layer_index}.npy"
        np.save(output_path, embeddings)

        logger.info(
            "Saved layer %d embeddings with shape %s to %s",
            layer_index,
            embeddings.shape,
            output_path,
        )

    logger.info("Saved labels to %s", labels_path)


if __name__ == "__main__":
    main()
