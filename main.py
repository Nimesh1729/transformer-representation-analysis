"""Extract and save transformer embeddings for the sentence dataset."""

import numpy as np

from src.extraction.dataset_embeddings import load_sentence_dataset
from src.extraction.embedding_extractor import (
    extract_cls_embeddings,
    extract_mean_embeddings,
)
from src.models.load_model import load_model, load_tokenizer
from src.utils.cli import parse_args
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.paths import get_embeddings_dir
from src.utils.reproducibility import set_seed


def main() -> None:
    """Extract CLS and mean-pooled embeddings from the sentence dataset."""
    logger = get_logger(__name__)

    args = parse_args()

    config = load_config(args.config)

    model_name = config["model"]["name"]
    seed = config["system"]["seed"]

    set_seed(seed)

    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name)

    dataset = load_sentence_dataset("data/astronomy_sentences.csv")
    texts = dataset["text"].tolist()
    labels = dataset["label"].to_numpy()

    logger.info("Experiment: %s", config["experiment"]["name"])
    logger.info("Model: %s", model_name)

    cls_embeddings = extract_cls_embeddings(
        texts=texts,
        tokenizer=tokenizer,
        model=model,
    )
    mean_embeddings = extract_mean_embeddings(
        texts=texts,
        tokenizer=tokenizer,
        model=model,
    )

    experiment_name = config["experiment"]["name"]

    output_dir = get_embeddings_dir(
        experiment_name,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    cls_embeddings_path = output_dir / "cls_embeddings.npy"
    mean_embeddings_path = output_dir / "mean_embeddings.npy"
    labels_path = output_dir / "labels.npy"

    np.save(cls_embeddings_path, cls_embeddings)
    np.save(mean_embeddings_path, mean_embeddings)
    np.save(labels_path, labels)

    logger.info("Loaded %d sentences.", len(texts))
    logger.info("CLS embeddings shape: %s", cls_embeddings.shape)
    logger.info("Mean embeddings shape: %s", mean_embeddings.shape)
    logger.info("Labels shape: %s", labels.shape)
    logger.info("Saved CLS embeddings to %s", cls_embeddings_path)
    logger.info("Saved mean embeddings to %s", mean_embeddings_path)
    logger.info("Saved labels to %s", labels_path)


if __name__ == "__main__":
    main()
