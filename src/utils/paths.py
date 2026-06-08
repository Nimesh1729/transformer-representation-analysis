"""Path utilities for experiment outputs."""

from pathlib import Path


def get_experiment_dir(experiment_name: str) -> Path:
    """Return the root output directory for an experiment.

    Args:
        experiment_name: Name of the experiment.

    Returns:
        Path to the experiment output directory.
    """
    return Path("outputs") / experiment_name


def get_embeddings_dir(experiment_name: str) -> Path:
    """Return the embeddings directory for an experiment."""
    return get_experiment_dir(experiment_name) / "embeddings"


def get_analysis_dir(experiment_name: str) -> Path:
    """Return the analysis directory for an experiment."""
    return get_experiment_dir(experiment_name) / "analysis"


def get_visualizations_dir(experiment_name: str) -> Path:
    """Return the visualizations directory for an experiment."""
    return get_experiment_dir(experiment_name) / "visualizations"


def get_layer_embeddings_dir(experiment_name: str) -> Path:
    """Return the layer embeddings directory for an experiment."""
    return get_experiment_dir(experiment_name) / "layer_embeddings"


def get_attention_dir(experiment_name: str) -> Path:
    """Return the attention directory for an experiment."""
    return get_experiment_dir(experiment_name) / "attention"


def ensure_dir(path: Path) -> Path:
    """Create directory if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_logs_dir(experiment_name: str) -> Path:
    """Return logs directory for an experiment.

    Args:
        experiment_name: Name of the experiment.

    Returns:
        Path to the experiment log directory.
    """
    return Path("logs") / experiment_name
