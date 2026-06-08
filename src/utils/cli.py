"""Command-line utilities."""

import argparse


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to configuration file.",
    )

    parser.add_argument(
        "--embedding-type",
        choices=["cls", "mean"],
        default="mean",
        help="Embedding type to analyze.",
    )

    return parser.parse_args()
