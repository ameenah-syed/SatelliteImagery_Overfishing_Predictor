"""CLI entrypoint for the overfishing-risk pipeline.

Usage:
    python -m src.pipeline features   # ingest satellite + AIS data, build feature table
    python -m src.pipeline train      # train baseline + improved model, evaluate
    python -m src.pipeline all        # both, in order
"""

from __future__ import annotations

import argparse
import json
import logging

from src.features import build_dataset
from src.modeling import train as train_module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=["features", "train", "all"],
        help="which stage of the pipeline to run",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    if args.stage in ("features", "all"):
        build_dataset.build_and_save()

    if args.stage in ("train", "all"):
        summary = train_module.run()
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
