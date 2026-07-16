"""Orchestrate ingestion + feature engineering into one training table.

For every month in the study period: fetch/cache the satellite rasters,
tile them into per-cell patches, engineer zonal-statistic features, and
join with the Global Fishing Watch apparent-fishing-effort label for that
cell/month. Cells with satellite coverage but no AIS-detected fishing
activity are real negatives (fishing_hours = 0), not missing data.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import config
from src.features import engineering
from src.ingestion import fishing_effort, satellite

logger = logging.getLogger(__name__)


def build(year: int = config.YEAR, months: list[int] = config.MONTHS) -> pd.DataFrame:
    monthly_frames = []
    for month in months:
        logger.info("processing %04d-%02d", year, month)
        sst_path = satellite.fetch_month(
            config.SST_DATASET_ID, config.SST_VARIABLE, year, month
        )
        chl_path = satellite.fetch_month(
            config.CHL_DATASET_ID, config.CHL_VARIABLE, year, month
        )
        feats = engineering.build_month_features(sst_path, chl_path, year, month)
        labels = fishing_effort.load_month_effort(year, month)
        labels = labels[["cell_ll_lat", "cell_ll_lon", "fishing_hours", "hours", "mmsi_present"]]
        labels["cell_ll_lat"] = np.round(labels["cell_ll_lat"], 4)
        labels["cell_ll_lon"] = np.round(labels["cell_ll_lon"], 4)

        merged = feats.merge(labels, on=["cell_ll_lat", "cell_ll_lon"], how="left")
        merged["fishing_hours"] = merged["fishing_hours"].fillna(0.0)
        merged["hours"] = merged["hours"].fillna(0.0)
        merged["mmsi_present"] = merged["mmsi_present"].fillna(0).astype(int)
        monthly_frames.append(merged)

    table = pd.concat(monthly_frames, ignore_index=True)
    logger.info("built feature table: %d rows, %d cols", *table.shape)
    return table


def build_and_save(year: int = config.YEAR, months: list[int] = config.MONTHS) -> pd.DataFrame:
    table = build(year, months)
    config.FEATURE_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(config.FEATURE_TABLE_PATH, index=False)
    logger.info("saved %s", config.FEATURE_TABLE_PATH)
    return table


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_and_save()
