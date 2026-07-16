"""Download and filter Global Fishing Watch AIS-based apparent fishing
effort (the prediction label) from the public Zenodo archive of GFW's
static dataset (no API key required, non-commercial use).

https://zenodo.org/records/14982712 — "Global AIS-based Apparent Fishing
Effort Dataset". Each yearly archive contains one CSV per month, gridded
at 0.1 degrees, broken down by flag state and gear type. We download the
archive once, extract only the months we need (without unzipping the
whole multi-GB archive), and aggregate to the study bounding box.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import pandas as pd
import requests

from src import config

logger = logging.getLogger(__name__)

ZENODO_FILE_URL = (
    "https://zenodo.org/api/records/{record}/files/{filename}/content"
)

CSV_COLUMNS = [
    "date", "year", "month", "cell_ll_lat", "cell_ll_lon",
    "flag", "geartype", "hours", "fishing_hours", "mmsi_present",
]


def _zip_path(year: int) -> Path:
    return config.FISHING_RAW_DIR / config.GFW_FILE_TEMPLATE.format(year=year)


def download_year_zip(year: int, timeout: int = 600) -> Path:
    """Download (or reuse cached) the yearly fleet-monthly zip archive."""
    out_path = _zip_path(year)
    if out_path.exists() and out_path.stat().st_size > 0:
        logger.info("cached %s", out_path.name)
        return out_path

    filename = config.GFW_FILE_TEMPLATE.format(year=year)
    url = ZENODO_FILE_URL.format(record=config.GFW_ZENODO_RECORD, filename=filename)
    logger.info("downloading %s (this is ~100-150MB, one-time)", url)
    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        tmp_path = out_path.with_suffix(".part")
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        tmp_path.rename(out_path)
    logger.info("downloaded %s", out_path.name)
    return out_path


def _extract_month_csv(zip_path: Path, year: int, month: int) -> Path:
    csv_name = config.GFW_CSV_TEMPLATE.format(year=year, month=month)
    dest = config.FISHING_RAW_DIR / csv_name
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(csv_name) as src, open(dest, "wb") as out:
            out.write(src.read())
    return dest


def load_month_effort(year: int, month: int) -> pd.DataFrame:
    """Return per-cell aggregated fishing effort for the study bbox for one month.

    Columns: cell_ll_lat, cell_ll_lon, year, month, hours, fishing_hours,
    mmsi_present (all summed across flag states and gear types).
    """
    zip_path = download_year_zip(year)
    csv_path = _extract_month_csv(zip_path, year, month)

    lat_min, lat_max = config.LAT_MIN, config.LAT_MAX
    lon_min, lon_max = config.LON_MIN, config.LON_MAX

    chunks = []
    for chunk in pd.read_csv(csv_path, usecols=CSV_COLUMNS, chunksize=500_000):
        mask = (
            chunk["cell_ll_lat"].between(lat_min, lat_max)
            & chunk["cell_ll_lon"].between(lon_min, lon_max)
        )
        sub = chunk.loc[mask]
        if len(sub):
            chunks.append(sub)

    if not chunks:
        return pd.DataFrame(
            columns=["cell_ll_lat", "cell_ll_lon", "year", "month",
                     "hours", "fishing_hours", "mmsi_present"]
        )

    region = pd.concat(chunks, ignore_index=True)
    agg = (
        region.groupby(["cell_ll_lat", "cell_ll_lon"], as_index=False)
        .agg(
            hours=("hours", "sum"),
            fishing_hours=("fishing_hours", "sum"),
            mmsi_present=("mmsi_present", "sum"),
        )
    )
    agg["year"] = year
    agg["month"] = month
    return agg


def load_all(year: int = config.YEAR, months: list[int] = config.MONTHS) -> pd.DataFrame:
    frames = [load_month_effort(year, m) for m in months]
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = load_all()
    print(df.describe())
    print(len(df), "cell-months with any AIS fishing activity")
