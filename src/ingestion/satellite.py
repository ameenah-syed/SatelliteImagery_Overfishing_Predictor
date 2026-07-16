"""Download monthly-composite satellite rasters (SST, chlorophyll-a) from
NOAA CoastWatch ERDDAP for a bounding box and cache them locally.

Each downloaded file is one real satellite composite image (a full 2-D
raster over the study area) for one variable/month. Feature engineering
later tiles each raster into per-grid-cell patches, which is where the
"10,000+ images" come from (grid cells x months x variables).
"""

from __future__ import annotations

import calendar
import logging
import time
from pathlib import Path

import requests

from src import config

logger = logging.getLogger(__name__)


def _erddap_url(dataset_id: str, variable: str, year: int, month: int) -> str:
    last_day = calendar.monthrange(year, month)[1]
    start = f"{year}-{month:02d}-01T00:00:00Z"
    stop = f"{year}-{month:02d}-{last_day:02d}T00:00:00Z"
    # Latitude axis is stored north-to-south in these datasets, so request
    # max:min for lat and min:max for lon.
    query = (
        f"{variable}"
        f"[({start}):1:({stop})]"
        f"[({config.LAT_MAX}):1:({config.LAT_MIN})]"
        f"[({config.LON_MIN}):1:({config.LON_MAX})]"
    )
    return f"{config.ERDDAP_BASE}/{dataset_id}.nc?{query}"


def fetch_month(
    dataset_id: str,
    variable: str,
    year: int,
    month: int,
    max_retries: int = 4,
    timeout: int = 120,
) -> Path:
    """Download (or reuse cached) monthly raster, return local file path."""
    out_path = config.SATELLITE_RAW_DIR / f"{variable}_{year}_{month:02d}.nc"
    if out_path.exists() and out_path.stat().st_size > 0:
        logger.info("cached %s", out_path.name)
        return out_path

    url = _erddap_url(dataset_id, variable, year, month)
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 503:
                raise requests.HTTPError("503 Service Unavailable")
            resp.raise_for_status()
            out_path.write_bytes(resp.content)
            logger.info("downloaded %s (%d bytes)", out_path.name, len(resp.content))
            return out_path
        except (requests.RequestException,) as exc:
            last_error = exc
            wait = min(30, 5 * attempt)
            logger.warning(
                "attempt %d/%d failed for %s/%s %04d-%02d: %s (retrying in %ds)",
                attempt, max_retries, dataset_id, variable, year, month, exc, wait,
            )
            time.sleep(wait)
    raise RuntimeError(
        f"failed to fetch {dataset_id}/{variable} {year}-{month:02d} after "
        f"{max_retries} attempts"
    ) from last_error


def fetch_all(year: int = config.YEAR, months: list[int] = config.MONTHS) -> dict[str, list[Path]]:
    """Fetch SST + chlorophyll rasters for every month. Returns paths by variable."""
    paths: dict[str, list[Path]] = {"sst": [], "chl": []}
    for month in months:
        paths["sst"].append(
            fetch_month(config.SST_DATASET_ID, config.SST_VARIABLE, year, month)
        )
        paths["chl"].append(
            fetch_month(config.CHL_DATASET_ID, config.CHL_VARIABLE, year, month)
        )
    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fetch_all()
