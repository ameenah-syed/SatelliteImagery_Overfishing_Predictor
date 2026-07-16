"""Tile a satellite raster into patches aligned to the fishing-effort grid.

Each NOAA composite is a native ~4km-resolution raster. The Global Fishing
Watch labels live on a coarser 0.1 degree grid, so each label cell covers
roughly a 2x2 to 3x3 block of native pixels. We group native pixels by the
0.1 degree cell they fall in -- each group is one spatial "image patch" --
so that feature engineering can compute zonal statistics (mean/std/extrema/
gradient) per patch instead of per raw pixel.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from src import config


def load_variable(path, variable: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one month's raster. Returns (lat 1D, lon 1D, data 2D [lat, lon])."""
    ds = xr.open_dataset(path)
    data = ds[variable]
    if "time" in data.dims:
        data = data.isel(time=0)
    lat = data["latitude"].values.astype(np.float64)
    lon = data["longitude"].values.astype(np.float64)
    values = data.values.astype(np.float64)
    ds.close()
    return lat, lon, values


def cell_index(coord: np.ndarray, cell_size: float = config.CELL_SIZE_DEG) -> np.ndarray:
    """Map raw coordinates to the lower-left corner of their grid cell."""
    return np.round(np.floor(coord / cell_size) * cell_size, 4)


def gradient_magnitude(values: np.ndarray) -> np.ndarray:
    """Local spatial gradient magnitude (proxy for oceanic fronts, a known
    driver of fish -- and therefore fishing vessel -- aggregation)."""
    with np.errstate(invalid="ignore"):
        gy, gx = np.gradient(values)
    return np.sqrt(gx ** 2 + gy ** 2)


def build_patches(lat: np.ndarray, lon: np.ndarray, values: np.ndarray) -> dict:
    """Group native pixels (and their local gradient) by destination cell.

    Returns {(cell_lat, cell_lon): {"values": np.ndarray, "gradient": np.ndarray}}
    """
    grad = gradient_magnitude(values)
    lat_cells = cell_index(lat)
    lon_cells = cell_index(lon)

    patches: dict = {}
    for i, clat in enumerate(lat_cells):
        row_vals = values[i]
        row_grad = grad[i]
        for j, clon in enumerate(lon_cells):
            v = row_vals[j]
            if np.isnan(v):
                continue
            key = (clat, clon)
            entry = patches.setdefault(key, {"values": [], "gradient": []})
            entry["values"].append(v)
            entry["gradient"].append(row_grad[j])
    return {
        k: {"values": np.array(v["values"]), "gradient": np.array(v["gradient"])}
        for k, v in patches.items()
    }
