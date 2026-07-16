"""Turn raw per-cell satellite patches into a numeric feature table.

Each row is one (grid cell, month) sample -- one "image" -- described by
zonal statistics of the raw pixels in that cell, computed independently
for SST and chlorophyll, plus derived spatial and temporal features.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.features import tiling


def _patch_stats(prefix: str, values: np.ndarray, gradient: np.ndarray) -> dict:
    # Gradient magnitude is undefined (NaN) at pixels adjacent to a
    # cloud-masked neighbor, even though the pixel's own value is valid.
    # Fall back to 0 (no detectable front) if every pixel in the patch is
    # affected, which only happens in heavily-masked cells.
    valid_grad = gradient[~np.isnan(gradient)]
    if valid_grad.size == 0:
        front_strength = front_max = 0.0
    else:
        front_strength = float(np.mean(valid_grad))
        front_max = float(np.max(valid_grad))
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_range": float(np.max(values) - np.min(values)),
        f"{prefix}_front_strength": front_strength,
        f"{prefix}_front_max": front_max,
        f"{prefix}_n_pixels": int(values.size),
    }


def build_month_features(sst_path, chl_path, year: int, month: int) -> pd.DataFrame:
    """Build a per-cell feature table for one month from two raster files."""
    sst_lat, sst_lon, sst_vals = tiling.load_variable(sst_path, "sstMask")
    chl_lat, chl_lon, chl_vals = tiling.load_variable(chl_path, "chlorophyll")

    # Chlorophyll is strongly log-normally distributed; work in log space.
    chl_vals = np.log1p(chl_vals)

    sst_patches = tiling.build_patches(sst_lat, sst_lon, sst_vals)
    chl_patches = tiling.build_patches(chl_lat, chl_lon, chl_vals)

    cells = set(sst_patches) & set(chl_patches)
    rows = []
    for (clat, clon) in cells:
        sp = sst_patches[(clat, clon)]
        cp = chl_patches[(clat, clon)]
        row = {"cell_ll_lat": clat, "cell_ll_lon": clon, "year": year, "month": month}
        row.update(_patch_stats("sst", sp["values"], sp["gradient"]))
        row.update(_patch_stats("chl", cp["values"], cp["gradient"]))
        # thermal-chlorophyll front colocation: strong gradients in both
        # fields at once is a classic signature of upwelling edges where
        # fish -- and fishing vessels -- concentrate.
        row["front_colocation"] = row["sst_front_strength"] * row["chl_front_strength"]
        row["lat"] = clat + 0.05
        row["lon"] = clon + 0.05
        angle = 2 * math.pi * (month - 1) / 12
        row["month_sin"] = math.sin(angle)
        row["month_cos"] = math.cos(angle)
        rows.append(row)

    return pd.DataFrame(rows)
