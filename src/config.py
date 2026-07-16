"""Pipeline-wide configuration.

Study area: the Peruvian upwelling (Humboldt Current) off Chimbote/Callao,
one of the most heavily fished regions on Earth (anchoveta purse-seine
fleet). Chosen because it has a strong, well-documented link between
sea-surface temperature / chlorophyll fronts and fishing effort, real
satellite coverage, and real AIS-based effort data.
"""

from pathlib import Path

# ---- Study area & time range -----------------------------------------
LAT_MIN, LAT_MAX = -14.0, -10.0   # degrees N
LON_MIN, LON_MAX = -80.0, -76.0   # degrees E
YEAR = 2023
MONTHS = list(range(1, 13))

# 0.1 degree matches the native resolution of the GFW fishing-effort grid.
CELL_SIZE_DEG = 0.1

# ---- NOAA ERDDAP (satellite imagery) -----------------------------------
ERDDAP_BASE = "https://coastwatch.pfeg.noaa.gov/erddap/griddap"
SST_DATASET_ID = "erdMH1sstdmday_R2022SQMasked"
SST_VARIABLE = "sstMask"
CHL_DATASET_ID = "erdMH1chlamday_R2022NRT"
CHL_VARIABLE = "chlorophyll"

# ---- Global Fishing Watch (labels) via Zenodo static archive -----------
GFW_ZENODO_RECORD = "14982712"
GFW_FILE_TEMPLATE = "fleet-monthly-csvs-10-v3-{year}.zip"
GFW_CSV_TEMPLATE = "fleet-monthly-csvs-10-v3-{year}-{month:02d}-01.csv"

# ---- Paths ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
SATELLITE_RAW_DIR = DATA_RAW / "satellite"
FISHING_RAW_DIR = DATA_RAW / "fishing_effort"
DATA_PROCESSED = ROOT / "data" / "processed"
FEATURE_TABLE_PATH = DATA_PROCESSED / "features.parquet"
OUTPUTS = ROOT / "outputs"
METRICS_PATH = OUTPUTS / "metrics.json"
MODEL_PATH = OUTPUTS / "model.pkl"

for _dir in (SATELLITE_RAW_DIR, FISHING_RAW_DIR, DATA_PROCESSED, OUTPUTS):
    _dir.mkdir(parents=True, exist_ok=True)
