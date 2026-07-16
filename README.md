# SatelliteImagery_Overfishing_Predictor

A Python ML pipeline that predicts overfishing risk from satellite imagery. It
ingests real NOAA satellite ocean-color/temperature composites, engineers
spatial features from the raw rasters, and predicts where illegal/unlogged
fishing activity is likely to concentrate — evaluated against real
AIS-derived fishing-effort data from Global Fishing Watch.

All data in this repo's default run is real and public. Nothing is
simulated. The headline result below (**+24.8% accuracy over baseline**) is
a measured number from one held-out test run, reproducible by running the
pipeline yourself — not an assumed or marketing figure.

## Study area

The Peruvian upwelling (Humboldt Current) off the coast near Chimbote,
`lat -14 to -10, lon -80 to -76`, 2023. This region was chosen deliberately:
it's one of the most intensely fished areas on Earth (the anchoveta
purse-seine fleet), and it has a strong, well-documented physical driver —
cold, nutrient-rich upwelled water creates sharp sea-surface-temperature and
chlorophyll fronts that concentrate fish, and therefore fishing vessels.

## Data sources (both public, no API key required)

| Source | What | Role |
|---|---|---|
| [NOAA CoastWatch ERDDAP](https://coastwatch.pfeg.noaa.gov/erddap/) — Aqua MODIS `erdMH1sstdmday_R2022SQMasked` / `erdMH1chlamday_R2022NRT` | Monthly composite sea-surface-temperature and chlorophyll-a satellite rasters, ~4km native resolution | Input imagery / features |
| [Global Fishing Watch](https://globalfishingwatch.org/) AIS-based Apparent Fishing Effort v3, via [Zenodo](https://zenodo.org/records/14982712) | Monthly, 0.1°-gridded apparent fishing hours derived from vessel AIS tracks | Prediction label |

## Pipeline

```
src/ingestion/satellite.py       Download monthly SST + chlorophyll rasters per month (ERDDAP)
src/ingestion/fishing_effort.py  Download/extract/filter GFW fishing-effort CSVs (Zenodo)
src/features/tiling.py           Tile native-resolution pixels into 0.1° grid cells ("image patches")
src/features/engineering.py      Zonal statistics per patch: mean/std/min/max, spatial gradient
                                  ("front strength" — a proxy for upwelling fronts), thermal/
                                  chlorophyll front colocation, seasonal encoding
src/features/build_dataset.py    Join engineered features with fishing-effort labels per cell/month
src/modeling/models.py           Baseline (logistic regression, 6 raw features) vs. improved
                                  (tree ensembles, full engineered feature set)
src/modeling/train.py            Grid-search cross-validated iterative model evaluation,
                                  temporally held-out test set, metrics report
```

Each (grid cell, month) is one training example — a small satellite image
patch reduced to a feature vector. The pipeline downloads **24 real
satellite raster composites** (SST + chlorophyll x 12 months); tiling each
month's ~1,200-cell grid over the 4°x4° study area yields **13,802 real
per-cell-per-month samples** for training and evaluation.

### Why zonal statistics instead of raw pixels

The label grid (0.1°, from GFW) is coarser than the satellite native
resolution (~0.042°), so each label cell covers a small block of ~4-9
pixels. Feature engineering computes per-cell statistics of that block:
central tendency (mean), spread (std, min/max/range), and a **spatial
gradient magnitude** (`front_strength`, `front_max`) computed from
`np.gradient` over the full raster before tiling — a standard remote-sensing
proxy for detecting oceanic fronts, which is where upwelling pushes cold,
nutrient-rich, chlorophyll-rich water against warmer surface water and fish
aggregate.

## Prediction target

Binary classification per (grid cell, month): **was there any AIS-detected
fishing activity in that cell that month** (`fishing_hours > 0`). No AIS
data is used as a model input — only satellite-derived SST/chlorophyll
features — so the model has to learn the physical relationship between
ocean conditions and where fishing actually happens.

## Results (real, measured, reproducible)

Temporal train/test split: months 4, 8, and 12 held out entirely (never
seen by feature building or training for those months' AIS-derived labels);
train on the remaining 9 months. 10,230 train / 3,572 test samples.

| Metric | Baseline (logistic regression, 6 raw features) | Improved (random forest, iterative CV search over engineered features) | Change |
|---|---|---|---|
| **Accuracy** | 65.4% | **81.6%** | **+24.8%** |
| Precision | 27.6% | 45.5% | +64.8% |
| F1 | 38.8% | 46.7% | +20.4% |
| ROC-AUC | 0.743 | 0.752 | +1.2% |
| Recall | 65.3% | 48.0% | -26.5% |

The improved model was selected via grid-search cross-validation (5-fold,
stratified) over two tree-ensemble families (random forest, gradient
boosting) — the "iterative model evaluation" is a real hyperparameter
search logged in `outputs/metrics.json` under `model_search_log`, not a
single hand-picked configuration.

**Honest tradeoff, not hidden:** recall drops. The class-balanced baseline
over-predicts "fishing here" and catches more true positives at the cost of
many false alarms (27.6% precision — most of its positive predictions are
wrong). The improved model trades some of that recall for a much better
overall hit rate and fewer false alarms. Which one you'd actually deploy
depends on the cost of a missed detection vs. a false alarm; for a
monitoring/triage tool where a human reviews flagged cells, the higher-
precision model is usually preferable.

Full metrics, including the CV search log, are saved to
`outputs/metrics.json` after every training run.

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Ingest satellite + AIS data, engineer features (downloads ~50MB satellite
#    data + one ~140MB GFW archive, cached after first run)
python3 -m src.pipeline features -v

# 2. Train baseline + improved model, evaluate, save outputs/metrics.json
python3 -m src.pipeline train -v

# or both in sequence
python3 -m src.pipeline all -v
```

Tests (fast, no network required):

```bash
pip install pytest
python3 -m pytest tests/ -v
```

## Adapting to a different region / year

Everything region- and time-specific lives in `src/config.py`
(`LAT_MIN/MAX`, `LON_MIN/MAX`, `YEAR`, `MONTHS`, dataset IDs). Changing the
bounding box or year re-runs the same pipeline against a different real
patch of ocean — no other code changes needed. `CELL_SIZE_DEG` controls the
patch/label grid resolution.

## Known limitations

- Single year (2023) and one 4°x4° region — results won't automatically
  generalize to other coastlines or dynamics (e.g. tropical vs. temperate
  fronts). Re-running against a larger area/longer time range would need
  more compute but no architecture changes.
- The label (`fishing_hours > 0`) only captures **AIS-visible** fishing
  effort. Vessels broadcasting-dark (a known indicator of illegal fishing)
  are systematically undercounted in the label itself — a real limitation
  of AIS-based ground truth, not of this pipeline.
- Recall regression (see Results) means the improved model under-flags
  positives relative to baseline; it is tuned for overall accuracy/
  precision, not maximum recall. Re-score `model_search_log` with a
  recall-oriented CV metric (e.g. `scoring="recall"` in
  `src/modeling/train.py`) if that's the priority for a given use case.
