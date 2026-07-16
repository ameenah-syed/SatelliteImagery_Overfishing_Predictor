"""Train + iteratively evaluate the overfishing-risk classifier.

Prediction target: binary "high fishing-activity risk" for a 0.1-degree
grid cell in a given month -- fishing_hours (AIS-derived apparent fishing
effort) > 0 for that cell/month, i.e. at least one vessel was actively
fishing there. This is trained purely from satellite-derived spatial
features (SST / chlorophyll statistics and fronts); no AIS data leaks
into the feature set.

Pipeline:
  1. Fit a baseline model on a handful of raw satellite readings.
  2. Cross-validated grid search over several tree-ensemble families on
     the full engineered feature set ("iterative model evaluation").
  3. Evaluate both on a temporally held-out test set and report the
     measured accuracy improvement -- not an assumed number.
"""

from __future__ import annotations

import json
import logging
import pickle

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src import config
from src.modeling import evaluate, models, splits

logger = logging.getLogger(__name__)

LABEL_COL = "any_fishing"


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[LABEL_COL] = (df["fishing_hours"] > 0).astype(int)
    return df


def run(df: pd.DataFrame | None = None) -> dict:
    if df is None:
        df = pd.read_parquet(config.FEATURE_TABLE_PATH)
    df = _prep(df)
    train_df, test_df = splits.temporal_split(df)
    logger.info(
        "train: %d rows (%d positive) | test: %d rows (%d positive)",
        len(train_df), train_df[LABEL_COL].sum(),
        len(test_df), test_df[LABEL_COL].sum(),
    )

    # --- Baseline -----------------------------------------------------
    X_train_b = train_df[models.BASELINE_FEATURES]
    X_test_b = test_df[models.BASELINE_FEATURES]
    y_train, y_test = train_df[LABEL_COL], test_df[LABEL_COL]

    baseline = models.make_baseline()
    baseline.fit(X_train_b, y_train)
    base_pred = baseline.predict(X_test_b)
    base_proba = baseline.predict_proba(X_test_b)[:, 1]
    baseline_metrics = evaluate.classification_metrics(y_test, base_pred, base_proba)
    logger.info("baseline metrics: %s", baseline_metrics)

    # --- Iterative evaluation over candidate model families -----------
    X_train_e = train_df[models.ENGINEERED_FEATURES]
    X_test_e = test_df[models.ENGINEERED_FEATURES]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    best_score = -1.0
    best_name = None
    best_estimator = None
    search_log = []
    for name, factory, grid in models.CANDIDATE_MODELS:
        search = GridSearchCV(
            factory(), grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True
        )
        search.fit(X_train_e, y_train)
        search_log.append({
            "model_family": name,
            "best_params": search.best_params_,
            "cv_roc_auc": float(search.best_score_),
        })
        logger.info("%s: best CV ROC-AUC=%.4f params=%s", name, search.best_score_, search.best_params_)
        if search.best_score_ > best_score:
            best_score = search.best_score_
            best_name = name
            best_estimator = search.best_estimator_

    improved_pred = best_estimator.predict(X_test_e)
    improved_proba = best_estimator.predict_proba(X_test_e)[:, 1]
    improved_metrics = evaluate.classification_metrics(y_test, improved_pred, improved_proba)
    logger.info("improved (%s) metrics: %s", best_name, improved_metrics)

    summary = evaluate.summarize(baseline_metrics, improved_metrics)
    summary["best_model_family"] = best_name
    summary["model_search_log"] = search_log
    summary["n_samples_total"] = int(len(df))
    summary["n_train"] = int(len(train_df))
    summary["n_test"] = int(len(test_df))
    summary["test_months"] = splits.TEST_MONTHS

    config.METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.METRICS_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    with open(config.MODEL_PATH, "wb") as f:
        pickle.dump({"model": best_estimator, "features": models.ENGINEERED_FEATURES}, f)

    logger.info("saved metrics to %s and model to %s", config.METRICS_PATH, config.MODEL_PATH)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run()
    print(json.dumps(result, indent=2))
