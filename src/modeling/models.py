"""Model definitions.

Baseline: a small, weakly-regularized linear model on a handful of raw
satellite readings -- what you'd ship on day one, no feature engineering.

Improved: a tree ensemble over the full engineered feature set (zonal
stats, spatial gradients / front strength, front colocation, seasonal
encoding), selected via cross-validated search over several candidate
model families and hyperparameters ("iterative model evaluation").
"""

from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASELINE_FEATURES = ["sst_mean", "chl_mean", "lat", "lon", "month_sin", "month_cos"]

ENGINEERED_FEATURES = [
    "sst_mean", "sst_std", "sst_min", "sst_max", "sst_range",
    "sst_front_strength", "sst_front_max",
    "chl_mean", "chl_std", "chl_min", "chl_max", "chl_range",
    "chl_front_strength", "chl_front_max",
    "front_colocation", "lat", "lon", "month_sin", "month_cos",
]


def make_baseline() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


# (name, estimator factory, hyperparameter grid) -- searched exhaustively
# with cross-validation in src/modeling/train.py.
CANDIDATE_MODELS = [
    (
        "random_forest",
        lambda **p: RandomForestClassifier(class_weight="balanced", random_state=0, **p),
        {
            "n_estimators": [200, 400],
            "max_depth": [4, 8, None],
            "min_samples_leaf": [1, 4],
        },
    ),
    (
        "gradient_boosting",
        lambda **p: GradientBoostingClassifier(random_state=0, **p),
        {
            "n_estimators": [100, 200],
            "max_depth": [2, 3],
            "learning_rate": [0.05, 0.1],
        },
    ),
]
