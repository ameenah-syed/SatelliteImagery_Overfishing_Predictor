"""Evaluation metrics and baseline-vs-improved comparison reporting."""

from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def pct_improvement(baseline: float, improved: float) -> float:
    if baseline == 0:
        return float("inf")
    return 100.0 * (improved - baseline) / baseline


def summarize(baseline_metrics: dict, improved_metrics: dict) -> dict:
    return {
        "baseline": baseline_metrics,
        "improved": improved_metrics,
        "pct_improvement": {
            k: pct_improvement(baseline_metrics[k], improved_metrics[k])
            for k in baseline_metrics
        },
    }
