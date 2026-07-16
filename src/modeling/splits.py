"""Temporal train/test split.

Splitting by month (rather than randomly by row) prevents leakage: rows
from the same month and nearby cells are spatially/temporally correlated,
so a random row split would let the model "see the answer" via a
near-duplicate neighbor in the training set. Held-out months are ones the
model has never seen any data from, in either features or labels.
"""

from __future__ import annotations

import pandas as pd

TEST_MONTHS = [4, 8, 12]  # spread across seasons, held out entirely


def temporal_split(df: pd.DataFrame, test_months: list[int] = TEST_MONTHS):
    train = df[~df["month"].isin(test_months)].reset_index(drop=True)
    test = df[df["month"].isin(test_months)].reset_index(drop=True)
    return train, test
