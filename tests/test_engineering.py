import numpy as np

from src.features.engineering import _patch_stats


def test_patch_stats_basic():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    gradient = np.array([0.1, 0.2, 0.1, 0.4])
    stats = _patch_stats("sst", values, gradient)

    assert stats["sst_mean"] == 2.5
    assert stats["sst_min"] == 1.0
    assert stats["sst_max"] == 4.0
    assert stats["sst_range"] == 3.0
    assert stats["sst_n_pixels"] == 4
    assert np.isclose(stats["sst_front_strength"], gradient.mean())
    assert stats["sst_front_max"] == 0.4
