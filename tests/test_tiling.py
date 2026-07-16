import numpy as np

from src.features import tiling


def test_cell_index_snaps_to_grid():
    coords = np.array([-14.03, -14.0, -13.96, -10.01])
    cells = tiling.cell_index(coords, cell_size=0.1)
    assert np.allclose(cells, [-14.1, -14.0, -14.0, -10.1])


def test_build_patches_groups_pixels_and_skips_nan():
    lat = np.array([-10.0, -9.95, -10.15])
    lon = np.array([-80.0, -79.95, -80.15])
    values = np.array([
        [1.0, 2.0, np.nan],
        [3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0],
    ])
    patches = tiling.build_patches(lat, lon, values)
    # (-10.0, -80.0) cell should contain the first 2x2 block minus the NaN
    key = (-10.0, -80.0)
    assert key in patches
    assert sorted(patches[key]["values"].tolist()) == [1.0, 2.0, 3.0, 4.0]
    # the cell containing only the NaN pixel's row/col partner should not
    # include it
    assert not any(np.isnan(p["values"]).any() for p in patches.values())


def test_gradient_magnitude_zero_on_flat_field():
    flat = np.ones((5, 5)) * 3.0
    grad = tiling.gradient_magnitude(flat)
    assert np.allclose(grad, 0.0)
