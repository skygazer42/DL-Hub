import numpy as np

from dlhub.data.synthetic import SyntheticClassificationConfig, make_linearly_separable_classification_numpy


def test_compact_classification_numpy_shapes_and_labels() -> None:
    x, y = make_linearly_separable_classification_numpy(
        SyntheticClassificationConfig(num_samples=200, num_features=3, noise_std=0.1, seed=0)
    )
    assert x.shape == (200, 3)
    assert y.shape == (200,)
    assert x.dtype == np.float64
    assert y.dtype == np.int64

    # Both classes should appear with high probability at this scale.
    assert set(np.unique(y)).issubset({0, 1})
    assert 0 in y and 1 in y
