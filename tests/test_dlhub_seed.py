import numpy as np
import pytest

from dlhub.seed import set_seed


def test_set_seed_makes_numpy_reproducible() -> None:
    set_seed(123)
    a = np.random.normal(size=16)
    set_seed(123)
    b = np.random.normal(size=16)
    assert np.allclose(a, b)


def test_set_seed_makes_torch_reproducible_when_available() -> None:
    torch = pytest.importorskip("torch")

    set_seed(123)
    a = torch.randn(16)
    set_seed(123)
    b = torch.randn(16)
    assert torch.allclose(a, b)
