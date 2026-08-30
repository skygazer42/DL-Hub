import numpy as np
import pytest

from dlhub.metrics import accuracy_numpy, accuracy_torch

torch = pytest.importorskip("torch")


def test_accuracy_numpy_rejects_empty_and_broadcastable_mismatched_inputs() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        accuracy_numpy([], [])
    with pytest.raises(ValueError, match="same shape"):
        accuracy_numpy(np.array([[1], [0]]), np.array([1, 0]))


def test_accuracy_numpy_supports_non_numeric_labels_but_rejects_nan() -> None:
    assert accuracy_numpy(["cat", "dog"], ["cat", "cat"]) == 0.5
    with pytest.raises(ValueError, match="finite"):
        accuracy_numpy([np.nan], [np.nan])


def test_accuracy_torch_validates_shapes_and_empty_tensors() -> None:
    with pytest.raises(ValueError, match="at least 2 dimensions"):
        accuracy_torch(torch.tensor([0.1, 0.9]), torch.tensor([1]))
    with pytest.raises(ValueError, match="non-empty"):
        accuracy_torch(torch.empty((0, 2)), torch.empty((0,), dtype=torch.long))
    with pytest.raises(ValueError, match="target shape"):
        accuracy_torch(torch.zeros((2, 2)), torch.zeros((2, 1), dtype=torch.long))


def test_accuracy_torch_supports_dense_predictions_and_rejects_nan_logits() -> None:
    logits = torch.tensor([[[[3.0, -1.0]], [[-1.0, 3.0]]]])
    targets = torch.tensor([[[0, 1]]])
    assert accuracy_torch(logits, targets) == 1.0

    with pytest.raises(ValueError, match="finite"):
        accuracy_torch(torch.tensor([[float("nan"), 1.0]]), torch.tensor([1]))
