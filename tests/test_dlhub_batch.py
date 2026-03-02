import pytest


torch = pytest.importorskip("torch")


def test_to_device_moves_nested_structures() -> None:
    from dlhub.training.batch import to_device

    device = torch.device("cpu")
    batch = (
        torch.randn(2, 3),
        {"x": torch.randn(1), "meta": {"id": 123}},
        [torch.randn(4), "keep"],
    )

    moved = to_device(batch, device=device)

    assert isinstance(moved, tuple)
    assert moved[0].device == device
    assert moved[1]["x"].device == device
    assert moved[1]["meta"]["id"] == 123
    assert moved[2][0].device == device
    assert moved[2][1] == "keep"

