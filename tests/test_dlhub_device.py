import pytest


def test_resolve_device_cpu() -> None:
    pytest.importorskip("torch")

    from dlhub.device import resolve_device

    info = resolve_device("cpu")
    assert info.name == "cpu"
    assert str(info.torch_device) == "cpu"
