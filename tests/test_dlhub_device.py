import pytest

from dlhub.device import resolve_device

torch = pytest.importorskip("torch")


def test_resolve_device_cpu() -> None:
    info = resolve_device("cpu")
    assert info.name == "cpu"
    assert str(info.torch_device) == "cpu"


def test_resolve_device_normalizes_case_and_whitespace() -> None:
    info = resolve_device("  CPU  ")
    assert info.name == "cpu"
    assert str(info.torch_device) == "cpu"


@pytest.mark.parametrize("requested", ["cuda:", "cuda:-1", "cuda:01", "cuda:gpu", "cuda:0:1"])
def test_resolve_device_rejects_invalid_cuda_syntax_before_availability_check(
    requested: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(ValueError, match="Invalid CUDA device"):
        resolve_device(requested)


def test_resolve_device_rejects_out_of_range_cuda_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    with pytest.raises(RuntimeError, match=r"index 1 is out of range.*1 visible"):
        resolve_device("cuda:1")


def test_resolve_device_accepts_visible_cuda_index(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    info = resolve_device("CUDA:1")

    assert info.name == "cuda:1"
    assert str(info.torch_device) == "cuda:1"


def test_resolve_device_rejects_non_string_request() -> None:
    with pytest.raises(TypeError, match="device must be a string or None"):
        resolve_device(1)  # type: ignore[arg-type]
