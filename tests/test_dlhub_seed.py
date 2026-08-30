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


def test_set_seed_propagates_torch_seeding_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")

    def fail_manual_seed(seed: int) -> None:
        del seed
        raise RuntimeError("simulated torch seeding failure")

    monkeypatch.setattr(torch, "manual_seed", fail_manual_seed)

    with pytest.raises(RuntimeError, match="simulated torch seeding failure"):
        set_seed(123)


def test_set_seed_can_enable_deterministic_torch_algorithms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    calls: list[tuple[object, ...]] = []

    monkeypatch.setattr(torch, "manual_seed", lambda seed: calls.append(("seed", seed)))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch,
        "use_deterministic_algorithms",
        lambda enabled, *, warn_only=False: calls.append(("deterministic", enabled, warn_only)),
    )
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", False)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)

    set_seed(321, deterministic=True, warn_only=True)

    assert calls == [("seed", 321), ("deterministic", True, True)]
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
