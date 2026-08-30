import os
import stat

import pytest

torch = pytest.importorskip("torch")


def test_save_and_load_checkpoint_round_trip(tmp_path) -> None:
    from dlhub.checkpoint import load_checkpoint, save_checkpoint

    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Make params non-default.
    x = torch.randn(8, 3)
    y = torch.randn(8, 2)
    loss = torch.nn.MSELoss()(model(x), y)
    loss.backward()
    optimizer.step()

    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(ckpt_path, model=model, optimizer=optimizer, epoch=3, extra={"tag": "demo"})

    model2 = torch.nn.Linear(3, 2)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.1)
    meta = load_checkpoint(ckpt_path, model=model2, optimizer=optimizer2, map_location="cpu")

    assert meta["epoch"] == 3
    assert meta["extra"]["tag"] == "demo"

    for (k1, v1), (k2, v2) in zip(
        model.state_dict().items(), model2.state_dict().items(), strict=True
    ):
        assert k1 == k2
        torch.testing.assert_close(v1, v2)


def test_load_checkpoint_never_implicitly_falls_back_to_unsafe_pickle(
    monkeypatch, tmp_path
) -> None:
    from dlhub.checkpoint import load_checkpoint

    calls: list[dict[str, object]] = []

    def reject_weights_only(path, **kwargs):
        del path
        calls.append(dict(kwargs))
        raise TypeError("weights_only is unavailable")

    monkeypatch.setattr(torch, "load", reject_weights_only)

    with pytest.raises(RuntimeError, match="Safe checkpoint loading"):
        load_checkpoint(tmp_path / "legacy.pt", model=torch.nn.Linear(3, 2))

    assert calls == [{"map_location": "cpu", "weights_only": True}]


def test_load_checkpoint_allows_explicit_trusted_legacy_fallback(monkeypatch, tmp_path) -> None:
    from dlhub.checkpoint import load_checkpoint

    source = torch.nn.Linear(3, 2)
    payload = {"model_state": source.state_dict(), "epoch": 7, "extra": {"trusted": True}}
    calls: list[dict[str, object]] = []

    def load_legacy(path, **kwargs):
        del path
        calls.append(dict(kwargs))
        if kwargs.get("weights_only") is True:
            raise TypeError("weights_only is unavailable")
        return payload

    monkeypatch.setattr(torch, "load", load_legacy)

    target = torch.nn.Linear(3, 2)
    with pytest.warns(RuntimeWarning, match="arbitrary code"):
        meta = load_checkpoint(
            tmp_path / "legacy.pt",
            model=target,
            allow_unsafe_legacy=True,
        )

    assert calls == [
        {"map_location": "cpu", "weights_only": True},
        {"map_location": "cpu"},
    ]
    assert meta == {"epoch": 7, "extra": {"trusted": True}}

    for expected, actual in zip(source.parameters(), target.parameters(), strict=True):
        torch.testing.assert_close(expected, actual)


def test_interrupted_checkpoint_save_preserves_last_good_file_and_cleans_temp(
    monkeypatch, tmp_path
) -> None:
    from dlhub.checkpoint import load_checkpoint, save_checkpoint

    ckpt_path = tmp_path / "ckpt.pt"
    model = torch.nn.Linear(3, 2)
    save_checkpoint(ckpt_path, model=model, epoch=4, extra={"status": "last-good"})
    original = ckpt_path.read_bytes()

    def interrupt_save(payload, handle) -> None:
        del payload
        handle.write(b"partial-checkpoint")
        raise KeyboardInterrupt("simulated interruption")

    with monkeypatch.context() as patch:
        patch.setattr(torch, "save", interrupt_save)
        with pytest.raises(KeyboardInterrupt, match="simulated interruption"):
            save_checkpoint(ckpt_path, model=model, epoch=5)

    assert ckpt_path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [ckpt_path]

    restored = torch.nn.Linear(3, 2)
    meta = load_checkpoint(ckpt_path, model=restored)
    assert meta == {"epoch": 4, "extra": {"status": "last-good"}}


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits")
def test_checkpoint_replacement_preserves_existing_mode(tmp_path) -> None:
    from dlhub.checkpoint import save_checkpoint

    ckpt_path = tmp_path / "ckpt.pt"
    model = torch.nn.Linear(3, 2)
    save_checkpoint(ckpt_path, model=model, epoch=1)
    ckpt_path.chmod(0o640)

    save_checkpoint(ckpt_path, model=model, epoch=2)

    assert stat.S_IMODE(ckpt_path.stat().st_mode) == 0o640
