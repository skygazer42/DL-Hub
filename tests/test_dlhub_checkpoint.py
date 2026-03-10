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
