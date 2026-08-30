from collections import defaultdict, namedtuple, OrderedDict

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


def test_to_device_preserves_common_container_types() -> None:
    from dlhub.training.batch import to_device

    Pair = namedtuple("Pair", ["features", "label"])
    batch = OrderedDict(
        [
            ("pair", Pair(torch.randn(2), "keep")),
            ("groups", defaultdict(list, {"items": [torch.randn(1)]})),
        ]
    )

    moved = to_device(batch, device=torch.device("cpu"))

    assert isinstance(moved, OrderedDict)
    assert isinstance(moved["pair"], Pair)
    assert moved["pair"].features.device.type == "cpu"
    assert isinstance(moved["groups"], defaultdict)
    assert moved["groups"].default_factory is list
    assert moved["groups"]["items"][0].device.type == "cpu"
