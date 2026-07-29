import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")


def test_vision_zoo_lists_100_plus_arches() -> None:
    from dlhub.vision.zoo import list_vision_arches

    arches = list_vision_arches()
    assert len(arches) >= 100
    assert "tv:resnet18" in arches
    assert "dl:resnet18" in arches
    assert any(a.startswith("tvseg:") for a in arches)
    assert any(a.startswith("tvdet:") for a in arches)
    assert any(a.startswith("tvflow:") for a in arches)
    assert any(a.startswith("tvvideo:") for a in arches)

    import torchvision
    from torchvision.models import list_models

    quant_mod = getattr(torchvision.models, "quantization", None)
    if quant_mod is not None and list_models(quant_mod):
        assert any(a.startswith("tvq:") for a in arches)


def test_vision_zoo_can_build_classifier_model_smoke() -> None:
    from dlhub.vision.zoo import build_torchvision_model

    model = build_torchvision_model("tv:resnet18", num_classes=4)
    model.eval()

    x = torch.zeros(2, 3, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert isinstance(y, torch.Tensor)
    assert tuple(y.shape) == (2, 4)


def test_timm_builder_filters_unsupported_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    from dlhub.vision import zoo

    received: dict[str, object] = {}
    sentinel = object()

    class FakeTimm:
        @staticmethod
        def create_model(name: str, *, pretrained: bool, num_classes: int):
            received.update(
                name=name,
                pretrained=pretrained,
                num_classes=num_classes,
            )
            return sentinel

    monkeypatch.setattr(zoo, "_import_timm", lambda: FakeTimm)

    assert zoo.build_timm_model("timm:example", num_classes=4) is sentinel
    assert received == {
        "name": "example",
        "pretrained": False,
        "num_classes": 4,
    }


def test_vision_zoo_can_build_quantized_classifier_model_smoke() -> None:
    import torchvision
    from torchvision.models import list_models

    quant_mod = getattr(torchvision.models, "quantization", None)
    if quant_mod is None:
        pytest.skip("torchvision.models.quantization not available")
    if "quantized_resnet18" not in set(list_models(quant_mod)):
        pytest.skip("quantized_resnet18 not available")

    from dlhub.vision.zoo import build_torchvision_model

    model = build_torchvision_model("tvq:resnet18", num_classes=4)
    model.eval()

    x = torch.zeros(2, 3, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert isinstance(y, torch.Tensor)
    assert tuple(y.shape) == (2, 4)
