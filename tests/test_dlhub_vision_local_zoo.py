import pytest


torch = pytest.importorskip("torch")


def test_local_vision_zoo_lists_100_plus_arches() -> None:
    from dlhub.vision.local_zoo import list_local_arches

    arches = list_local_arches()
    assert len(arches) >= 100
    assert "dl:resnet18" in arches
    assert "dl:vgg16" in arches
    assert "dl:densenet121" in arches
    assert "dl:efficientnet_b0" in arches
    assert "dl:vit_tiny" in arches


@pytest.mark.parametrize("arch_id", ["dl:resnet18", "dl:vgg16", "dl:vit_tiny"])
def test_local_vision_zoo_can_build_classifier_smoke(arch_id: str) -> None:
    from dlhub.vision.local_zoo import build_local_model

    model = build_local_model(arch_id, in_channels=1, num_classes=4, image_size=64)
    model.eval()

    x = torch.zeros(2, 1, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert isinstance(y, torch.Tensor)
    assert tuple(y.shape) == (2, 4)

