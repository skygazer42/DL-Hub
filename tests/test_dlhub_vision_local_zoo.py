import pytest


torch = pytest.importorskip("torch")


def test_local_vision_zoo_lists_200_plus_arches() -> None:
    from dlhub.vision.local_zoo import list_local_arches

    arches = list_local_arches()
    assert len(arches) >= 200
    assert "dl:lenet5" in arches
    assert "dl:alexnet" in arches
    assert "dl:resnet18" in arches
    assert "dl:vgg16" in arches
    assert "dl:densenet121" in arches
    assert "dl:efficientnet_b0" in arches
    assert "dl:vit_tiny" in arches
    assert "dl:googlenet" in arches
    assert "dl:regnetx_400mf" in arches
    assert "dl:swin_tiny" in arches
    assert "dl:poolformer_tiny" in arches
    assert "dl:gmlp_tiny" in arches
    assert "dl:resmlp_tiny" in arches
    assert "dl:mobilevit_tiny" in arches
    assert "dl:coatnet_tiny" in arches
    assert "dl:fnet_tiny" in arches
    assert "dl:pvt_tiny" in arches
    assert "dl:eca_resnet18" in arches
    assert "dl:cbam_resnet18" in arches


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


@pytest.mark.parametrize(
    ("arch_id", "width_mult"),
    [
        ("dl:lenet", 1.0),
        ("dl:googlenet", 0.5),
        ("dl:xception_tiny", 0.5),
        ("dl:darknet_tiny", 0.5),
        ("dl:cspdarknet_tiny", 0.5),
        ("dl:regnetx_400mf", 0.5),
        ("dl:regnety_400mf", 0.5),
        ("dl:shufflenetv1", 0.5),
        ("dl:mnasnet", 0.5),
        ("dl:ghostnet", 0.5),
        ("dl:mobileone", 0.5),
        ("dl:swin", 0.5),
        ("dl:poolformer", 0.5),
        ("dl:gmlp", 0.5),
        ("dl:resmlp", 0.5),
        ("dl:gmlp_tiny_p16", 0.5),
        ("dl:mobilevit", 0.5),
        ("dl:coatnet", 0.5),
        ("dl:fnet", 0.5),
        ("dl:fnet_tiny_p16", 0.5),
        ("dl:pvt", 0.5),
        ("dl:pvt_tiny_p8", 0.5),
        ("dl:eca_resnet", 0.5),
        ("dl:cbam_resnet", 0.5),
    ],
)
def test_local_vision_zoo_more_arches_smoke(arch_id: str, width_mult: float) -> None:
    from dlhub.vision.local_zoo import build_local_model

    model = build_local_model(arch_id, in_channels=1, num_classes=4, image_size=64, width_mult=float(width_mult))
    model.eval()

    x = torch.zeros(2, 1, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert isinstance(y, torch.Tensor)
    assert tuple(y.shape) == (2, 4)
