import pytest

torch = pytest.importorskip("torch")


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    from tracks.vision.synthetic_shapes import DataConfig, get_dataloaders

    train_loader, _ = get_dataloaders(
        DataConfig(
            num_samples=64, batch_size=8, image_size=64, val_fraction=0.2, seed=0, num_workers=0
        )
    )
    return next(iter(train_loader))


@pytest.mark.parametrize(
    "arch",
    [
        "vgg",
        "resnet18",
        "resnet50",
        "resnext50",
        "densenet",
        "squeezenet",
        "mobilenetv2",
        "efficientnetb0",
        "repvgg",
    ],
)
def test_vision_cnn_backbones_forward_loss_backward_smoke(arch: str) -> None:
    from tracks.vision.lesson_09_cnn_backbones_compact_classification.model import (
        ModelConfig,
        build_model,
    )

    x, y = _batch()
    model = build_model(
        ModelConfig(arch=arch, in_channels=1, num_classes=4, width_mult=0.5, dropout=0.0)
    )
    logits = model(x)
    assert tuple(logits.shape) == (8, 4)

    loss = torch.nn.CrossEntropyLoss()(logits, y)
    assert torch.isfinite(loss)
    loss.backward()


def test_vision_torchvision_backbone_forward_loss_backward_smoke() -> None:
    pytest.importorskip("torchvision")

    from tracks.vision.lesson_09_cnn_backbones_compact_classification.model import (
        ModelConfig,
        build_model,
        list_supported_arches,
    )

    arches = list_supported_arches()
    assert "tv:resnet18" in arches

    x, y = _batch()
    model = build_model(
        ModelConfig(arch="tv:resnet18", in_channels=1, num_classes=4, width_mult=1.0, dropout=0.0)
    )
    logits = model(x)
    assert tuple(logits.shape) == (8, 4)

    loss = torch.nn.CrossEntropyLoss()(logits, y)
    assert torch.isfinite(loss)
    loss.backward()


def test_vision_torchvision_quantized_backbone_forward_loss_backward_smoke() -> None:
    torchvision = pytest.importorskip("torchvision")
    from torchvision.models import list_models

    quant_mod = getattr(torchvision.models, "quantization", None)
    if quant_mod is None:
        pytest.skip("torchvision.models.quantization not available")
    if "quantized_resnet18" not in set(list_models(quant_mod)):
        pytest.skip("quantized_resnet18 not available")

    from tracks.vision.lesson_09_cnn_backbones_compact_classification.model import (
        ModelConfig,
        build_model,
    )

    x, y = _batch()
    model = build_model(
        ModelConfig(arch="tvq:resnet18", in_channels=1, num_classes=4, width_mult=1.0, dropout=0.0)
    )
    logits = model(x)
    assert tuple(logits.shape) == (8, 4)

    loss = torch.nn.CrossEntropyLoss()(logits, y)
    assert torch.isfinite(loss)
    loss.backward()


def test_vision_repvgg_switch_to_deploy_keeps_shape() -> None:
    from tracks.vision.lesson_09_cnn_backbones_compact_classification.model import RepVGGClassifier

    x, _ = _batch()
    model = RepVGGClassifier(
        in_channels=1, num_classes=4, width_mult=0.5, dropout=0.0, deploy=False
    )
    out1 = model(x)
    assert tuple(out1.shape) == (8, 4)

    model.switch_to_deploy()
    out2 = model(x)
    assert tuple(out2.shape) == (8, 4)
