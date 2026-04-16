import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_face_parsing_batch_contract_and_loss_smoke() -> None:
    from tracks.vision.lesson_37_synthetic_face_parsing.data import (
        DataConfig,
        SyntheticFaceParsingDataset,
        get_dataloaders,
    )
    from tracks.vision.lesson_37_synthetic_face_parsing.model import (
        FaceParsingConfig,
        FaceParsingSegmenter,
        mean_iou,
        parsing_loss,
    )

    cfg = DataConfig(
        num_samples=32,
        batch_size=4,
        image_size=48,
        val_fraction=0.2,
        seed=0,
        num_workers=0,
        num_classes=6,
    )

    dataset = SyntheticFaceParsingDataset(cfg)
    image, mask = dataset[0]
    assert tuple(image.shape) == (1, 48, 48)
    assert tuple(mask.shape) == (48, 48)
    assert mask.dtype == torch.long
    assert int(mask.max().item()) < 6

    train_loader, _ = get_dataloaders(cfg)
    images, masks = next(iter(train_loader))
    assert tuple(images.shape) == (4, 1, 48, 48)
    assert tuple(masks.shape) == (4, 48, 48)

    model = FaceParsingSegmenter(FaceParsingConfig(in_channels=1, hidden_channels=24, num_classes=6))
    logits = model(images)
    assert tuple(logits.shape) == (4, 6, 48, 48)

    loss = parsing_loss(logits, masks)
    assert torch.isfinite(loss)
    assert 0.0 <= mean_iou(logits.detach(), masks, num_classes=6) <= 1.0
    loss.backward()


def test_face_parsing_training_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tracks.vision.lesson_37_synthetic_face_parsing.data import DataConfig
    from tracks.vision.lesson_37_synthetic_face_parsing.model import FaceParsingConfig
    from tracks.vision.lesson_37_synthetic_face_parsing.train import TrainConfig, run_training

    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path))
    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=1e-3,
            seed=42,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_face_parsing_smoke",
        ),
        DataConfig(
            num_samples=48,
            batch_size=4,
            image_size=48,
            val_fraction=0.2,
            seed=5,
            num_workers=0,
            num_classes=6,
        ),
        FaceParsingConfig(in_channels=1, hidden_channels=24, num_classes=6),
    )

    assert exit_code == 0
    run_dir = tmp_path / "vision" / "lesson_37_synthetic_face_parsing" / "pytest_face_parsing_smoke"
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "logs" / "train.log").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()

    metrics = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(metrics) == 1
    assert metrics[0]["epoch"] == 1
