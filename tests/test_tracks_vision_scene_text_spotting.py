import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_vision_scene_text_spotting_batch_contract_and_loss_smoke() -> None:
    from tracks.vision.lesson_85_synthetic_scene_text_spotting.data import (
        DataConfig,
        SyntheticSceneTextSpottingDataset,
        get_dataloaders,
    )
    from tracks.vision.lesson_85_synthetic_scene_text_spotting.model import (
        ModelConfig,
        SceneTextSpotter,
        scene_text_spotting_loss,
        sequence_word_accuracy,
    )

    cfg = DataConfig(
        num_samples=40,
        batch_size=4,
        image_size=40,
        text_length=4,
        val_fraction=0.25,
        seed=11,
        num_workers=0,
        in_channels=1,
        noise_std=0.02,
    )
    ds = SyntheticSceneTextSpottingDataset(cfg)
    image, target = ds[0]
    assert tuple(image.shape) == (1, 40, 40)
    assert set(target.keys()) == {"score_map", "text_tokens", "first_token"}
    assert tuple(target["score_map"].shape) == (1, 40, 40)
    assert tuple(target["text_tokens"].shape) == (4,)
    assert tuple(target["first_token"].shape) == ()
    assert image.dtype == torch.float32
    assert target["score_map"].dtype == torch.float32
    assert target["text_tokens"].dtype == torch.long
    assert target["first_token"].dtype == torch.long

    train_loader, _ = get_dataloaders(cfg)
    batch_images, batch_targets = next(iter(train_loader))
    assert tuple(batch_images.shape) == (4, 1, 40, 40)
    assert tuple(batch_targets["score_map"].shape) == (4, 1, 40, 40)
    assert tuple(batch_targets["text_tokens"].shape) == (4, 4)
    assert tuple(batch_targets["first_token"].shape) == (4,)

    model = SceneTextSpotter(
        ModelConfig(
            in_channels=1,
            text_length=4,
            hidden_channels=24,
            family="spotter_v1",
            variant="spotter_v1_tiny",
            width_mult=1.0,
        )
    )
    outputs = model(batch_images)
    assert set(outputs.keys()) == {"score_map", "seq_logits", "aux_char_logits"}
    assert tuple(outputs["score_map"].shape) == (4, 1, 40, 40)
    assert tuple(outputs["seq_logits"].shape) == (4, 4, 37)
    assert tuple(outputs["aux_char_logits"].shape) == (4, 37)

    loss, parts = scene_text_spotting_loss(outputs, batch_targets)
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"det_loss", "rec_loss", "aux_loss"}
    assert float(parts["det_loss"]) >= 0.0
    assert float(parts["rec_loss"]) >= 0.0
    assert float(parts["aux_loss"]) >= 0.0
    assert 0.0 <= sequence_word_accuracy(outputs["seq_logits"], batch_targets["text_tokens"]) <= 1.0
    loss.backward()


def test_vision_scene_text_spotting_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.vision.lesson_85_synthetic_scene_text_spotting.data import DataConfig
    from tracks.vision.lesson_85_synthetic_scene_text_spotting.model import ModelConfig
    from tracks.vision.lesson_85_synthetic_scene_text_spotting.train import (
        TrainConfig,
        run_training,
    )

    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path))

    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=1e-3,
            seed=42,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_scene_text_spotting_smoke",
        ),
        DataConfig(
            num_samples=48,
            batch_size=4,
            image_size=40,
            text_length=4,
            val_fraction=0.25,
            seed=3,
            num_workers=0,
            in_channels=1,
            noise_std=0.02,
        ),
        ModelConfig(
            in_channels=1,
            text_length=4,
            hidden_channels=24,
            family="spotter_v1",
            variant="spotter_v1_tiny",
            width_mult=1.0,
        ),
    )
    assert exit_code == 0

    run_dir = (
        tmp_path
        / "vision"
        / "lesson_85_synthetic_scene_text_spotting"
        / "pytest_scene_text_spotting_smoke"
    )
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
    record = metrics[0]
    for key in (
        "train_loss",
        "train_det_loss",
        "train_rec_loss",
        "train_aux_loss",
        "train_word_acc",
        "eval_loss",
        "eval_det_loss",
        "eval_rec_loss",
        "eval_aux_loss",
        "eval_word_acc",
    ):
        assert key in record
        assert float(record[key]) >= 0.0
