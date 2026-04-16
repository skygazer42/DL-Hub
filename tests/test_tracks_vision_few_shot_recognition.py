import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_few_shot_recognition_episode_shapes() -> None:
    from tracks.vision.lesson_65_synthetic_few_shot_recognition.data import (
        DataConfig,
        get_dataloaders,
    )

    train_loader, _ = get_dataloaders(
        DataConfig(
            num_episodes=24,
            batch_size=3,
            num_ways=4,
            shots=2,
            queries_per_class=3,
            image_size=48,
            val_fraction=0.25,
            seed=0,
            num_workers=0,
            in_channels=1,
        )
    )
    batch = next(iter(train_loader))

    assert batch["support_images"].shape == (3, 8, 1, 48, 48)
    assert batch["support_labels"].shape == (3, 8)
    assert batch["query_images"].shape == (3, 12, 1, 48, 48)
    assert batch["query_labels"].shape == (3, 12)
    assert len(batch["class_names"]) == 3
    assert len(batch["class_names"][0]) == 4

    for episode_labels in batch["support_labels"]:
        uniques = torch.unique(episode_labels)
        torch.testing.assert_close(uniques, torch.arange(4, dtype=episode_labels.dtype))


def test_few_shot_recognition_model_forward_and_metrics() -> None:
    from tracks.vision.lesson_65_synthetic_few_shot_recognition.data import (
        DataConfig,
        get_dataloaders,
    )
    from tracks.vision.lesson_65_synthetic_few_shot_recognition.model import (
        ModelConfig,
        PrototypicalFewShotRecognizer,
        episode_accuracy,
        prototypical_loss,
    )

    train_loader, _ = get_dataloaders(
        DataConfig(
            num_episodes=18,
            batch_size=2,
            num_ways=3,
            shots=2,
            queries_per_class=2,
            image_size=48,
            val_fraction=0.25,
            seed=1,
            num_workers=0,
            in_channels=1,
        )
    )
    batch = next(iter(train_loader))
    model = PrototypicalFewShotRecognizer(
        ModelConfig(
            in_channels=1,
            hidden_channels=16,
            embedding_dim=24,
            dropout=0.0,
        )
    )

    outputs = model(batch)
    assert set(outputs) == {"support_embeddings", "query_embeddings", "prototypes", "logits"}
    assert outputs["support_embeddings"].shape == (2, 6, 24)
    assert outputs["query_embeddings"].shape == (2, 6, 24)
    assert outputs["prototypes"].shape == (2, 3, 24)
    assert outputs["logits"].shape == (2, 6, 3)

    loss = prototypical_loss(outputs["logits"], batch["query_labels"])
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    acc = episode_accuracy(outputs["logits"], batch["query_labels"])
    assert 0.0 <= acc <= 1.0


def test_few_shot_recognition_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.vision.lesson_65_synthetic_few_shot_recognition.data import DataConfig
    from tracks.vision.lesson_65_synthetic_few_shot_recognition.train import (
        TrainConfig,
        run_training,
    )

    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path))

    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=2e-3,
            seed=7,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_few_shot_recognition_smoke",
            hidden_channels=16,
            embedding_dim=24,
            dropout=0.0,
        ),
        DataConfig(
            num_episodes=24,
            batch_size=3,
            num_ways=3,
            shots=2,
            queries_per_class=2,
            image_size=48,
            val_fraction=0.25,
            seed=4,
            num_workers=0,
            in_channels=1,
        ),
    )
    assert exit_code == 0

    run_dir = (
        tmp_path
        / "vision"
        / "lesson_65_synthetic_few_shot_recognition"
        / "pytest_few_shot_recognition_smoke"
    )
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "logs" / "train.log").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()

    metric_row = json.loads((run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert metric_row["epoch"] == 1
    assert metric_row["train_loss"] >= 0.0
    assert 0.0 <= metric_row["train_episode_acc"] <= 1.0
    assert metric_row["eval_loss"] >= 0.0
    assert 0.0 <= metric_row["eval_episode_acc"] <= 1.0
