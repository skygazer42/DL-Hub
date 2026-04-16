import json
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")


def test_multimodal_thumb_position_reasoning_batch_shapes() -> None:
    from tracks.multimodal.lesson_58_thumb_position_vlm_reasoning.data import DataConfig, get_dataloaders

    cfg = DataConfig(
        num_samples=48,
        batch_size=8,
        image_size=64,
        max_text_length=16,
        val_fraction=0.25,
        seed=0,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(cfg)
    batch = next(iter(train_loader))

    assert batch["image"].shape == (8, 1, 64, 64)
    assert batch["query_ids"].shape == (8, 16)
    assert batch["query_mask"].shape == (8, 16)
    assert batch["target_thumb_position"].shape == (8,)
    assert batch["target_thumb_position"].dtype == torch.long
    assert torch.all(batch["target_thumb_position"] >= 0)
    assert torch.all(batch["target_thumb_position"] < 3)
    assert len(batch["query_text"]) == 8
    for token in ("estimate", "thumb", "position", "low", "middle", "high", "query"):
        assert token in vocab.token_to_id


def test_multimodal_thumb_position_reasoning_model_outputs() -> None:
    from tracks.multimodal.lesson_58_thumb_position_vlm_reasoning.data import DataConfig, get_dataloaders
    from tracks.multimodal.lesson_58_thumb_position_vlm_reasoning.model import (
        ThumbPositionReasoningConfig,
        ToyThumbPositionReasoningModel,
        compute_accuracy,
        thumb_position_loss,
    )

    data_cfg = DataConfig(
        num_samples=32,
        batch_size=8,
        image_size=64,
        max_text_length=16,
        val_fraction=0.25,
        seed=1,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(data_cfg)
    batch = next(iter(train_loader))

    model = ToyThumbPositionReasoningModel(
        ThumbPositionReasoningConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            num_classes=3,
            hidden_dim=72,
            text_dim=32,
            vision_width=40,
        )
    )
    outputs = model(batch)

    assert set(outputs) >= {"logits"}
    assert outputs["logits"].shape == (8, 3)

    loss = thumb_position_loss(outputs["logits"], batch["target_thumb_position"])
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    acc = compute_accuracy(outputs["logits"], batch["target_thumb_position"])
    assert 0.0 <= acc <= 1.0


def test_multimodal_thumb_position_reasoning_training_smoke(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.multimodal.lesson_58_thumb_position_vlm_reasoning.data import DataConfig
    from tracks.multimodal.lesson_58_thumb_position_vlm_reasoning.train import TrainConfig, run_training

    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path / "outputs"))
    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=1e-3,
            weight_decay=1e-4,
            seed=58,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_thumb_position_reasoning_smoke",
            hidden_dim=72,
            text_dim=32,
            vision_width=40,
        ),
        DataConfig(
            num_samples=64,
            batch_size=8,
            image_size=64,
            max_text_length=16,
            val_fraction=0.25,
            seed=7,
            num_workers=0,
        ),
    )

    assert exit_code == 0
    run_dir = (
        tmp_path
        / "outputs"
        / "multimodal"
        / "lesson_58_thumb_position_vlm_reasoning"
        / "pytest_thumb_position_reasoning_smoke"
    )
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "vocab.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()

    metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(metrics) == 1
    assert metrics[0]["epoch"] == 1
    for key in ("train_loss", "train_accuracy", "eval_loss", "eval_accuracy"):
        assert key in metrics[0]
        assert float(metrics[0][key]) >= 0.0


def test_multimodal_thumb_position_reasoning_dry_run() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_58_thumb_position_vlm_reasoning",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "tracks.multimodal.lesson_58_thumb_position_vlm_reasoning.train" in proc.stdout
