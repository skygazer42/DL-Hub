import json
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")


def test_multimodal_finger_spread_reasoning_batch_shapes() -> None:
    from tracks.multimodal.lesson_57_finger_spread_vlm_reasoning.data import DataConfig, get_dataloaders

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
    assert batch["target_finger_spread"].shape == (8, 1)
    assert batch["target_finger_spread"].dtype == torch.float32
    assert torch.all(batch["target_finger_spread"] >= 0.0)
    assert torch.all(batch["target_finger_spread"] <= 1.0)
    assert len(batch["query_text"]) == 8
    assert "estimate" in vocab.token_to_id
    assert "finger" in vocab.token_to_id
    assert "spread" in vocab.token_to_id
    assert "angle" in vocab.token_to_id


def test_multimodal_finger_spread_reasoning_model_outputs() -> None:
    from tracks.multimodal.lesson_57_finger_spread_vlm_reasoning.data import DataConfig, get_dataloaders
    from tracks.multimodal.lesson_57_finger_spread_vlm_reasoning.model import (
        FingerSpreadReasoningConfig,
        CompactFingerSpreadReasoningModel,
        compute_mae,
        finger_spread_loss,
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

    model = CompactFingerSpreadReasoningModel(
        FingerSpreadReasoningConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            hidden_dim=72,
            text_dim=32,
            vision_width=40,
        )
    )
    outputs = model(batch)

    assert set(outputs) >= {"prediction"}
    assert outputs["prediction"].shape == (8, 1)

    loss = finger_spread_loss(outputs["prediction"], batch["target_finger_spread"])
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    mae = compute_mae(outputs["prediction"], batch["target_finger_spread"])
    assert 0.0 <= mae <= 1.0


def test_multimodal_finger_spread_reasoning_training_smoke(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.multimodal.lesson_57_finger_spread_vlm_reasoning.data import DataConfig
    from tracks.multimodal.lesson_57_finger_spread_vlm_reasoning.train import TrainConfig, run_training

    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path / "outputs"))
    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=1e-3,
            weight_decay=1e-4,
            seed=57,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_finger_spread_reasoning_smoke",
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
        / "lesson_57_finger_spread_vlm_reasoning"
        / "pytest_finger_spread_reasoning_smoke"
    )
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "vocab.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()

    metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(metrics) == 1
    assert metrics[0]["epoch"] == 1
    for key in ("train_loss", "train_mae", "eval_loss", "eval_mae"):
        assert key in metrics[0]
        assert float(metrics[0][key]) >= 0.0


def test_multimodal_finger_spread_reasoning_dry_run() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_57_finger_spread_vlm_reasoning",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "tracks.multimodal.lesson_57_finger_spread_vlm_reasoning.train" in proc.stdout
