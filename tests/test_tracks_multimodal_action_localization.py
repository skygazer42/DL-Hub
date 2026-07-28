import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_multimodal_action_localization_batch_shapes() -> None:
    from tracks.multimodal.lesson_32_video_text_action_localization.data import (
        DataConfig,
        get_dataloaders,
    )

    cfg = DataConfig(
        num_samples=48,
        batch_size=8,
        num_frames=10,
        feature_dim=24,
        max_text_length=10,
        val_fraction=0.25,
        seed=0,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(cfg)
    batch = next(iter(train_loader))

    assert batch["video_features"].shape == (8, 10, 24)
    assert batch["query_ids"].shape == (8, 10)
    assert batch["attention_mask"].shape == (8, 10)
    assert batch["action_mask"].shape == (8, 10)
    assert batch["segment"].shape == (8, 2)
    assert len(batch["query_text"]) == 8
    assert len(batch["action_type"]) == 8
    assert "locate" in vocab.token_to_id
    assert "action" in vocab.token_to_id
    assert "start" in vocab.token_to_id
    assert "end" in vocab.token_to_id


def test_multimodal_action_localization_model_outputs() -> None:
    from tracks.multimodal.lesson_32_video_text_action_localization.data import (
        DataConfig,
        get_dataloaders,
    )
    from tracks.multimodal.lesson_32_video_text_action_localization.model import (
        CompactActionLocalizationModel,
        ActionLocalizationModelConfig,
        action_localization_loss,
    )

    data_cfg = DataConfig(
        num_samples=32,
        batch_size=8,
        num_frames=10,
        feature_dim=24,
        max_text_length=10,
        val_fraction=0.25,
        seed=1,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(data_cfg)
    batch = next(iter(train_loader))

    model = CompactActionLocalizationModel(
        ActionLocalizationModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            num_frames=data_cfg.num_frames,
            feature_dim=data_cfg.feature_dim,
            hidden_dim=40,
            text_dim=32,
        )
    )
    outputs = model(batch)

    assert set(outputs) >= {"mask_logits", "pred_segments"}
    assert outputs["mask_logits"].shape == (8, 10)
    assert outputs["pred_segments"].shape == (8, 2)

    losses = action_localization_loss(
        mask_logits=outputs["mask_logits"],
        action_mask=batch["action_mask"],
    )
    assert set(losses) >= {"loss", "mask_loss"}
    assert losses["loss"].ndim == 0
    assert torch.isfinite(losses["loss"])


def test_multimodal_action_localization_training_smoke() -> None:
    run_dir = (
        _repo_root()
        / "outputs"
        / "multimodal"
        / "lesson_32_video_text_action_localization"
        / "pytest_action_localization_smoke"
    )
    if run_dir.exists():
        shutil.rmtree(run_dir)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracks.multimodal.lesson_32_video_text_action_localization.train",
            "--epochs",
            "1",
            "--num-samples",
            "64",
            "--batch-size",
            "8",
            "--num-frames",
            "10",
            "--feature-dim",
            "24",
            "--max-text-length",
            "10",
            "--max-train-batches",
            "2",
            "--max-eval-batches",
            "1",
            "--device",
            "cpu",
            "--run-name",
            "pytest_action_localization_smoke",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()
