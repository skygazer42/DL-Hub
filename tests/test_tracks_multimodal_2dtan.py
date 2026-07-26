import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_multimodal_2dtan_batch_shapes() -> None:
    from tracks.multimodal.lesson_15_2dtan_toy_temporal_grounding.data import (
        DataConfig,
        get_dataloaders,
    )

    cfg = DataConfig(
        num_samples=48,
        batch_size=8,
        num_frames=8,
        image_size=20,
        max_text_length=16,
        val_fraction=0.25,
        seed=0,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(cfg)
    batch = next(iter(train_loader))

    assert batch["video"].shape == (8, 8, 3, 20, 20)
    assert batch["query_ids"].shape == (8, 16)
    assert batch["attention_mask"].shape == (8, 16)
    assert batch["map_labels"].shape == (8, 8, 8)
    assert batch["map_mask"].shape == (8, 8, 8)
    assert batch["segment"].shape == (8, 2)
    assert len(batch["query_text"]) == 8
    assert len(batch["event_type"]) == 8
    assert "when" in vocab.token_to_id
    assert "does" in vocab.token_to_id
    assert "move" in vocab.token_to_id
    assert "left" in vocab.token_to_id
    assert "right" in vocab.token_to_id
    assert "flash" in vocab.token_to_id


def test_multimodal_2dtan_model_outputs() -> None:
    from tracks.multimodal.lesson_15_2dtan_toy_temporal_grounding.data import (
        DataConfig,
        get_dataloaders,
    )
    from tracks.multimodal.lesson_15_2dtan_toy_temporal_grounding.model import (
        TwoDtanModelConfig,
        ToyTwoDtanTemporalGroundingModel,
        temporal_map_loss,
    )

    data_cfg = DataConfig(
        num_samples=32,
        batch_size=8,
        num_frames=8,
        image_size=20,
        max_text_length=16,
        val_fraction=0.25,
        seed=1,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(data_cfg)
    batch = next(iter(train_loader))

    model = ToyTwoDtanTemporalGroundingModel(
        TwoDtanModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            num_frames=data_cfg.num_frames,
            hidden_dim=48,
            vision_width=32,
            text_dim=32,
        )
    )
    outputs = model(batch)

    assert set(outputs) >= {"score_map", "map_features", "pred_segments"}
    assert outputs["score_map"].shape == (8, 8, 8)
    assert outputs["map_features"].shape == (8, 8, 8, 48)
    assert outputs["pred_segments"].shape == (8, 2)

    losses = temporal_map_loss(
        score_map=outputs["score_map"],
        map_labels=batch["map_labels"],
        map_mask=batch["map_mask"],
    )
    assert set(losses) >= {"loss", "map_loss"}
    assert losses["loss"].ndim == 0
    assert torch.isfinite(losses["loss"])


def test_multimodal_2dtan_training_smoke() -> None:
    run_dir = (
        _repo_root()
        / "outputs"
        / "multimodal"
        / "lesson_15_2dtan_toy_temporal_grounding"
        / "pytest_2dtan_smoke"
    )
    if run_dir.exists():
        shutil.rmtree(run_dir)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracks.multimodal.lesson_15_2dtan_toy_temporal_grounding.train",
            "--epochs",
            "1",
            "--num-samples",
            "64",
            "--batch-size",
            "8",
            "--num-frames",
            "8",
            "--image-size",
            "20",
            "--max-text-length",
            "16",
            "--max-train-batches",
            "2",
            "--max-eval-batches",
            "1",
            "--device",
            "cpu",
            "--run-name",
            "pytest_2dtan_smoke",
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
