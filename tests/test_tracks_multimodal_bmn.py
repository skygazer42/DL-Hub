import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_multimodal_bmn_batch_shapes() -> None:
    from tracks.multimodal.lesson_14_bmn_toy_temporal_grounding.data import (
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
    assert batch["start_labels"].shape == (8, 8)
    assert batch["end_labels"].shape == (8, 8)
    assert batch["proposal_labels"].shape == (8, 8, 8)
    assert batch["proposal_mask"].shape == (8, 8, 8)
    assert batch["segment"].shape == (8, 2)
    assert len(batch["query_text"]) == 8
    assert len(batch["event_type"]) == 8
    assert "when" in vocab.token_to_id
    assert "does" in vocab.token_to_id
    assert "move" in vocab.token_to_id
    assert "left" in vocab.token_to_id
    assert "right" in vocab.token_to_id
    assert "flash" in vocab.token_to_id


def test_multimodal_bmn_model_outputs() -> None:
    from tracks.multimodal.lesson_14_bmn_toy_temporal_grounding.data import (
        DataConfig,
        get_dataloaders,
    )
    from tracks.multimodal.lesson_14_bmn_toy_temporal_grounding.model import (
        BmnModelConfig,
        ToyBmnTemporalGroundingModel,
        temporal_grounding_loss,
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

    model = ToyBmnTemporalGroundingModel(
        BmnModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            num_frames=data_cfg.num_frames,
            hidden_dim=48,
            vision_width=32,
            text_dim=32,
        )
    )
    outputs = model(batch)

    assert set(outputs) >= {"start_logits", "end_logits", "proposal_scores", "pred_segments"}
    assert outputs["start_logits"].shape == (8, 8)
    assert outputs["end_logits"].shape == (8, 8)
    assert outputs["proposal_scores"].shape == (8, 8, 8)
    assert outputs["pred_segments"].shape == (8, 2)

    losses = temporal_grounding_loss(
        start_logits=outputs["start_logits"],
        end_logits=outputs["end_logits"],
        proposal_scores=outputs["proposal_scores"],
        start_labels=batch["start_labels"],
        end_labels=batch["end_labels"],
        proposal_labels=batch["proposal_labels"],
        proposal_mask=batch["proposal_mask"],
    )
    assert set(losses) >= {"loss", "start_loss", "end_loss", "proposal_loss"}
    assert losses["loss"].ndim == 0
    assert torch.isfinite(losses["loss"])


def test_multimodal_bmn_training_smoke() -> None:
    run_dir = (
        _repo_root()
        / "outputs"
        / "multimodal"
        / "lesson_14_bmn_toy_temporal_grounding"
        / "pytest_bmn_smoke"
    )
    if run_dir.exists():
        shutil.rmtree(run_dir)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracks.multimodal.lesson_14_bmn_toy_temporal_grounding.train",
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
            "pytest_bmn_smoke",
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
