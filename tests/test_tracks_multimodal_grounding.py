import shutil
import subprocess
import sys
from pathlib import Path

import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_multimodal_grounding_batch_shapes() -> None:
    from tracks.multimodal.lesson_04_grounding_toy_refexp.data import DataConfig, get_dataloaders

    cfg = DataConfig(
        num_samples=48,
        batch_size=8,
        image_size=32,
        grid_size=4,
        max_text_length=8,
        val_fraction=0.25,
        seed=0,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(cfg)
    batch = next(iter(train_loader))

    assert batch["image"].shape == (8, 3, 32, 32)
    assert batch["input_ids"].shape == (8, 8)
    assert batch["attention_mask"].shape == (8, 8)
    assert batch["target_cell"].shape == (8,)
    assert batch["target_box"].shape == (8, 4)
    assert batch["target_delta"].shape == (8, 4)
    assert "find" in vocab.token_to_id
    assert "locate" in vocab.token_to_id
    assert "object" in vocab.token_to_id
    assert "top" in vocab.token_to_id
    assert "left" in vocab.token_to_id


def test_multimodal_grounding_model_outputs() -> None:
    from tracks.multimodal.lesson_04_grounding_toy_refexp.data import DataConfig, get_dataloaders
    from tracks.multimodal.lesson_04_grounding_toy_refexp.model import (
        GroundingLossConfig,
        GroundingModelConfig,
        ToyGroundingModel,
        grounding_loss,
    )

    data_cfg = DataConfig(
        num_samples=32,
        batch_size=8,
        image_size=32,
        grid_size=4,
        max_text_length=8,
        val_fraction=0.25,
        seed=1,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(data_cfg)
    batch = next(iter(train_loader))

    model = ToyGroundingModel(
        GroundingModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            image_size=data_cfg.image_size,
            grid_size=data_cfg.grid_size,
            hidden_dim=48,
            vision_width=32,
            text_dim=32,
        )
    )
    outputs = model(batch)

    assert set(outputs) >= {"cell_logits", "box_deltas", "pred_boxes"}
    assert outputs["cell_logits"].shape == (8, 16)
    assert outputs["box_deltas"].shape == (8, 16, 4)
    assert outputs["pred_boxes"].shape == (8, 4)

    losses = grounding_loss(
        cell_logits=outputs["cell_logits"],
        box_deltas=outputs["box_deltas"],
        target_cell=batch["target_cell"],
        target_delta=batch["target_delta"],
        cfg=GroundingLossConfig(box_weight=2.0),
    )
    assert set(losses) >= {"loss", "cell_loss", "box_loss"}
    assert losses["loss"].ndim == 0
    assert torch.isfinite(losses["loss"])


def test_multimodal_grounding_training_smoke() -> None:
    run_dir = (
        _repo_root()
        / "outputs"
        / "multimodal"
        / "lesson_04_grounding_toy_refexp"
        / "pytest_grounding_smoke"
    )
    if run_dir.exists():
        shutil.rmtree(run_dir)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracks.multimodal.lesson_04_grounding_toy_refexp.train",
            "--epochs",
            "1",
            "--num-samples",
            "64",
            "--batch-size",
            "8",
            "--image-size",
            "32",
            "--grid-size",
            "4",
            "--max-text-length",
            "8",
            "--max-train-batches",
            "2",
            "--max-eval-batches",
            "1",
            "--device",
            "cpu",
            "--run-name",
            "pytest_grounding_smoke",
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
