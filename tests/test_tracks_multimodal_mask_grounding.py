import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_multimodal_mask_grounding_batch_shapes() -> None:
    from tracks.multimodal.lesson_05_mask_grounding_compact_refexp.data import (
        DataConfig,
        get_dataloaders,
    )

    cfg = DataConfig(
        num_samples=48,
        batch_size=8,
        image_size=32,
        mask_size=8,
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
    assert batch["target_mask"].shape == (8, 1, 8, 8)
    assert "segment" in vocab.token_to_id
    assert "mask" in vocab.token_to_id
    assert "highlight" in vocab.token_to_id
    assert "top" in vocab.token_to_id
    assert "left" in vocab.token_to_id


def test_multimodal_mask_grounding_model_outputs() -> None:
    from tracks.multimodal.lesson_05_mask_grounding_compact_refexp.data import (
        DataConfig,
        get_dataloaders,
    )
    from tracks.multimodal.lesson_05_mask_grounding_compact_refexp.model import (
        MaskGroundingLossConfig,
        MaskGroundingModelConfig,
        CompactMaskGroundingModel,
        mask_grounding_loss,
    )

    data_cfg = DataConfig(
        num_samples=32,
        batch_size=8,
        image_size=32,
        mask_size=8,
        max_text_length=8,
        val_fraction=0.25,
        seed=1,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(data_cfg)
    batch = next(iter(train_loader))

    model = CompactMaskGroundingModel(
        MaskGroundingModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            image_size=data_cfg.image_size,
            mask_size=data_cfg.mask_size,
            hidden_dim=48,
            vision_width=32,
            text_dim=32,
        )
    )
    outputs = model(batch)

    assert set(outputs) >= {"mask_logits", "pred_mask"}
    assert outputs["mask_logits"].shape == (8, 1, 8, 8)
    assert outputs["pred_mask"].shape == (8, 1, 8, 8)

    losses = mask_grounding_loss(
        mask_logits=outputs["mask_logits"],
        target_mask=batch["target_mask"],
        cfg=MaskGroundingLossConfig(dice_weight=1.0),
    )
    assert set(losses) >= {"loss", "bce_loss", "dice_loss"}
    assert losses["loss"].ndim == 0
    assert torch.isfinite(losses["loss"])


def test_multimodal_mask_grounding_training_smoke() -> None:
    run_dir = (
        _repo_root()
        / "outputs"
        / "multimodal"
        / "lesson_05_mask_grounding_compact_refexp"
        / "pytest_mask_grounding_smoke"
    )
    if run_dir.exists():
        shutil.rmtree(run_dir)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracks.multimodal.lesson_05_mask_grounding_compact_refexp.train",
            "--epochs",
            "1",
            "--num-samples",
            "64",
            "--batch-size",
            "8",
            "--image-size",
            "32",
            "--mask-size",
            "8",
            "--max-text-length",
            "8",
            "--max-train-batches",
            "2",
            "--max-eval-batches",
            "1",
            "--device",
            "cpu",
            "--run-name",
            "pytest_mask_grounding_smoke",
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
