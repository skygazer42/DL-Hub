import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_multimodal_owlvit_batch_shapes() -> None:
    from tracks.multimodal.lesson_10_owlvit_toy_open_vocab_detection.data import (
        DataConfig,
        get_dataloaders,
    )

    cfg = DataConfig(
        num_samples=48,
        batch_size=8,
        image_size=32,
        grid_size=4,
        max_text_length=6,
        val_fraction=0.25,
        seed=0,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(cfg)
    batch = next(iter(train_loader))

    assert batch["image"].shape == (8, 3, 32, 32)
    assert batch["query_ids"].shape == (8, 6)
    assert batch["attention_mask"].shape == (8, 6)
    assert batch["target_present"].shape == (8,)
    assert batch["target_cell"].shape == (8,)
    assert batch["target_box"].shape == (8, 4)
    assert batch["target_delta"].shape == (8, 4)
    assert "detect" in vocab.token_to_id
    assert "find" in vocab.token_to_id
    assert "red" in vocab.token_to_id
    assert "square" in vocab.token_to_id
    assert "circle" in vocab.token_to_id


def test_multimodal_owlvit_model_outputs() -> None:
    from tracks.multimodal.lesson_10_owlvit_toy_open_vocab_detection.data import (
        DataConfig,
        get_dataloaders,
    )
    from tracks.multimodal.lesson_10_owlvit_toy_open_vocab_detection.model import (
        OwlVitLossConfig,
        OwlVitModelConfig,
        ToyOwlVitModel,
        owlvit_loss,
    )

    data_cfg = DataConfig(
        num_samples=32,
        batch_size=8,
        image_size=32,
        grid_size=4,
        max_text_length=6,
        val_fraction=0.25,
        seed=1,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(data_cfg)
    batch = next(iter(train_loader))

    model = ToyOwlVitModel(
        OwlVitModelConfig(
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

    assert set(outputs) >= {"presence_logit", "cell_logits", "pred_boxes"}
    assert outputs["presence_logit"].shape == (8,)
    assert outputs["cell_logits"].shape == (8, 16)
    assert outputs["pred_boxes"].shape == (8, 4)

    losses = owlvit_loss(
        presence_logit=outputs["presence_logit"],
        cell_logits=outputs["cell_logits"],
        box_deltas=outputs["box_deltas"],
        target_present=batch["target_present"],
        target_cell=batch["target_cell"],
        target_delta=batch["target_delta"],
        cfg=OwlVitLossConfig(box_weight=2.0),
    )
    assert set(losses) >= {"loss", "presence_loss", "cell_loss", "box_loss"}
    assert losses["loss"].ndim == 0
    assert torch.isfinite(losses["loss"])


def test_multimodal_owlvit_training_smoke() -> None:
    run_dir = (
        _repo_root()
        / "outputs"
        / "multimodal"
        / "lesson_10_owlvit_toy_open_vocab_detection"
        / "pytest_owlvit_smoke"
    )
    if run_dir.exists():
        shutil.rmtree(run_dir)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracks.multimodal.lesson_10_owlvit_toy_open_vocab_detection.train",
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
            "6",
            "--max-train-batches",
            "2",
            "--max-eval-batches",
            "1",
            "--device",
            "cpu",
            "--run-name",
            "pytest_owlvit_smoke",
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
