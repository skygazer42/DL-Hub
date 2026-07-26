import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_multimodal_pedestrian_attributes_batch_shapes() -> None:
    from tracks.multimodal.lesson_33_pedestrian_attribute_recognition.data import (
        DataConfig,
        get_dataloaders,
    )

    cfg = DataConfig(
        num_samples=40,
        batch_size=8,
        image_size=24,
        max_text_length=10,
        val_fraction=0.25,
        seed=0,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(cfg)
    batch = next(iter(train_loader))

    assert batch["image"].shape == (8, 3, 24, 24)
    assert batch["input_ids"].shape == (8, 10)
    assert batch["attention_mask"].shape == (8, 10)
    assert batch["attribute_id"].shape == (8,)
    assert batch["attribute_id"].dtype == torch.long
    assert len(batch["query_text"]) == 8
    assert len(batch["attribute_tokens"]) == 8
    assert "pedestrian" in vocab.token_to_id
    assert "backpack" in vocab.token_to_id
    assert "hoodie" in vocab.token_to_id


def test_multimodal_pedestrian_attributes_model_outputs() -> None:
    from tracks.multimodal.lesson_33_pedestrian_attribute_recognition.data import (
        DataConfig,
        get_dataloaders,
    )
    from tracks.multimodal.lesson_33_pedestrian_attribute_recognition.model import (
        ModelConfig,
        ToyPedestrianAttributeModel,
        attribute_retrieval_loss,
        retrieval_accuracy,
        recall_at_k,
    )

    data_cfg = DataConfig(
        num_samples=32,
        batch_size=8,
        image_size=24,
        max_text_length=10,
        val_fraction=0.25,
        seed=1,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(data_cfg)
    batch = next(iter(train_loader))

    model = ToyPedestrianAttributeModel(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            max_text_length=data_cfg.max_text_length,
            image_size=data_cfg.image_size,
            embed_dim=32,
            vision_width=32,
            text_width=32,
            init_temperature=0.07,
        )
    )
    outputs = model(batch)

    assert set(outputs) >= {"image_embed", "text_embed", "logits_per_image", "logits_per_text"}
    assert outputs["image_embed"].shape == (8, 32)
    assert outputs["text_embed"].shape == (8, 32)
    assert outputs["logits_per_image"].shape == (8, 8)
    assert outputs["logits_per_text"].shape == (8, 8)

    loss = attribute_retrieval_loss(outputs["logits_per_image"], outputs["logits_per_text"])
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    i2t_acc, t2i_acc = retrieval_accuracy(outputs["logits_per_image"], outputs["logits_per_text"])
    i2t_r3, t2i_r3 = recall_at_k(outputs["logits_per_image"], outputs["logits_per_text"], k=3)
    for metric in (i2t_acc, t2i_acc, i2t_r3, t2i_r3):
        assert 0.0 <= metric <= 1.0


def test_multimodal_pedestrian_attributes_training_smoke() -> None:
    run_dir = (
        _repo_root()
        / "outputs"
        / "multimodal"
        / "lesson_33_pedestrian_attribute_recognition"
        / "pytest_pedestrian_attributes_smoke"
    )
    if run_dir.exists():
        shutil.rmtree(run_dir)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracks.multimodal.lesson_33_pedestrian_attribute_recognition.train",
            "--epochs",
            "1",
            "--num-samples",
            "64",
            "--batch-size",
            "8",
            "--image-size",
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
            "pytest_pedestrian_attributes_smoke",
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
