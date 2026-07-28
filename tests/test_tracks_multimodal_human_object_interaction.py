import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_multimodal_human_object_interaction_batch_shapes() -> None:
    from tracks.multimodal.lesson_29_human_object_interaction_reasoning.data import (
        DataConfig,
        get_dataloaders,
    )

    cfg = DataConfig(
        num_samples=48,
        batch_size=8,
        num_regions=6,
        feature_dim=16,
        max_query_length=10,
        val_fraction=0.25,
        seed=0,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(cfg)
    batch = next(iter(train_loader))

    assert batch["region_features"].shape == (8, 6, 16)
    assert batch["region_boxes"].shape == (8, 6, 4)
    assert batch["query_ids"].shape == (8, 10)
    assert batch["query_mask"].shape == (8, 10)
    assert batch["labels"].shape == (8,)
    assert batch["labels"].dtype == torch.long
    assert len(batch["query_text"]) == 8
    assert len(batch["answer_text"]) == 8
    assert "person" in vocab.token_to_id
    assert "holding" in vocab.token_to_id
    assert "cup" in vocab.token_to_id
    assert "book" in vocab.token_to_id


def test_multimodal_human_object_interaction_model_outputs() -> None:
    from tracks.multimodal.lesson_29_human_object_interaction_reasoning.data import (
        DataConfig,
        get_dataloaders,
    )
    from tracks.multimodal.lesson_29_human_object_interaction_reasoning.model import (
        HoiReasoningConfig,
        CompactHoiReasoningModel,
        hoi_accuracy,
        hoi_loss,
    )

    data_cfg = DataConfig(
        num_samples=32,
        batch_size=8,
        num_regions=6,
        feature_dim=16,
        max_query_length=10,
        val_fraction=0.25,
        seed=1,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(data_cfg)
    batch = next(iter(train_loader))

    model = CompactHoiReasoningModel(
        HoiReasoningConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            num_classes=2,
            feature_dim=16,
            text_dim=32,
            hidden_dim=48,
        )
    )
    outputs = model(batch)

    assert set(outputs) >= {"logits", "probabilities"}
    assert outputs["logits"].shape == (8, 2)
    assert outputs["probabilities"].shape == (8, 2)
    assert torch.allclose(
        outputs["probabilities"].sum(dim=1),
        torch.ones(8, dtype=outputs["probabilities"].dtype),
        atol=1e-5,
    )

    loss = hoi_loss(outputs["logits"], batch["labels"])
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    acc = hoi_accuracy(outputs["logits"], batch["labels"])
    assert 0.0 <= acc <= 1.0


def test_multimodal_human_object_interaction_training_smoke() -> None:
    run_dir = (
        _repo_root()
        / "outputs"
        / "multimodal"
        / "lesson_29_human_object_interaction_reasoning"
        / "pytest_hoi_reasoning_smoke"
    )
    if run_dir.exists():
        shutil.rmtree(run_dir)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracks.multimodal.lesson_29_human_object_interaction_reasoning.train",
            "--epochs",
            "1",
            "--num-samples",
            "64",
            "--batch-size",
            "8",
            "--num-regions",
            "6",
            "--feature-dim",
            "16",
            "--max-query-length",
            "10",
            "--max-train-batches",
            "2",
            "--max-eval-batches",
            "1",
            "--device",
            "cpu",
            "--run-name",
            "pytest_hoi_reasoning_smoke",
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
