import shutil
import subprocess
import sys
from pathlib import Path

import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_multimodal_reasoning_batch_shapes() -> None:
    from tracks.multimodal.lesson_24_multimodal_reasoning.data import (
        DataConfig,
        get_dataloaders,
    )

    cfg = DataConfig(
        num_samples=64,
        batch_size=8,
        image_size=16,
        max_facts_length=12,
        max_query_length=8,
        val_fraction=0.25,
        seed=0,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(cfg)
    batch = next(iter(train_loader))

    assert batch["image"].shape == (8, 3, 16, 16)
    assert batch["facts_ids"].shape == (8, 12)
    assert batch["facts_mask"].shape == (8, 12)
    assert batch["query_ids"].shape == (8, 8)
    assert batch["query_mask"].shape == (8, 8)
    assert batch["labels"].shape == (8,)
    assert batch["labels"].dtype == torch.long
    assert len(batch["answer_text"]) == 8
    assert len(batch["query_text"]) == 8
    assert "candidate_a" in vocab.token_to_id
    assert "candidate_b" in vocab.token_to_id
    assert "metal" in vocab.token_to_id
    assert "wood" in vocab.token_to_id


def test_multimodal_reasoning_model_outputs() -> None:
    from tracks.multimodal.lesson_24_multimodal_reasoning.data import (
        DataConfig,
        get_dataloaders,
    )
    from tracks.multimodal.lesson_24_multimodal_reasoning.model import (
        MultimodalReasoningConfig,
        ToyMultimodalReasoningModel,
        reasoning_loss,
    )

    data_cfg = DataConfig(
        num_samples=32,
        batch_size=8,
        image_size=16,
        max_facts_length=12,
        max_query_length=8,
        val_fraction=0.25,
        seed=1,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(data_cfg)
    batch = next(iter(train_loader))

    model = ToyMultimodalReasoningModel(
        MultimodalReasoningConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            num_classes=2,
            hidden_dim=48,
            text_dim=32,
            vision_width=32,
        )
    )
    outputs = model(batch)

    assert set(outputs) >= {"logits", "probs"}
    assert outputs["logits"].shape == (8, 2)
    assert outputs["probs"].shape == (8, 2)
    assert torch.allclose(
        outputs["probs"].sum(dim=1),
        torch.ones(8, dtype=outputs["probs"].dtype),
        atol=1e-5,
    )

    loss = reasoning_loss(outputs["logits"], batch["labels"])
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_multimodal_reasoning_training_smoke() -> None:
    run_dir = (
        _repo_root()
        / "outputs"
        / "multimodal"
        / "lesson_24_multimodal_reasoning"
        / "pytest_multimodal_reasoning_smoke"
    )
    if run_dir.exists():
        shutil.rmtree(run_dir)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracks.multimodal.lesson_24_multimodal_reasoning.train",
            "--epochs",
            "1",
            "--num-samples",
            "64",
            "--batch-size",
            "8",
            "--image-size",
            "16",
            "--max-facts-length",
            "12",
            "--max-query-length",
            "8",
            "--max-train-batches",
            "2",
            "--max-eval-batches",
            "1",
            "--device",
            "cpu",
            "--run-name",
            "pytest_multimodal_reasoning_smoke",
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
