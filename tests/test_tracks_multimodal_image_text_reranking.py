import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_multimodal_image_text_reranking_batch_shapes() -> None:
    from tracks.multimodal.lesson_26_image_text_reranking.data import DataConfig, get_dataloaders

    cfg = DataConfig(
        num_samples=40,
        batch_size=8,
        image_size=20,
        num_candidates=4,
        max_text_length=10,
        val_fraction=0.25,
        seed=0,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(cfg)
    batch = next(iter(train_loader))

    assert batch["image"].shape == (8, 3, 20, 20)
    assert batch["candidate_input_ids"].shape == (8, 4, 10)
    assert batch["candidate_attention_mask"].shape == (8, 4, 10)
    assert batch["label_index"].shape == (8,)
    assert batch["label_index"].dtype == torch.long
    assert len(batch["query_text"]) == 8
    assert len(batch["candidate_texts"]) == 8
    assert "photo" in vocab.token_to_id
    assert "striped" in vocab.token_to_id
    assert "circle" in vocab.token_to_id


def test_multimodal_image_text_reranking_model_outputs() -> None:
    from tracks.multimodal.lesson_26_image_text_reranking.data import DataConfig, get_dataloaders
    from tracks.multimodal.lesson_26_image_text_reranking.model import (
        ImageTextRerankerConfig,
        CompactImageTextReranker,
        reranking_accuracy,
        reranking_loss,
    )

    data_cfg = DataConfig(
        num_samples=32,
        batch_size=8,
        image_size=20,
        num_candidates=5,
        max_text_length=10,
        val_fraction=0.25,
        seed=1,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(data_cfg)
    batch = next(iter(train_loader))

    model = CompactImageTextReranker(
        ImageTextRerankerConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            num_candidates=data_cfg.num_candidates,
            max_text_length=data_cfg.max_text_length,
            embed_dim=32,
            vision_width=32,
            text_width=32,
            hidden_dim=48,
        )
    )
    outputs = model(batch)

    assert set(outputs) >= {"scores", "image_embed", "candidate_embed", "probabilities"}
    assert outputs["scores"].shape == (8, 5)
    assert outputs["image_embed"].shape == (8, 32)
    assert outputs["candidate_embed"].shape == (8, 5, 32)
    assert outputs["probabilities"].shape == (8, 5)
    assert torch.allclose(
        outputs["probabilities"].sum(dim=1),
        torch.ones(8, dtype=outputs["probabilities"].dtype),
        atol=1e-5,
    )

    loss = reranking_loss(outputs["scores"], batch["label_index"])
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    accuracy = reranking_accuracy(outputs["scores"], batch["label_index"])
    assert 0.0 <= accuracy <= 1.0


def test_multimodal_image_text_reranking_training_smoke() -> None:
    run_dir = (
        _repo_root()
        / "outputs"
        / "multimodal"
        / "lesson_26_image_text_reranking"
        / "pytest_image_text_reranking_smoke"
    )
    if run_dir.exists():
        shutil.rmtree(run_dir)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracks.multimodal.lesson_26_image_text_reranking.train",
            "--epochs",
            "1",
            "--num-samples",
            "64",
            "--batch-size",
            "8",
            "--image-size",
            "20",
            "--num-candidates",
            "5",
            "--max-text-length",
            "10",
            "--max-train-batches",
            "2",
            "--max-eval-batches",
            "1",
            "--device",
            "cpu",
            "--run-name",
            "pytest_image_text_reranking_smoke",
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
