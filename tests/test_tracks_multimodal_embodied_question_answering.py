import shutil
import subprocess
import sys
from pathlib import Path

import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_multimodal_embodied_qa_batch_shapes() -> None:
    from tracks.multimodal.lesson_23_embodied_question_answering.data import DataConfig, get_dataloaders

    cfg = DataConfig(
        num_samples=48,
        batch_size=8,
        trajectory_len=6,
        image_size=20,
        max_question_length=12,
        val_fraction=0.25,
        seed=0,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(cfg)
    batch = next(iter(train_loader))

    assert batch["trajectory"].shape == (8, 6, 2)
    assert batch["observations"].shape == (8, 6, 3, 20, 20)
    assert batch["question_ids"].shape == (8, 12)
    assert batch["question_mask"].shape == (8, 12)
    assert batch["answer_id"].shape == (8,)
    assert batch["target_step"].shape == (8,)
    assert len(batch["question_text"]) == 8
    assert len(batch["answer_text"]) == 8
    assert "where" in vocab.token_to_id
    assert "goal" in vocab.token_to_id
    assert "left" in vocab.token_to_id
    assert "right" in vocab.token_to_id


def test_multimodal_embodied_qa_model_outputs() -> None:
    from tracks.multimodal.lesson_23_embodied_question_answering.data import DataConfig, get_dataloaders
    from tracks.multimodal.lesson_23_embodied_question_answering.model import (
        EmbodiedQaConfig,
        ToyEmbodiedQaModel,
        eqa_accuracy,
        eqa_loss,
    )

    data_cfg = DataConfig(
        num_samples=32,
        batch_size=8,
        trajectory_len=6,
        image_size=20,
        max_question_length=12,
        val_fraction=0.25,
        seed=1,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(data_cfg)
    batch = next(iter(train_loader))

    model = ToyEmbodiedQaModel(
        EmbodiedQaConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            max_question_length=data_cfg.max_question_length,
            trajectory_len=data_cfg.trajectory_len,
            num_answers=4,
            hidden_dim=48,
            vision_width=32,
            traj_width=24,
            text_width=32,
        )
    )
    outputs = model(batch)

    assert set(outputs) >= {"logits", "fused_state", "nav_state"}
    assert outputs["logits"].shape == (8, 4)
    assert outputs["fused_state"].shape == (8, 48)
    assert outputs["nav_state"].shape == (8, 48)

    loss = eqa_loss(outputs["logits"], batch["answer_id"])
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    acc = eqa_accuracy(outputs["logits"], batch["answer_id"])
    assert 0.0 <= acc <= 1.0


def test_multimodal_embodied_qa_training_smoke() -> None:
    run_dir = (
        _repo_root()
        / "outputs"
        / "multimodal"
        / "lesson_23_embodied_question_answering"
        / "pytest_embodied_qa_smoke"
    )
    if run_dir.exists():
        shutil.rmtree(run_dir)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracks.multimodal.lesson_23_embodied_question_answering.train",
            "--epochs",
            "1",
            "--num-samples",
            "64",
            "--batch-size",
            "8",
            "--trajectory-len",
            "6",
            "--image-size",
            "20",
            "--max-question-length",
            "12",
            "--max-train-batches",
            "2",
            "--max-eval-batches",
            "1",
            "--device",
            "cpu",
            "--run-name",
            "pytest_embodied_qa_smoke",
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
