import shutil
import subprocess
import sys
from pathlib import Path

import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_multimodal_video_vlm_batch_shapes() -> None:
    from tracks.multimodal.lesson_13_video_vlm_toy_temporal_qa.data import (
        DataConfig,
        get_dataloaders,
    )

    cfg = DataConfig(
        num_samples=48,
        batch_size=8,
        seq_len=4,
        image_size=20,
        max_text_length=16,
        val_fraction=0.25,
        seed=0,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(cfg)
    batch = next(iter(train_loader))

    assert batch["video"].shape == (8, 4, 3, 20, 20)
    assert batch["prompt_ids"].shape == (8, 16)
    assert batch["input_ids"].shape == (8, 16)
    assert batch["labels"].shape == (8, 16)
    assert batch["attention_mask"].shape == (8, 16)
    assert len(batch["task_type"]) == 8
    assert "what" in vocab.token_to_id
    assert "color" in vocab.token_to_id
    assert "shape" in vocab.token_to_id
    assert "moving" in vocab.token_to_id
    assert "left" in vocab.token_to_id
    assert "yes" in vocab.token_to_id
    assert "no" in vocab.token_to_id


def test_multimodal_video_vlm_model_outputs() -> None:
    from tracks.multimodal.lesson_13_video_vlm_toy_temporal_qa.data import (
        DataConfig,
        get_dataloaders,
    )
    from tracks.multimodal.lesson_13_video_vlm_toy_temporal_qa.model import (
        ToyVideoVlmModel,
        VideoVlmModelConfig,
        qa_loss,
    )

    data_cfg = DataConfig(
        num_samples=32,
        batch_size=8,
        seq_len=4,
        image_size=20,
        max_text_length=16,
        val_fraction=0.25,
        seed=1,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(data_cfg)
    batch = next(iter(train_loader))

    model = ToyVideoVlmModel(
        VideoVlmModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            bos_id=vocab.bos_id,
            eos_id=vocab.eos_id,
            sep_id=vocab.sep_id,
            max_text_length=data_cfg.max_text_length,
            seq_len=data_cfg.seq_len,
            hidden_dim=48,
            vision_width=32,
            embed_dim=32,
        )
    )
    outputs = model(batch)

    assert set(outputs) >= {"logits", "video_tokens"}
    assert outputs["logits"].shape == (8, 16, vocab.size)
    assert outputs["video_tokens"].shape[0] == 8
    assert outputs["video_tokens"].shape[-1] == 48

    loss = qa_loss(outputs["logits"], batch["labels"])
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_multimodal_video_vlm_training_smoke() -> None:
    run_dir = (
        _repo_root()
        / "outputs"
        / "multimodal"
        / "lesson_13_video_vlm_toy_temporal_qa"
        / "pytest_video_vlm_smoke"
    )
    if run_dir.exists():
        shutil.rmtree(run_dir)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracks.multimodal.lesson_13_video_vlm_toy_temporal_qa.train",
            "--epochs",
            "1",
            "--num-samples",
            "64",
            "--batch-size",
            "8",
            "--seq-len",
            "4",
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
            "pytest_video_vlm_smoke",
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
