import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_multimodal_scene_text_recognition_batch_shapes() -> None:
    from tracks.multimodal.lesson_27_scene_text_vlm_recognition.data import (
        DataConfig,
        get_dataloaders,
    )

    cfg = DataConfig(
        num_samples=40,
        batch_size=8,
        image_size=24,
        max_text_length=12,
        val_fraction=0.25,
        seed=0,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(cfg)
    batch = next(iter(train_loader))

    assert batch["image"].shape == (8, 3, 24, 24)
    assert batch["prompt_ids"].shape == (8, 12)
    assert batch["prompt_mask"].shape == (8, 12)
    assert batch["label_ids"].shape == (8,)
    assert batch["label_ids"].dtype == torch.long
    assert len(batch["scene_text"]) == 8
    assert len(batch["prompt_text"]) == 8
    assert "read" in vocab.token_to_id
    assert "text" in vocab.token_to_id
    assert "alpha" in vocab.token_to_id
    assert "beta" in vocab.token_to_id


def test_multimodal_scene_text_recognition_model_outputs() -> None:
    from tracks.multimodal.lesson_27_scene_text_vlm_recognition.data import (
        DataConfig,
        get_dataloaders,
    )
    from tracks.multimodal.lesson_27_scene_text_vlm_recognition.model import (
        SceneTextRecognizerConfig,
        CompactSceneTextRecognizer,
        recognition_accuracy,
        recognition_loss,
    )

    data_cfg = DataConfig(
        num_samples=32,
        batch_size=8,
        image_size=24,
        max_text_length=12,
        val_fraction=0.25,
        seed=1,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(data_cfg)
    batch = next(iter(train_loader))

    model = CompactSceneTextRecognizer(
        SceneTextRecognizerConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            num_words=4,
            hidden_dim=48,
            vision_width=32,
            text_width=32,
        )
    )
    outputs = model(batch)

    assert set(outputs) >= {"logits", "probs"}
    assert outputs["logits"].shape == (8, 4)
    assert outputs["probs"].shape == (8, 4)
    assert torch.allclose(
        outputs["probs"].sum(dim=1),
        torch.ones(8, dtype=outputs["probs"].dtype),
        atol=1e-5,
    )

    loss = recognition_loss(outputs["logits"], batch["label_ids"])
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    acc = recognition_accuracy(outputs["logits"], batch["label_ids"])
    assert 0.0 <= acc <= 1.0


def test_multimodal_scene_text_recognition_training_smoke() -> None:
    run_dir = (
        _repo_root()
        / "outputs"
        / "multimodal"
        / "lesson_27_scene_text_vlm_recognition"
        / "pytest_scene_text_recognition_smoke"
    )
    if run_dir.exists():
        shutil.rmtree(run_dir)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracks.multimodal.lesson_27_scene_text_vlm_recognition.train",
            "--epochs",
            "1",
            "--num-samples",
            "64",
            "--batch-size",
            "8",
            "--image-size",
            "24",
            "--max-text-length",
            "12",
            "--max-train-batches",
            "2",
            "--max-eval-batches",
            "1",
            "--device",
            "cpu",
            "--run-name",
            "pytest_scene_text_recognition_smoke",
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
