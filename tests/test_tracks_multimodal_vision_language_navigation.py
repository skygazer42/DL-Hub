import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_multimodal_vln_batch_shapes() -> None:
    from tracks.multimodal.lesson_25_vision_language_navigation.data import (
        DataConfig,
        get_dataloaders,
    )

    cfg = DataConfig(
        num_samples=48,
        batch_size=8,
        grid_size=7,
        max_instruction_length=12,
        val_fraction=0.25,
        seed=0,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(cfg)
    batch = next(iter(train_loader))

    assert batch["observation"].shape == (8, 3, 7, 7)
    assert batch["instruction_ids"].shape == (8, 12)
    assert batch["instruction_mask"].shape == (8, 12)
    assert batch["actions"].shape == (8,)
    assert batch["actions"].dtype == torch.long
    assert batch["agent_pos"].shape == (8, 2)
    assert batch["goal_pos"].shape == (8, 2)
    assert len(batch["instruction_text"]) == 8
    assert "north" in vocab.token_to_id
    assert "south" in vocab.token_to_id
    assert "east" in vocab.token_to_id
    assert "west" in vocab.token_to_id


def test_multimodal_vln_model_outputs() -> None:
    from tracks.multimodal.lesson_25_vision_language_navigation.data import (
        DataConfig,
        get_dataloaders,
    )
    from tracks.multimodal.lesson_25_vision_language_navigation.model import (
        VisionLanguageNavigationConfig,
        CompactVisionLanguageNavigationModel,
        navigation_accuracy,
        navigation_loss,
    )

    data_cfg = DataConfig(
        num_samples=32,
        batch_size=8,
        grid_size=7,
        max_instruction_length=12,
        val_fraction=0.25,
        seed=1,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(data_cfg)
    batch = next(iter(train_loader))

    model = CompactVisionLanguageNavigationModel(
        VisionLanguageNavigationConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            num_actions=4,
            hidden_dim=48,
            text_dim=32,
            vision_width=32,
        )
    )
    outputs = model(batch)

    assert set(outputs) >= {"logits", "policy"}
    assert outputs["logits"].shape == (8, 4)
    assert outputs["policy"].shape == (8, 4)
    assert torch.allclose(
        outputs["policy"].sum(dim=1),
        torch.ones(8, dtype=outputs["policy"].dtype),
        atol=1e-5,
    )

    loss = navigation_loss(outputs["logits"], batch["actions"])
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    acc = navigation_accuracy(outputs["logits"], batch["actions"])
    assert 0.0 <= acc <= 1.0


def test_multimodal_vln_training_smoke() -> None:
    run_dir = (
        _repo_root()
        / "outputs"
        / "multimodal"
        / "lesson_25_vision_language_navigation"
        / "pytest_vln_smoke"
    )
    if run_dir.exists():
        shutil.rmtree(run_dir)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracks.multimodal.lesson_25_vision_language_navigation.train",
            "--epochs",
            "1",
            "--num-samples",
            "64",
            "--batch-size",
            "8",
            "--grid-size",
            "7",
            "--max-instruction-length",
            "12",
            "--max-train-batches",
            "2",
            "--max-eval-batches",
            "1",
            "--device",
            "cpu",
            "--run-name",
            "pytest_vln_smoke",
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
