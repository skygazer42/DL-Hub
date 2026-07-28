import json
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")


def test_multimodal_hand_pose_reasoning_batch_shapes() -> None:
    from tracks.multimodal.lesson_51_hand_pose_vlm_reasoning.data import DataConfig, get_dataloaders

    cfg = DataConfig(
        num_samples=48,
        batch_size=8,
        image_size=64,
        max_text_length=16,
        val_fraction=0.25,
        seed=0,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(cfg)
    batch = next(iter(train_loader))

    assert batch["image"].shape == (8, 1, 64, 64)
    assert batch["query_ids"].shape == (8, 16)
    assert batch["query_mask"].shape == (8, 16)
    assert batch["target_keypoints"].shape == (8, 20)
    assert len(batch["query_text"]) == 8
    assert "estimate" in vocab.token_to_id
    assert "hand" in vocab.token_to_id
    assert "pose" in vocab.token_to_id
    assert "wrist" in vocab.token_to_id
    assert "thumb" in vocab.token_to_id


def test_multimodal_hand_pose_reasoning_model_outputs() -> None:
    from tracks.multimodal.lesson_51_hand_pose_vlm_reasoning.data import DataConfig, get_dataloaders
    from tracks.multimodal.lesson_51_hand_pose_vlm_reasoning.model import (
        HandPoseReasoningConfig,
        CompactHandPoseReasoningModel,
        hand_pose_loss,
        keypoint_l2,
    )

    data_cfg = DataConfig(
        num_samples=32,
        batch_size=8,
        image_size=64,
        max_text_length=16,
        val_fraction=0.25,
        seed=1,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(data_cfg)
    batch = next(iter(train_loader))

    model = CompactHandPoseReasoningModel(
        HandPoseReasoningConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            hidden_dim=72,
            text_dim=32,
            vision_width=40,
            num_keypoints=10,
        )
    )
    outputs = model(batch)

    assert set(outputs) >= {"pred_keypoints"}
    assert outputs["pred_keypoints"].shape == (8, 20)
    assert torch.all(outputs["pred_keypoints"] >= 0.0)
    assert torch.all(outputs["pred_keypoints"] <= 1.0)

    loss = hand_pose_loss(outputs["pred_keypoints"], batch["target_keypoints"])
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    l2 = keypoint_l2(outputs["pred_keypoints"], batch["target_keypoints"])
    assert l2.shape == (8,)
    assert torch.all(l2 >= 0.0)


def test_multimodal_hand_pose_reasoning_training_smoke(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.multimodal.lesson_51_hand_pose_vlm_reasoning.data import DataConfig
    from tracks.multimodal.lesson_51_hand_pose_vlm_reasoning.train import TrainConfig, run_training

    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path / "outputs"))
    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=1e-3,
            weight_decay=1e-4,
            seed=42,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_hand_pose_reasoning_smoke",
            hidden_dim=72,
            text_dim=32,
            vision_width=40,
        ),
        DataConfig(
            num_samples=64,
            batch_size=8,
            image_size=64,
            max_text_length=16,
            val_fraction=0.25,
            seed=7,
            num_workers=0,
        ),
    )

    assert exit_code == 0
    run_dir = (
        tmp_path
        / "outputs"
        / "multimodal"
        / "lesson_51_hand_pose_vlm_reasoning"
        / "pytest_hand_pose_reasoning_smoke"
    )
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "vocab.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()

    metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(metrics) == 1
    assert metrics[0]["epoch"] == 1
    for key in ("train_loss", "train_mean_l2", "eval_loss", "eval_mean_l2"):
        assert key in metrics[0]
        assert float(metrics[0][key]) >= 0.0


def test_multimodal_hand_pose_reasoning_dry_run() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_51_hand_pose_vlm_reasoning",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "tracks.multimodal.lesson_51_hand_pose_vlm_reasoning.train" in proc.stdout

