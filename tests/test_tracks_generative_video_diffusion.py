import pytest

torch = pytest.importorskip("torch")


def test_toy_video_diffusion_data_and_model_contract() -> None:
    from tracks.generative.lesson_45_toy_video_diffusion.data import DataConfig, get_dataloaders
    from tracks.generative.lesson_45_toy_video_diffusion.model import (
        DiffusionSchedule,
        ModelConfig,
        ToyVideoDiffusionModel,
        q_sample,
    )

    train_loader, _ = get_dataloaders(
        DataConfig(
            num_samples=36,
            batch_size=4,
            image_size=16,
            num_frames=4,
            motion_dim=3,
            seed=0,
            num_workers=0,
            val_fraction=0.25,
        )
    )
    keyframe, motion_code, target_video = next(iter(train_loader))

    assert tuple(keyframe.shape) == (4, 1, 16, 16)
    assert tuple(motion_code.shape) == (4, 3)
    assert tuple(target_video.shape) == (4, 1, 4, 16, 16)
    assert torch.all(keyframe >= 0.0)
    assert torch.all(keyframe <= 1.0)
    assert torch.all(target_video >= 0.0)
    assert torch.all(target_video <= 1.0)
    assert not torch.allclose(target_video[:, :, 0], target_video[:, :, -1])

    cfg = ModelConfig(
        in_channels=1,
        hidden_channels=12,
        motion_dim=3,
        time_embed_dim=12,
    )
    schedule = DiffusionSchedule(num_steps=10)
    model = ToyVideoDiffusionModel(cfg)

    noise = torch.randn_like(target_video)
    timesteps = torch.randint(low=0, high=schedule.num_steps, size=(4,), dtype=torch.long)
    xt = q_sample(schedule, target_video, timesteps, noise)
    pred_noise = model(xt=xt, keyframe=keyframe, motion_code=motion_code, timesteps=timesteps)
    sampled = model.sample(
        schedule=schedule,
        keyframe=keyframe,
        motion_code=motion_code,
        device=torch.device("cpu"),
        num_steps=5,
    )

    assert tuple(pred_noise.shape) == (4, 1, 4, 16, 16)
    assert tuple(sampled.shape) == (4, 1, 4, 16, 16)
    assert torch.all(sampled >= 0.0)
    assert torch.all(sampled <= 1.0)


def test_toy_video_diffusion_training_smoke(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tracks.generative.lesson_45_toy_video_diffusion.data import DataConfig
    from tracks.generative.lesson_45_toy_video_diffusion.model import (
        DiffusionSchedule,
        ModelConfig,
    )
    from tracks.generative.lesson_45_toy_video_diffusion.train import TrainConfig, run_training

    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path))
    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=1e-3,
            seed=45,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_video_diffusion_smoke",
            num_sample_steps=5,
        ),
        DataConfig(
            num_samples=40,
            batch_size=4,
            image_size=16,
            num_frames=4,
            motion_dim=3,
            seed=7,
            num_workers=0,
            val_fraction=0.25,
        ),
        ModelConfig(
            in_channels=1,
            hidden_channels=12,
            motion_dim=3,
            time_embed_dim=12,
        ),
        DiffusionSchedule(num_steps=10),
    )

    assert exit_code == 0
    run_dir = (
        tmp_path
        / "generative"
        / "lesson_45_toy_video_diffusion"
        / "pytest_video_diffusion_smoke"
    )
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "samples.pt").is_file()
    assert (run_dir / "video_diffusion_triplets.pt").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()
