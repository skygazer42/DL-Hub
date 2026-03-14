import pytest

torch = pytest.importorskip("torch")


def test_vision_video_mot_shapes_and_loss_smoke() -> None:
    from tracks.vision.lesson_14_video_mot_basics.data import DataConfig, get_dataloaders
    from tracks.vision.lesson_14_video_mot_basics.model import ModelConfig, build_model

    train_loader, _ = get_dataloaders(
        DataConfig(
            num_samples=32,
            batch_size=4,
            seq_len=4,
            image_size=64,
            max_objects=3,
            num_classes=3,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
            noise_std=0.05,
            min_box_size=6,
            max_box_size=12,
            max_speed=2.0,
        )
    )
    video, targets = next(iter(train_loader))
    assert tuple(video.shape) == (4, 4, 3, 64, 64)
    assert set(targets.keys()) == {"boxes", "labels", "present"}
    assert tuple(targets["boxes"].shape) == (4, 3, 4)
    assert tuple(targets["labels"].shape) == (4, 3)
    assert tuple(targets["present"].shape) == (4, 3)

    model = build_model(
        ModelConfig(
            arch="mot2d:sort_tiny",
            in_channels=3,
            num_classes=3,
            seq_len=4,
            image_size=64,
            width_mult=0.5,
            dropout=0.0,
        )
    )
    out = model.track(video)
    assert set(out.keys()) >= {"track_boxes", "track_scores", "cls_logits"}

    max_objects = int(targets["boxes"].shape[1])
    pred_boxes = out["track_boxes"][:, :max_objects, :]
    pred_scores = out["track_scores"][:, :max_objects]
    pred_cls = out["cls_logits"][:, :max_objects, :]

    box_target = targets["boxes"]
    label_target = targets["labels"]
    present_target = targets["present"]

    present_sum = present_target.sum().clamp(min=1.0)
    box_err = torch.nn.functional.smooth_l1_loss(pred_boxes, box_target, reduction="none").mean(dim=-1)
    box_loss = (box_err * present_target).sum() / present_sum
    score_loss = torch.nn.functional.binary_cross_entropy(
        pred_scores.clamp(min=1e-6, max=1.0 - 1e-6), present_target
    )

    cls_loss = torch.tensor(0.0)
    pos_mask = present_target > 0.5
    if bool(pos_mask.any()):
        cls_loss = torch.nn.functional.cross_entropy(pred_cls[pos_mask], label_target[pos_mask])

    loss = box_loss + score_loss + cls_loss
    assert torch.isfinite(loss)
    loss.backward()


def test_vision_video_mot_train_parse_args_list_arch_families(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    import sys

    from tracks.vision.lesson_14_video_mot_basics.train import parse_args

    monkeypatch.setattr(sys, "argv", ["prog", "--list-arch-families", "--list-sort", "alpha"])
    with pytest.raises(SystemExit) as exc:
        parse_args()

    assert exc.value.code == 0
    lines = [ln.strip() for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert "sort" in lines
    assert "bytetrack" in lines
    assert "mht" in lines
    assert len(lines) >= 80
    assert lines == sorted(lines)


def test_vision_video_mot_train_parse_args_list_arch_family_filter(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    import sys

    from tracks.vision.lesson_14_video_mot_basics.train import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--list-arch", "--arch-family", "sort", "--list-sort", "alpha"],
    )
    with pytest.raises(SystemExit) as exc:
        parse_args()

    assert exc.value.code == 0
    lines = [ln.strip() for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert lines == ["mot2d:sort_base", "mot2d:sort_small", "mot2d:sort_tiny"]


def test_vision_video_mot_train_parse_args_arch_family_requires_list_arch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    import sys

    from tracks.vision.lesson_14_video_mot_basics.train import parse_args

    monkeypatch.setattr(sys, "argv", ["prog", "--arch-family", "sort"])
    with pytest.raises(SystemExit) as exc:
        parse_args()

    assert exc.value.code == 2
    assert "--arch-family" in capsys.readouterr().err


def test_vision_video_mot_train_parse_args_print_config_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    import json
    import sys

    from tracks.vision.lesson_14_video_mot_basics.train import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--print-config", "--epochs", "3", "--arch", "mot2d:bytetrack_tiny"],
    )
    with pytest.raises(SystemExit) as exc:
        parse_args()

    assert exc.value.code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["train"]["epochs"] == 3
    assert payload["model"]["arch"] == "mot2d:bytetrack_tiny"


def test_vision_video_mot_cli_list_arch_does_not_print_pynvml_warning() -> None:
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracks.vision.lesson_14_video_mot_basics.train",
            "--list-arch",
            "--arch-family",
            "sort",
            "--list-sort",
            "alpha",
        ],
        cwd=str(repo_root),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    assert lines == ["mot2d:sort_base", "mot2d:sort_small", "mot2d:sort_tiny"]
    assert "pynvml package is deprecated" not in proc.stderr
