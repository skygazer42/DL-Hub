from pathlib import Path

import pytest

from dlhub import artifacts
from dlhub.cli_utils import format_arch_fidelity, print_limited, summarize_output


def test_print_limited_zero_limit_does_not_consume_input(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_if_consumed():
        raise AssertionError("input must not be consumed")
        yield "unreachable"

    print_limited(fail_if_consumed(), limit=0)

    assert capsys.readouterr().out == ""


def test_print_limited_validates_counts_before_consuming_input() -> None:
    with pytest.raises(TypeError, match="limit must be an integer"):
        print_limited([], limit=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tail must be >= 0"):
        print_limited([], limit=0, tail=-1)


def test_arch_fidelity_annotation_distinguishes_local_and_external_sources(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert format_arch_fidelity("tv:resnet18") == (
        "tv:resnet18\tfidelity=external\tsource=external-package"
    )
    assert format_arch_fidelity("vlm:llava_tiny") == (
        "vlm:llava_tiny\tfidelity=baseline-alias\tsource=dlhub/multimodal/vlm/llava.py"
    )
    assert format_arch_fidelity("aes:aes_clip_tiny") == (
        "aes:aes_clip_tiny\tfidelity=baseline-alias\t"
        "source=dlhub/vision/aesthetic_assessment/aes_clip.py"
    )

    print_limited(["diff:sdxl_tiny"], annotate_fidelity=True)

    assert capsys.readouterr().out.strip() == (
        "diff:sdxl_tiny\tfidelity=baseline-alias\tsource=dlhub/generative/diffusion/sdxl.py"
    )


def test_summarize_output_handles_recursive_containers() -> None:
    recursive: list[object] = []
    recursive.append(recursive)

    summary = summarize_output(recursive)

    assert "<cycle:list>" in summary


@pytest.mark.parametrize("nrow", [True, 0, -1, 1.5])
def test_save_image_validates_nrow_before_loading_torchvision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, nrow: object
) -> None:
    monkeypatch.setattr(
        artifacts,
        "_load_torchvision_save_image",
        lambda: pytest.fail("torchvision must not load for invalid input"),
    )

    with pytest.raises((TypeError, ValueError), match="nrow"):
        artifacts.save_image_if_available(
            object(),
            tmp_path / "grid.png",
            nrow=nrow,  # type: ignore[arg-type]
        )


def test_save_image_rejects_directory_destinations_before_loading_torchvision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        artifacts,
        "_load_torchvision_save_image",
        lambda: pytest.fail("torchvision must not load for invalid input"),
    )

    with pytest.raises(IsADirectoryError):
        artifacts.save_image_if_available(object(), tmp_path)
