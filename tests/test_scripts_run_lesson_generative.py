"""Tests for scripts/run_lesson.py: generative track lessons."""

import subprocess
import sys

from _run_lesson_helpers import _repo_root


def test_run_lesson_dry_run_resolves_generative_latent_diffusion_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_04_compact_latent_diffusion",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_04_compact_latent_diffusion.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_consistency_model_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_05_compact_consistency_model",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_05_compact_consistency_model.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_flow_matching_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_06_compact_flow_matching",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_06_compact_flow_matching.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_rectified_flow_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_07_compact_rectified_flow",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_07_compact_rectified_flow.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_diffusion_transformer_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_08_compact_diffusion_transformer",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_08_compact_diffusion_transformer.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_conditional_gan_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_09_compact_conditional_gan",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_09_compact_conditional_gan.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_diffusion_image_editing_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_10_compact_diffusion_image_editing",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_10_compact_diffusion_image_editing.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_controlnet_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_11_compact_controlnet",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_11_compact_controlnet.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_layout_to_image_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_12_compact_layout_to_image",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_12_compact_layout_to_image.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_text_to_image_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_13_compact_text_to_image_diffusion",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_13_compact_text_to_image_diffusion.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_diffusion_inpainting_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_14_compact_diffusion_inpainting",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_14_compact_diffusion_inpainting.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_diffusion_super_resolution_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_15_compact_diffusion_super_resolution",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_15_compact_diffusion_super_resolution.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_diffusion_deblurring_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_16_compact_diffusion_deblurring",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_16_compact_diffusion_deblurring.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_diffusion_denoising_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_17_compact_diffusion_denoising",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_17_compact_diffusion_denoising.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_diffusion_deraining_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_18_compact_diffusion_deraining",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_18_compact_diffusion_deraining.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_diffusion_dehazing_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_19_compact_diffusion_dehazing",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_19_compact_diffusion_dehazing.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_diffusion_reflection_removal_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_20_compact_diffusion_reflection_removal",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_20_compact_diffusion_reflection_removal.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_reference_editing_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_31_compact_diffusion_reference_editing",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_31_compact_diffusion_reference_editing.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_layout_preserving_editing_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_32_compact_diffusion_layout_preserving_editing",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_32_compact_diffusion_layout_preserving_editing.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_masked_reference_editing_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_33_compact_diffusion_masked_reference_editing",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_33_compact_diffusion_masked_reference_editing.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_layout_reference_fusion_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_34_compact_diffusion_layout_reference_fusion",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_34_compact_diffusion_layout_reference_fusion.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_box_mask_editing_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_35_compact_diffusion_box_mask_editing",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_35_compact_diffusion_box_mask_editing.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_layout_subject_fusion_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_36_compact_diffusion_layout_subject_fusion",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_36_compact_diffusion_layout_subject_fusion.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_polygon_mask_editing_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_37_compact_diffusion_polygon_mask_editing",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_37_compact_diffusion_polygon_mask_editing.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_layout_attribute_fusion_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_38_compact_diffusion_layout_attribute_fusion",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_38_compact_diffusion_layout_attribute_fusion.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_scribble_mask_editing_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_39_compact_diffusion_scribble_mask_editing",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_39_compact_diffusion_scribble_mask_editing.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_layout_style_fusion_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_40_compact_diffusion_layout_style_fusion",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_40_compact_diffusion_layout_style_fusion.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_stroke_mask_editing_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_41_compact_diffusion_stroke_mask_editing",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_41_compact_diffusion_stroke_mask_editing.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_layout_palette_fusion_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_42_compact_diffusion_layout_palette_fusion",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_42_compact_diffusion_layout_palette_fusion.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_path_mask_editing_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_43_compact_diffusion_path_mask_editing",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_43_compact_diffusion_path_mask_editing.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_layout_lighting_fusion_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_44_compact_diffusion_layout_lighting_fusion",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_44_compact_diffusion_layout_lighting_fusion.train" in proc.stdout
