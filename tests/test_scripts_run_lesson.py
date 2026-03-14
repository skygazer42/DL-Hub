import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_run_lesson_lists_tracks() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/run_lesson.py", "--list"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "Tracks:" in proc.stdout
    assert "multimodal" in proc.stdout
    assert "vision" in proc.stdout


def test_run_lesson_lists_lessons_for_track() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/run_lesson.py", "vision", "--list"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "Lessons (vision):" in proc.stdout
    assert "lesson_01_mnist_lenet" in proc.stdout
    assert "lesson_14_video_mot_basics" in proc.stdout


def test_run_lesson_dry_run_resolves_train_module() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/run_lesson.py", "vision", "lesson_01_mnist_lenet", "--dry-run"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_01_mnist_lenet.train" in proc.stdout


def test_run_lesson_lists_lessons_for_multimodal_track() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/run_lesson.py", "multimodal", "--list"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "Lessons (multimodal):" in proc.stdout
    assert "lesson_01_clip_toy_retrieval" in proc.stdout
    assert "lesson_02_blip_toy_captioning" in proc.stdout
    assert "lesson_03_llava_toy_instruction_vlm" in proc.stdout
    assert "lesson_04_grounding_toy_refexp" in proc.stdout
    assert "lesson_05_mask_grounding_toy_refexp" in proc.stdout
    assert "lesson_06_flamingo_toy_interleaved_vlm" in proc.stdout
    assert "lesson_07_qformer_toy_bridge_vlm" in proc.stdout
    assert "lesson_08_perceiver_resampler_toy_vlm" in proc.stdout
    assert "lesson_09_paligemma_toy_siglip_decoder_vlm" in proc.stdout
    assert "lesson_10_owlvit_toy_open_vocab_detection" in proc.stdout
    assert "lesson_11_grounded_sam_toy_open_vocab_segmentation" in proc.stdout
    assert "lesson_12_key_value_ocr_toy_doc_vlm" in proc.stdout
    assert "lesson_13_video_vlm_toy_temporal_qa" in proc.stdout
    assert "lesson_14_bmn_toy_temporal_grounding" in proc.stdout
    assert "lesson_15_2dtan_toy_temporal_grounding" in proc.stdout
    assert "lesson_16_multiscale_2dtan_toy_temporal_grounding" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_01_clip_toy_retrieval",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_01_clip_toy_retrieval.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_blip_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_02_blip_toy_captioning",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_02_blip_toy_captioning.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_llava_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_03_llava_toy_instruction_vlm",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_03_llava_toy_instruction_vlm.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_grounding_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_04_grounding_toy_refexp",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_04_grounding_toy_refexp.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_mask_grounding_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_05_mask_grounding_toy_refexp",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_05_mask_grounding_toy_refexp.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_flamingo_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_06_flamingo_toy_interleaved_vlm",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_06_flamingo_toy_interleaved_vlm.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_qformer_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_07_qformer_toy_bridge_vlm",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_07_qformer_toy_bridge_vlm.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_perceiver_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_08_perceiver_resampler_toy_vlm",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_08_perceiver_resampler_toy_vlm.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_paligemma_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_09_paligemma_toy_siglip_decoder_vlm",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_09_paligemma_toy_siglip_decoder_vlm.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_owlvit_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_10_owlvit_toy_open_vocab_detection",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_10_owlvit_toy_open_vocab_detection.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_grounded_sam_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_11_grounded_sam_toy_open_vocab_segmentation",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_11_grounded_sam_toy_open_vocab_segmentation.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_key_value_ocr_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_12_key_value_ocr_toy_doc_vlm",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_12_key_value_ocr_toy_doc_vlm.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_video_vlm_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_13_video_vlm_toy_temporal_qa",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_13_video_vlm_toy_temporal_qa.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_bmn_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_14_bmn_toy_temporal_grounding",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_14_bmn_toy_temporal_grounding.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_2dtan_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_15_2dtan_toy_temporal_grounding",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_15_2dtan_toy_temporal_grounding.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_multiscale_2dtan_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_16_multiscale_2dtan_toy_temporal_grounding",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_16_multiscale_2dtan_toy_temporal_grounding.train" in proc.stdout
