"""Tests for scripts/run_lesson.py: multimodal track lessons."""

import subprocess
import sys

from _run_lesson_helpers import _repo_root


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


def test_run_lesson_dry_run_resolves_multimodal_prompt_learning_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_18_prompt_learning_vlm",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_18_prompt_learning_vlm.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_audio_text_understanding_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_19_audio_text_understanding",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_19_audio_text_understanding.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_audio_visual_learning_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_20_audio_visual_learning",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_20_audio_visual_learning.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_audio_grounded_retrieval_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_21_audio_grounded_retrieval",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_21_audio_grounded_retrieval.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_audio_visual_event_localization_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_22_audio_visual_event_localization",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_22_audio_visual_event_localization.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_embodied_question_answering_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_23_embodied_question_answering",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_23_embodied_question_answering.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_reasoning_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_24_multimodal_reasoning",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_24_multimodal_reasoning.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_vision_language_navigation_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_25_vision_language_navigation",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_25_vision_language_navigation.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_image_text_reranking_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_26_image_text_reranking",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_26_image_text_reranking.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_scene_text_recognition_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_27_scene_text_vlm_recognition",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_27_scene_text_vlm_recognition.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_document_vlm_reasoning_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_28_document_vlm_reasoning",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_28_document_vlm_reasoning.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_human_object_interaction_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_29_human_object_interaction_reasoning",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_29_human_object_interaction_reasoning.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_person_search_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_31_person_search_attribute_retrieval",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_31_person_search_attribute_retrieval.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_action_localization_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_32_video_text_action_localization",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_32_video_text_action_localization.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_pedestrian_attributes_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_33_pedestrian_attribute_recognition",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_33_pedestrian_attribute_recognition.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_action_recognition_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_34_video_text_action_recognition",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_34_video_text_action_recognition.train" in proc.stdout
