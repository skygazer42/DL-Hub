"""Tests for scripts/run_lesson.py: multimodal face/hand/gesture lessons."""

import subprocess
import sys

from _run_lesson_helpers import _repo_root


def test_run_lesson_dry_run_resolves_multimodal_gaze_estimation_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_30_vision_language_gaze_estimation",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_30_vision_language_gaze_estimation.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_face_alignment_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_45_face_alignment_vlm_reasoning",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_45_face_alignment_vlm_reasoning.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_face_detection_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_46_face_detection_vlm_reasoning",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_46_face_detection_vlm_reasoning.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_face_retrieval_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_47_face_retrieval_vlm_reasoning",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_47_face_retrieval_vlm_reasoning.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_face_pose_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_48_face_pose_vlm_reasoning",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_48_face_pose_vlm_reasoning.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_face_gaze_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_49_face_gaze_vlm_reasoning",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_49_face_gaze_vlm_reasoning.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_person_pose_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_50_person_pose_vlm_reasoning",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_50_person_pose_vlm_reasoning.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_hand_pose_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_51_hand_pose_vlm_reasoning",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_51_hand_pose_vlm_reasoning.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_gesture_reasoning_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_52_gesture_vlm_reasoning",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_52_gesture_vlm_reasoning.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_finger_count_reasoning_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_53_finger_count_vlm_reasoning",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_53_finger_count_vlm_reasoning.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_handedness_reasoning_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_54_handedness_vlm_reasoning",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_54_handedness_vlm_reasoning.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_palm_orientation_reasoning_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_55_palm_orientation_vlm_reasoning",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_55_palm_orientation_vlm_reasoning.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_sign_digit_reasoning_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_56_sign_digit_vlm_reasoning",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_56_sign_digit_vlm_reasoning.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_finger_spread_reasoning_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_57_finger_spread_vlm_reasoning",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_57_finger_spread_vlm_reasoning.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_thumb_position_reasoning_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_58_thumb_position_vlm_reasoning",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_58_thumb_position_vlm_reasoning.train" in proc.stdout
