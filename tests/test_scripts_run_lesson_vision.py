"""Tests for scripts/run_lesson.py: vision track lessons."""

import subprocess
import sys

from _run_lesson_helpers import _repo_root


def test_run_lesson_dry_run_resolves_vision_monocular_depth_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_19_synthetic_monocular_depth_estimation",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_19_synthetic_monocular_depth_estimation.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_lane_detection_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_20_synthetic_lane_detection",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_20_synthetic_lane_detection.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_lane_topology_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_21_synthetic_lane_topology_estimation",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_21_synthetic_lane_topology_estimation.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_road_scene_understanding_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_22_synthetic_road_scene_understanding",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_22_synthetic_road_scene_understanding.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_image_dehazing_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_23_synthetic_image_dehazing",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_23_synthetic_image_dehazing.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_reflection_removal_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_24_synthetic_reflection_removal",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_24_synthetic_reflection_removal.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_image_fusion_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_25_synthetic_image_fusion",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_25_synthetic_image_fusion.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_text_detection_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_26_synthetic_text_detection",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_26_synthetic_text_detection.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_edge_detection_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_27_synthetic_edge_detection",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_27_synthetic_edge_detection.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_salient_object_detection_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_28_synthetic_salient_object_detection",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_28_synthetic_salient_object_detection.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_camouflaged_object_detection_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_29_synthetic_camouflaged_object_detection",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_29_synthetic_camouflaged_object_detection.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_salient_object_boxes_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_30_synthetic_salient_object_detection_boxes",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_30_synthetic_salient_object_detection_boxes.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_interactive_segmentation_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_31_synthetic_interactive_segmentation",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_31_synthetic_interactive_segmentation.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_face_landmark_detection_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_32_synthetic_face_landmark_detection",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_32_synthetic_face_landmark_detection.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_face_liveness_detection_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_33_synthetic_face_liveness_detection",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_33_synthetic_face_liveness_detection.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_license_plate_recognition_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_34_synthetic_license_plate_recognition",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_34_synthetic_license_plate_recognition.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_pose_6d_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_35_synthetic_6d_pose_estimation",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_35_synthetic_6d_pose_estimation.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_face_retrieval_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_46_synthetic_face_retrieval",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_46_synthetic_face_retrieval.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_face_pose_estimation_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_47_synthetic_face_pose_estimation",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_47_synthetic_face_pose_estimation.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_gaze_estimation_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_48_synthetic_gaze_estimation",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_48_synthetic_gaze_estimation.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_human_pose_estimation_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_49_synthetic_human_pose_estimation",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_49_synthetic_human_pose_estimation.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_hand_pose_estimation_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_50_synthetic_hand_pose_estimation",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_50_synthetic_hand_pose_estimation.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_gesture_recognition_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_51_synthetic_gesture_recognition",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_51_synthetic_gesture_recognition.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_finger_count_estimation_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_52_synthetic_finger_count_estimation",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_52_synthetic_finger_count_estimation.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_handedness_classification_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_53_synthetic_handedness_classification",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_53_synthetic_handedness_classification.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_palm_orientation_estimation_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_54_synthetic_palm_orientation_estimation",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_54_synthetic_palm_orientation_estimation.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_sign_digit_classification_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_55_synthetic_sign_digit_classification",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_55_synthetic_sign_digit_classification.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_finger_spread_estimation_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_56_synthetic_finger_spread_estimation",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_56_synthetic_finger_spread_estimation.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_thumb_position_classification_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_57_synthetic_thumb_position_classification",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_57_synthetic_thumb_position_classification.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_finger_curvature_estimation_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_58_synthetic_finger_curvature_estimation",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_58_synthetic_finger_curvature_estimation.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_thumb_contact_classification_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_59_synthetic_thumb_contact_classification",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_59_synthetic_thumb_contact_classification.train" in proc.stdout
