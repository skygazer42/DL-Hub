# Batch 42 Vision-Heavy Trackification Manifest

Batch 42 continues the covered-surface trackification loop with a vision-heavy batch.
It targets first-class `dlhub/*` surfaces that already exist but still have no direct
teaching-first `tracks/*` lesson on their home track. The batch leans toward vision
because that track still holds the largest cluster of uncovered surfaces, while keeping
one pointcloud and one generative lane so the covered-surface backlog keeps shrinking
across domains.

Selection rules for this batch:

- fixed `10 worktree + 10 lane` execution
- only choose directions that already exist as first-class `dlhub/*` implementations
- only choose directions that still lack a direct `tracks/*` lesson on their target track
- keep lessons compact-first and CPU-friendly
- keep shared integration ownership limited to repo indexes and `run_lesson` coverage
- replace `pointcloud/reconstruction` with `vision/anomaly_detection` because `tracks/pointcloud/lesson_07_pointnet_compact_reconstruction/` already covers the reconstruction teaching slot

Start point:

- base branch: `main`
- branch anchor commit: `docs: mark batch41 gaps merged`
- integration branch: `batch420-integration`
- worktree root: `.worktrees/`

## Lane Table

| lane_id | branch | worktree | surface | slug | lane_owned_paths | integration_owned_paths | verification_cmds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L01 | `batch420-layout-generation-vision` | `.worktrees/batch420-layout-generation-vision` | `tracks/vision` | `layout_generation_vision_lesson` | `tracks/vision/lesson_82_synthetic_layout_generation/`, `tests/test_tracks_vision_layout_generation.py` | `README.md`, `tracks/pointcloud/README.md`, `tracks/vision/README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_layout_generation.py`<br>`python scripts/run_lesson.py vision lesson_82_synthetic_layout_generation --dry-run` |
| L02 | `batch420-panoptic-segmentation-vision` | `.worktrees/batch420-panoptic-segmentation-vision` | `tracks/vision` | `panoptic_segmentation_vision_lesson` | `tracks/vision/lesson_83_synthetic_panoptic_segmentation/`, `tests/test_tracks_vision_panoptic_segmentation.py` | `README.md`, `tracks/pointcloud/README.md`, `tracks/vision/README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_panoptic_segmentation.py`<br>`python scripts/run_lesson.py vision lesson_83_synthetic_panoptic_segmentation --dry-run` |
| L03 | `batch420-medical-segmentation-vision` | `.worktrees/batch420-medical-segmentation-vision` | `tracks/vision` | `medical_segmentation_vision_lesson` | `tracks/vision/lesson_84_synthetic_medical_segmentation/`, `tests/test_tracks_vision_medical_segmentation.py` | `README.md`, `tracks/pointcloud/README.md`, `tracks/vision/README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_medical_segmentation.py`<br>`python scripts/run_lesson.py vision lesson_84_synthetic_medical_segmentation --dry-run` |
| L04 | `batch420-scene-text-spotting-vision` | `.worktrees/batch420-scene-text-spotting-vision` | `tracks/vision` | `scene_text_spotting_vision_lesson` | `tracks/vision/lesson_85_synthetic_scene_text_spotting/`, `tests/test_tracks_vision_scene_text_spotting.py` | `README.md`, `tracks/pointcloud/README.md`, `tracks/vision/README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_scene_text_spotting.py`<br>`python scripts/run_lesson.py vision lesson_85_synthetic_scene_text_spotting --dry-run` |
| L05 | `batch420-co-segmentation-vision` | `.worktrees/batch420-co-segmentation-vision` | `tracks/vision` | `co_segmentation_vision_lesson` | `tracks/vision/lesson_86_synthetic_co_segmentation/`, `tests/test_tracks_vision_co_segmentation.py` | `README.md`, `tracks/pointcloud/README.md`, `tracks/vision/README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_co_segmentation.py`<br>`python scripts/run_lesson.py vision lesson_86_synthetic_co_segmentation --dry-run` |
| L06 | `batch420-action-recognition-vision` | `.worktrees/batch420-action-recognition-vision` | `tracks/vision` | `action_recognition_vision_lesson` | `tracks/vision/lesson_87_synthetic_action_recognition/`, `tests/test_tracks_vision_action_recognition.py` | `README.md`, `tracks/pointcloud/README.md`, `tracks/vision/README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_action_recognition.py`<br>`python scripts/run_lesson.py vision lesson_87_synthetic_action_recognition --dry-run` |
| L07 | `batch420-reid-vision` | `.worktrees/batch420-reid-vision` | `tracks/vision` | `reid_vision_lesson` | `tracks/vision/lesson_88_synthetic_reid/`, `tests/test_tracks_vision_reid.py` | `README.md`, `tracks/pointcloud/README.md`, `tracks/vision/README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_reid.py`<br>`python scripts/run_lesson.py vision lesson_88_synthetic_reid --dry-run` |
| L08 | `batch420-anomaly-detection-vision` | `.worktrees/batch420-anomaly-detection-vision` | `tracks/vision` | `anomaly_detection_vision_lesson` | `tracks/vision/lesson_89_synthetic_anomaly_detection/`, `tests/test_tracks_vision_anomaly_detection.py` | `README.md`, `tracks/pointcloud/README.md`, `tracks/vision/README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_anomaly_detection.py`<br>`python scripts/run_lesson.py vision lesson_89_synthetic_anomaly_detection --dry-run` |
| L09 | `batch420-pointcloud-registration-pointcloud` | `.worktrees/batch420-pointcloud-registration-pointcloud` | `tracks/pointcloud` | `pointcloud_registration_track_lesson` | `tracks/pointcloud/lesson_36_compact_pointcloud_registration/`, `tests/test_tracks_pointcloud_registration.py` | `README.md`, `tracks/pointcloud/README.md`, `tracks/vision/README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_pointcloud_registration.py`<br>`python scripts/run_lesson.py pointcloud lesson_36_compact_pointcloud_registration --dry-run` |
| L10 | `batch420-world-models-generative` | `.worktrees/batch420-world-models-generative` | `tracks/generative` | `world_models_generative_lesson` | `tracks/generative/lesson_51_compact_world_models/`, `tests/test_tracks_generative_world_models.py` | `README.md`, `tracks/pointcloud/README.md`, `tracks/vision/README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_generative_world_models.py`<br>`python scripts/run_lesson.py generative lesson_51_compact_world_models --dry-run` |

## Integration Branch Ownership

`batch420-integration` owns only the shared surfaces required after the 10 lane merges:

- `README.md`
- `tracks/pointcloud/README.md`
- `tracks/vision/README.md`
- `tracks/generative/README.md`
- `tests/test_scripts_run_lesson.py`
- any shared verification-only updates required to make the merged batch pass on `main`

## Merge Order

1. `batch420-layout-generation-vision`
2. `batch420-panoptic-segmentation-vision`
3. `batch420-medical-segmentation-vision`
4. `batch420-scene-text-spotting-vision`
5. `batch420-co-segmentation-vision`
6. `batch420-action-recognition-vision`
7. `batch420-reid-vision`
8. `batch420-anomaly-detection-vision`
9. `batch420-pointcloud-registration-pointcloud`
10. `batch420-world-models-generative`

## Exit Criteria

- each lane has at least one local commit on its own branch
- each lane passes its manifest smoke commands from its own worktree
- integration updates the top-level lesson counts and the pointcloud/vision/generative track tables for all 10 new lessons
- integration branch passes fresh shared verification before merge back to `main`
