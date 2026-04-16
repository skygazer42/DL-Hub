# Batch 40 Pointcloud-Heavy Trackification Manifest

Batch 40 continues the second-pass trackification loop by converting additional covered
first-class `dlhub/*` surfaces into runnable `tracks/*` lessons. This batch is intentionally
pointcloud-heavy: it targets the remaining 3D directions that still lack teaching-first track
coverage, then rounds the batch out with two uncovered vision surfaces that also already exist as
first-class implementations.

Selection rules for this batch:

- fixed `10 worktree + 10 lane` execution
- prefer covered `dlhub/*` directions that still have no corresponding `tracks/*` lesson
- bias toward compact synthetic tensors that stay CPU-friendly
- keep shared integration ownership limited to repo indexes and `run_lesson` coverage

Start point:

- base branch: `main`
- branch anchor commit: `docs: mark batch39 gaps merged`
- integration branch: `batch400-integration`
- worktree root: `.worktrees/`

## Lane Table

| lane_id | branch | worktree | surface | slug | lane_owned_paths | integration_owned_paths | verification_cmds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L01 | `batch400-gaussian-splatting-pointcloud` | `.worktrees/batch400-gaussian-splatting-pointcloud` | `tracks/pointcloud` | `gaussian_splatting_pointcloud_lesson` | `tracks/pointcloud/lesson_26_toy_gaussian_splatting/`, `tests/test_tracks_pointcloud_gaussian_splatting.py` | `README.md`, `tracks/pointcloud/README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_pointcloud_gaussian_splatting.py`<br>`python scripts/run_lesson.py pointcloud lesson_26_toy_gaussian_splatting --dry-run` |
| L02 | `batch400-detection3d-pointcloud` | `.worktrees/batch400-detection3d-pointcloud` | `tracks/pointcloud` | `detection3d_pointcloud_lesson` | `tracks/pointcloud/lesson_27_toy_3d_object_detection/`, `tests/test_tracks_pointcloud_detection3d.py` | `README.md`, `tracks/pointcloud/README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_pointcloud_detection3d.py`<br>`python scripts/run_lesson.py pointcloud lesson_27_toy_3d_object_detection --dry-run` |
| L03 | `batch400-segmentation3d-pointcloud` | `.worktrees/batch400-segmentation3d-pointcloud` | `tracks/pointcloud` | `segmentation3d_pointcloud_lesson` | `tracks/pointcloud/lesson_28_toy_3d_semantic_segmentation/`, `tests/test_tracks_pointcloud_segmentation3d.py` | `README.md`, `tracks/pointcloud/README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_pointcloud_segmentation3d.py`<br>`python scripts/run_lesson.py pointcloud lesson_28_toy_3d_semantic_segmentation --dry-run` |
| L04 | `batch400-instance-segmentation3d-pointcloud` | `.worktrees/batch400-instance-segmentation3d-pointcloud` | `tracks/pointcloud` | `instance_segmentation3d_pointcloud_lesson` | `tracks/pointcloud/lesson_29_toy_3d_instance_segmentation/`, `tests/test_tracks_pointcloud_instance_segmentation3d.py` | `README.md`, `tracks/pointcloud/README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_pointcloud_instance_segmentation3d.py`<br>`python scripts/run_lesson.py pointcloud lesson_29_toy_3d_instance_segmentation --dry-run` |
| L05 | `batch400-tracking3d-pointcloud` | `.worktrees/batch400-tracking3d-pointcloud` | `tracks/pointcloud` | `tracking3d_pointcloud_lesson` | `tracks/pointcloud/lesson_30_toy_3d_object_tracking/`, `tests/test_tracks_pointcloud_tracking3d.py` | `README.md`, `tracks/pointcloud/README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_pointcloud_tracking3d.py`<br>`python scripts/run_lesson.py pointcloud lesson_30_toy_3d_object_tracking --dry-run` |
| L06 | `batch400-open-vocabulary-3d-pointcloud` | `.worktrees/batch400-open-vocabulary-3d-pointcloud` | `tracks/pointcloud` | `open_vocabulary_3d_pointcloud_lesson` | `tracks/pointcloud/lesson_31_toy_open_vocabulary_3d/`, `tests/test_tracks_pointcloud_open_vocabulary_3d.py` | `README.md`, `tracks/pointcloud/README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_pointcloud_open_vocabulary_3d.py`<br>`python scripts/run_lesson.py pointcloud lesson_31_toy_open_vocabulary_3d --dry-run` |
| L07 | `batch400-pointcloud-forecasting` | `.worktrees/batch400-pointcloud-forecasting` | `tracks/pointcloud` | `pointcloud_forecasting_track_lesson` | `tracks/pointcloud/lesson_32_toy_pointcloud_forecasting/`, `tests/test_tracks_pointcloud_forecasting.py` | `README.md`, `tracks/pointcloud/README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_pointcloud_forecasting.py`<br>`python scripts/run_lesson.py pointcloud lesson_32_toy_pointcloud_forecasting --dry-run` |
| L08 | `batch400-pointcloud-anomaly-detection` | `.worktrees/batch400-pointcloud-anomaly-detection` | `tracks/pointcloud` | `pointcloud_anomaly_detection_track_lesson` | `tracks/pointcloud/lesson_33_toy_pointcloud_anomaly_detection/`, `tests/test_tracks_pointcloud_anomaly_detection.py` | `README.md`, `tracks/pointcloud/README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_pointcloud_anomaly_detection.py`<br>`python scripts/run_lesson.py pointcloud lesson_33_toy_pointcloud_anomaly_detection --dry-run` |
| L09 | `batch400-image-deweathering-vision` | `.worktrees/batch400-image-deweathering-vision` | `tracks/vision` | `image_deweathering_vision_lesson` | `tracks/vision/lesson_76_synthetic_image_deweathering/`, `tests/test_tracks_vision_image_deweathering.py` | `README.md`, `tracks/pointcloud/README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_image_deweathering.py`<br>`python scripts/run_lesson.py vision lesson_76_synthetic_image_deweathering --dry-run` |
| L10 | `batch400-transparent-depth-estimation-vision` | `.worktrees/batch400-transparent-depth-estimation-vision` | `tracks/vision` | `transparent_depth_estimation_vision_lesson` | `tracks/vision/lesson_77_synthetic_transparent_depth_estimation/`, `tests/test_tracks_vision_transparent_depth_estimation.py` | `README.md`, `tracks/pointcloud/README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_transparent_depth_estimation.py`<br>`python scripts/run_lesson.py vision lesson_77_synthetic_transparent_depth_estimation --dry-run` |

## Integration Branch Ownership

`batch400-integration` owns only the shared surfaces required after the 10 lane merges:

- `README.md`
- `tracks/pointcloud/README.md`
- `tracks/vision/README.md`
- `tests/test_scripts_run_lesson.py`
- any shared verification-only updates required to make the merged batch pass on `main`

## Merge Order

1. `batch400-gaussian-splatting-pointcloud`
2. `batch400-detection3d-pointcloud`
3. `batch400-segmentation3d-pointcloud`
4. `batch400-instance-segmentation3d-pointcloud`
5. `batch400-tracking3d-pointcloud`
6. `batch400-open-vocabulary-3d-pointcloud`
7. `batch400-pointcloud-forecasting`
8. `batch400-pointcloud-anomaly-detection`
9. `batch400-image-deweathering-vision`
10. `batch400-transparent-depth-estimation-vision`

## Exit Criteria

- each lane has at least one local commit on its own branch
- each lane passes its manifest smoke commands from its own worktree
- integration updates the top-level lesson counts and pointcloud/vision track tables for all 10 new lessons
- integration branch passes fresh shared verification before merge back to `main`
