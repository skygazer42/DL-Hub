# Batch 41 Cross-Domain Trackification Manifest

Batch 41 resumes the covered-surface trackification loop with a balanced cross-domain batch.
It targets first-class `dlhub/*` surfaces that are already implemented but still have no
teaching-first `tracks/*` lesson. This batch mixes vision, pointcloud, and generative lanes so
the remaining uncovered surfaces keep shrinking without over-concentrating on a single track.

Selection rules for this batch:

- fixed `10 worktree + 10 lane` execution
- only choose directions that already exist as first-class `dlhub/*` implementations
- only choose directions that still lack a `tracks/*` lesson
- keep lessons compact-first and CPU-friendly
- keep shared integration ownership limited to repo indexes and `run_lesson` coverage

Start point:

- base branch: `main`
- branch anchor commit: `docs: mark batch40 gaps merged`
- integration branch: `batch410-integration`
- worktree root: `.worktrees/`

## Lane Table

| lane_id | branch | worktree | surface | slug | lane_owned_paths | integration_owned_paths | verification_cmds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L01 | `batch410-image-relighting-vision` | `.worktrees/batch410-image-relighting-vision` | `tracks/vision` | `image_relighting_vision_lesson` | `tracks/vision/lesson_78_synthetic_image_relighting/`, `tests/test_tracks_vision_image_relighting.py` | `README.md`, `tracks/pointcloud/README.md`, `tracks/vision/README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_image_relighting.py`<br>`python scripts/run_lesson.py vision lesson_78_synthetic_image_relighting --dry-run` |
| L02 | `batch410-transparent-object-segmentation-vision` | `.worktrees/batch410-transparent-object-segmentation-vision` | `tracks/vision` | `transparent_object_segmentation_vision_lesson` | `tracks/vision/lesson_79_synthetic_transparent_object_segmentation/`, `tests/test_tracks_vision_transparent_object_segmentation.py` | `README.md`, `tracks/pointcloud/README.md`, `tracks/vision/README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_transparent_object_segmentation.py`<br>`python scripts/run_lesson.py vision lesson_79_synthetic_transparent_object_segmentation --dry-run` |
| L03 | `batch410-event-camera-understanding-vision` | `.worktrees/batch410-event-camera-understanding-vision` | `tracks/vision` | `event_camera_understanding_vision_lesson` | `tracks/vision/lesson_80_synthetic_event_camera_understanding/`, `tests/test_tracks_vision_event_camera_understanding.py` | `README.md`, `tracks/pointcloud/README.md`, `tracks/vision/README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_event_camera_understanding.py`<br>`python scripts/run_lesson.py vision lesson_80_synthetic_event_camera_understanding --dry-run` |
| L04 | `batch410-shadow-detection-vision` | `.worktrees/batch410-shadow-detection-vision` | `tracks/vision` | `shadow_detection_vision_lesson` | `tracks/vision/lesson_81_synthetic_shadow_detection/`, `tests/test_tracks_vision_shadow_detection.py` | `README.md`, `tracks/pointcloud/README.md`, `tracks/vision/README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_shadow_detection.py`<br>`python scripts/run_lesson.py vision lesson_81_synthetic_shadow_detection --dry-run` |
| L05 | `batch410-pointcloud-upsampling-pointcloud` | `.worktrees/batch410-pointcloud-upsampling-pointcloud` | `tracks/pointcloud` | `pointcloud_upsampling_track_lesson` | `tracks/pointcloud/lesson_34_compact_pointcloud_upsampling/`, `tests/test_tracks_pointcloud_upsampling.py` | `README.md`, `tracks/pointcloud/README.md`, `tracks/vision/README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_pointcloud_upsampling.py`<br>`python scripts/run_lesson.py pointcloud lesson_34_compact_pointcloud_upsampling --dry-run` |
| L06 | `batch410-shape-correspondence3d-pointcloud` | `.worktrees/batch410-shape-correspondence3d-pointcloud` | `tracks/pointcloud` | `shape_correspondence_3d_track_lesson` | `tracks/pointcloud/lesson_35_compact_shape_correspondence_3d/`, `tests/test_tracks_pointcloud_shape_correspondence3d.py` | `README.md`, `tracks/pointcloud/README.md`, `tracks/vision/README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_pointcloud_shape_correspondence3d.py`<br>`python scripts/run_lesson.py pointcloud lesson_35_compact_shape_correspondence_3d --dry-run` |
| L07 | `batch410-text-to-3d-generative` | `.worktrees/batch410-text-to-3d-generative` | `tracks/generative` | `text_to_3d_generative_lesson` | `tracks/generative/lesson_47_compact_text_to_3d/`, `tests/test_tracks_generative_text_to_3d.py` | `README.md`, `tracks/pointcloud/README.md`, `tracks/vision/README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_generative_text_to_3d.py`<br>`python scripts/run_lesson.py generative lesson_47_compact_text_to_3d --dry-run` |
| L08 | `batch410-image-to-3d-generative` | `.worktrees/batch410-image-to-3d-generative` | `tracks/generative` | `image_to_3d_generative_lesson` | `tracks/generative/lesson_48_compact_image_to_3d/`, `tests/test_tracks_generative_image_to_3d.py` | `README.md`, `tracks/pointcloud/README.md`, `tracks/vision/README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_generative_image_to_3d.py`<br>`python scripts/run_lesson.py generative lesson_48_compact_image_to_3d --dry-run` |
| L09 | `batch410-text-to-video-generative` | `.worktrees/batch410-text-to-video-generative` | `tracks/generative` | `text_to_video_generative_lesson` | `tracks/generative/lesson_49_compact_text_to_video/`, `tests/test_tracks_generative_text_to_video.py` | `README.md`, `tracks/pointcloud/README.md`, `tracks/vision/README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_generative_text_to_video.py`<br>`python scripts/run_lesson.py generative lesson_49_compact_text_to_video --dry-run` |
| L10 | `batch410-video-to-video-generative` | `.worktrees/batch410-video-to-video-generative` | `tracks/generative` | `video_to_video_generative_lesson` | `tracks/generative/lesson_50_compact_video_to_video/`, `tests/test_tracks_generative_video_to_video.py` | `README.md`, `tracks/pointcloud/README.md`, `tracks/vision/README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_generative_video_to_video.py`<br>`python scripts/run_lesson.py generative lesson_50_compact_video_to_video --dry-run` |

## Integration Branch Ownership

`batch410-integration` owns only the shared surfaces required after the 10 lane merges:

- `README.md`
- `tracks/pointcloud/README.md`
- `tracks/vision/README.md`
- `tracks/generative/README.md`
- `tests/test_scripts_run_lesson.py`
- any shared verification-only updates required to make the merged batch pass on `main`

## Merge Order

1. `batch410-image-relighting-vision`
2. `batch410-transparent-object-segmentation-vision`
3. `batch410-event-camera-understanding-vision`
4. `batch410-shadow-detection-vision`
5. `batch410-pointcloud-upsampling-pointcloud`
6. `batch410-shape-correspondence3d-pointcloud`
7. `batch410-text-to-3d-generative`
8. `batch410-image-to-3d-generative`
9. `batch410-text-to-video-generative`
10. `batch410-video-to-video-generative`

## Exit Criteria

- each lane has at least one local commit on its own branch
- each lane passes its manifest smoke commands from its own worktree
- integration updates the top-level lesson counts and the pointcloud/vision/generative track tables for all 10 new lessons
- integration branch passes fresh shared verification before merge back to `main`
