# Batch 39 Video Vision Trackification Manifest

Batch 39 continues the second-pass trackification loop by converting the remaining covered
video-first `dlhub/vision/*` directions into runnable `tracks/vision/*` lessons. The emphasis is
to expose compact temporal contracts that mirror the already-landed first-class vision surfaces
without reusing the full surface implementation.

Selection rules for this batch:

- fixed `10 worktree + 10 lane` execution
- restrict this batch to covered `dlhub/vision/*` video directions not yet expressed as `tracks/*`
- favor lessons with compact temporal tensors and CPU-friendly synthetic clips
- keep shared integration ownership limited to repository indexes and `run_lesson` coverage

Start point:

- base branch: `main`
- branch anchor commit: `docs: mark batch38 gaps merged`
- integration branch: `batch390-integration`
- worktree root: `.worktrees/`

## Lane Table

| lane_id | branch | worktree | surface | slug | lane_owned_paths | integration_owned_paths | verification_cmds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L01 | `batch390-video-object-detection-vision` | `.worktrees/batch390-video-object-detection-vision` | `tracks/vision` | `video_object_detection_vision_lesson` | `tracks/vision/lesson_66_synthetic_video_object_detection/`, `tests/test_tracks_vision_video_object_detection.py` | `README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_video_object_detection.py`<br>`python scripts/run_lesson.py vision lesson_66_synthetic_video_object_detection --dry-run` |
| L02 | `batch390-video-stabilization-vision` | `.worktrees/batch390-video-stabilization-vision` | `tracks/vision` | `video_stabilization_vision_lesson` | `tracks/vision/lesson_67_synthetic_video_stabilization/`, `tests/test_tracks_vision_video_stabilization.py` | `README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_video_stabilization.py`<br>`python scripts/run_lesson.py vision lesson_67_synthetic_video_stabilization --dry-run` |
| L03 | `batch390-video-frame-interpolation-vision` | `.worktrees/batch390-video-frame-interpolation-vision` | `tracks/vision` | `video_frame_interpolation_vision_lesson` | `tracks/vision/lesson_68_synthetic_video_frame_interpolation/`, `tests/test_tracks_vision_video_frame_interpolation.py` | `README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_video_frame_interpolation.py`<br>`python scripts/run_lesson.py vision lesson_68_synthetic_video_frame_interpolation --dry-run` |
| L04 | `batch390-video-restoration-vision` | `.worktrees/batch390-video-restoration-vision` | `tracks/vision` | `video_restoration_vision_lesson` | `tracks/vision/lesson_69_synthetic_video_restoration/`, `tests/test_tracks_vision_video_restoration.py` | `README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_video_restoration.py`<br>`python scripts/run_lesson.py vision lesson_69_synthetic_video_restoration --dry-run` |
| L05 | `batch390-video-understanding-vision` | `.worktrees/batch390-video-understanding-vision` | `tracks/vision` | `video_understanding_vision_lesson` | `tracks/vision/lesson_70_synthetic_video_understanding/`, `tests/test_tracks_vision_video_understanding.py` | `README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_video_understanding.py`<br>`python scripts/run_lesson.py vision lesson_70_synthetic_video_understanding --dry-run` |
| L06 | `batch390-video-summarization-vision` | `.worktrees/batch390-video-summarization-vision` | `tracks/vision` | `video_summarization_vision_lesson` | `tracks/vision/lesson_71_synthetic_video_summarization/`, `tests/test_tracks_vision_video_summarization.py` | `README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_video_summarization.py`<br>`python scripts/run_lesson.py vision lesson_71_synthetic_video_summarization --dry-run` |
| L07 | `batch390-video-enhancement-vision` | `.worktrees/batch390-video-enhancement-vision` | `tracks/vision` | `video_enhancement_vision_lesson` | `tracks/vision/lesson_72_synthetic_video_enhancement/`, `tests/test_tracks_vision_video_enhancement.py` | `README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_video_enhancement.py`<br>`python scripts/run_lesson.py vision lesson_72_synthetic_video_enhancement --dry-run` |
| L08 | `batch390-video-object-segmentation-vision` | `.worktrees/batch390-video-object-segmentation-vision` | `tracks/vision` | `video_object_segmentation_vision_lesson` | `tracks/vision/lesson_73_synthetic_video_object_segmentation/`, `tests/test_tracks_vision_video_object_segmentation.py` | `README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_video_object_segmentation.py`<br>`python scripts/run_lesson.py vision lesson_73_synthetic_video_object_segmentation --dry-run` |
| L09 | `batch390-video-instance-segmentation-vision` | `.worktrees/batch390-video-instance-segmentation-vision` | `tracks/vision` | `video_instance_segmentation_vision_lesson` | `tracks/vision/lesson_74_synthetic_video_instance_segmentation/`, `tests/test_tracks_vision_video_instance_segmentation.py` | `README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_video_instance_segmentation.py`<br>`python scripts/run_lesson.py vision lesson_74_synthetic_video_instance_segmentation --dry-run` |
| L10 | `batch390-video-matting-vision` | `.worktrees/batch390-video-matting-vision` | `tracks/vision` | `video_matting_vision_lesson` | `tracks/vision/lesson_75_synthetic_video_matting/`, `tests/test_tracks_vision_video_matting.py` | `README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_video_matting.py`<br>`python scripts/run_lesson.py vision lesson_75_synthetic_video_matting --dry-run` |

## Integration Branch Ownership

`batch390-integration` owns only the shared surfaces required after the 10 lane merges:

- `README.md`
- `tracks/vision/README.md`
- `tests/test_scripts_run_lesson.py`
- any shared verification-only updates required to make the merged batch pass on `main`

## Merge Order

1. `batch390-video-object-detection-vision`
2. `batch390-video-stabilization-vision`
3. `batch390-video-frame-interpolation-vision`
4. `batch390-video-restoration-vision`
5. `batch390-video-understanding-vision`
6. `batch390-video-summarization-vision`
7. `batch390-video-enhancement-vision`
8. `batch390-video-object-segmentation-vision`
9. `batch390-video-instance-segmentation-vision`
10. `batch390-video-matting-vision`

## Exit Criteria

- each lane has at least one local commit on its own branch
- each lane passes its manifest smoke commands from its own worktree
- integration updates the top-level lesson counts and vision lesson tables for all 10 new lessons
- integration branch passes fresh shared verification before merge back to `main`
