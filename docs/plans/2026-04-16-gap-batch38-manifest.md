# Batch 38 Trackification Manifest

Batch 38 reopens the fixed `10 worktree + 10 lane` loop after the original register was exhausted by
switching to a second-pass trackification mode. In this mode, a subset of directions already
implemented as first-class `dlhub/*` surfaces is decomposed into new teaching-first `tracks/*`
lessons so the topic pool continues to land as runnable educational lanes.

Selection rules for this batch:

- fixed `10 worktree + 10 lane` execution
- prefer direct user-tag rows currently marked `covered` on `dlhub/*` surfaces
- convert those rows into new `tracks/*` lessons with minimal synthetic contracts
- bias toward lessons that can be derived from adjacent teaching neighbors already on the track

Start point:

- base branch: `main`
- branch anchor commit: `plan: seed batch38 exhaustion checkpoint`
- integration branch: `batch380-integration`
- worktree root: `.worktrees/`

## Lane Table

| lane_id | branch | worktree | surface | slug | lane_owned_paths | integration_owned_paths | verification_cmds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L01 | `batch380-image-deraining-vision` | `.worktrees/batch380-image-deraining-vision` | `tracks/vision` | `image_deraining_vision_lesson` | `tracks/vision/lesson_60_synthetic_image_deraining/`, `tests/test_tracks_vision_image_deraining.py` | `README.md`, `tracks/vision/README.md`, `tracks/pointcloud/README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_image_deraining.py`<br>`python scripts/run_lesson.py vision lesson_60_synthetic_image_deraining --dry-run` |
| L02 | `batch380-image-retrieval-vision` | `.worktrees/batch380-image-retrieval-vision` | `tracks/vision` | `image_retrieval_vision_lesson` | `tracks/vision/lesson_61_synthetic_image_retrieval/`, `tests/test_tracks_vision_image_retrieval.py` | `README.md`, `tracks/vision/README.md`, `tracks/pointcloud/README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_image_retrieval.py`<br>`python scripts/run_lesson.py vision lesson_61_synthetic_image_retrieval --dry-run` |
| L03 | `batch380-image-matching-vision` | `.worktrees/batch380-image-matching-vision` | `tracks/vision` | `image_matching_vision_lesson` | `tracks/vision/lesson_62_synthetic_image_matching/`, `tests/test_tracks_vision_image_matching.py` | `README.md`, `tracks/vision/README.md`, `tracks/pointcloud/README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_image_matching.py`<br>`python scripts/run_lesson.py vision lesson_62_synthetic_image_matching --dry-run` |
| L04 | `batch380-image-stitching-vision` | `.worktrees/batch380-image-stitching-vision` | `tracks/vision` | `image_stitching_vision_lesson` | `tracks/vision/lesson_63_synthetic_image_stitching/`, `tests/test_tracks_vision_image_stitching.py` | `README.md`, `tracks/vision/README.md`, `tracks/pointcloud/README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_image_stitching.py`<br>`python scripts/run_lesson.py vision lesson_63_synthetic_image_stitching --dry-run` |
| L05 | `batch380-fine-grained-vision` | `.worktrees/batch380-fine-grained-vision` | `tracks/vision` | `fine_grained_recognition_vision_lesson` | `tracks/vision/lesson_64_synthetic_fine_grained_recognition/`, `tests/test_tracks_vision_fine_grained_recognition.py` | `README.md`, `tracks/vision/README.md`, `tracks/pointcloud/README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_fine_grained_recognition.py`<br>`python scripts/run_lesson.py vision lesson_64_synthetic_fine_grained_recognition --dry-run` |
| L06 | `batch380-few-shot-vision` | `.worktrees/batch380-few-shot-vision` | `tracks/vision` | `few_shot_recognition_vision_lesson` | `tracks/vision/lesson_65_synthetic_few_shot_recognition/`, `tests/test_tracks_vision_few_shot_recognition.py` | `README.md`, `tracks/vision/README.md`, `tracks/pointcloud/README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_few_shot_recognition.py`<br>`python scripts/run_lesson.py vision lesson_65_synthetic_few_shot_recognition --dry-run` |
| L07 | `batch380-pointcloud-completion` | `.worktrees/batch380-pointcloud-completion` | `tracks/pointcloud` | `pointcloud_completion_track_lesson` | `tracks/pointcloud/lesson_24_toy_pointcloud_completion/`, `tests/test_tracks_pointcloud_completion.py` | `README.md`, `tracks/vision/README.md`, `tracks/pointcloud/README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_pointcloud_completion.py`<br>`python scripts/run_lesson.py pointcloud lesson_24_toy_pointcloud_completion --dry-run` |
| L08 | `batch380-scene-flow-pointcloud` | `.worktrees/batch380-scene-flow-pointcloud` | `tracks/pointcloud` | `scene_flow_pointcloud_lesson` | `tracks/pointcloud/lesson_25_toy_scene_flow_estimation/`, `tests/test_tracks_pointcloud_scene_flow.py` | `README.md`, `tracks/vision/README.md`, `tracks/pointcloud/README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_pointcloud_scene_flow.py`<br>`python scripts/run_lesson.py pointcloud lesson_25_toy_scene_flow_estimation --dry-run` |
| L09 | `batch380-video-diffusion-generative` | `.worktrees/batch380-video-diffusion-generative` | `tracks/generative` | `video_diffusion_generative_lesson` | `tracks/generative/lesson_45_toy_video_diffusion/`, `tests/test_tracks_generative_video_diffusion.py` | `README.md`, `tracks/vision/README.md`, `tracks/pointcloud/README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_generative_video_diffusion.py`<br>`python scripts/run_lesson.py generative lesson_45_toy_video_diffusion --dry-run` |
| L10 | `batch380-image-to-video-generative` | `.worktrees/batch380-image-to-video-generative` | `tracks/generative` | `image_to_video_generative_lesson` | `tracks/generative/lesson_46_toy_image_to_video_diffusion/`, `tests/test_tracks_generative_image_to_video_diffusion.py` | `README.md`, `tracks/vision/README.md`, `tracks/pointcloud/README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_generative_image_to_video_diffusion.py`<br>`python scripts/run_lesson.py generative lesson_46_toy_image_to_video_diffusion --dry-run` |

## Integration Branch Ownership

`batch380-integration` owns only the shared surfaces required after the 10 lane merges:

- `README.md`
- `tracks/vision/README.md`
- `tracks/pointcloud/README.md`
- `tracks/generative/README.md`
- `tests/test_scripts_run_lesson.py`
- any shared verification-only updates required to make the merged batch pass on `main`

## Merge Order

1. `batch380-image-deraining-vision`
2. `batch380-image-retrieval-vision`
3. `batch380-image-matching-vision`
4. `batch380-image-stitching-vision`
5. `batch380-fine-grained-vision`
6. `batch380-few-shot-vision`
7. `batch380-pointcloud-completion`
8. `batch380-scene-flow-pointcloud`
9. `batch380-video-diffusion-generative`
10. `batch380-image-to-video-generative`

## Exit Criteria

- each lane has at least one local commit on its own branch
- each lane passes its manifest smoke commands from its own worktree
- integration updates the top-level lesson counts and track lesson tables for all 10 new lessons
- integration branch passes fresh shared verification before merge back to `main`
