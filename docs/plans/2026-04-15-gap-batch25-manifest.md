# Batch 25 Track-Fill Manifest

Batch 25 keeps the fixed `10 worktree + 10 lane` loop moving immediately after the Batch 24 merge
back. This round pushes further into agent-style LLM prompting, restoration generation, person and
action understanding, and low-data text learning while preserving the same balanced track-first
cadence.

Selection rules locked for this batch:

- fixed `10 worktree + 10 lane` execution
- track-first coverage across `tracks/llm`, `tracks/generative`, `tracks/multimodal`,
  `tracks/vision`, and `tracks/nlp`
- max 2 lanes per surface
- remaining direct user-tag mappings first, then nearest teaching-neighbor fill for the same track

Start point:

- base branch: `main`
- branch anchor commit: `plan: seed batch25 track-fill manifest`
- integration branch: `batch250-integration`
- worktree root: `.worktrees/`

## Lane Table

| lane_id | branch | worktree | surface | slug | lane_owned_paths | integration_owned_paths | verification_cmds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L01 | `batch250-reflection-memory-agent` | `.worktrees/batch250-reflection-memory-agent` | `tracks/llm` | `reflection_memory_agent_llm_lesson` | `tracks/llm/lesson_18_toy_reflection_memory_agent/`, `tests/test_tracks_llm_reflection_memory_agent.py` | `README.md`, `tracks/llm/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_llm_reflection_memory_agent.py`<br>`python scripts/run_lesson.py llm lesson_18_toy_reflection_memory_agent --dry-run` |
| L02 | `batch250-plan-execute-prompting` | `.worktrees/batch250-plan-execute-prompting` | `tracks/llm` | `plan_execute_prompting_llm_lesson` | `tracks/llm/lesson_19_toy_plan_execute_prompting/`, `tests/test_tracks_llm_plan_execute_prompting.py` | `README.md`, `tracks/llm/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_llm_plan_execute_prompting.py`<br>`python scripts/run_lesson.py llm lesson_19_toy_plan_execute_prompting --dry-run` |
| L03 | `batch250-diffusion-dehazing` | `.worktrees/batch250-diffusion-dehazing` | `tracks/generative` | `diffusion_dehazing_generative_lesson` | `tracks/generative/lesson_19_toy_diffusion_dehazing/`, `tests/test_tracks_generative_diffusion_dehazing.py` | `README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_generative_diffusion_dehazing.py`<br>`python scripts/run_lesson.py generative lesson_19_toy_diffusion_dehazing --dry-run` |
| L04 | `batch250-diffusion-reflection-removal` | `.worktrees/batch250-diffusion-reflection-removal` | `tracks/generative` | `diffusion_reflection_removal_generative_lesson` | `tracks/generative/lesson_20_toy_diffusion_reflection_removal/`, `tests/test_tracks_generative_diffusion_reflection_removal.py` | `README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_generative_diffusion_reflection_removal.py`<br>`python scripts/run_lesson.py generative lesson_20_toy_diffusion_reflection_removal --dry-run` |
| L05 | `batch250-pedestrian-attributes` | `.worktrees/batch250-pedestrian-attributes` | `tracks/multimodal` | `pedestrian_attribute_multimodal_lesson` | `tracks/multimodal/lesson_33_pedestrian_attribute_recognition/`, `tests/test_tracks_multimodal_pedestrian_attributes.py` | `README.md`, `tracks/multimodal/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_multimodal_pedestrian_attributes.py`<br>`python scripts/run_lesson.py multimodal lesson_33_pedestrian_attribute_recognition --dry-run` |
| L06 | `batch250-action-recognition` | `.worktrees/batch250-action-recognition` | `tracks/multimodal` | `action_recognition_multimodal_lesson` | `tracks/multimodal/lesson_34_video_text_action_recognition/`, `tests/test_tracks_multimodal_action_recognition.py` | `README.md`, `tracks/multimodal/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_multimodal_action_recognition.py`<br>`python scripts/run_lesson.py multimodal lesson_34_video_text_action_recognition --dry-run` |
| L07 | `batch250-license-plate-recognition` | `.worktrees/batch250-license-plate-recognition` | `tracks/vision` | `license_plate_recognition_lesson` | `tracks/vision/lesson_34_synthetic_license_plate_recognition/`, `tests/test_tracks_vision_license_plate_recognition.py` | `README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_license_plate_recognition.py`<br>`python scripts/run_lesson.py vision lesson_34_synthetic_license_plate_recognition --dry-run` |
| L08 | `batch250-6d-pose-estimation` | `.worktrees/batch250-6d-pose-estimation` | `tracks/vision` | `pose_estimation_6d_lesson` | `tracks/vision/lesson_35_synthetic_6d_pose_estimation/`, `tests/test_tracks_vision_6d_pose_estimation.py` | `README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_6d_pose_estimation.py`<br>`python scripts/run_lesson.py vision lesson_35_synthetic_6d_pose_estimation --dry-run` |
| L09 | `batch250-meta-few-shot-text` | `.worktrees/batch250-meta-few-shot-text` | `tracks/nlp` | `few_shot_nlp_lesson` | `tracks/nlp/lesson_24_toy_meta_few_shot_text_classification/`, `tests/test_tracks_nlp_meta_few_shot.py` | `README.md`, `tracks/nlp/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_nlp_meta_few_shot.py`<br>`python scripts/run_lesson.py nlp lesson_24_toy_meta_few_shot_text_classification --dry-run` |
| L10 | `batch250-low-shot-intent` | `.worktrees/batch250-low-shot-intent` | `tracks/nlp` | `low_shot_nlp_lesson` | `tracks/nlp/lesson_25_toy_low_shot_intent_detection/`, `tests/test_tracks_nlp_low_shot_intent.py` | `README.md`, `tracks/nlp/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_nlp_low_shot_intent.py`<br>`python scripts/run_lesson.py nlp lesson_25_toy_low_shot_intent_detection --dry-run` |

## Integration Branch Ownership

`batch250-integration` owns only the shared surfaces required after the 10 lane merges:

- `README.md`
- `tracks/llm/README.md`
- `tracks/generative/README.md`
- `tracks/multimodal/README.md`
- `tracks/vision/README.md`
- `tracks/nlp/README.md`
- `tests/test_scripts_run_lesson.py`
- any shared verification-only updates required to make the merged track batch pass on `main`

## Merge Order

1. `batch250-reflection-memory-agent`
2. `batch250-plan-execute-prompting`
3. `batch250-diffusion-dehazing`
4. `batch250-diffusion-reflection-removal`
5. `batch250-pedestrian-attributes`
6. `batch250-action-recognition`
7. `batch250-license-plate-recognition`
8. `batch250-6d-pose-estimation`
9. `batch250-meta-few-shot-text`
10. `batch250-low-shot-intent`

## Exit Criteria

- each lane has at least one local commit on its own branch
- each lane passes its manifest smoke commands from its own worktree
- integration updates the top-level lesson counts and track lesson tables for all 10 new lessons
- integration branch passes fresh shared verification before merge back to `main`
