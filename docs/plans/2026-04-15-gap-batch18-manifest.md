# Batch 18 Track-Fill Manifest

Batch 18 keeps the fixed `10 worktree + 10 lane` rhythm introduced in Batch 17, but shifts the
lesson fill toward prompt/adaptation, audio-grounded multimodal teaching, and autonomous-driving
vision follow-ups. The selection still stays track-first so teaching coverage catches up with the
already expanded `dlhub/` surfaces.

Selection rules locked for this batch:

- fixed `10 worktree + 10 lane` execution
- track-first coverage across `tracks/llm`, `tracks/generative`, `tracks/multimodal`,
  `tracks/vision`, and `tracks/nlp`
- max 2 lanes per surface
- user tag pool first, then neighbor fill where the tag pool had no clean direct lesson mapping

Start point:

- base branch: `main`
- branch anchor commit: `plan: seed batch18 track-fill manifest`
- integration branch: `batch180-integration`
- worktree root: `.worktrees/`

## Lane Table

| lane_id | branch | worktree | surface | slug | lane_owned_paths | integration_owned_paths | verification_cmds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L01 | `batch180-instruction-tuning` | `.worktrees/batch180-instruction-tuning` | `tracks/llm` | `instruction_tuning_llm_lesson` | `tracks/llm/lesson_04_toy_instruction_tuning/`, `tests/test_tracks_llm_instruction_tuning.py` | `README.md`, `tracks/llm/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_llm_instruction_tuning.py`<br>`python scripts/run_lesson.py llm lesson_04_toy_instruction_tuning --dry-run` |
| L02 | `batch180-prefix-tuning` | `.worktrees/batch180-prefix-tuning` | `tracks/llm` | `prefix_tuning_llm_lesson` | `tracks/llm/lesson_05_toy_prefix_tuning/`, `tests/test_tracks_llm_prefix_tuning.py` | `README.md`, `tracks/llm/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_llm_prefix_tuning.py`<br>`python scripts/run_lesson.py llm lesson_05_toy_prefix_tuning --dry-run` |
| L03 | `batch180-consistency-model` | `.worktrees/batch180-consistency-model` | `tracks/generative` | `consistency_model_lesson` | `tracks/generative/lesson_05_toy_consistency_model/`, `tests/test_tracks_generative_consistency_model.py` | `README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_generative_consistency_model.py`<br>`python scripts/run_lesson.py generative lesson_05_toy_consistency_model --dry-run` |
| L04 | `batch180-flow-matching` | `.worktrees/batch180-flow-matching` | `tracks/generative` | `flow_matching_lesson` | `tracks/generative/lesson_06_toy_flow_matching/`, `tests/test_tracks_generative_flow_matching.py` | `README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_generative_flow_matching.py`<br>`python scripts/run_lesson.py generative lesson_06_toy_flow_matching --dry-run` |
| L05 | `batch180-audio-text-understanding-lesson` | `.worktrees/batch180-audio-text-understanding-lesson` | `tracks/multimodal` | `audio_text_understanding_lesson` | `tracks/multimodal/lesson_19_audio_text_understanding/`, `tests/test_tracks_multimodal_audio_text_understanding.py` | `README.md`, `tracks/multimodal/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_multimodal_audio_text_understanding.py`<br>`python scripts/run_lesson.py multimodal lesson_19_audio_text_understanding --dry-run` |
| L06 | `batch180-audio-visual-learning-lesson` | `.worktrees/batch180-audio-visual-learning-lesson` | `tracks/multimodal` | `audio_visual_learning_lesson` | `tracks/multimodal/lesson_20_audio_visual_learning/`, `tests/test_tracks_multimodal_audio_visual_learning.py` | `README.md`, `tracks/multimodal/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_multimodal_audio_visual_learning.py`<br>`python scripts/run_lesson.py multimodal lesson_20_audio_visual_learning --dry-run` |
| L07 | `batch180-lane-detection-lesson` | `.worktrees/batch180-lane-detection-lesson` | `tracks/vision` | `lane_detection_lesson` | `tracks/vision/lesson_20_synthetic_lane_detection/`, `tests/test_tracks_vision_lane_detection.py` | `README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_lane_detection.py`<br>`python scripts/run_lesson.py vision lesson_20_synthetic_lane_detection --dry-run` |
| L08 | `batch180-lane-topology-lesson` | `.worktrees/batch180-lane-topology-lesson` | `tracks/vision` | `lane_topology_estimation_lesson` | `tracks/vision/lesson_21_synthetic_lane_topology_estimation/`, `tests/test_tracks_vision_lane_topology_estimation.py` | `README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_lane_topology_estimation.py`<br>`python scripts/run_lesson.py vision lesson_21_synthetic_lane_topology_estimation --dry-run` |
| L09 | `batch180-prompt-tuning-nlp` | `.worktrees/batch180-prompt-tuning-nlp` | `tracks/nlp` | `prompt_tuning_nlp_lesson` | `tracks/nlp/lesson_10_toy_prompt_tuning_classifier/`, `tests/test_tracks_nlp_prompt_tuning.py` | `README.md`, `tracks/nlp/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_nlp_prompt_tuning.py`<br>`python scripts/run_lesson.py nlp lesson_10_toy_prompt_tuning_classifier --dry-run` |
| L10 | `batch180-few-shot-text-classification` | `.worktrees/batch180-few-shot-text-classification` | `tracks/nlp` | `few_shot_text_classification_lesson` | `tracks/nlp/lesson_11_toy_few_shot_text_classification/`, `tests/test_tracks_nlp_few_shot_text_classification.py` | `README.md`, `tracks/nlp/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_nlp_few_shot_text_classification.py`<br>`python scripts/run_lesson.py nlp lesson_11_toy_few_shot_text_classification --dry-run` |

## Integration Branch Ownership

`batch180-integration` owns only the shared surfaces required after the 10 lane merges:

- `README.md`
- `tracks/llm/README.md`
- `tracks/generative/README.md`
- `tracks/multimodal/README.md`
- `tracks/vision/README.md`
- `tracks/nlp/README.md`
- `tests/test_scripts_run_lesson.py`
- any shared verification-only updates required to make the merged track batch pass on `main`

## Merge Order

1. `batch180-instruction-tuning`
2. `batch180-prefix-tuning`
3. `batch180-consistency-model`
4. `batch180-flow-matching`
5. `batch180-audio-text-understanding-lesson`
6. `batch180-audio-visual-learning-lesson`
7. `batch180-lane-detection-lesson`
8. `batch180-lane-topology-lesson`
9. `batch180-prompt-tuning-nlp`
10. `batch180-few-shot-text-classification`

## Exit Criteria

- each lane has at least one local commit on its own branch
- each lane passes its manifest smoke commands from its own worktree
- integration updates the top-level lesson counts and track lesson tables for all 10 new lessons
- integration branch passes fresh shared verification before merge back to `main`
