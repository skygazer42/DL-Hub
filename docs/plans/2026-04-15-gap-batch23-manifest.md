# Batch 23 Track-Fill Manifest

Batch 23 keeps the fixed `10 worktree + 10 lane` loop moving after the Batch 22 merge-back. The
remaining direct user-tag mappings now sit mostly in self-supervision, restoration, interaction,
and adversarial robustness topics, with one LLM neighbor-fill lane to preserve the same balanced
track-first cadence.

Selection rules locked for this batch:

- fixed `10 worktree + 10 lane` execution
- track-first coverage across `tracks/llm`, `tracks/generative`, `tracks/multimodal`,
  `tracks/vision`, and `tracks/nlp`
- max 2 lanes per surface
- remaining direct user-tag mappings first, then nearest teaching-neighbor fill for the same track

Start point:

- base branch: `main`
- branch anchor commit: `plan: seed batch23 track-fill manifest`
- integration branch: `batch230-integration`
- worktree root: `.worktrees/`

## Lane Table

| lane_id | branch | worktree | surface | slug | lane_owned_paths | integration_owned_paths | verification_cmds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L01 | `batch230-replaced-token-detection` | `.worktrees/batch230-replaced-token-detection` | `tracks/llm` | `self_supervised_transformer_llm_lesson` | `tracks/llm/lesson_14_toy_replaced_token_detection_transformer/`, `tests/test_tracks_llm_replaced_token_detection.py` | `README.md`, `tracks/llm/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_llm_replaced_token_detection.py`<br>`python scripts/run_lesson.py llm lesson_14_toy_replaced_token_detection_transformer --dry-run` |
| L02 | `batch230-llm-judge` | `.worktrees/batch230-llm-judge` | `tracks/llm` | `llm_judge_lesson` | `tracks/llm/lesson_15_toy_llm_judge/`, `tests/test_tracks_llm_judge.py` | `README.md`, `tracks/llm/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_llm_judge.py`<br>`python scripts/run_lesson.py llm lesson_15_toy_llm_judge --dry-run` |
| L03 | `batch230-diffusion-super-resolution` | `.worktrees/batch230-diffusion-super-resolution` | `tracks/generative` | `diffusion_super_resolution_generative_lesson` | `tracks/generative/lesson_15_toy_diffusion_super_resolution/`, `tests/test_tracks_generative_diffusion_super_resolution.py` | `README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_generative_diffusion_super_resolution.py`<br>`python scripts/run_lesson.py generative lesson_15_toy_diffusion_super_resolution --dry-run` |
| L04 | `batch230-diffusion-deblurring` | `.worktrees/batch230-diffusion-deblurring` | `tracks/generative` | `diffusion_deblurring_generative_lesson` | `tracks/generative/lesson_16_toy_diffusion_deblurring/`, `tests/test_tracks_generative_diffusion_deblurring.py` | `README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_generative_diffusion_deblurring.py`<br>`python scripts/run_lesson.py generative lesson_16_toy_diffusion_deblurring --dry-run` |
| L05 | `batch230-human-object-interaction` | `.worktrees/batch230-human-object-interaction` | `tracks/multimodal` | `human_object_interaction_multimodal_lesson` | `tracks/multimodal/lesson_29_human_object_interaction_reasoning/`, `tests/test_tracks_multimodal_human_object_interaction.py` | `README.md`, `tracks/multimodal/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_multimodal_human_object_interaction.py`<br>`python scripts/run_lesson.py multimodal lesson_29_human_object_interaction_reasoning --dry-run` |
| L06 | `batch230-gaze-estimation` | `.worktrees/batch230-gaze-estimation` | `tracks/multimodal` | `gaze_estimation_multimodal_lesson` | `tracks/multimodal/lesson_30_vision_language_gaze_estimation/`, `tests/test_tracks_multimodal_gaze_estimation.py` | `README.md`, `tracks/multimodal/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_multimodal_gaze_estimation.py`<br>`python scripts/run_lesson.py multimodal lesson_30_vision_language_gaze_estimation --dry-run` |
| L07 | `batch230-salient-object-boxes` | `.worktrees/batch230-salient-object-boxes` | `tracks/vision` | `salient_object_boxes_lesson` | `tracks/vision/lesson_30_synthetic_salient_object_detection_boxes/`, `tests/test_tracks_vision_salient_object_boxes.py` | `README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_salient_object_boxes.py`<br>`python scripts/run_lesson.py vision lesson_30_synthetic_salient_object_detection_boxes --dry-run` |
| L08 | `batch230-interactive-segmentation` | `.worktrees/batch230-interactive-segmentation` | `tracks/vision` | `interactive_segmentation_lesson` | `tracks/vision/lesson_31_synthetic_interactive_segmentation/`, `tests/test_tracks_vision_interactive_segmentation.py` | `README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_interactive_segmentation.py`<br>`python scripts/run_lesson.py vision lesson_31_synthetic_interactive_segmentation --dry-run` |
| L09 | `batch230-adversarial-text-classification` | `.worktrees/batch230-adversarial-text-classification` | `tracks/nlp` | `adversarial_text_classification_lesson` | `tracks/nlp/lesson_20_toy_adversarial_text_classification/`, `tests/test_tracks_nlp_adversarial_text_classification.py` | `README.md`, `tracks/nlp/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_nlp_adversarial_text_classification.py`<br>`python scripts/run_lesson.py nlp lesson_20_toy_adversarial_text_classification --dry-run` |
| L10 | `batch230-adversarial-example-detection` | `.worktrees/batch230-adversarial-example-detection` | `tracks/nlp` | `adversarial_example_detection_lesson` | `tracks/nlp/lesson_21_toy_adversarial_example_detection/`, `tests/test_tracks_nlp_adversarial_example_detection.py` | `README.md`, `tracks/nlp/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_nlp_adversarial_example_detection.py`<br>`python scripts/run_lesson.py nlp lesson_21_toy_adversarial_example_detection --dry-run` |

## Integration Branch Ownership

`batch230-integration` owns only the shared surfaces required after the 10 lane merges:

- `README.md`
- `tracks/llm/README.md`
- `tracks/generative/README.md`
- `tracks/multimodal/README.md`
- `tracks/vision/README.md`
- `tracks/nlp/README.md`
- `tests/test_scripts_run_lesson.py`
- any shared verification-only updates required to make the merged track batch pass on `main`

## Merge Order

1. `batch230-replaced-token-detection`
2. `batch230-llm-judge`
3. `batch230-diffusion-super-resolution`
4. `batch230-diffusion-deblurring`
5. `batch230-human-object-interaction`
6. `batch230-gaze-estimation`
7. `batch230-salient-object-boxes`
8. `batch230-interactive-segmentation`
9. `batch230-adversarial-text-classification`
10. `batch230-adversarial-example-detection`

## Exit Criteria

- each lane has at least one local commit on its own branch
- each lane passes its manifest smoke commands from its own worktree
- integration updates the top-level lesson counts and track lesson tables for all 10 new lessons
- integration branch passes fresh shared verification before merge back to `main`
