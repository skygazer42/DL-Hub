# Batch 19 Track-Fill Manifest

Batch 19 continues the fixed `10 worktree + 10 lane` loop, using the remaining direct gaps from
the user tag pool and one adjacent teaching continuation per surface where the direct pool only
supplied a single clean lesson mapping. The track-first policy remains unchanged so `tracks/`
keeps catching up with the already wider `dlhub/` coverage.

Selection rules locked for this batch:

- fixed `10 worktree + 10 lane` execution
- track-first coverage across `tracks/llm`, `tracks/generative`, `tracks/multimodal`,
  `tracks/vision`, and `tracks/nlp`
- max 2 lanes per surface
- user tag pool first, then neighbor fill where the tag pool had no clean direct lesson mapping

Start point:

- base branch: `main`
- branch anchor commit: `plan: seed batch19 track-fill manifest`
- integration branch: `batch190-integration`
- worktree root: `.worktrees/`

## Lane Table

| lane_id | branch | worktree | surface | slug | lane_owned_paths | integration_owned_paths | verification_cmds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L01 | `batch190-preference-optimization` | `.worktrees/batch190-preference-optimization` | `tracks/llm` | `preference_optimization_llm_lesson` | `tracks/llm/lesson_06_compact_preference_optimization/`, `tests/test_tracks_llm_preference_optimization.py` | `README.md`, `tracks/llm/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_llm_preference_optimization.py`<br>`python scripts/run_lesson.py llm lesson_06_compact_preference_optimization --dry-run` |
| L02 | `batch190-reward-modeling` | `.worktrees/batch190-reward-modeling` | `tracks/llm` | `reward_modeling_llm_lesson` | `tracks/llm/lesson_07_compact_reward_modeling/`, `tests/test_tracks_llm_reward_modeling.py` | `README.md`, `tracks/llm/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_llm_reward_modeling.py`<br>`python scripts/run_lesson.py llm lesson_07_compact_reward_modeling --dry-run` |
| L03 | `batch190-rectified-flow` | `.worktrees/batch190-rectified-flow` | `tracks/generative` | `rectified_flow_lesson` | `tracks/generative/lesson_07_compact_rectified_flow/`, `tests/test_tracks_generative_rectified_flow.py` | `README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_generative_rectified_flow.py`<br>`python scripts/run_lesson.py generative lesson_07_compact_rectified_flow --dry-run` |
| L04 | `batch190-diffusion-transformer` | `.worktrees/batch190-diffusion-transformer` | `tracks/generative` | `diffusion_transformer_lesson` | `tracks/generative/lesson_08_compact_diffusion_transformer/`, `tests/test_tracks_generative_diffusion_transformer.py` | `README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_generative_diffusion_transformer.py`<br>`python scripts/run_lesson.py generative lesson_08_compact_diffusion_transformer --dry-run` |
| L05 | `batch190-audio-grounded-retrieval` | `.worktrees/batch190-audio-grounded-retrieval` | `tracks/multimodal` | `audio_grounded_retrieval_lesson` | `tracks/multimodal/lesson_21_audio_grounded_retrieval/`, `tests/test_tracks_multimodal_audio_grounded_retrieval.py` | `README.md`, `tracks/multimodal/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_multimodal_audio_grounded_retrieval.py`<br>`python scripts/run_lesson.py multimodal lesson_21_audio_grounded_retrieval --dry-run` |
| L06 | `batch190-audio-visual-event-localization` | `.worktrees/batch190-audio-visual-event-localization` | `tracks/multimodal` | `audio_visual_event_localization_lesson` | `tracks/multimodal/lesson_22_audio_visual_event_localization/`, `tests/test_tracks_multimodal_audio_visual_event_localization.py` | `README.md`, `tracks/multimodal/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_multimodal_audio_visual_event_localization.py`<br>`python scripts/run_lesson.py multimodal lesson_22_audio_visual_event_localization --dry-run` |
| L07 | `batch190-road-scene-understanding` | `.worktrees/batch190-road-scene-understanding` | `tracks/vision` | `road_scene_understanding_lesson` | `tracks/vision/lesson_22_synthetic_road_scene_understanding/`, `tests/test_tracks_vision_road_scene_understanding.py` | `README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_road_scene_understanding.py`<br>`python scripts/run_lesson.py vision lesson_22_synthetic_road_scene_understanding --dry-run` |
| L08 | `batch190-image-dehazing` | `.worktrees/batch190-image-dehazing` | `tracks/vision` | `image_dehazing_lesson` | `tracks/vision/lesson_23_synthetic_image_dehazing/`, `tests/test_tracks_vision_image_dehazing.py` | `README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_image_dehazing.py`<br>`python scripts/run_lesson.py vision lesson_23_synthetic_image_dehazing --dry-run` |
| L09 | `batch190-in-context-text-classification` | `.worktrees/batch190-in-context-text-classification` | `tracks/nlp` | `in_context_text_classification_lesson` | `tracks/nlp/lesson_12_compact_in_context_text_classification/`, `tests/test_tracks_nlp_in_context_text_classification.py` | `README.md`, `tracks/nlp/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_nlp_in_context_text_classification.py`<br>`python scripts/run_lesson.py nlp lesson_12_compact_in_context_text_classification --dry-run` |
| L10 | `batch190-masked-language-modeling` | `.worktrees/batch190-masked-language-modeling` | `tracks/nlp` | `masked_language_modeling_lesson` | `tracks/nlp/lesson_13_compact_masked_language_modeling/`, `tests/test_tracks_nlp_masked_language_modeling.py` | `README.md`, `tracks/nlp/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_nlp_masked_language_modeling.py`<br>`python scripts/run_lesson.py nlp lesson_13_compact_masked_language_modeling --dry-run` |

## Integration Branch Ownership

`batch190-integration` owns only the shared surfaces required after the 10 lane merges:

- `README.md`
- `tracks/llm/README.md`
- `tracks/generative/README.md`
- `tracks/multimodal/README.md`
- `tracks/vision/README.md`
- `tracks/nlp/README.md`
- `tests/test_scripts_run_lesson.py`
- any shared verification-only updates required to make the merged track batch pass on `main`

## Merge Order

1. `batch190-preference-optimization`
2. `batch190-reward-modeling`
3. `batch190-rectified-flow`
4. `batch190-diffusion-transformer`
5. `batch190-audio-grounded-retrieval`
6. `batch190-audio-visual-event-localization`
7. `batch190-road-scene-understanding`
8. `batch190-image-dehazing`
9. `batch190-in-context-text-classification`
10. `batch190-masked-language-modeling`

## Exit Criteria

- each lane has at least one local commit on its own branch
- each lane passes its manifest smoke commands from its own worktree
- integration updates the top-level lesson counts and track lesson tables for all 10 new lessons
- integration branch passes fresh shared verification before merge back to `main`
