# Batch 24 Track-Fill Manifest

Batch 24 keeps the fixed `10 worktree + 10 lane` loop moving after the Batch 23 merge-back. This
round expands the remaining direct user-tag pool into memory-centric chat training, restoration
generation, person-centric multimodal retrieval, action localization, face analysis, and
weak/self-supervised text learning while preserving the same balanced track-first cadence.

Selection rules locked for this batch:

- fixed `10 worktree + 10 lane` execution
- track-first coverage across `tracks/llm`, `tracks/generative`, `tracks/multimodal`,
  `tracks/vision`, and `tracks/nlp`
- max 2 lanes per surface
- remaining direct user-tag mappings first, then nearest teaching-neighbor fill for the same track

Start point:

- base branch: `main`
- branch anchor commit: `plan: seed batch24 track-fill manifest`
- integration branch: `batch240-integration`
- worktree root: `.worktrees/`

## Lane Table

| lane_id | branch | worktree | surface | slug | lane_owned_paths | integration_owned_paths | verification_cmds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L01 | `batch240-multi-turn-memory-sft` | `.worktrees/batch240-multi-turn-memory-sft` | `tracks/llm` | `multi_turn_memory_sft_llm_lesson` | `tracks/llm/lesson_16_compact_multi_turn_memory_sft/`, `tests/test_tracks_llm_multi_turn_memory.py` | `README.md`, `tracks/llm/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_llm_multi_turn_memory.py`<br>`python scripts/run_lesson.py llm lesson_16_compact_multi_turn_memory_sft --dry-run` |
| L02 | `batch240-self-refine-prompting` | `.worktrees/batch240-self-refine-prompting` | `tracks/llm` | `self_refine_prompting_llm_lesson` | `tracks/llm/lesson_17_compact_self_refine_prompting/`, `tests/test_tracks_llm_self_refine_prompting.py` | `README.md`, `tracks/llm/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_llm_self_refine_prompting.py`<br>`python scripts/run_lesson.py llm lesson_17_compact_self_refine_prompting --dry-run` |
| L03 | `batch240-diffusion-denoising` | `.worktrees/batch240-diffusion-denoising` | `tracks/generative` | `diffusion_denoising_generative_lesson` | `tracks/generative/lesson_17_compact_diffusion_denoising/`, `tests/test_tracks_generative_diffusion_denoising.py` | `README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_generative_diffusion_denoising.py`<br>`python scripts/run_lesson.py generative lesson_17_compact_diffusion_denoising --dry-run` |
| L04 | `batch240-diffusion-deraining` | `.worktrees/batch240-diffusion-deraining` | `tracks/generative` | `diffusion_deraining_generative_lesson` | `tracks/generative/lesson_18_compact_diffusion_deraining/`, `tests/test_tracks_generative_diffusion_deraining.py` | `README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_generative_diffusion_deraining.py`<br>`python scripts/run_lesson.py generative lesson_18_compact_diffusion_deraining --dry-run` |
| L05 | `batch240-person-search-retrieval` | `.worktrees/batch240-person-search-retrieval` | `tracks/multimodal` | `person_search_multimodal_lesson` | `tracks/multimodal/lesson_31_person_search_attribute_retrieval/`, `tests/test_tracks_multimodal_person_search.py` | `README.md`, `tracks/multimodal/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_multimodal_person_search.py`<br>`python scripts/run_lesson.py multimodal lesson_31_person_search_attribute_retrieval --dry-run` |
| L06 | `batch240-action-localization` | `.worktrees/batch240-action-localization` | `tracks/multimodal` | `temporal_action_localization_multimodal_lesson` | `tracks/multimodal/lesson_32_video_text_action_localization/`, `tests/test_tracks_multimodal_action_localization.py` | `README.md`, `tracks/multimodal/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_multimodal_action_localization.py`<br>`python scripts/run_lesson.py multimodal lesson_32_video_text_action_localization --dry-run` |
| L07 | `batch240-face-landmarks` | `.worktrees/batch240-face-landmarks` | `tracks/vision` | `face_landmark_detection_lesson` | `tracks/vision/lesson_32_synthetic_face_landmark_detection/`, `tests/test_tracks_vision_face_landmark_detection.py` | `README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_face_landmark_detection.py`<br>`python scripts/run_lesson.py vision lesson_32_synthetic_face_landmark_detection --dry-run` |
| L08 | `batch240-face-liveness` | `.worktrees/batch240-face-liveness` | `tracks/vision` | `face_liveness_detection_lesson` | `tracks/vision/lesson_33_synthetic_face_liveness_detection/`, `tests/test_tracks_vision_face_liveness_detection.py` | `README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_face_liveness_detection.py`<br>`python scripts/run_lesson.py vision lesson_33_synthetic_face_liveness_detection --dry-run` |
| L09 | `batch240-weak-supervision` | `.worktrees/batch240-weak-supervision` | `tracks/nlp` | `weak_supervision_text_classification_lesson` | `tracks/nlp/lesson_22_compact_weak_supervision_text_classification/`, `tests/test_tracks_nlp_weak_supervision.py` | `README.md`, `tracks/nlp/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_nlp_weak_supervision.py`<br>`python scripts/run_lesson.py nlp lesson_22_compact_weak_supervision_text_classification --dry-run` |
| L10 | `batch240-sentence-denoising` | `.worktrees/batch240-sentence-denoising` | `tracks/nlp` | `sentence_denoising_nlp_lesson` | `tracks/nlp/lesson_23_compact_sentence_denoising_autoencoder/`, `tests/test_tracks_nlp_sentence_denoising.py` | `README.md`, `tracks/nlp/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_nlp_sentence_denoising.py`<br>`python scripts/run_lesson.py nlp lesson_23_compact_sentence_denoising_autoencoder --dry-run` |

## Integration Branch Ownership

`batch240-integration` owns only the shared surfaces required after the 10 lane merges:

- `README.md`
- `tracks/llm/README.md`
- `tracks/generative/README.md`
- `tracks/multimodal/README.md`
- `tracks/vision/README.md`
- `tracks/nlp/README.md`
- `tests/test_scripts_run_lesson.py`
- any shared verification-only updates required to make the merged track batch pass on `main`

## Merge Order

1. `batch240-multi-turn-memory-sft`
2. `batch240-self-refine-prompting`
3. `batch240-diffusion-denoising`
4. `batch240-diffusion-deraining`
5. `batch240-person-search-retrieval`
6. `batch240-action-localization`
7. `batch240-face-landmarks`
8. `batch240-face-liveness`
9. `batch240-weak-supervision`
10. `batch240-sentence-denoising`

## Exit Criteria

- each lane has at least one local commit on its own branch
- each lane passes its manifest smoke commands from its own worktree
- integration updates the top-level lesson counts and track lesson tables for all 10 new lessons
- integration branch passes fresh shared verification before merge back to `main`
