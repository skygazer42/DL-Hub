# Batch 21 Track-Fill Manifest

Batch 21 keeps the fixed `10 worktree + 10 lane` loop, but the direct user-tag pool on the
teaching tracks is now mostly exhausted. This batch therefore mixes the remaining clean direct
track mappings with neighbor-fill continuations that extend the next adjacent teaching concepts on
each surface while still staying anchored to the user tag vocabulary.

Selection rules locked for this batch:

- fixed `10 worktree + 10 lane` execution
- track-first coverage across `tracks/llm`, `tracks/generative`, `tracks/multimodal`,
  `tracks/vision`, and `tracks/nlp`
- max 2 lanes per surface
- remaining direct user-tag mappings first, then nearest teaching-neighbor fill for the same track

Start point:

- base branch: `main`
- branch anchor commit: `plan: seed batch21 track-fill manifest`
- integration branch: `batch210-integration`
- worktree root: `.worktrees/`

## Lane Table

| lane_id | branch | worktree | surface | slug | lane_owned_paths | integration_owned_paths | verification_cmds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L01 | `batch210-grpo-alignment` | `.worktrees/batch210-grpo-alignment` | `tracks/llm` | `grpo_alignment_llm_lesson` | `tracks/llm/lesson_10_compact_grpo_alignment/`, `tests/test_tracks_llm_grpo_alignment.py` | `README.md`, `tracks/llm/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_llm_grpo_alignment.py`<br>`python scripts/run_lesson.py llm lesson_10_compact_grpo_alignment --dry-run` |
| L02 | `batch210-rag-language-model` | `.worktrees/batch210-rag-language-model` | `tracks/llm` | `rag_language_model_llm_lesson` | `tracks/llm/lesson_11_compact_rag_language_model/`, `tests/test_tracks_llm_rag_language_model.py` | `README.md`, `tracks/llm/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_llm_rag_language_model.py`<br>`python scripts/run_lesson.py llm lesson_11_compact_rag_language_model --dry-run` |
| L03 | `batch210-controlnet` | `.worktrees/batch210-controlnet` | `tracks/generative` | `controlnet_generative_lesson` | `tracks/generative/lesson_11_compact_controlnet/`, `tests/test_tracks_generative_controlnet.py` | `README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_generative_controlnet.py`<br>`python scripts/run_lesson.py generative lesson_11_compact_controlnet --dry-run` |
| L04 | `batch210-layout-to-image` | `.worktrees/batch210-layout-to-image` | `tracks/generative` | `layout_to_image_generative_lesson` | `tracks/generative/lesson_12_compact_layout_to_image/`, `tests/test_tracks_generative_layout_to_image.py` | `README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_generative_layout_to_image.py`<br>`python scripts/run_lesson.py generative lesson_12_compact_layout_to_image --dry-run` |
| L05 | `batch210-vision-language-navigation` | `.worktrees/batch210-vision-language-navigation` | `tracks/multimodal` | `vision_language_navigation_lesson` | `tracks/multimodal/lesson_25_vision_language_navigation/`, `tests/test_tracks_multimodal_vision_language_navigation.py` | `README.md`, `tracks/multimodal/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_multimodal_vision_language_navigation.py`<br>`python scripts/run_lesson.py multimodal lesson_25_vision_language_navigation --dry-run` |
| L06 | `batch210-image-text-reranking` | `.worktrees/batch210-image-text-reranking` | `tracks/multimodal` | `image_text_reranking_lesson` | `tracks/multimodal/lesson_26_image_text_reranking/`, `tests/test_tracks_multimodal_image_text_reranking.py` | `README.md`, `tracks/multimodal/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_multimodal_image_text_reranking.py`<br>`python scripts/run_lesson.py multimodal lesson_26_image_text_reranking --dry-run` |
| L07 | `batch210-text-detection` | `.worktrees/batch210-text-detection` | `tracks/vision` | `text_detection_lesson` | `tracks/vision/lesson_26_synthetic_text_detection/`, `tests/test_tracks_vision_text_detection.py` | `README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_text_detection.py`<br>`python scripts/run_lesson.py vision lesson_26_synthetic_text_detection --dry-run` |
| L08 | `batch210-edge-detection` | `.worktrees/batch210-edge-detection` | `tracks/vision` | `edge_detection_lesson` | `tracks/vision/lesson_27_synthetic_edge_detection/`, `tests/test_tracks_vision_edge_detection.py` | `README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_edge_detection.py`<br>`python scripts/run_lesson.py vision lesson_27_synthetic_edge_detection --dry-run` |
| L09 | `batch210-text-clustering` | `.worktrees/batch210-text-clustering` | `tracks/nlp` | `text_clustering_lesson` | `tracks/nlp/lesson_16_compact_text_clustering/`, `tests/test_tracks_nlp_text_clustering.py` | `README.md`, `tracks/nlp/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_nlp_text_clustering.py`<br>`python scripts/run_lesson.py nlp lesson_16_compact_text_clustering --dry-run` |
| L10 | `batch210-text-anomaly-detection` | `.worktrees/batch210-text-anomaly-detection` | `tracks/nlp` | `text_anomaly_detection_lesson` | `tracks/nlp/lesson_17_compact_text_anomaly_detection/`, `tests/test_tracks_nlp_text_anomaly_detection.py` | `README.md`, `tracks/nlp/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_nlp_text_anomaly_detection.py`<br>`python scripts/run_lesson.py nlp lesson_17_compact_text_anomaly_detection --dry-run` |

## Integration Branch Ownership

`batch210-integration` owns only the shared surfaces required after the 10 lane merges:

- `README.md`
- `tracks/llm/README.md`
- `tracks/generative/README.md`
- `tracks/multimodal/README.md`
- `tracks/vision/README.md`
- `tracks/nlp/README.md`
- `tests/test_scripts_run_lesson.py`
- any shared verification-only updates required to make the merged track batch pass on `main`

## Merge Order

1. `batch210-grpo-alignment`
2. `batch210-rag-language-model`
3. `batch210-controlnet`
4. `batch210-layout-to-image`
5. `batch210-vision-language-navigation`
6. `batch210-image-text-reranking`
7. `batch210-text-detection`
8. `batch210-edge-detection`
9. `batch210-text-clustering`
10. `batch210-text-anomaly-detection`

## Exit Criteria

- each lane has at least one local commit on its own branch
- each lane passes its manifest smoke commands from its own worktree
- integration updates the top-level lesson counts and track lesson tables for all 10 new lessons
- integration branch passes fresh shared verification before merge back to `main`
