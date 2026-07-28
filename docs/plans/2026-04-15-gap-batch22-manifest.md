# Batch 22 Track-Fill Manifest

Batch 22 continues the fixed `10 worktree + 10 lane` loop after the Batch 21 merge-back. The
remaining clean direct user-tag mappings now mostly live in interpretability, OCR/text reading,
saliency/camouflage vision, and training-technique topics, so this batch mixes those direct lanes
with one LLM neighbor-fill continuation to keep the track-first balance intact.

Selection rules locked for this batch:

- fixed `10 worktree + 10 lane` execution
- track-first coverage across `tracks/llm`, `tracks/generative`, `tracks/multimodal`,
  `tracks/vision`, and `tracks/nlp`
- max 2 lanes per surface
- remaining direct user-tag mappings first, then nearest teaching-neighbor fill for the same track

Start point:

- base branch: `main`
- branch anchor commit: `plan: seed batch22 track-fill manifest`
- integration branch: `batch220-integration`
- worktree root: `.worktrees/`

## Lane Table

| lane_id | branch | worktree | surface | slug | lane_owned_paths | integration_owned_paths | verification_cmds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L01 | `batch220-transformer-interpretability` | `.worktrees/batch220-transformer-interpretability` | `tracks/llm` | `transformer_interpretability_llm_lesson` | `tracks/llm/lesson_12_compact_transformer_interpretability/`, `tests/test_tracks_llm_transformer_interpretability.py` | `README.md`, `tracks/llm/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_llm_transformer_interpretability.py`<br>`python scripts/run_lesson.py llm lesson_12_compact_transformer_interpretability --dry-run` |
| L02 | `batch220-tool-calling-agent` | `.worktrees/batch220-tool-calling-agent` | `tracks/llm` | `tool_calling_agent_llm_lesson` | `tracks/llm/lesson_13_compact_tool_calling_agent/`, `tests/test_tracks_llm_tool_calling.py` | `README.md`, `tracks/llm/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_llm_tool_calling.py`<br>`python scripts/run_lesson.py llm lesson_13_compact_tool_calling_agent --dry-run` |
| L03 | `batch220-text-to-image` | `.worktrees/batch220-text-to-image` | `tracks/generative` | `text_to_image_diffusion_generative_lesson` | `tracks/generative/lesson_13_compact_text_to_image_diffusion/`, `tests/test_tracks_generative_text_to_image.py` | `README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_generative_text_to_image.py`<br>`python scripts/run_lesson.py generative lesson_13_compact_text_to_image_diffusion --dry-run` |
| L04 | `batch220-diffusion-inpainting` | `.worktrees/batch220-diffusion-inpainting` | `tracks/generative` | `diffusion_inpainting_generative_lesson` | `tracks/generative/lesson_14_compact_diffusion_inpainting/`, `tests/test_tracks_generative_inpainting.py` | `README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_generative_inpainting.py`<br>`python scripts/run_lesson.py generative lesson_14_compact_diffusion_inpainting --dry-run` |
| L05 | `batch220-scene-text-vlm-recognition` | `.worktrees/batch220-scene-text-vlm-recognition` | `tracks/multimodal` | `scene_text_recognition_multimodal_lesson` | `tracks/multimodal/lesson_27_scene_text_vlm_recognition/`, `tests/test_tracks_multimodal_scene_text_recognition.py` | `README.md`, `tracks/multimodal/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_multimodal_scene_text_recognition.py`<br>`python scripts/run_lesson.py multimodal lesson_27_scene_text_vlm_recognition --dry-run` |
| L06 | `batch220-document-vlm-reasoning` | `.worktrees/batch220-document-vlm-reasoning` | `tracks/multimodal` | `document_ocr_reasoning_multimodal_lesson` | `tracks/multimodal/lesson_28_document_vlm_reasoning/`, `tests/test_tracks_multimodal_document_vlm_reasoning.py` | `README.md`, `tracks/multimodal/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_multimodal_document_vlm_reasoning.py`<br>`python scripts/run_lesson.py multimodal lesson_28_document_vlm_reasoning --dry-run` |
| L07 | `batch220-salient-object-detection` | `.worktrees/batch220-salient-object-detection` | `tracks/vision` | `salient_object_detection_lesson` | `tracks/vision/lesson_28_synthetic_salient_object_detection/`, `tests/test_tracks_vision_salient_object_detection.py` | `README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_salient_object_detection.py`<br>`python scripts/run_lesson.py vision lesson_28_synthetic_salient_object_detection --dry-run` |
| L08 | `batch220-camouflaged-object-detection` | `.worktrees/batch220-camouflaged-object-detection` | `tracks/vision` | `camouflaged_object_detection_lesson` | `tracks/vision/lesson_29_synthetic_camouflaged_object_detection/`, `tests/test_tracks_vision_camouflaged_object_detection.py` | `README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_camouflaged_object_detection.py`<br>`python scripts/run_lesson.py vision lesson_29_synthetic_camouflaged_object_detection --dry-run` |
| L09 | `batch220-topic-modeling` | `.worktrees/batch220-topic-modeling` | `tracks/nlp` | `topic_modeling_lesson` | `tracks/nlp/lesson_18_compact_topic_modeling/`, `tests/test_tracks_nlp_topic_modeling.py` | `README.md`, `tracks/nlp/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_nlp_topic_modeling.py`<br>`python scripts/run_lesson.py nlp lesson_18_compact_topic_modeling --dry-run` |
| L10 | `batch220-distilled-text-classifier` | `.worktrees/batch220-distilled-text-classifier` | `tracks/nlp` | `distilled_text_classifier_lesson` | `tracks/nlp/lesson_19_compact_distilled_text_classifier/`, `tests/test_tracks_nlp_distilled_text_classifier.py` | `README.md`, `tracks/nlp/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_nlp_distilled_text_classifier.py`<br>`python scripts/run_lesson.py nlp lesson_19_compact_distilled_text_classifier --dry-run` |

## Integration Branch Ownership

`batch220-integration` owns only the shared surfaces required after the 10 lane merges:

- `README.md`
- `tracks/llm/README.md`
- `tracks/generative/README.md`
- `tracks/multimodal/README.md`
- `tracks/vision/README.md`
- `tracks/nlp/README.md`
- `tests/test_scripts_run_lesson.py`
- any shared verification-only updates required to make the merged track batch pass on `main`

## Merge Order

1. `batch220-transformer-interpretability`
2. `batch220-tool-calling-agent`
3. `batch220-text-to-image`
4. `batch220-diffusion-inpainting`
5. `batch220-scene-text-vlm-recognition`
6. `batch220-document-vlm-reasoning`
7. `batch220-salient-object-detection`
8. `batch220-camouflaged-object-detection`
9. `batch220-topic-modeling`
10. `batch220-distilled-text-classifier`

## Exit Criteria

- each lane has at least one local commit on its own branch
- each lane passes its manifest smoke commands from its own worktree
- integration updates the top-level lesson counts and track lesson tables for all 10 new lessons
- integration branch passes fresh shared verification before merge back to `main`
