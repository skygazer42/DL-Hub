# Batch 28 Track-Fill Manifest

Batch 28 continues the fixed `10 worktree + 10 lane` loop immediately after the Batch 27 merge
back. This round extends the prompting block on the LLM track, pushes the generative track toward
composition and variation, keeps the multimodal and vision tracks on face-centric continuations,
and advances the NLP dialog block beyond state tracking.

Selection rules locked for this batch:

- fixed `10 worktree + 10 lane` execution
- track-first coverage across `tracks/llm`, `tracks/generative`, `tracks/multimodal`,
  `tracks/vision`, and `tracks/nlp`
- max 2 lanes per surface
- nearest teaching-neighbor fill after the current end of each track

Start point:

- base branch: `main`
- branch anchor commit: `plan: seed batch28 dialog-and-face track-fill manifest`
- integration branch: `batch280-integration`
- worktree root: `.worktrees/`

## Lane Table

| lane_id | branch | worktree | surface | slug | lane_owned_paths | integration_owned_paths | verification_cmds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L01 | `batch280-debate-prompting` | `.worktrees/batch280-debate-prompting` | `tracks/llm` | `debate_prompting_llm_lesson` | `tracks/llm/lesson_24_toy_debate_prompting/`, `tests/test_tracks_llm_debate_prompting.py` | `README.md`, `tracks/llm/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_llm_debate_prompting.py`<br>`python scripts/run_lesson.py llm lesson_24_toy_debate_prompting --dry-run` |
| L02 | `batch280-verifier-guided-prompting` | `.worktrees/batch280-verifier-guided-prompting` | `tracks/llm` | `verifier_guided_prompting_llm_lesson` | `tracks/llm/lesson_25_toy_verifier_guided_prompting/`, `tests/test_tracks_llm_verifier_guided_prompting.py` | `README.md`, `tracks/llm/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_llm_verifier_guided_prompting.py`<br>`python scripts/run_lesson.py llm lesson_25_toy_verifier_guided_prompting --dry-run` |
| L03 | `batch280-compositional-generation` | `.worktrees/batch280-compositional-generation` | `tracks/generative` | `compositional_generation_generative_lesson` | `tracks/generative/lesson_25_toy_diffusion_compositional_generation/`, `tests/test_tracks_generative_compositional_generation.py` | `README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_generative_compositional_generation.py`<br>`python scripts/run_lesson.py generative lesson_25_toy_diffusion_compositional_generation --dry-run` |
| L04 | `batch280-image-variation` | `.worktrees/batch280-image-variation` | `tracks/generative` | `image_variation_generative_lesson` | `tracks/generative/lesson_26_toy_diffusion_image_variation/`, `tests/test_tracks_generative_image_variation.py` | `README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_generative_image_variation.py`<br>`python scripts/run_lesson.py generative lesson_26_toy_diffusion_image_variation --dry-run` |
| L05 | `batch280-face-attribute-vlm` | `.worktrees/batch280-face-attribute-vlm` | `tracks/multimodal` | `face_attribute_multimodal_lesson` | `tracks/multimodal/lesson_39_face_attribute_vlm_reasoning/`, `tests/test_tracks_multimodal_face_attribute.py` | `README.md`, `tracks/multimodal/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_multimodal_face_attribute.py`<br>`python scripts/run_lesson.py multimodal lesson_39_face_attribute_vlm_reasoning --dry-run` |
| L06 | `batch280-face-caption-vlm` | `.worktrees/batch280-face-caption-vlm` | `tracks/multimodal` | `face_caption_multimodal_lesson` | `tracks/multimodal/lesson_40_face_caption_vlm_grounding/`, `tests/test_tracks_multimodal_face_caption.py` | `README.md`, `tracks/multimodal/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_multimodal_face_caption.py`<br>`python scripts/run_lesson.py multimodal lesson_40_face_caption_vlm_grounding --dry-run` |
| L07 | `batch280-face-attribute-vision` | `.worktrees/batch280-face-attribute-vision` | `tracks/vision` | `face_attribute_vision_lesson` | `tracks/vision/lesson_40_synthetic_face_attribute_recognition/`, `tests/test_tracks_vision_face_attribute_recognition.py` | `README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_face_attribute_recognition.py`<br>`python scripts/run_lesson.py vision lesson_40_synthetic_face_attribute_recognition --dry-run` |
| L08 | `batch280-face-occlusion-vision` | `.worktrees/batch280-face-occlusion-vision` | `tracks/vision` | `face_occlusion_vision_lesson` | `tracks/vision/lesson_41_synthetic_face_occlusion_estimation/`, `tests/test_tracks_vision_face_occlusion_estimation.py` | `README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_face_occlusion_estimation.py`<br>`python scripts/run_lesson.py vision lesson_41_synthetic_face_occlusion_estimation --dry-run` |
| L09 | `batch280-dialog-response-selection` | `.worktrees/batch280-dialog-response-selection` | `tracks/nlp` | `dialog_response_selection_nlp_lesson` | `tracks/nlp/lesson_30_toy_dialog_response_selection/`, `tests/test_tracks_nlp_dialog_response_selection.py` | `README.md`, `tracks/nlp/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_nlp_dialog_response_selection.py`<br>`python scripts/run_lesson.py nlp lesson_30_toy_dialog_response_selection --dry-run` |
| L10 | `batch280-slot-carryover` | `.worktrees/batch280-slot-carryover` | `tracks/nlp` | `slot_carryover_nlp_lesson` | `tracks/nlp/lesson_31_toy_slot_carryover_prediction/`, `tests/test_tracks_nlp_slot_carryover.py` | `README.md`, `tracks/nlp/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_nlp_slot_carryover.py`<br>`python scripts/run_lesson.py nlp lesson_31_toy_slot_carryover_prediction --dry-run` |

## Integration Branch Ownership

`batch280-integration` owns only the shared surfaces required after the 10 lane merges:

- `README.md`
- `tracks/llm/README.md`
- `tracks/generative/README.md`
- `tracks/multimodal/README.md`
- `tracks/vision/README.md`
- `tracks/nlp/README.md`
- `tests/test_scripts_run_lesson.py`
- any shared verification-only updates required to make the merged track batch pass on `main`

## Merge Order

1. `batch280-debate-prompting`
2. `batch280-verifier-guided-prompting`
3. `batch280-compositional-generation`
4. `batch280-image-variation`
5. `batch280-face-attribute-vlm`
6. `batch280-face-caption-vlm`
7. `batch280-face-attribute-vision`
8. `batch280-face-occlusion-vision`
9. `batch280-dialog-response-selection`
10. `batch280-slot-carryover`

## Exit Criteria

- each lane has at least one local commit on its own branch
- each lane passes its manifest smoke commands from its own worktree
- integration updates the top-level lesson counts and track lesson tables for all 10 new lessons
- integration branch passes fresh shared verification before merge back to `main`
