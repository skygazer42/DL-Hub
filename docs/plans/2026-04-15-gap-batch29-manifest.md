# Batch 29 Track-Fill Manifest

Batch 29 keeps the fixed `10 worktree + 10 lane` loop moving immediately after the Batch 28 merge
back. This round continues prompt-side control on the LLM track, pushes the generative track
toward reference-conditioned creation, extends the face-centric multimodal and vision blocks, and
keeps the NLP track on task-oriented dialog continuation.

Selection rules locked for this batch:

- fixed `10 worktree + 10 lane` execution
- track-first coverage across `tracks/llm`, `tracks/generative`, `tracks/multimodal`,
  `tracks/vision`, and `tracks/nlp`
- max 2 lanes per surface
- nearest teaching-neighbor fill after the current end of each track

Start point:

- base branch: `main`
- branch anchor commit: `plan: seed batch29 face-dialog continuation manifest`
- integration branch: `batch290-integration`
- worktree root: `.worktrees/`

## Lane Table

| lane_id | branch | worktree | surface | slug | lane_owned_paths | integration_owned_paths | verification_cmds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L01 | `batch290-process-supervision-prompting` | `.worktrees/batch290-process-supervision-prompting` | `tracks/llm` | `process_supervision_prompting_llm_lesson` | `tracks/llm/lesson_26_toy_process_supervision_prompting/`, `tests/test_tracks_llm_process_supervision_prompting.py` | `README.md`, `tracks/llm/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_llm_process_supervision_prompting.py`<br>`python scripts/run_lesson.py llm lesson_26_toy_process_supervision_prompting --dry-run` |
| L02 | `batch290-self-correction-prompting` | `.worktrees/batch290-self-correction-prompting` | `tracks/llm` | `self_correction_prompting_llm_lesson` | `tracks/llm/lesson_27_toy_self_correction_prompting/`, `tests/test_tracks_llm_self_correction_prompting.py` | `README.md`, `tracks/llm/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_llm_self_correction_prompting.py`<br>`python scripts/run_lesson.py llm lesson_27_toy_self_correction_prompting --dry-run` |
| L03 | `batch290-reference-guided-generation` | `.worktrees/batch290-reference-guided-generation` | `tracks/generative` | `reference_guided_generation_generative_lesson` | `tracks/generative/lesson_27_toy_diffusion_reference_guided_generation/`, `tests/test_tracks_generative_reference_guided_generation.py` | `README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_generative_reference_guided_generation.py`<br>`python scripts/run_lesson.py generative lesson_27_toy_diffusion_reference_guided_generation --dry-run` |
| L04 | `batch290-subject-driven-generation` | `.worktrees/batch290-subject-driven-generation` | `tracks/generative` | `subject_driven_generation_generative_lesson` | `tracks/generative/lesson_28_toy_diffusion_subject_driven_generation/`, `tests/test_tracks_generative_subject_driven_generation.py` | `README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_generative_subject_driven_generation.py`<br>`python scripts/run_lesson.py generative lesson_28_toy_diffusion_subject_driven_generation --dry-run` |
| L05 | `batch290-face-occlusion-vlm` | `.worktrees/batch290-face-occlusion-vlm` | `tracks/multimodal` | `face_occlusion_multimodal_lesson` | `tracks/multimodal/lesson_41_face_occlusion_vlm_reasoning/`, `tests/test_tracks_multimodal_face_occlusion.py` | `README.md`, `tracks/multimodal/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_multimodal_face_occlusion.py`<br>`python scripts/run_lesson.py multimodal lesson_41_face_occlusion_vlm_reasoning --dry-run` |
| L06 | `batch290-face-region-grounding-vlm` | `.worktrees/batch290-face-region-grounding-vlm` | `tracks/multimodal` | `face_region_grounding_multimodal_lesson` | `tracks/multimodal/lesson_42_face_region_grounding_vlm/`, `tests/test_tracks_multimodal_face_region_grounding.py` | `README.md`, `tracks/multimodal/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_multimodal_face_region_grounding.py`<br>`python scripts/run_lesson.py multimodal lesson_42_face_region_grounding_vlm --dry-run` |
| L07 | `batch290-face-expression-vision` | `.worktrees/batch290-face-expression-vision` | `tracks/vision` | `face_expression_vision_lesson` | `tracks/vision/lesson_42_synthetic_face_expression_recognition/`, `tests/test_tracks_vision_face_expression_recognition.py` | `README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_face_expression_recognition.py`<br>`python scripts/run_lesson.py vision lesson_42_synthetic_face_expression_recognition --dry-run` |
| L08 | `batch290-deepfake-detection-vision` | `.worktrees/batch290-deepfake-detection-vision` | `tracks/vision` | `deepfake_detection_vision_lesson` | `tracks/vision/lesson_43_synthetic_deepfake_detection/`, `tests/test_tracks_vision_deepfake_detection.py` | `README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_deepfake_detection.py`<br>`python scripts/run_lesson.py vision lesson_43_synthetic_deepfake_detection --dry-run` |
| L09 | `batch290-dialog-act-prediction` | `.worktrees/batch290-dialog-act-prediction` | `tracks/nlp` | `dialog_act_prediction_nlp_lesson` | `tracks/nlp/lesson_32_toy_dialog_act_prediction/`, `tests/test_tracks_nlp_dialog_act_prediction.py` | `README.md`, `tracks/nlp/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_nlp_dialog_act_prediction.py`<br>`python scripts/run_lesson.py nlp lesson_32_toy_dialog_act_prediction --dry-run` |
| L10 | `batch290-dialog-intent-prediction` | `.worktrees/batch290-dialog-intent-prediction` | `tracks/nlp` | `dialog_intent_prediction_nlp_lesson` | `tracks/nlp/lesson_33_toy_dialog_intent_prediction/`, `tests/test_tracks_nlp_dialog_intent_prediction.py` | `README.md`, `tracks/nlp/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_nlp_dialog_intent_prediction.py`<br>`python scripts/run_lesson.py nlp lesson_33_toy_dialog_intent_prediction --dry-run` |

## Integration Branch Ownership

`batch290-integration` owns only the shared surfaces required after the 10 lane merges:

- `README.md`
- `tracks/llm/README.md`
- `tracks/generative/README.md`
- `tracks/multimodal/README.md`
- `tracks/vision/README.md`
- `tracks/nlp/README.md`
- `tests/test_scripts_run_lesson.py`
- any shared verification-only updates required to make the merged track batch pass on `main`

## Merge Order

1. `batch290-process-supervision-prompting`
2. `batch290-self-correction-prompting`
3. `batch290-reference-guided-generation`
4. `batch290-subject-driven-generation`
5. `batch290-face-occlusion-vlm`
6. `batch290-face-region-grounding-vlm`
7. `batch290-face-expression-vision`
8. `batch290-deepfake-detection-vision`
9. `batch290-dialog-act-prediction`
10. `batch290-dialog-intent-prediction`

## Exit Criteria

- each lane has at least one local commit on its own branch
- each lane passes its manifest smoke commands from its own worktree
- integration updates the top-level lesson counts and track lesson tables for all 10 new lessons
- integration branch passes fresh shared verification before merge back to `main`
