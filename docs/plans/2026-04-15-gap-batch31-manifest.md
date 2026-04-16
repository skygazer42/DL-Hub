# Batch 31 Track-Fill Manifest

Batch 31 starts immediately after the Batch 30 merge-back and keeps the fixed `10 worktree + 10
lane` loop moving. This round continues the prompt-control arc on the LLM track with citation and
schema constraints, extends the generative control block into reference editing and layout-safe
editing, pushes the face-centric multimodal arc into alignment and detection reasoning, broadens
the face vision block into retrieval and pose estimation, and keeps the NLP track on compact dialog
state-decision modeling.

Selection rules locked for this batch:

- fixed `10 worktree + 10 lane` execution
- track-first coverage across `tracks/llm`, `tracks/generative`, `tracks/multimodal`,
  `tracks/vision`, and `tracks/nlp`
- max 2 lanes per surface
- nearest teaching-neighbor fill after the current end of each track

Start point:

- base branch: `main`
- branch anchor commit: `docs: mark batch30 gaps merged`
- integration branch: `batch310-integration`
- worktree root: `.worktrees/`

## Lane Table

| lane_id | branch | worktree | surface | slug | lane_owned_paths | integration_owned_paths | verification_cmds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L01 | `batch310-citation-grounded-prompting` | `.worktrees/batch310-citation-grounded-prompting` | `tracks/llm` | `citation_grounded_prompting_llm_lesson` | `tracks/llm/lesson_30_toy_citation_grounded_prompting/`, `tests/test_tracks_llm_citation_grounded_prompting.py` | `README.md`, `tracks/llm/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_llm_citation_grounded_prompting.py`<br>`python scripts/run_lesson.py llm lesson_30_toy_citation_grounded_prompting --dry-run` |
| L02 | `batch310-schema-constrained-prompting` | `.worktrees/batch310-schema-constrained-prompting` | `tracks/llm` | `schema_constrained_prompting_llm_lesson` | `tracks/llm/lesson_31_toy_schema_constrained_prompting/`, `tests/test_tracks_llm_schema_constrained_prompting.py` | `README.md`, `tracks/llm/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_llm_schema_constrained_prompting.py`<br>`python scripts/run_lesson.py llm lesson_31_toy_schema_constrained_prompting --dry-run` |
| L03 | `batch310-reference-editing` | `.worktrees/batch310-reference-editing` | `tracks/generative` | `reference_editing_generative_lesson` | `tracks/generative/lesson_31_toy_diffusion_reference_editing/`, `tests/test_tracks_generative_reference_editing.py` | `README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_generative_reference_editing.py`<br>`python scripts/run_lesson.py generative lesson_31_toy_diffusion_reference_editing --dry-run` |
| L04 | `batch310-layout-preserving-editing` | `.worktrees/batch310-layout-preserving-editing` | `tracks/generative` | `layout_preserving_editing_generative_lesson` | `tracks/generative/lesson_32_toy_diffusion_layout_preserving_editing/`, `tests/test_tracks_generative_layout_preserving_editing.py` | `README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_generative_layout_preserving_editing.py`<br>`python scripts/run_lesson.py generative lesson_32_toy_diffusion_layout_preserving_editing --dry-run` |
| L05 | `batch310-face-alignment-vlm` | `.worktrees/batch310-face-alignment-vlm` | `tracks/multimodal` | `face_alignment_multimodal_lesson` | `tracks/multimodal/lesson_45_face_alignment_vlm_reasoning/`, `tests/test_tracks_multimodal_face_alignment_reasoning.py` | `README.md`, `tracks/multimodal/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_multimodal_face_alignment_reasoning.py`<br>`python scripts/run_lesson.py multimodal lesson_45_face_alignment_vlm_reasoning --dry-run` |
| L06 | `batch310-face-detection-vlm` | `.worktrees/batch310-face-detection-vlm` | `tracks/multimodal` | `face_detection_multimodal_lesson` | `tracks/multimodal/lesson_46_face_detection_vlm_reasoning/`, `tests/test_tracks_multimodal_face_detection_reasoning.py` | `README.md`, `tracks/multimodal/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_multimodal_face_detection_reasoning.py`<br>`python scripts/run_lesson.py multimodal lesson_46_face_detection_vlm_reasoning --dry-run` |
| L07 | `batch310-face-retrieval-vision` | `.worktrees/batch310-face-retrieval-vision` | `tracks/vision` | `face_retrieval_vision_lesson` | `tracks/vision/lesson_46_synthetic_face_retrieval/`, `tests/test_tracks_vision_face_retrieval.py` | `README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_face_retrieval.py`<br>`python scripts/run_lesson.py vision lesson_46_synthetic_face_retrieval --dry-run` |
| L08 | `batch310-face-pose-vision` | `.worktrees/batch310-face-pose-vision` | `tracks/vision` | `face_pose_vision_lesson` | `tracks/vision/lesson_47_synthetic_face_pose_estimation/`, `tests/test_tracks_vision_face_pose_estimation.py` | `README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_face_pose_estimation.py`<br>`python scripts/run_lesson.py vision lesson_47_synthetic_face_pose_estimation --dry-run` |
| L09 | `batch310-dialog-slot-prediction` | `.worktrees/batch310-dialog-slot-prediction` | `tracks/nlp` | `dialog_slot_prediction_nlp_lesson` | `tracks/nlp/lesson_36_toy_dialog_slot_prediction/`, `tests/test_tracks_nlp_dialog_slot_prediction.py` | `README.md`, `tracks/nlp/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_nlp_dialog_slot_prediction.py`<br>`python scripts/run_lesson.py nlp lesson_36_toy_dialog_slot_prediction --dry-run` |
| L10 | `batch310-dialog-outcome-prediction` | `.worktrees/batch310-dialog-outcome-prediction` | `tracks/nlp` | `dialog_outcome_prediction_nlp_lesson` | `tracks/nlp/lesson_37_toy_dialog_outcome_prediction/`, `tests/test_tracks_nlp_dialog_outcome_prediction.py` | `README.md`, `tracks/nlp/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_nlp_dialog_outcome_prediction.py`<br>`python scripts/run_lesson.py nlp lesson_37_toy_dialog_outcome_prediction --dry-run` |

## Integration Branch Ownership

`batch310-integration` owns only the shared surfaces required after the 10 lane merges:

- `README.md`
- `tracks/llm/README.md`
- `tracks/generative/README.md`
- `tracks/multimodal/README.md`
- `tracks/vision/README.md`
- `tracks/nlp/README.md`
- `tests/test_scripts_run_lesson.py`
- any shared verification-only updates required to make the merged track batch pass on `main`

## Merge Order

1. `batch310-citation-grounded-prompting`
2. `batch310-schema-constrained-prompting`
3. `batch310-reference-editing`
4. `batch310-layout-preserving-editing`
5. `batch310-face-alignment-vlm`
6. `batch310-face-detection-vlm`
7. `batch310-face-retrieval-vision`
8. `batch310-face-pose-vision`
9. `batch310-dialog-slot-prediction`
10. `batch310-dialog-outcome-prediction`

## Exit Criteria

- each lane has at least one local commit on its own branch
- each lane passes its manifest smoke commands from its own worktree
- integration updates the top-level lesson counts and track lesson tables for all 10 new lessons
- integration branch passes fresh shared verification before merge back to `main`
