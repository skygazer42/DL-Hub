# Batch 32 Track-Fill Manifest

Batch 32 starts immediately after the Batch 31 merge-back and keeps the fixed `10 worktree + 10
lane` loop moving. This round continues the LLM structured-output arc with JSON and
function-signature control, extends the generative editing block into masked reference editing and
layout-reference fusion, pushes the face-centric multimodal arc into retrieval and pose reasoning,
shifts the vision continuation into gaze and human-pose regression, and keeps the NLP track inside
compact dialog-state decision modeling.

Selection rules locked for this batch:

- fixed `10 worktree + 10 lane` execution
- track-first coverage across `tracks/llm`, `tracks/generative`, `tracks/multimodal`,
  `tracks/vision`, and `tracks/nlp`
- max 2 lanes per surface
- nearest teaching-neighbor fill after the current end of each track

Start point:

- base branch: `main`
- branch anchor commit: `docs: mark batch31 gaps merged`
- integration branch: `batch320-integration`
- worktree root: `.worktrees/`

## Lane Table

| lane_id | branch | worktree | surface | slug | lane_owned_paths | integration_owned_paths | verification_cmds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L01 | `batch320-json-constrained-prompting` | `.worktrees/batch320-json-constrained-prompting` | `tracks/llm` | `json_constrained_prompting_llm_lesson` | `tracks/llm/lesson_32_toy_json_constrained_prompting/`, `tests/test_tracks_llm_json_constrained_prompting.py` | `README.md`, `tracks/llm/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_llm_json_constrained_prompting.py`<br>`python scripts/run_lesson.py llm lesson_32_toy_json_constrained_prompting --dry-run` |
| L02 | `batch320-function-signature-prompting` | `.worktrees/batch320-function-signature-prompting` | `tracks/llm` | `function_signature_prompting_llm_lesson` | `tracks/llm/lesson_33_toy_function_signature_prompting/`, `tests/test_tracks_llm_function_signature_prompting.py` | `README.md`, `tracks/llm/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_llm_function_signature_prompting.py`<br>`python scripts/run_lesson.py llm lesson_33_toy_function_signature_prompting --dry-run` |
| L03 | `batch320-masked-reference-editing` | `.worktrees/batch320-masked-reference-editing` | `tracks/generative` | `masked_reference_editing_generative_lesson` | `tracks/generative/lesson_33_toy_diffusion_masked_reference_editing/`, `tests/test_tracks_generative_masked_reference_editing.py` | `README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_generative_masked_reference_editing.py`<br>`python scripts/run_lesson.py generative lesson_33_toy_diffusion_masked_reference_editing --dry-run` |
| L04 | `batch320-layout-reference-fusion` | `.worktrees/batch320-layout-reference-fusion` | `tracks/generative` | `layout_reference_fusion_generative_lesson` | `tracks/generative/lesson_34_toy_diffusion_layout_reference_fusion/`, `tests/test_tracks_generative_layout_reference_fusion.py` | `README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_generative_layout_reference_fusion.py`<br>`python scripts/run_lesson.py generative lesson_34_toy_diffusion_layout_reference_fusion --dry-run` |
| L05 | `batch320-face-retrieval-vlm` | `.worktrees/batch320-face-retrieval-vlm` | `tracks/multimodal` | `face_retrieval_multimodal_lesson` | `tracks/multimodal/lesson_47_face_retrieval_vlm_reasoning/`, `tests/test_tracks_multimodal_face_retrieval_reasoning.py` | `README.md`, `tracks/multimodal/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_multimodal_face_retrieval_reasoning.py`<br>`python scripts/run_lesson.py multimodal lesson_47_face_retrieval_vlm_reasoning --dry-run` |
| L06 | `batch320-face-pose-vlm` | `.worktrees/batch320-face-pose-vlm` | `tracks/multimodal` | `face_pose_multimodal_lesson` | `tracks/multimodal/lesson_48_face_pose_vlm_reasoning/`, `tests/test_tracks_multimodal_face_pose_reasoning.py` | `README.md`, `tracks/multimodal/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_multimodal_face_pose_reasoning.py`<br>`python scripts/run_lesson.py multimodal lesson_48_face_pose_vlm_reasoning --dry-run` |
| L07 | `batch320-gaze-estimation-vision` | `.worktrees/batch320-gaze-estimation-vision` | `tracks/vision` | `gaze_estimation_vision_lesson` | `tracks/vision/lesson_48_synthetic_gaze_estimation/`, `tests/test_tracks_vision_gaze_estimation.py` | `README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_gaze_estimation.py`<br>`python scripts/run_lesson.py vision lesson_48_synthetic_gaze_estimation --dry-run` |
| L08 | `batch320-human-pose-vision` | `.worktrees/batch320-human-pose-vision` | `tracks/vision` | `human_pose_estimation_vision_lesson` | `tracks/vision/lesson_49_synthetic_human_pose_estimation/`, `tests/test_tracks_vision_human_pose_estimation.py` | `README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_human_pose_estimation.py`<br>`python scripts/run_lesson.py vision lesson_49_synthetic_human_pose_estimation --dry-run` |
| L09 | `batch320-dialog-satisfaction-prediction` | `.worktrees/batch320-dialog-satisfaction-prediction` | `tracks/nlp` | `dialog_satisfaction_prediction_nlp_lesson` | `tracks/nlp/lesson_38_toy_dialog_satisfaction_prediction/`, `tests/test_tracks_nlp_dialog_satisfaction_prediction.py` | `README.md`, `tracks/nlp/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_nlp_dialog_satisfaction_prediction.py`<br>`python scripts/run_lesson.py nlp lesson_38_toy_dialog_satisfaction_prediction --dry-run` |
| L10 | `batch320-dialog-escalation-risk` | `.worktrees/batch320-dialog-escalation-risk` | `tracks/nlp` | `dialog_escalation_risk_nlp_lesson` | `tracks/nlp/lesson_39_toy_dialog_escalation_risk_prediction/`, `tests/test_tracks_nlp_dialog_escalation_risk_prediction.py` | `README.md`, `tracks/nlp/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_nlp_dialog_escalation_risk_prediction.py`<br>`python scripts/run_lesson.py nlp lesson_39_toy_dialog_escalation_risk_prediction --dry-run` |

## Integration Branch Ownership

`batch320-integration` owns only the shared surfaces required after the 10 lane merges:

- `README.md`
- `tracks/llm/README.md`
- `tracks/generative/README.md`
- `tracks/multimodal/README.md`
- `tracks/vision/README.md`
- `tracks/nlp/README.md`
- `tests/test_scripts_run_lesson.py`
- any shared verification-only updates required to make the merged track batch pass on `main`

## Merge Order

1. `batch320-json-constrained-prompting`
2. `batch320-function-signature-prompting`
3. `batch320-masked-reference-editing`
4. `batch320-layout-reference-fusion`
5. `batch320-face-retrieval-vlm`
6. `batch320-face-pose-vlm`
7. `batch320-gaze-estimation-vision`
8. `batch320-human-pose-vision`
9. `batch320-dialog-satisfaction-prediction`
10. `batch320-dialog-escalation-risk`

## Exit Criteria

- each lane has at least one local commit on its own branch
- each lane passes its manifest smoke commands from its own worktree
- integration updates the top-level lesson counts and track lesson tables for all 10 new lessons
- integration branch passes fresh shared verification before merge back to `main`
