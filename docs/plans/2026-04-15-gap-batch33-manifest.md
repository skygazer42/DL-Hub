# Batch 33 Track-Fill Manifest

Batch 33 starts immediately after the Batch 32 merge-back and keeps the fixed `10 worktree + 10
lane` loop moving. This round continues the LLM structured-output arc with XML and regex
constraints, pushes the generative editing block into box-mask editing and layout-subject fusion,
extends the multimodal face/body reasoning line into gaze and person-pose queries, shifts the
vision continuation into hand-pose and gesture recognition, and keeps the NLP track inside compact
dialog triage decision modeling.

Selection rules locked for this batch:

- fixed `10 worktree + 10 lane` execution
- track-first coverage across `tracks/llm`, `tracks/generative`, `tracks/multimodal`,
  `tracks/vision`, and `tracks/nlp`
- max 2 lanes per surface
- nearest teaching-neighbor fill after the current end of each track

Start point:

- base branch: `main`
- branch anchor commit: `docs: mark batch32 gaps merged`
- integration branch: `batch330-integration`
- worktree root: `.worktrees/`

## Lane Table

| lane_id | branch | worktree | surface | slug | lane_owned_paths | integration_owned_paths | verification_cmds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L01 | `batch330-xml-constrained-prompting` | `.worktrees/batch330-xml-constrained-prompting` | `tracks/llm` | `xml_constrained_prompting_llm_lesson` | `tracks/llm/lesson_34_toy_xml_constrained_prompting/`, `tests/test_tracks_llm_xml_constrained_prompting.py` | `README.md`, `tracks/llm/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_llm_xml_constrained_prompting.py`<br>`python scripts/run_lesson.py llm lesson_34_toy_xml_constrained_prompting --dry-run` |
| L02 | `batch330-regex-constrained-prompting` | `.worktrees/batch330-regex-constrained-prompting` | `tracks/llm` | `regex_constrained_prompting_llm_lesson` | `tracks/llm/lesson_35_toy_regex_constrained_prompting/`, `tests/test_tracks_llm_regex_constrained_prompting.py` | `README.md`, `tracks/llm/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_llm_regex_constrained_prompting.py`<br>`python scripts/run_lesson.py llm lesson_35_toy_regex_constrained_prompting --dry-run` |
| L03 | `batch330-box-mask-editing` | `.worktrees/batch330-box-mask-editing` | `tracks/generative` | `box_mask_editing_generative_lesson` | `tracks/generative/lesson_35_toy_diffusion_box_mask_editing/`, `tests/test_tracks_generative_box_mask_editing.py` | `README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_generative_box_mask_editing.py`<br>`python scripts/run_lesson.py generative lesson_35_toy_diffusion_box_mask_editing --dry-run` |
| L04 | `batch330-layout-subject-fusion` | `.worktrees/batch330-layout-subject-fusion` | `tracks/generative` | `layout_subject_fusion_generative_lesson` | `tracks/generative/lesson_36_toy_diffusion_layout_subject_fusion/`, `tests/test_tracks_generative_layout_subject_fusion.py` | `README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_generative_layout_subject_fusion.py`<br>`python scripts/run_lesson.py generative lesson_36_toy_diffusion_layout_subject_fusion --dry-run` |
| L05 | `batch330-face-gaze-vlm` | `.worktrees/batch330-face-gaze-vlm` | `tracks/multimodal` | `face_gaze_multimodal_lesson` | `tracks/multimodal/lesson_49_face_gaze_vlm_reasoning/`, `tests/test_tracks_multimodal_face_gaze_reasoning.py` | `README.md`, `tracks/multimodal/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_multimodal_face_gaze_reasoning.py`<br>`python scripts/run_lesson.py multimodal lesson_49_face_gaze_vlm_reasoning --dry-run` |
| L06 | `batch330-person-pose-vlm` | `.worktrees/batch330-person-pose-vlm` | `tracks/multimodal` | `person_pose_multimodal_lesson` | `tracks/multimodal/lesson_50_person_pose_vlm_reasoning/`, `tests/test_tracks_multimodal_person_pose_reasoning.py` | `README.md`, `tracks/multimodal/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_multimodal_person_pose_reasoning.py`<br>`python scripts/run_lesson.py multimodal lesson_50_person_pose_vlm_reasoning --dry-run` |
| L07 | `batch330-hand-pose-vision` | `.worktrees/batch330-hand-pose-vision` | `tracks/vision` | `hand_pose_estimation_vision_lesson` | `tracks/vision/lesson_50_synthetic_hand_pose_estimation/`, `tests/test_tracks_vision_hand_pose_estimation.py` | `README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_hand_pose_estimation.py`<br>`python scripts/run_lesson.py vision lesson_50_synthetic_hand_pose_estimation --dry-run` |
| L08 | `batch330-gesture-recognition-vision` | `.worktrees/batch330-gesture-recognition-vision` | `tracks/vision` | `gesture_recognition_vision_lesson` | `tracks/vision/lesson_51_synthetic_gesture_recognition/`, `tests/test_tracks_vision_gesture_recognition.py` | `README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_gesture_recognition.py`<br>`python scripts/run_lesson.py vision lesson_51_synthetic_gesture_recognition --dry-run` |
| L09 | `batch330-dialog-priority-prediction` | `.worktrees/batch330-dialog-priority-prediction` | `tracks/nlp` | `dialog_priority_prediction_nlp_lesson` | `tracks/nlp/lesson_40_toy_dialog_priority_prediction/`, `tests/test_tracks_nlp_dialog_priority_prediction.py` | `README.md`, `tracks/nlp/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_nlp_dialog_priority_prediction.py`<br>`python scripts/run_lesson.py nlp lesson_40_toy_dialog_priority_prediction --dry-run` |
| L10 | `batch330-dialog-transfer-prediction` | `.worktrees/batch330-dialog-transfer-prediction` | `tracks/nlp` | `dialog_transfer_prediction_nlp_lesson` | `tracks/nlp/lesson_41_toy_dialog_transfer_prediction/`, `tests/test_tracks_nlp_dialog_transfer_prediction.py` | `README.md`, `tracks/nlp/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_nlp_dialog_transfer_prediction.py`<br>`python scripts/run_lesson.py nlp lesson_41_toy_dialog_transfer_prediction --dry-run` |

## Integration Branch Ownership

`batch330-integration` owns only the shared surfaces required after the 10 lane merges:

- `README.md`
- `tracks/llm/README.md`
- `tracks/generative/README.md`
- `tracks/multimodal/README.md`
- `tracks/vision/README.md`
- `tracks/nlp/README.md`
- `tests/test_scripts_run_lesson.py`
- any shared verification-only updates required to make the merged track batch pass on `main`

## Merge Order

1. `batch330-xml-constrained-prompting`
2. `batch330-regex-constrained-prompting`
3. `batch330-box-mask-editing`
4. `batch330-layout-subject-fusion`
5. `batch330-face-gaze-vlm`
6. `batch330-person-pose-vlm`
7. `batch330-hand-pose-vision`
8. `batch330-gesture-recognition-vision`
9. `batch330-dialog-priority-prediction`
10. `batch330-dialog-transfer-prediction`

## Exit Criteria

- each lane has at least one local commit on its own branch
- each lane passes its manifest smoke commands from its own worktree
- integration updates the top-level lesson counts and track lesson tables for all 10 new lessons
- integration branch passes fresh shared verification before merge back to `main`
